"""Chroma-backed vector store for document chunks.

ChromaDB is imported lazily so the module can load without the optional
``chromadb`` dependency installed.

Collection naming convention:
    ``entity_{entity_id}__{model_slug}_v{version}``

where ``model_slug`` is the embedding model name with non-alphanumeric chars
replaced by underscores and ``version`` is the schema version integer.
"""

from __future__ import annotations

import json
import logging
import re

from evidence_enrichment.core.retrieval.models import Chunk, RetrievalResult

_SCHEMA_VERSION = 1
_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_]{1,510}[a-zA-Z0-9]$")


def _sanitize(text: str) -> str:
    """Replace non-alphanumeric chars with underscores."""
    return _SAFE_RE.sub("_", text)


def _to_chroma_where(where: dict) -> dict:
    """Convert a flat multi-key filter dict to Chroma-compatible ``$and`` form.

    Chroma ≥ 0.6 requires exactly one top-level operator in a ``where`` clause.
    A plain dict like ``{"a": "x", "b": "y"}`` raises
    ``Expected where to have exactly one operator``.  This helper rewrites it to
    ``{"$and": [{"a": {"$eq": "x"}}, {"b": {"$eq": "y"}}]}``.

    Single-key dicts and dicts that already start with ``$`` are returned as-is
    so we don't double-wrap already-valid filters.
    """
    if len(where) <= 1:
        return where
    first_key = next(iter(where))
    if first_key.startswith("$"):
        # Already uses an operator at the top level (e.g. {"$and": [...]})
        return where
    return {"$and": [{k: {"$eq": v}} for k, v in where.items()]}


def _serialise_table(table_data: list[list[str]] | None) -> str:
    """Serialise table_data to a compact JSON string suitable for Chroma metadata."""
    if not table_data:
        return ""
    return json.dumps(table_data, ensure_ascii=False, separators=(",", ":"))


def _deserialise_table(value: str | None) -> list[list[str]] | None:
    """Deserialise table_data from the Chroma metadata JSON string."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
        # Validate structure: must be list[list[str]]
        if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
            return parsed
    except (json.JSONDecodeError, TypeError):
        logging.warning("Failed to deserialise table_data_json: %r", value)
    return None


def _collection_name(entity_id: str, model: str) -> str:
    model_slug = _sanitize(model)
    entity_slug = _sanitize(entity_id)
    name = f"entity_{entity_slug}__{model_slug}_v{_SCHEMA_VERSION}"
    # Chroma requires collection names 3–512 chars, starting and ending
    # with [a-zA-Z0-9].  Truncate then strip any trailing underscores.
    name = name[:512].rstrip("_")
    if len(name) < 3:
        name = (name + "xxx")[:3]
    return name


class ChromaVectorStore:
    """Persistent Chroma vector store with per-entity collections.

    Parameters
    ----------
    persist_path:
        Directory path for Chroma's persistent storage.
    embedding_model:
        Embedding model name (used in collection naming).
    """

    def __init__(
        self, persist_path: str, embedding_model: str = "text-embedding-3-small"
    ) -> None:
        self.persist_path = persist_path
        self.embedding_model = embedding_model
        self._client: object | None = None  # lazy

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import chromadb  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "chromadb package is required for vector storage. "
                    "Install with: pip install 'evidence_enrichment[retrieval]'"
                ) from exc
            self._client = chromadb.PersistentClient(path=self.persist_path)
        return self._client

    def _get_collection(self, entity_id: str) -> object:
        client = self._get_client()
        name = _collection_name(entity_id, self.embedding_model)
        return client.get_or_create_collection(  # type: ignore[union-attr]
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        entity_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks and embeddings.

        Stale chunks for the same ``document_url`` are removed *after* the new
        chunks are written so that a failed upsert never leaves the document
        with zero indexed chunks.  The sequence is:

        1. Write new chunks via ``collection.upsert()`` (idempotent).
        2. For each document_url, query all currently indexed IDs.
        3. Delete only the IDs that are stale (not part of the new write).

        This is still not fully atomic, but a failure at step 1 leaves the old
        data intact, and a failure at step 3 leaves at most a few extra stale
        chunks rather than zero chunks.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have the same length, "
                f"got {len(chunks)} chunks and {len(embeddings)} embeddings"
            )
        collection = self._get_collection(entity_id)

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "document_url": c.document_url,
                "chunk_type": c.chunk_type,
                "index": c.index,
                "char_count": c.char_count,
                "content_hash": c.content_hash,
                # Hierarchical fields — written for new chunks; absent on legacy chunks
                "section_id": c.section_id,
                "section_path_str": c.section_path_str,
                "section_level": c.section_level,
                "chunk_role": c.chunk_role,
                "parent_section_id": c.parent_section_id,
                "token_count": c.token_count,
                # table_data is a nested list — serialise to JSON string for Chroma
                # (Chroma only accepts scalar metadata values).  Empty string = absent.
                **({"table_data_json": _serialise_table(c.table_data)} if c.table_data else {}),
                **({"page": c.page} if c.page is not None else {}),
            }
            for c in chunks
        ]
        # Step 1: write new chunks first so old data is preserved on failure.
        collection.upsert(  # type: ignore[union-attr]
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Step 2 & 3: purge stale IDs per document_url.
        new_ids_set = set(ids)
        for doc_url in {c.document_url for c in chunks}:
            try:
                existing = collection.get(  # type: ignore[union-attr]
                    where={"document_url": doc_url},
                    include=[],
                )
                stale_ids = [
                    i for i in (existing.get("ids") or []) if i not in new_ids_set
                ]
                if stale_ids:
                    collection.delete(ids=stale_ids)  # type: ignore[union-attr]
            except Exception as exc:
                logging.warning(
                    "Failed to purge stale chunks for %s: %s",
                    doc_url,
                    exc,
                )

    def evict_document(self, entity_id: str, document_url: str) -> None:
        """Delete all indexed chunks for *document_url* from the entity collection.

        Called when re-indexing a document produces zero chunks (e.g. the page
        now falls below min_size or returns no parseable content) so that stale
        evidence from the previous index is not left queryable.
        """
        collection = self._get_collection(entity_id)
        try:
            existing = collection.get(  # type: ignore[union-attr]
                where={"document_url": document_url},
                include=[],
            )
            stale_ids = existing.get("ids") or []
            if stale_ids:
                collection.delete(ids=stale_ids)  # type: ignore[union-attr]
        except Exception as exc:
            logging.warning(
                "Failed to evict chunks for %s: %s",
                document_url,
                exc,
            )

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def query(
        self,
        entity_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Query for the top_k most similar chunks.

        Parameters
        ----------
        where:
            Optional Chroma metadata filter dict, e.g.
            ``{"document_url": "https://..."}`` for document-scoped retrieval.
        """
        collection = self._get_collection(entity_id)
        # Over-fetch 2x then let the retriever rerank
        n_results = top_k * 2
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = _to_chroma_where(where)

        try:
            results = collection.query(**query_kwargs)  # type: ignore[union-attr]
        except Exception as exc:
            logging.warning("Chroma query failed for entity %s: %s", entity_id, exc)
            return []

        hits: list[RetrievalResult] = []
        if not results or not results.get("ids"):
            return hits

        ids_list = results["ids"][0]
        docs_list = results["documents"][0]
        metas_list = results["metadatas"][0]
        dists_list = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(
            ids_list, docs_list, metas_list, dists_list
        ):
            # Convert cosine distance → similarity (Chroma returns L2 or cosine dist)
            vector_score = max(0.0, 1.0 - dist)
            # page may be absent from legacy chunks — keep None in that case
            _raw_page = meta.get("page")
            _page: int | None = int(_raw_page) if _raw_page is not None else None
            chunk = Chunk(
                chunk_id=chunk_id,
                document_url=meta.get("document_url", ""),
                content_hash=meta.get("content_hash", ""),
                index=int(meta.get("index", 0)),
                content=doc,
                chunk_type=meta.get("chunk_type", "text"),
                char_count=int(meta.get("char_count", len(doc))),
                # Hierarchical fields — default gracefully for legacy chunks
                section_id=meta.get("section_id", ""),
                section_path_str=meta.get("section_path_str", ""),
                section_level=int(meta.get("section_level", 0)),
                parent_section_id=meta.get("parent_section_id", ""),
                chunk_role=meta.get("chunk_role", "content"),
                page=_page,
                token_count=int(meta.get("token_count", 0)),
                table_data=_deserialise_table(meta.get("table_data_json")),
            )
            hits.append(
                RetrievalResult(
                    chunk=chunk,
                    score=vector_score,
                    vector_score=vector_score,
                )
            )

        return hits

    def get_section_metadata_for_document(
        self,
        entity_id: str,
        document_url: str,
    ) -> list[dict]:
        """Return lightweight section metadata for every chunk of *document_url*.

        Each returned dict contains ``section_id``, ``section_path_str``, and
        ``parent_section_id``.  Callers use this to reconstruct
        ``_descendant_map`` without fetching full chunk content or embeddings
        (a single metadata-only ``collection.get`` is issued).

        Returns an empty list when the collection does not yet exist or the
        document has no indexed chunks.
        """
        collection = self._get_collection(entity_id)
        try:
            result = collection.get(  # type: ignore[union-attr]
                where={"document_url": {"$eq": document_url}},
                include=["metadatas"],
            )
        except Exception as exc:
            logging.warning(
                "get_section_metadata_for_document failed for %s (entity=%s): %s",
                document_url,
                entity_id,
                exc,
            )
            return []

        metas = result.get("metadatas") or []
        out: list[dict] = []
        for m in metas:
            sid = m.get("section_id", "")
            path = m.get("section_path_str", "")
            parent = m.get("parent_section_id", "")
            if sid:
                out.append({"section_id": sid, "section_path_str": path,
                             "parent_section_id": parent})
        return out

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_name_for(self, entity_id: str) -> str:
        """Return the Chroma collection name for an entity (for testing/debugging)."""
        return _collection_name(entity_id, self.embedding_model)

    def delete_collection(self, entity_id: str) -> None:
        """Delete the collection for an entity (primarily for test cleanup)."""
        client = self._get_client()
        name = _collection_name(entity_id, self.embedding_model)
        try:
            client.delete_collection(name)  # type: ignore[union-attr]
        except Exception:
            pass
