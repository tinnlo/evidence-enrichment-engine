"""Chroma-backed vector store for document chunks.

ChromaDB is imported lazily so the module can load without the optional
``chromadb`` dependency installed.

Collection naming convention:
    ``entity_{entity_id}__{model_slug}_v{version}``

where ``model_slug`` is the embedding model name with non-alphanumeric chars
replaced by underscores and ``version`` is the schema version integer.
"""

from __future__ import annotations

import re

from evidence_enrichment.core.retrieval.models import Chunk, RetrievalResult

_SCHEMA_VERSION = 1
_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_]{1,510}[a-zA-Z0-9]$")


def _sanitize(text: str) -> str:
    """Replace non-alphanumeric chars with underscores."""
    return _SAFE_RE.sub("_", text)


def _collection_name(entity_id: str, model: str) -> str:
    model_slug = _sanitize(model)
    entity_slug = _sanitize(entity_id)
    name = f"entity_{entity_slug}__{model_slug}_v{_SCHEMA_VERSION}"
    # Chroma requires collection names 3–512 chars
    name = name[:512]
    if len(name) < 3:
        name = name.ljust(3, "x")
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

    def __init__(self, persist_path: str, embedding_model: str = "text-embedding-3-small") -> None:
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
        """Idempotent upsert of chunks and their embeddings.

        Existing chunks with the same chunk_id are overwritten.
        """
        if not chunks:
            return
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
            }
            for c in chunks
        ]
        collection.upsert(  # type: ignore[union-attr]
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
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
            query_kwargs["where"] = where

        try:
            results = collection.query(**query_kwargs)  # type: ignore[union-attr]
        except Exception:
            return []

        hits: list[RetrievalResult] = []
        if not results or not results.get("ids"):
            return hits

        ids_list = results["ids"][0]
        docs_list = results["documents"][0]
        metas_list = results["metadatas"][0]
        dists_list = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            # Convert cosine distance → similarity (Chroma returns L2 or cosine dist)
            vector_score = max(0.0, 1.0 - dist)
            chunk = Chunk(
                chunk_id=chunk_id,
                document_url=meta.get("document_url", ""),
                content_hash=meta.get("content_hash", ""),
                index=int(meta.get("index", 0)),
                content=doc,
                chunk_type=meta.get("chunk_type", "text"),
                char_count=int(meta.get("char_count", len(doc))),
            )
            hits.append(
                RetrievalResult(
                    chunk=chunk,
                    score=vector_score,
                    vector_score=vector_score,
                )
            )

        return hits

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
