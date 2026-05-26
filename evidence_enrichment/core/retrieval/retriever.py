"""Hybrid retriever: vector + keyword + table-type boost.

Scoring formula (configurable weights, defaults from the approved plan):
    score = w_vector * vector_score + w_keyword * keyword_score + w_table * table_boost

where:
    - vector_score:  cosine similarity from Chroma (0–1)
    - keyword_score: simple BM25-inspired term overlap (0–1)
    - table_boost:   0.1 if chunk is a table AND query contains numeric/financial
                     keywords, else 0.0

Retrieval is document-scoped: all queries filter by ``document_url`` to
preserve per-document claim attribution.
"""

from __future__ import annotations

import logging
import math
import re

from evidence_enrichment.core.models.contracts import ParsedDocument
from evidence_enrichment.core.retrieval.chunker import TableAwareChunker
from evidence_enrichment.core.retrieval.embedder import OpenAIEmbedder, PartialEmbedError
from evidence_enrichment.core.retrieval.models import Chunk, RetrievalResult
from evidence_enrichment.core.retrieval.store import ChromaVectorStore

# Keywords that signal a numeric/financial query — triggers table boost
_NUMERIC_KEYWORDS = frozenset(
    [
        "revenue",
        "profit",
        "loss",
        "earnings",
        "ebitda",
        "eps",
        "shares",
        "employees",
        "headcount",
        "staff",
        "salary",
        "wage",
        "cost",
        "price",
        "total",
        "sum",
        "count",
        "number",
        "million",
        "billion",
        "thousand",
        "percent",
        "%",
        "$",
        "€",
        "£",
        "¥",
        "quarterly",
        "annual",
        "fiscal",
        "q1",
        "q2",
        "q3",
        "q4",
        "fy",
        "ytd",
    ]
)


def _keyword_score(query: str, chunk_content: str) -> float:
    """Simple term-overlap keyword score in [0, 1].

    Tokenises both strings on whitespace/punctuation and computes the
    fraction of unique query terms present in the chunk.
    """
    tok = re.compile(r"\w+")
    query_terms = {t.lower() for t in tok.findall(query)}
    chunk_terms = {t.lower() for t in tok.findall(chunk_content)}
    if not query_terms:
        return 0.0
    overlap = query_terms & chunk_terms
    return len(overlap) / len(query_terms)


def _table_boost(query: str, chunk: Chunk) -> float:
    """Return 0.1 if the chunk is a table and the query is numeric-flavoured."""
    if chunk.chunk_type != "table":
        return 0.0
    query_lower = query.lower()
    query_tokens = set(re.findall(r"\w+", query_lower))
    if query_tokens & _NUMERIC_KEYWORDS or any(
        sym in query_lower for sym in ("$", "€", "£", "¥", "%")
    ):
        return 0.1
    return 0.0


class IndexingPartialError(Exception):
    """Raised when embedding succeeded but the vector-store upsert failed.

    Carries the chunks that were embedded (and therefore likely billed) so the
    caller can accrue accurate FinOps cost without over-charging pre-embed
    failures.
    """

    def __init__(self, message: str, embedded_chunks: list["Chunk"]) -> None:
        super().__init__(message)
        self.embedded_chunks: list["Chunk"] = embedded_chunks


class QueryPartialError(Exception):
    """Raised when ``embed_query()`` succeeded but a later retrieval step failed.

    The query embedding was billed before the failure (store.query, reranking,
    etc.) so the caller must accrue the query's char count rather than treating
    the call as free.  Carries ``query_chars`` (length of the embedded query
    string) for accurate FinOps accounting.
    """

    def __init__(self, message: str, query_chars: int) -> None:
        super().__init__(message)
        self.query_chars: int = query_chars


class HybridRetriever:
    """Orchestrate chunk → embed → store → hybrid-rerank retrieval.

    Parameters
    ----------
    entity_id:
        Identifier for the entity being enriched (used for Chroma collection
        namespacing).
    store:
        ChromaVectorStore instance.
    embedder:
        OpenAIEmbedder (or compatible object with embed_texts/embed_query).
    chunker:
        TableAwareChunker.
    top_k:
        Number of results to return after reranking.
    weights:
        Tuple of (w_vector, w_keyword, w_table).  Must sum to 1.0.
    """

    def __init__(
        self,
        entity_id: str,
        store: ChromaVectorStore,
        embedder: OpenAIEmbedder,
        chunker: TableAwareChunker | None = None,
        top_k: int = 5,
        weights: tuple[float, float, float] = (0.7, 0.2, 0.1),
    ) -> None:
        self.entity_id = entity_id
        self.store = store
        self.embedder = embedder
        self.chunker = chunker or TableAwareChunker()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        if len(weights) != 3:
            raise ValueError(
                f"weights must be a 3-tuple (w_vector, w_keyword, w_table), got length {len(weights)}"
            )
        if any(not math.isfinite(w) for w in weights):
            raise ValueError(
                f"Each weight must be a finite number, got weights={weights}"
            )
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError(
                f"Each weight must be in [0.0, 1.0], got weights={weights}"
            )
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Retrieval weights must sum to 1.0, got {weight_sum} (weights={weights})"
            )
        self.w_vector, self.w_keyword, self.w_table = weights

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_document(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk and embed a document, upsert into Chroma, return chunks."""
        chunks = self.chunker.chunk(document)
        if not chunks:
            # Evict any stale chunks from a previous index so that a page
            # that now yields no content doesn't leave old evidence queryable.
            self.store.evict_document(self.entity_id, document.url)
            return []
        texts = [c.content for c in chunks]
        try:
            embeddings = self.embedder.embed_texts(texts)
        except PartialEmbedError as exc:
            # Some batches succeeded and were billed; re-raise as IndexingPartialError
            # carrying only the chunks that were actually embedded so the coordinator
            # can accrue their exact cost.
            raise IndexingPartialError(
                f"embed_texts partial failure for {document.url}: {exc}",
                embedded_chunks=chunks[: exc.completed_count],
            ) from exc
        try:
            self.store.upsert(self.entity_id, chunks, embeddings)
        except Exception as exc:
            raise IndexingPartialError(
                f"upsert failed for {document.url}: {exc}",
                embedded_chunks=chunks,
            ) from exc
        return chunks

    def evict_document(self, document_url: str) -> None:
        """Remove all indexed chunks for *document_url* from the vector store.

        Safe to call when ``index_document`` fails mid-way (e.g. after
        ``embed_texts`` succeeds but ``upsert`` raises) to prevent stale
        vectors from a previous run remaining queryable.
        """
        try:
            self.store.evict_document(self.entity_id, document_url)
        except Exception:
            # Best-effort: log and continue — eviction failure must not mask
            # the original indexing error.
            logging.warning(
                "evict_document failed for %s (entity=%s); stale chunks may remain",
                document_url,
                self.entity_id,
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        document_url: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and rerank chunks from a single document.

        Parameters
        ----------
        query:
            The natural language query (typically the field being enriched).
        document_url:
            Filter retrieval to chunks from this document only.
        top_k:
            Override instance-level top_k.
        """
        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k override must be >= 1, got {top_k}")
            k = top_k
        else:
            k = self.top_k
        query_embedding = self.embedder.embed_query(query)
        if not query_embedding:
            return []

        # embed_query succeeded — any failure from here on should be surfaced
        # as QueryPartialError so callers can bill the embedding spend.
        try:
            # Over-fetch 2x, then rerank
            raw_hits = self.store.query(
                entity_id=self.entity_id,
                query_embedding=query_embedding,
                top_k=k * 2,
                where={"document_url": document_url},
            )

            if not raw_hits:
                return []

            # Apply hybrid scoring
            scored: list[RetrievalResult] = []
            for hit in raw_hits:
                kw = _keyword_score(query, hit.chunk.content)
                tb = _table_boost(query, hit.chunk)
                hybrid = (
                    self.w_vector * hit.vector_score
                    + self.w_keyword * kw
                    + self.w_table * tb
                )
                scored.append(
                    RetrievalResult(
                        chunk=hit.chunk,
                        score=hybrid,
                        vector_score=hit.vector_score,
                        keyword_score=kw,
                        table_boost_score=tb,
                        is_table_boost=tb > 0,
                    )
                )

            # Sort descending by final score, return top_k
            scored.sort(key=lambda r: r.score, reverse=True)
            for i, r in enumerate(scored):
                r.rank = i + 1

            return scored[:k]
        except QueryPartialError:
            raise
        except Exception as exc:
            raise QueryPartialError(
                f"retrieve failed after embed_query for {document_url}: {exc}",
                query_chars=len(query),
            ) from exc


class HierarchicalRetriever:
    """Two-stage section-pruning retriever for hierarchically chunked documents.

    Stage 1: Embed query → retrieve top ``section_top_k`` *section_summary*
             chunks by cosine similarity.  The returned ``section_id``s form
             the candidate set for Stage 2.  If no section_summary chunks
             exist (legacy collection or flat-chunked document), Stage 2 runs
             without a section filter.

    Stage 2: For each selected section_id, expand to include all descendant
             section_ids via ``_descendant_map`` (so selecting a parent such as
             "Financial Review" also retrieves leaf chunks under
             "Financial Review > Revenue").  Query ``top_k * 2`` content/table
             chunks per expanded section_id, then hybrid-rerank the merged pool.

    Zero-hit fallback: if Stage 2 returns nothing (e.g. all section-scoped
    queries are empty), retry with ``document_url`` filter only.

    Descendant map lifecycle:
    - Populated from the ``SectionNode`` tree at ``index_document()`` time.
    - On a fresh instance pointed at a pre-existing persisted collection,
      ``_ensure_descendant_map()`` rebuilds it lazily from persisted chunk
      metadata (one ``collection.get`` per document_url, no embeddings
      transferred).  The rebuild uses persisted ``parent_section_id`` edges
      (exact) for chunks indexed with F11+, or falls back to a
      ``section_path_str`` prefix heuristic for older legacy chunks.

    ``_select_sections`` guarantees at least the top-1 section hit is kept
    even when all hits fall below ``section_min_score``.
    """

    def __init__(
        self,
        entity_id: str,
        store: "ChromaVectorStore",
        embedder: "OpenAIEmbedder",
        chunker: object | None = None,
        top_k: int = 5,
        weights: tuple[float, float, float, float, float] = (0.5, 0.15, 0.1, 0.15, 0.1),
        section_top_k: int = 3,
        section_min_score: float = 0.25,
    ) -> None:
        self.entity_id = entity_id
        self.store = store
        self.embedder = embedder
        # Accept HierarchicalChunker or TableAwareChunker or None
        self.chunker = chunker
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        if len(weights) != 5:
            raise ValueError(
                f"HierarchicalRetriever weights must be a 5-tuple, got length {len(weights)}"
            )
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(
                f"HierarchicalRetriever weights must sum to 1.0, got {sum(weights)}"
            )
        (
            self.w_vector,
            self.w_keyword,
            self.w_table,
            self.w_section_summary,
            self.w_role_table,
        ) = weights
        self.section_top_k = section_top_k
        self.section_min_score = section_min_score
        # Descendant map: section_id → frozenset of section_ids (self + all descendants).
        # Populated at index_document time; keyed by document_url.
        # Used by Stage 2 to expand a selected parent section to all its children.
        # On a fresh retriever instance pointed at a pre-existing persisted collection,
        # the map starts empty and is rebuilt lazily from stored chunk metadata the
        # first time retrieve() encounters an unknown document_url.
        self._descendant_map: dict[str, dict[str, frozenset[str]]] = {}
        # doc_url → {section_id: frozenset}

    # ------------------------------------------------------------------
    # Descendant-map helpers
    # ------------------------------------------------------------------

    def _ensure_descendant_map(self, document_url: str) -> None:
        """Lazily populate ``_descendant_map[document_url]`` from persisted chunk metadata.

        Called at the start of Stage-2 expansion when ``document_url`` is absent
        from the in-memory map (i.e. this retriever instance did not perform the
        original ``index_document`` call).  Uses ``store.get_section_metadata_for_document``
        which issues a single metadata-only ``collection.get`` against the persisted
        Chroma collection — no embeddings are transferred.

        The rebuild uses the persisted ``parent_section_id`` edge (written since F11)
        to reconstruct the exact parent→child adjacency.  For legacy chunks that
        predate F11 (``parent_section_id`` absent / empty), it falls back to the
        path-prefix heuristic — which is imprecise for repeated headings but still
        better than no expansion at all.

        No-op when the map entry already exists (populated by ``index_document``).
        """
        if document_url in self._descendant_map:
            return
        try:
            metas = self.store.get_section_metadata_for_document(
                self.entity_id, document_url
            )
        except Exception as exc:
            logging.warning(
                "_ensure_descendant_map: metadata fetch failed for %s: %s",
                document_url,
                exc,
            )
            return
        if not metas:
            return

        # Collect unique (section_id, parent_section_id) pairs.
        # Multiple chunks may share the same section_id; deduplicate.
        children_of: dict[str, set[str]] = {}  # parent_sid → set of child sids
        all_sids: set[str] = set()
        has_parent_edges = False
        for m in metas:
            sid = m["section_id"]
            parent = m.get("parent_section_id", "")
            all_sids.add(sid)
            if parent:
                has_parent_edges = True
                children_of.setdefault(parent, set()).add(sid)

        if has_parent_edges:
            # --- Primary path: reconstruct from persisted parent edges (F11+) ---
            # Ensure every sid appears as a key even if it has no children.
            for sid in all_sids:
                children_of.setdefault(sid, set())

            memo: dict[str, frozenset[str]] = {}

            def _desc(sid: str) -> frozenset[str]:
                if sid in memo:
                    return memo[sid]
                result: set[str] = {sid}
                for child in children_of.get(sid, set()):
                    result |= _desc(child)
                memo[sid] = frozenset(result)
                return memo[sid]

            for sid in all_sids:
                _desc(sid)
            self._descendant_map[document_url] = memo
        else:
            # --- Fallback: path-prefix heuristic for legacy chunks without parent edges ---
            class _MetaChunk:
                __slots__ = ("section_id", "section_path_str")
                def __init__(self, sid: str, path: str) -> None:
                    self.section_id = sid
                    self.section_path_str = path

            pseudo_chunks = [
                _MetaChunk(m["section_id"], m["section_path_str"]) for m in metas
            ]
            self._descendant_map[document_url] = _build_descendant_map_from_chunks(  # type: ignore[arg-type]
                pseudo_chunks
            )

    # ------------------------------------------------------------------
    # Indexing (delegated to chunker → HybridRetriever.index_document logic)
    # ------------------------------------------------------------------

    def index_document(self, document: "ParsedDocument") -> "list[Chunk]":
        """Chunk (using whichever chunker is configured) and upsert."""
        if self.chunker is None:
            from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker
            self.chunker = HierarchicalChunker()

        chunks = self.chunker.chunk(document)
        if not chunks:
            self.store.evict_document(self.entity_id, document.url)
            return []
        texts = [c.content for c in chunks]
        try:
            embeddings = self.embedder.embed_texts(texts)
        except PartialEmbedError as exc:
            raise IndexingPartialError(
                f"embed_texts partial failure for {document.url}: {exc}",
                embedded_chunks=chunks[: exc.completed_count],
            ) from exc
        try:
            self.store.upsert(self.entity_id, chunks, embeddings)
        except Exception as exc:
            raise IndexingPartialError(
                f"upsert failed for {document.url}: {exc}",
                embedded_chunks=chunks,
            ) from exc

        # Build/refresh the descendant map for this document from the section tree.
        # Fall back to chunk-derived paths when no section tree is available.
        if document.sections:
            self._descendant_map[document.url] = _build_descendant_map(document.sections)
        else:
            self._descendant_map[document.url] = _build_descendant_map_from_chunks(chunks)

        return chunks

    def evict_document(self, document_url: str) -> None:
        """Remove all indexed chunks for document_url from the vector store."""
        try:
            self.store.evict_document(self.entity_id, document_url)
        except Exception:
            logging.warning(
                "evict_document failed for %s (entity=%s); stale chunks may remain",
                document_url,
                self.entity_id,
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        document_url: str,
        top_k: int | None = None,
    ) -> "list[RetrievalResult]":
        """Run two-stage section-pruning retrieval for *query* scoped to *document_url*.

        Returns up to *top_k* ``RetrievalResult`` objects sorted by descending
        hybrid score.  Returns ``[]`` when the query cannot be embedded or the
        document has no indexed chunks.  Raises ``QueryPartialError`` on
        unexpected failures after embedding succeeds.
        """
        k = top_k if top_k is not None and top_k >= 1 else self.top_k

        query_embedding = self.embedder.embed_query(query)
        if not query_embedding:
            return []

        try:
            # ----------------------------------------------------------------
            # Stage 1: section_summary candidates
            # ----------------------------------------------------------------
            summary_hits = self.store.query(
                entity_id=self.entity_id,
                query_embedding=query_embedding,
                top_k=self.section_top_k * 2,
                where={"document_url": document_url, "chunk_role": "section_summary"},
            )
            selected_sections = self._select_sections(summary_hits)

            # ----------------------------------------------------------------
            # Stage 2: content/table chunks within selected sections
            # ----------------------------------------------------------------
            if selected_sections:
                # Ensure the descendant map is populated for this document_url.
                # This is a no-op when the same instance performed index_document();
                # on a fresh instance it rebuilds from persisted chunk metadata.
                self._ensure_descendant_map(document_url)

                # Expand each selected section to include all its descendants so
                # that selecting a parent section (e.g. "Financial Review") also
                # retrieves chunks from its children (e.g. "Financial Review > Revenue").
                doc_desc_map = self._descendant_map.get(document_url, {})
                expanded: list[str] = []
                seen_expanded: set[str] = set()
                for sid in selected_sections:
                    for desc_sid in doc_desc_map.get(sid, frozenset([sid])):
                        if desc_sid not in seen_expanded:
                            expanded.append(desc_sid)
                            seen_expanded.add(desc_sid)

                # Chroma supports $eq only on scalars — iterate and merge
                stage2_hits: list[RetrievalResult] = []
                seen_ids: set[str] = set()
                for section_id in expanded:
                    hits = self.store.query(
                        entity_id=self.entity_id,
                        query_embedding=query_embedding,
                        top_k=k * 2,
                        where={"document_url": document_url, "section_id": section_id},
                    )
                    for h in hits:
                        if h.chunk.chunk_id not in seen_ids:
                            stage2_hits.append(h)
                            seen_ids.add(h.chunk.chunk_id)
            else:
                # No section_summary chunks in this collection (legacy doc or
                # flat-chunked document) — query without section filter.
                stage2_hits = self.store.query(
                    entity_id=self.entity_id,
                    query_embedding=query_embedding,
                    top_k=k * 2,
                    where={"document_url": document_url},
                )

            # ----------------------------------------------------------------
            # Zero-hit fallback: remove section filter, keep document filter
            # ----------------------------------------------------------------
            if not stage2_hits:
                stage2_hits = self.store.query(
                    entity_id=self.entity_id,
                    query_embedding=query_embedding,
                    top_k=k * 2,
                    where={"document_url": document_url},
                )

            if not stage2_hits:
                return []

            # ----------------------------------------------------------------
            # Hybrid rerank with role boosts
            # ----------------------------------------------------------------
            scored: list[RetrievalResult] = []
            for hit in stage2_hits:
                kw = _keyword_score(query, hit.chunk.content)
                tb = _table_boost(query, hit.chunk)
                role_ss = 0.1 if hit.chunk.chunk_role == "section_summary" else 0.0
                role_tbl = 0.05 if hit.chunk.chunk_role == "table" else 0.0
                hybrid = (
                    self.w_vector * hit.vector_score
                    + self.w_keyword * kw
                    + self.w_table * tb
                    + self.w_section_summary * role_ss
                    + self.w_role_table * role_tbl
                )
                scored.append(
                    RetrievalResult(
                        chunk=hit.chunk,
                        score=hybrid,
                        vector_score=hit.vector_score,
                        keyword_score=kw,
                        table_boost_score=tb,
                        is_table_boost=tb > 0,
                    )
                )

            scored.sort(key=lambda r: r.score, reverse=True)
            for i, r in enumerate(scored):
                r.rank = i + 1
            return scored[:k]

        except QueryPartialError:
            raise
        except Exception as exc:
            raise QueryPartialError(
                f"hierarchical retrieve failed after embed_query for {document_url}: {exc}",
                query_chars=len(query),
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_sections(self, hits: "list[RetrievalResult]") -> list[str]:
        """Return up to section_top_k section_ids from Stage-1 hits.

        Guarantees at least the top-1 hit is included regardless of min_score.
        """
        if not hits:
            return []
        # Always include the best hit
        selected = [hits[0].chunk.section_id]
        for h in hits[1 : self.section_top_k]:
            if h.vector_score >= self.section_min_score:
                sid = h.chunk.section_id
                if sid not in selected:
                    selected.append(sid)
        return [s for s in selected if s]  # drop empty strings (legacy chunks)


# ---------------------------------------------------------------------------
# Descendant-map helpers
# ---------------------------------------------------------------------------


def _build_descendant_map(sections: "list") -> "dict[str, frozenset[str]]":
    """Build section_id → frozenset(self + all descendant section_ids) from a SectionNode list.

    Uses children_ids linkage from the section tree.  The result is used by
    HierarchicalRetriever to expand Stage-2 queries so that selecting a parent
    section also retrieves chunks from all its children.
    """
    # First pass: build children adjacency
    children: dict[str, list[str]] = {s.section_id: list(s.children_ids) for s in sections}

    # Second pass: DFS to compute transitive closure for each node
    memo: dict[str, frozenset[str]] = {}

    def _descendants(sid: str) -> frozenset[str]:
        if sid in memo:
            return memo[sid]
        result: set[str] = {sid}
        for child_id in children.get(sid, []):
            result |= _descendants(child_id)
        memo[sid] = frozenset(result)
        return memo[sid]

    for s in sections:
        _descendants(s.section_id)

    return memo


def _build_descendant_map_from_chunks(chunks: "list") -> "dict[str, frozenset[str]]":
    """Legacy fallback: build a descendant map from ``section_path_str`` prefix matching.

    Used only when chunks have no ``parent_section_id`` metadata (i.e. indexed
    before F11).  Treats section A as an ancestor of section B when B's
    pipe-joined path starts with A's path followed by ``|``.

    Known limitations (acceptable for legacy-only use):
    - Two sibling sections with the same heading share the same path prefix,
      so selecting one parent may incorrectly expand to the other's descendants.
    - Root-owned chunks have an empty path, so ``startswith("|")`` never
      matches any child — root ancestry is not reconstructed.

    Collections indexed with F11+ use ``_ensure_descendant_map``'s primary
    path (exact parent edges) instead of this function.
    """
    # Collect all unique section_ids and their path prefixes
    section_paths: dict[str, str] = {}  # section_id -> section_path_str
    for chunk in chunks:
        sid = chunk.section_id
        if sid and sid not in section_paths:
            section_paths[sid] = chunk.section_path_str

    result: dict[str, frozenset[str]] = {}
    for sid, path in section_paths.items():
        descendants: set[str] = {sid}
        for other_sid, other_path in section_paths.items():
            if other_sid != sid and other_path.startswith(path + "|"):
                descendants.add(other_sid)
        result[sid] = frozenset(descendants)
    return result
