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

import math
import re

from evidence_enrichment.core.models.contracts import ParsedDocument
from evidence_enrichment.core.retrieval.chunker import TableAwareChunker
from evidence_enrichment.core.retrieval.embedder import OpenAIEmbedder
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
        embeddings = self.embedder.embed_texts(texts)
        self.store.upsert(self.entity_id, chunks, embeddings)
        return chunks

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
                    is_table_boost=tb > 0,
                )
            )

        # Sort descending by final score, return top_k
        scored.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(scored):
            r.rank = i + 1

        return scored[:k]
