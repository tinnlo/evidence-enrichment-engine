"""Tests for HybridRetriever scoring, document scoping, and edge cases."""

from __future__ import annotations

import pytest

from evidence_enrichment.core.models.contracts import ParsedDocument
from evidence_enrichment.core.retrieval.chunker import TableAwareChunker
from evidence_enrichment.core.retrieval.models import Chunk
from evidence_enrichment.core.retrieval.retriever import HybridRetriever, _keyword_score, _table_boost
from evidence_enrichment.core.retrieval.store import ChromaVectorStore


# ---------------------------------------------------------------------------
# Fake embedder (no API calls)
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Returns deterministic fake embeddings based on content length % 8."""

    model = "fake-embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        vec = [0.0] * 8
        vec[len(text) % 8] = 1.0
        return vec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def retriever(tmp_path):
    import chromadb

    store = ChromaVectorStore(persist_path=str(tmp_path), embedding_model="fake-embedder")
    store._client = chromadb.EphemeralClient()

    chunker = TableAwareChunker(chunk_size=500, overlap=50, min_size=10)
    embedder = FakeEmbedder()

    return HybridRetriever(
        entity_id="test_entity",
        store=store,
        embedder=embedder,  # type: ignore[arg-type]
        chunker=chunker,
        top_k=5,
        weights=(0.7, 0.2, 0.1),
    )


def _make_doc(url: str, text: str, *, accepted: bool = True) -> ParsedDocument:
    import hashlib

    content_hash = hashlib.sha256(text.encode()).hexdigest()
    return ParsedDocument(
        url=url,
        title="Test",
        content_type="text/html",
        text=text,
        excerpt=text[:200],
        full_text=text,
        content_hash=content_hash,
        accepted_for_analysis=accepted,
    )


# ---------------------------------------------------------------------------
# Unit tests for scoring helpers
# ---------------------------------------------------------------------------

class TestKeywordScore:
    def test_full_overlap(self):
        score = _keyword_score("revenue profit", "The company revenue and profit grew.")
        assert score == 1.0

    def test_partial_overlap(self):
        score = _keyword_score("revenue employees", "Revenue grew last year.")
        assert 0.4 < score < 0.7

    def test_no_overlap(self):
        score = _keyword_score("purple unicorn", "Revenue and profit data.")
        assert score == 0.0

    def test_empty_query(self):
        score = _keyword_score("", "some content")
        assert score == 0.0


class TestTableBoost:
    def test_table_boost_numeric_query(self):
        chunk = Chunk(
            chunk_id="x" * 16, document_url="u", content_hash="h",
            index=0, content="data", chunk_type="table",
        )
        boost = _table_boost("What is the annual revenue?", chunk)
        assert boost == pytest.approx(0.1)

    def test_no_boost_text_chunk(self):
        chunk = Chunk(
            chunk_id="x" * 16, document_url="u", content_hash="h",
            index=0, content="data", chunk_type="text",
        )
        boost = _table_boost("What is the annual revenue?", chunk)
        assert boost == 0.0

    def test_no_boost_non_numeric_query(self):
        chunk = Chunk(
            chunk_id="x" * 16, document_url="u", content_hash="h",
            index=0, content="data", chunk_type="table",
        )
        boost = _table_boost("Where is the company headquartered?", chunk)
        assert boost == 0.0


# ---------------------------------------------------------------------------
# Integration tests (use FakeEmbedder + EphemeralClient)
# ---------------------------------------------------------------------------

class TestHybridScoringFormula:
    def test_scores_between_0_and_1(self, retriever):
        """All returned scores are in [0, 1]."""
        doc = _make_doc("https://example.com/a", "The company headquarters is in Berlin, Germany. " * 20)
        retriever.index_document(doc)
        results = retriever.retrieve("headquarters country", "https://example.com/a")
        for r in results:
            assert 0.0 <= r.score <= 1.0 + 1e-6  # tiny float tolerance

    def test_results_sorted_descending(self, retriever):
        """Results are sorted by score descending."""
        doc = _make_doc("https://example.com/b", "Berlin Germany headquarters office. " * 20)
        retriever.index_document(doc)
        results = retriever.retrieve("headquarters", "https://example.com/b")
        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_rank_field_populated(self, retriever):
        """Rank field is set on returned results."""
        doc = _make_doc("https://example.com/c", "data " * 100)
        retriever.index_document(doc)
        results = retriever.retrieve("data", "https://example.com/c")
        for i, r in enumerate(results, start=1):
            assert r.rank == i


class TestDocumentScoped:
    def test_only_target_document_returned(self, retriever):
        """Retrieve only returns chunks from the specified document_url."""
        doc_a = _make_doc("https://example.com/a", "Paris is the capital of France. " * 20)
        doc_b = _make_doc("https://example.com/b", "Berlin is the capital of Germany. " * 20)
        retriever.index_document(doc_a)
        retriever.index_document(doc_b)

        results = retriever.retrieve("capital city", "https://example.com/a")
        for r in results:
            assert r.chunk.document_url == "https://example.com/a"

    def test_empty_collection_returns_empty(self, retriever):
        """Querying a URL that was never indexed returns empty list."""
        results = retriever.retrieve("anything", "https://not-indexed.com/doc")
        assert results == []


class TestTopKLimiting:
    def test_results_limited_to_top_k(self, retriever):
        """Results are capped at top_k."""
        long_text = "sentence with information. " * 100
        doc = _make_doc("https://example.com/long", long_text)
        retriever.index_document(doc)
        results = retriever.retrieve("information", "https://example.com/long", top_k=3)
        assert len(results) <= 3

    def test_top_k_override(self, retriever):
        """Per-call top_k overrides instance default."""
        long_text = "content about headquarters location. " * 50
        doc = _make_doc("https://example.com/k", long_text)
        retriever.index_document(doc)
        results = retriever.retrieve("headquarters", "https://example.com/k", top_k=2)
        assert len(results) <= 2


class TestHybridRetrieverValidation:
    def _base_kwargs(self, tmp_path):
        import chromadb

        store = ChromaVectorStore(persist_path=str(tmp_path), embedding_model="fake-embedder")
        store._client = chromadb.EphemeralClient()
        return {
            "entity_id": "ent",
            "store": store,
            "embedder": FakeEmbedder(),  # type: ignore[arg-type]
            "chunker": TableAwareChunker(chunk_size=500, overlap=50, min_size=10),
        }

    def test_invalid_top_k_raises(self, tmp_path):
        """top_k < 1 raises ValueError at construction time."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            HybridRetriever(**self._base_kwargs(tmp_path), top_k=0, weights=(0.7, 0.2, 0.1))

    def test_weights_not_summing_to_one_raises(self, tmp_path):
        """Weights that don't sum to 1.0 raise ValueError at construction time."""
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            HybridRetriever(**self._base_kwargs(tmp_path), top_k=5, weights=(0.5, 0.2, 0.1))

    def test_weights_wrong_length_raises(self, tmp_path):
        """A weights tuple with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="3-tuple"):
            HybridRetriever(**self._base_kwargs(tmp_path), top_k=5, weights=(0.5, 0.5))  # type: ignore[arg-type]

    def test_negative_weight_raises(self, tmp_path):
        """A negative weight raises ValueError."""
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            HybridRetriever(**self._base_kwargs(tmp_path), top_k=5, weights=(-0.1, 0.9, 0.2))

    def test_nan_weight_raises(self, tmp_path):
        """A NaN weight raises ValueError."""
        import math
        with pytest.raises(ValueError, match="finite"):
            HybridRetriever(**self._base_kwargs(tmp_path), top_k=5, weights=(math.nan, 0.5, 0.5))

    def test_valid_weights_accepted(self, tmp_path):
        """Weights that sum to exactly 1.0 are accepted."""
        r = HybridRetriever(**self._base_kwargs(tmp_path), top_k=3, weights=(0.6, 0.3, 0.1))
        assert r.top_k == 3


# ---------------------------------------------------------------------------
# IndexingPartialError regression tests
# ---------------------------------------------------------------------------

class TestIndexingPartialError:
    """Regression tests: upsert-failure raises IndexingPartialError with embedded chunks."""

    def test_upsert_failure_raises_indexing_partial_error(self, tmp_path):
        """When upsert raises, index_document raises IndexingPartialError carrying chunks."""
        from unittest.mock import MagicMock
        from evidence_enrichment.core.retrieval.retriever import IndexingPartialError

        store = MagicMock()
        store.upsert.side_effect = RuntimeError("chroma unavailable")
        # evict_document must not raise (called during error handling elsewhere)
        store.evict_document.return_value = None

        embedder = FakeEmbedder()
        chunker = TableAwareChunker(chunk_size=500, overlap=50, min_size=10)
        r = HybridRetriever(
            entity_id="e",
            store=store,
            embedder=embedder,  # type: ignore[arg-type]
            chunker=chunker,
            top_k=3,
        )

        doc = _make_doc("https://example.com/a", "word " * 200)
        with pytest.raises(IndexingPartialError) as exc_info:
            r.index_document(doc)

        err = exc_info.value
        assert len(err.embedded_chunks) > 0, "embedded_chunks must be non-empty"
        assert all(hasattr(c, "content") for c in err.embedded_chunks)

    def test_pre_embed_failure_raises_plain_exception(self, tmp_path):
        """When chunking raises before embed, a plain exception propagates (no IndexingPartialError)."""
        from unittest.mock import MagicMock
        from evidence_enrichment.core.retrieval.retriever import IndexingPartialError

        store = MagicMock()
        bad_chunker = MagicMock()
        bad_chunker.chunk.side_effect = ValueError("chunker exploded")

        r = HybridRetriever(
            entity_id="e",
            store=store,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            chunker=bad_chunker,
            top_k=3,
        )

        doc = _make_doc("https://example.com/b", "word " * 100)
        with pytest.raises(ValueError, match="chunker exploded"):
            r.index_document(doc)
        # Must NOT be an IndexingPartialError (no embedding happened)
        try:
            r.index_document(doc)
        except IndexingPartialError:
            pytest.fail("Pre-embed failure should not raise IndexingPartialError")
        except ValueError:
            pass  # expected
