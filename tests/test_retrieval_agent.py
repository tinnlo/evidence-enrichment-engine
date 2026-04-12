"""Tests for the LangGraph adaptive retrieval agent.

All tests use a stub ``HybridRetriever`` to avoid Chroma/OpenAI dependencies.
"""

from __future__ import annotations

import pytest

from evidence_enrichment.core.retrieval.models import Chunk, RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str, url: str = "https://example.com/doc") -> Chunk:
    chunk_id = Chunk.make_id(url, 0, content[:8])
    return Chunk(
        chunk_id=chunk_id,
        document_url=url,
        content_hash=content[:8],
        index=0,
        content=content,
    )


def _make_result(score: float, content: str = "some text") -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(content), score=score)


class StubRetriever:
    """Minimal HybridRetriever stand-in for agent tests."""

    def __init__(
        self,
        entity_id: str = "acme",
        results_per_call: list[list[RetrievalResult]] | None = None,
        top_k: int = 5,
    ) -> None:
        self.entity_id = entity_id
        self.top_k = top_k
        # Each call to retrieve() pops from the front; last entry is reused.
        self._calls: list[list[RetrievalResult]] = results_per_call or [
            [_make_result(0.8)]
        ]
        self._call_index = 0
        self.retrieve_call_count = 0

    def retrieve(
        self,
        query: str,
        document_url: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.retrieve_call_count += 1
        idx = min(self._call_index, len(self._calls) - 1)
        self._call_index += 1
        return self._calls[idx]

    def index_document(self, document):  # pragma: no cover
        return []


# ---------------------------------------------------------------------------
# Import guard (langgraph may not be installed in minimal CI environments)
# ---------------------------------------------------------------------------

try:
    from evidence_enrichment.core.retrieval.agent import RetrievalAgent

    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AGENT_AVAILABLE, reason="langgraph not installed")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrievalAgentHappyPath:
    """Agent returns results on the first pass when quality is sufficient."""

    def test_returns_results_when_quality_sufficient(self):
        high_score_results = [_make_result(0.9), _make_result(0.85)]
        stub = StubRetriever(results_per_call=[high_score_results])
        agent = RetrievalAgent(stub, max_iterations=3)

        results = agent.retrieve("headquarters location", "https://example.com/doc")

        assert results == high_score_results
        assert agent.last_iterations == 1
        assert stub.retrieve_call_count == 1

    def test_entity_id_proxied_from_inner(self):
        stub = StubRetriever(entity_id="test-entity")
        agent = RetrievalAgent(stub)
        assert agent.entity_id == "test-entity"


class TestRetrievalAgentRefinement:
    """Agent iterates when initial quality is below threshold."""

    def test_refines_query_on_low_score(self):
        low_results = [_make_result(0.1)]
        high_results = [_make_result(0.9)]
        stub = StubRetriever(results_per_call=[low_results, high_results])
        agent = RetrievalAgent(stub, max_iterations=3)

        results = agent.retrieve("hq country", "https://example.com/doc")

        # Should have iterated twice (first low, then high)
        assert agent.last_iterations == 2
        assert stub.retrieve_call_count == 2
        assert results == high_results

    def test_stops_at_max_iterations(self):
        always_low = [_make_result(0.05)]
        stub = StubRetriever(
            results_per_call=[always_low, always_low, always_low, always_low]
        )
        agent = RetrievalAgent(stub, max_iterations=2)

        results = agent.retrieve("country", "https://example.com/doc")

        # Capped at max_iterations=2
        assert agent.last_iterations == 2
        assert stub.retrieve_call_count == 2
        assert results == always_low

    def test_empty_results_do_not_raise(self):
        stub = StubRetriever(results_per_call=[[], []])
        agent = RetrievalAgent(stub, max_iterations=2)

        results = agent.retrieve("query", "https://example.com/doc")

        assert results == []
        assert agent.last_iterations == 2
