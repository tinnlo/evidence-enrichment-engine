"""Tests for the LangGraph adaptive retrieval agent.

All tests use a stub ``HybridRetriever`` to avoid Chroma/OpenAI dependencies.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from evidence_enrichment.core.retrieval.agent import (
    RetrievalAgent,
    RetrievalMetrics,
    RetrievalPartialError,
)
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

        results, metrics = agent.retrieve_with_metrics("headquarters location", "https://example.com/doc")

        assert results == high_score_results
        assert agent.last_iterations == 1
        assert stub.retrieve_call_count == 1
        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.iterations == 1
        assert metrics.total_query_chars == len("headquarters location")
        assert metrics.query_char_history == [len("headquarters location")]

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

        results, metrics = agent.retrieve_with_metrics("hq country", "https://example.com/doc")

        # Should have iterated twice (first low, then high)
        assert agent.last_iterations == 2
        assert stub.retrieve_call_count == 2
        assert results == high_results
        assert metrics.iterations == 2

    def test_stops_at_max_iterations(self):
        always_low = [_make_result(0.05)]
        stub = StubRetriever(
            results_per_call=[always_low, always_low, always_low, always_low]
        )
        agent = RetrievalAgent(stub, max_iterations=2)

        results, metrics = agent.retrieve_with_metrics("country", "https://example.com/doc")

        # Capped at max_iterations=2
        assert agent.last_iterations == 2
        assert stub.retrieve_call_count == 2
        assert results == always_low
        assert metrics.iterations == 2

    def test_empty_results_do_not_raise(self):
        stub = StubRetriever(results_per_call=[[], []])
        agent = RetrievalAgent(stub, max_iterations=2)

        results, metrics = agent.retrieve_with_metrics("query", "https://example.com/doc")

        assert results == []
        assert agent.last_iterations == 2
        assert metrics.iterations == 2

    def test_returns_best_scored_pass_not_final_degraded_pass(self):
        """Agent returns the highest-scored pass even when later refinements degrade quality."""
        # All scores below 0.40 threshold so the agent runs to the iteration cap.
        best_results = [_make_result(0.35), _make_result(0.30)]  # iter 1 — best
        mid_results  = [_make_result(0.20)]                       # iter 2 — worse
        bad_results  = [_make_result(0.05)]                       # iter 3 — worst (cap hit)
        stub = StubRetriever(results_per_call=[best_results, mid_results, bad_results])
        agent = RetrievalAgent(stub, max_iterations=3)

        results, metrics = agent.retrieve_with_metrics("founding year", "https://example.com/doc")

        # best_results (iter 1, highest score) should be returned, not bad_results.
        assert results == best_results, (
            "expected best-scored pass (iter 1) but got the degraded final pass"
        )
        assert metrics.iterations == 3


class TestRetrievalAgentPartialError:
    """Agent raises RetrievalPartialError with exact history on mid-graph failures."""

    def test_evaluate_failure_after_first_retrieve_carries_exact_history(self):
        """score_chunks failing on iteration 2 must report iteration-1 history, not [len(query)]."""
        query = "hq country"
        low_results = [_make_result(0.1)]
        high_results = [_make_result(0.9)]
        stub = StubRetriever(results_per_call=[low_results, high_results])
        agent = RetrievalAgent(stub, max_iterations=3)

        call_count = 0

        def _fail_on_second_call(results, query_text):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("score_chunks boom")
            from evidence_enrichment.core.retrieval.evaluator import score_chunks as _real
            return _real(results, query_text)

        with patch(
            "evidence_enrichment.core.retrieval.agent.score_chunks",
            side_effect=_fail_on_second_call,
        ):
            with pytest.raises(RetrievalPartialError) as exc_info:
                agent.retrieve(query, "https://example.com/doc")

        err = exc_info.value
        # Two retrieve nodes ran; second evaluate raised — history must have 2 entries.
        assert len(err.partial_metrics.query_char_history) == 2, (
            "expected 2 entries in query_char_history (one per completed retrieve node)"
        )
        assert err.partial_metrics.total_query_chars == sum(
            err.partial_metrics.query_char_history
        )
        # The second evaluate never completed so best_results is from iteration 1
        # (low_results, the only confirmed scored pass).
        assert err.partial_results == low_results

    def test_refine_failure_carries_exact_history_and_prior_results(self):
        """refine_query failing after the first evaluate must carry iteration-1 state."""
        query = "revenue"
        low_results = [_make_result(0.15)]
        stub = StubRetriever(results_per_call=[low_results, low_results])
        agent = RetrievalAgent(stub, max_iterations=3)

        with patch(
            "evidence_enrichment.core.retrieval.agent.refine_query",
            side_effect=RuntimeError("refine boom"),
        ):
            with pytest.raises(RetrievalPartialError) as exc_info:
                agent.retrieve(query, "https://example.com/doc")

        err = exc_info.value
        # One retrieve ran before refine_query failed.
        assert len(err.partial_metrics.query_char_history) == 1
        assert err.partial_metrics.query_char_history[0] == len(query)
        assert err.partial_results == low_results

    def test_first_retrieve_failure_carries_no_phantom_cost(self):
        """inner.retrieve() failing on the very first URL accrues zero query chars.

        We cannot distinguish pre-embed from post-embed failures in the inner
        retriever, but with no accumulated hits the safest assumption is that
        embed_query may not have run — so no query chars are billed.
        """
        query = "founded year"
        stub = StubRetriever()
        stub.retrieve = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("db down"))  # type: ignore[method-assign]
        agent = RetrievalAgent(stub, max_iterations=3)

        with pytest.raises(RetrievalPartialError) as exc_info:
            agent.retrieve(query, "https://example.com/doc")

        err = exc_info.value
        assert err.partial_metrics.query_char_history == []
        assert err.partial_results == []

    def test_query_partial_error_from_inner_bills_query_chars_on_first_url(self):
        """QueryPartialError from inner.retrieve() bills query_chars even for first URL.

        embed_query succeeded inside HybridRetriever before the store.query failed,
        so _make_retrieve_node must include query chars regardless of successful_calls.
        """
        from evidence_enrichment.core.retrieval.retriever import QueryPartialError

        query = "founded year"
        qchars = len(query)

        def _raise_query_partial(q, document_url, top_k=None):
            raise QueryPartialError("store.query exploded", query_chars=qchars)

        stub = StubRetriever()
        stub.retrieve = _raise_query_partial  # type: ignore[method-assign]
        agent = RetrievalAgent(stub, max_iterations=3)

        with pytest.raises(RetrievalPartialError) as exc_info:
            agent.retrieve(query, "https://example.com/doc")

        err = exc_info.value
        # embed_query ran (QueryPartialError carries confirmed query_chars),
        # so history must include them even though successful_calls == 0.
        assert err.partial_metrics.query_char_history == [qchars]
        assert err.partial_results == []

    def test_retrieve_node_bills_query_chars_when_prior_url_call_succeeded(self):
        """_retrieve_node bills query chars whenever any prior URL call returned.

        Even if the successful call returned zero hits, successful_calls > 0
        means embed_query ran — so the failing iteration's chars must appear
        in query_char_history.
        """
        from evidence_enrichment.core.retrieval.agent import _RetrieveNodeError

        query = "employee count"
        call_count = 0

        def _succeed_then_fail(q, document_url, top_k=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # first URL: successful call, zero hits
            raise RuntimeError("second URL db down")

        stub = StubRetriever()
        stub.retrieve = _succeed_then_fail  # type: ignore[method-assign]
        agent = RetrievalAgent(stub, max_iterations=3)

        # Build the retrieve node directly (bypassing the full graph).
        snapshot = {"query_char_history": [], "best_results": [], "best_score": -1.0}
        retrieve_node = agent._make_retrieve_node(snapshot)

        state = {
            "query": query,
            "document_urls": ["https://example.com/doc1", "https://example.com/doc2"],
            "top_k": 5,
            "results": [],
            "score": 0.0,
            "iteration": 0,
            "query_char_history": [],
        }

        with pytest.raises(_RetrieveNodeError) as exc_info:
            retrieve_node(state)

        err = exc_info.value
        # successful_calls == 1 (first URL returned before second raised), so
        # query_chars must be included.
        assert err.query_char_history == [len(query)]


# ---------------------------------------------------------------------------
# D4 — Retriever parity: RetrievalAgent works identically with HybridRetriever
# and HierarchicalRetriever stub interfaces.
# ---------------------------------------------------------------------------

class StubHierarchicalRetriever(StubRetriever):
    """Minimal HierarchicalRetriever stand-in.

    Adds a ``section_top_k`` attribute (present on the real
    ``HierarchicalRetriever``) and a no-op ``_ensure_descendant_map`` to
    confirm that ``RetrievalAgent`` never calls private methods on its
    wrapped retriever — it only calls ``retrieve()``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.section_top_k = 3
        self._ensure_descendant_map_calls: int = 0

    def _ensure_descendant_map(self, document_url: str) -> None:  # pragma: no cover
        # Should never be called by RetrievalAgent.
        self._ensure_descendant_map_calls += 1


@pytest.mark.parametrize(
    "retriever_cls,label",
    [
        (StubRetriever, "hybrid"),
        (StubHierarchicalRetriever, "hierarchical"),
    ],
)
class TestRetrievalAgentRetrieverParity:
    """RetrievalAgent must behave identically regardless of which retriever it wraps.

    All tests use the stub interface — no Chroma or OpenAI calls.  The suite
    confirms that ``RetrievalAgent`` only depends on the public ``retrieve()``
    method, not on retriever-specific internals.
    """

    def test_happy_path_returns_results(self, retriever_cls, label):
        """Agent returns results for both retriever types."""
        stub = retriever_cls(results_per_call=[[_make_result(0.85)]])
        agent = RetrievalAgent(stub, max_iterations=3)
        results = agent.retrieve("headquarters country", "https://example.com/doc")
        assert results, f"[{label}] expected non-empty results"
        assert results[0].score == pytest.approx(0.85), f"[{label}] score mismatch"

    def test_refinement_loop_runs_for_both(self, retriever_cls, label):
        """Agent runs at least 2 iterations when first score is below threshold."""
        stub = retriever_cls(
            results_per_call=[
                [_make_result(0.3)],   # first call — below threshold
                [_make_result(0.9)],   # second call — above threshold
            ]
        )
        agent = RetrievalAgent(stub, max_iterations=3)
        results = agent.retrieve("headquarters", "https://example.com/doc")
        assert stub.retrieve_call_count >= 2, (
            f"[{label}] expected >=2 iterations, got {stub.retrieve_call_count}"
        )
        assert results, f"[{label}] expected results after refinement"

    def test_empty_results_handled_gracefully(self, retriever_cls, label):
        """Agent returns empty list when retriever consistently returns nothing."""
        stub = retriever_cls(results_per_call=[[]])
        agent = RetrievalAgent(stub, max_iterations=2)
        results = agent.retrieve("obscure query", "https://example.com/doc")
        assert isinstance(results, list), f"[{label}] expected list return type"

    def test_agent_does_not_call_private_methods(self, retriever_cls, label):
        """RetrievalAgent only calls retrieve() — no private retriever methods."""
        stub = retriever_cls(results_per_call=[[_make_result(0.8)]])
        agent = RetrievalAgent(stub, max_iterations=1)
        agent.retrieve("query", "https://example.com/doc")
        # StubHierarchicalRetriever tracks _ensure_descendant_map calls.
        if hasattr(stub, "_ensure_descendant_map_calls"):
            assert stub._ensure_descendant_map_calls == 0, (
                f"[{label}] RetrievalAgent must not call private retriever methods"
            )
