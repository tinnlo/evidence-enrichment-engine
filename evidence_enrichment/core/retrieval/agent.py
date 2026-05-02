"""LangGraph-based adaptive retrieval agent.

Wraps a ``HybridRetriever`` in a three-node StateGraph that iteratively
retrieves, evaluates chunk quality, and refines the query when the quality
score is below a threshold — up to a configurable maximum number of
iterations.

Graph topology
--------------

    retrieve ──► evaluate ──► (branch)
                                 │ quality OK  ──► END
                                 │ quality low ──► refine ──► retrieve

Nodes
-----
``_retrieve_node``
    Call ``inner.retrieve(state["query"], document_url)`` for every document
    URL and accumulate ``RetrievalResult`` objects into ``state["results"]``.

``_evaluate_node``
    Compute the quality score for the current results.  Store
    ``state["score"]`` and increment ``state["iteration"]``.

``_refine_node``
    Replace ``state["query"]`` with an enriched query produced by
    ``evaluator.refine_query``.

``_branch``
    Conditional edge: route to END when score ≥ threshold or the
    iteration cap is reached; otherwise route to ``refine``.

Public API
----------
The ``RetrievalAgent`` class mirrors the ``HybridRetriever`` interface that
``coordinator._get_retriever`` expects:

* ``entity_id`` — property, proxied from inner retriever
* ``index_document(doc)`` — delegated to inner retriever
* ``retrieve(query, document_url, top_k)`` — runs the graph, returns results
* ``last_iterations`` — number of graph iterations used in the last call
"""

from __future__ import annotations

from typing import Any, NamedTuple, TypedDict

try:
    from langgraph.graph import END, StateGraph  # type: ignore[import-untyped]

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

from evidence_enrichment.core.models.contracts import ParsedDocument
from evidence_enrichment.core.retrieval.evaluator import (
    is_quality_sufficient,
    refine_query,
    score_chunks,
)
from evidence_enrichment.core.retrieval.models import Chunk, RetrievalResult
from evidence_enrichment.core.retrieval.retriever import QueryPartialError

# Maximum refinement iterations before giving up and returning what we have.
_DEFAULT_MAX_ITERATIONS = 3


class _AgentState(TypedDict):
    """Mutable state threaded through all graph nodes.

    ``query_char_history`` is invocation-local: each ``retrieve()`` call
    starts with an empty list and each ``_retrieve_node`` appends the current
    query length.  Because this lives inside the graph state dict (not on the
    ``RetrievalAgent`` instance), concurrent invocations that share the same
    cached agent cannot race on this field.
    """

    query: str
    document_urls: list[str]
    top_k: int
    results: list[RetrievalResult]
    score: float
    iteration: int
    max_iterations: int
    query_char_history: list[int]  # per-iteration query lengths — invocation-local


class RetrievalMetrics(NamedTuple):
    """Request-scoped retrieval accounting returned by ``RetrievalAgent.retrieve()``.

    Carrying metrics in the return value (rather than storing them on the
    instance) makes them safe to read under concurrent ``run()`` calls that
    share the same cached retriever instance.
    """

    iterations: int
    total_query_chars: int
    query_char_history: list[int]  # exact per-iteration query lengths


class _RetrieveNodeError(Exception):
    """Internal sentinel raised from ``_retrieve_node`` on inner-retriever failure.

    Carries the per-iteration accounting accumulated *up to and including* the
    failing iteration so that ``retrieve()`` can build accurate partial metrics
    even when LangGraph swallows intermediate state updates.
    """

    def __init__(
        self,
        cause: BaseException,
        query_char_history: list[int],
        partial_results: list[RetrievalResult],
    ) -> None:
        super().__init__(str(cause))
        self.__cause__ = cause
        self.query_char_history = query_char_history
        self.partial_results = partial_results


class RetrievalPartialError(Exception):
    """Raised when the retrieval graph fails mid-execution.

    Carries the partial :class:`RetrievalMetrics` accumulated before the
    failure so callers can accrue any embedding spend that already occurred
    without reading mutable instance fields.  Also carries ``partial_results``
    — the best-so-far retrieval hits — so callers can use them as a fallback
    rather than discarding all retrieval context for that document.
    """

    def __init__(
        self,
        cause: BaseException,
        partial_metrics: RetrievalMetrics,
        partial_results: list[RetrievalResult],
    ) -> None:
        super().__init__(str(cause))
        self.__cause__ = cause
        self.partial_metrics = partial_metrics
        self.partial_results = partial_results


class RetrievalAgent:
    """Adaptive retrieval agent built on a LangGraph ``StateGraph``.

    Parameters
    ----------
    inner:
        A ``HybridRetriever`` instance that provides the actual retrieval
        and indexing operations.
    max_iterations:
        Maximum number of retrieve→evaluate→refine cycles before the agent
        returns whatever it has found so far.
    """

    def __init__(
        self, inner: Any, max_iterations: int = _DEFAULT_MAX_ITERATIONS
    ) -> None:
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is required for retrieval mode='agent'. "
                "Install it with: pip install 'evidence_enrichment[retrieval]'"
            )
        self.inner = inner
        self.max_iterations = max_iterations
        # Kept for backward-compat / diagnostics only.  FinOps-critical callers
        # must use the RetrievalMetrics returned by retrieve() instead.
        self.last_iterations: int = 0
        self.last_query_chars: int = 0
        self.last_total_query_chars: int = 0
        self.last_query_char_history: list[int] = []

    # ------------------------------------------------------------------
    # Public interface expected by coordinator
    # ------------------------------------------------------------------

    @property
    def entity_id(self) -> str:
        return self.inner.entity_id

    def index_document(self, document: ParsedDocument) -> list[Chunk]:
        """Delegate indexing to the inner retriever."""
        return self.inner.index_document(document)

    def evict_document(self, document_url: str) -> None:
        """Delegate stale-vector eviction to the inner retriever.

        Ensures that ``RetrievalAgent`` satisfies the same eviction contract as
        ``HybridRetriever`` so the coordinator's ``hasattr`` guard triggers
        correctly in ``retrieval.mode='agent'``.
        """
        if hasattr(self.inner, "evict_document"):
            self.inner.evict_document(document_url)

    def retrieve(
        self,
        query: str,
        document_url: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Run the adaptive retrieval graph and return results.

        Implements the same interface as ``HybridRetriever.retrieve()`` so
        ``RetrievalAgent`` remains a drop-in substitutable retriever.  Callers
        that also need FinOps accounting should use ``retrieve_with_metrics()``
        instead.

        Raises
        ------
        RetrievalPartialError
            If the graph fails mid-execution.  The exception carries
            ``partial_metrics`` for any embedding spend that already occurred
            and ``partial_results`` for the best-so-far retrieval hits.

        Parameters
        ----------
        query:
            Natural-language retrieval query.
        document_url:
            Restrict retrieval to chunks from this document.
        top_k:
            Override the inner retriever's default top_k.
        """
        results, _ = self.retrieve_with_metrics(query, document_url, top_k=top_k)
        return results

    def retrieve_with_metrics(
        self,
        query: str,
        document_url: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievalResult], RetrievalMetrics]:
        """Run the adaptive retrieval graph and return results with request-scoped metrics.

        Returns
        -------
        tuple of (results, metrics) where ``metrics`` is a :class:`RetrievalMetrics`
        namedtuple carrying the per-request iteration count and query char history.
        All accounting data lives in per-invocation closures, so concurrent calls
        sharing the same ``RetrievalAgent`` instance cannot race on accounting fields.

        Raises
        ------
        RetrievalPartialError
            If the graph fails mid-execution.  The exception carries
            ``partial_metrics`` for any embedding spend that already occurred
            and ``partial_results`` for the best-so-far retrieval hits.

        Parameters
        ----------
        query:
            Natural-language retrieval query.
        document_url:
            Restrict retrieval to chunks from this document.
        top_k:
            Override the inner retriever's default top_k.
        """
        effective_top_k = top_k if top_k is not None else self.inner.top_k

        # Per-invocation snapshot — updated by every node so that any graph
        # exception (including evaluate/refine failures) can report exact
        # history and best-so-far results without reading LangGraph internals.
        # ``best_results`` tracks the highest-scored retrieve pass so that a
        # later degraded refinement cannot overwrite better earlier evidence.
        snapshot: dict[str, Any] = {
            "query_char_history": [],
            "best_results": [],
            "best_score": -1.0,
        }

        # Build a fresh compiled graph whose node closures capture `snapshot`.
        graph = self._build_graph(snapshot)

        initial_state: _AgentState = {
            "query": query,
            "document_urls": [document_url],
            "top_k": effective_top_k,
            "results": [],
            "score": 0.0,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "query_char_history": [],
        }
        try:
            final_state = graph.invoke(initial_state)
        except _RetrieveNodeError as node_exc:
            # _retrieve_node raised — snapshot already carries exact history
            # including the failing iteration plus prior-iteration results.
            partial_history: list[int] = node_exc.query_char_history
            partial_res: list[RetrievalResult] = node_exc.partial_results
            partial_metrics = RetrievalMetrics(
                iterations=len(partial_history),
                total_query_chars=sum(partial_history),
                query_char_history=partial_history,
            )
            self.last_iterations = partial_metrics.iterations
            self.last_query_chars = partial_history[-1] if partial_history else 0
            self.last_total_query_chars = partial_metrics.total_query_chars
            self.last_query_char_history = partial_history
            raise RetrievalPartialError(
                node_exc.__cause__,  # type: ignore[arg-type]
                partial_metrics,
                partial_res,
            ) from node_exc.__cause__
        except Exception as exc:
            # Failure in evaluate/refine nodes — snapshot holds the exact
            # history and best-scored results from all completed retrieve+evaluate
            # cycles.  Use best_results (not the latest retrieve pass) so a
            # degraded later refinement cannot overwrite stronger earlier evidence.
            partial_history = list(snapshot["query_char_history"])
            # Only fabricate a cost floor when there is evidence that at least one
            # embed_query call completed (i.e. a retrieve node committed history).
            # An empty history means the graph failed before any embedding ran;
            # charging len(query) in that case violates the zero-charge invariant
            # for infrastructure / graph-wiring failures.
            partial_res = list(snapshot["best_results"])
            partial_metrics = RetrievalMetrics(
                iterations=len(partial_history),
                total_query_chars=sum(partial_history),
                query_char_history=partial_history,
            )
            self.last_iterations = partial_metrics.iterations
            self.last_query_chars = partial_history[-1] if partial_history else 0
            self.last_total_query_chars = partial_metrics.total_query_chars
            self.last_query_char_history = partial_history
            raise RetrievalPartialError(exc, partial_metrics, partial_res) from exc

        history: list[int] = list(final_state.get("query_char_history") or [])
        metrics = RetrievalMetrics(
            iterations=final_state["iteration"],
            total_query_chars=sum(history),
            query_char_history=history,
        )
        self.last_iterations = metrics.iterations
        self.last_query_chars = len(final_state["query"])
        self.last_total_query_chars = metrics.total_query_chars
        self.last_query_char_history = history
        # Return the best-scored results across all iterations, not just the
        # final pass.  Refinement can degrade quality on later iterations, so
        # snapshot["best_results"] is always >= final_state["results"] in score.
        # Fall back to final_state["results"] only if evaluate never ran
        # (e.g. max_iterations=0 edge case).
        best = snapshot["best_results"] or final_state["results"]
        return best, metrics

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, snapshot: dict[str, Any]) -> Any:
        """Build and compile a fresh graph whose node closures capture *snapshot*.

        Building per-invocation is cheap (pure Python object construction, no
        I/O) and gives true isolation: each ``retrieve()`` call gets its own
        nodes that update the same ``snapshot`` dict, so failures in any node
        always leave the caller with the most recent complete state.
        """
        graph: StateGraph = StateGraph(_AgentState)
        graph.add_node("retrieve", self._make_retrieve_node(snapshot))
        graph.add_node("evaluate", self._make_evaluate_node(snapshot))
        graph.add_node("refine", self._refine_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._branch,
            {"refine": "refine", "end": END},
        )
        graph.add_edge("refine", "retrieve")

        return graph.compile()

    # ------------------------------------------------------------------
    # Node factories
    # ------------------------------------------------------------------

    def _make_retrieve_node(self, snapshot: dict[str, Any]):  # noqa: ANN202
        """Return a retrieve node closure that updates *snapshot* on success."""

        def _retrieve_node(state: _AgentState) -> dict:
            """Retrieve chunks for every document URL in state."""
            # Record query chars only after the embedding call completes to
            # avoid billing a pre-embed failure (e.g. invalid top_k, preflight
            # validation) as a real embedding spend.
            prior_history = list(state.get("query_char_history") or [])
            query_chars = len(state["query"])
            # prior_results: best-so-far from snapshot (set after each evaluate).
            prior_results: list[RetrievalResult] = list(snapshot["best_results"])
            accumulated: list[RetrievalResult] = []
            # Track successful call count (not hit count) to decide whether
            # embed_query definitely ran before a later failure.
            successful_calls = 0
            try:
                for url in state["document_urls"]:
                    hits = self.inner.retrieve(
                        state["query"],
                        document_url=url,
                        top_k=state["top_k"],
                    )
                    successful_calls += 1
                    accumulated.extend(hits)
            except QueryPartialError as inner_exc:
                # embed_query succeeded before the inner failure — billing is
                # warranted regardless of successful_calls.
                failed_history = prior_history + [inner_exc.query_chars]
                raise _RetrieveNodeError(
                    inner_exc,
                    failed_history,
                    prior_results + accumulated,
                ) from inner_exc
            except Exception as inner_exc:
                # Unknown failure.  Include query chars only when at least one
                # prior retrieve() call returned (embed_query definitely ran).
                # A zero-call failure (first URL, before any embed) accrues nothing.
                if successful_calls > 0:
                    failed_history = prior_history + [query_chars]
                else:
                    failed_history = prior_history
                raise _RetrieveNodeError(
                    inner_exc,
                    failed_history,
                    prior_results + accumulated,
                ) from inner_exc
            # Commit history only on success — embed_query definitely ran.
            updated_history = prior_history + [query_chars]
            snapshot["query_char_history"] = updated_history
            return {"results": accumulated, "query_char_history": updated_history}

        return _retrieve_node

    def _make_evaluate_node(self, snapshot: dict[str, Any]):  # noqa: ANN202
        """Return an evaluate node closure that tracks best-so-far results."""

        def _evaluate_node(state: _AgentState) -> dict:
            """Score current results, increment iteration counter, and update best snapshot."""
            score = score_chunks(state["results"], state["query"])
            # Update best_results whenever this pass scores better than all prior passes.
            if score > snapshot["best_score"]:
                snapshot["best_score"] = score
                snapshot["best_results"] = list(state["results"])
            # Seed best_results on the first pass even if score is 0 so that any
            # evidence is always preferred over an empty fallback.
            elif not snapshot["best_results"]:
                snapshot["best_results"] = list(state["results"])
            return {"score": score, "iteration": state["iteration"] + 1}

        return _evaluate_node

    def _refine_node(self, state: _AgentState) -> dict:
        """Produce a refined query for the next retrieval pass."""
        new_query = refine_query(
            state["query"],
            state["results"],
            iteration=state["iteration"] - 1,  # already incremented in evaluate
        )
        return {"query": new_query}

    # ------------------------------------------------------------------
    # Conditional edge
    # ------------------------------------------------------------------

    @staticmethod
    def _branch(state: _AgentState) -> str:
        """Return 'end' when quality is sufficient or iteration cap is hit."""
        if is_quality_sufficient(state["score"]):
            return "end"
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "refine"
