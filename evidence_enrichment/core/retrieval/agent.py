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

from typing import Any, TypedDict

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

# Maximum refinement iterations before giving up and returning what we have.
_DEFAULT_MAX_ITERATIONS = 3


class _AgentState(TypedDict):
    """Mutable state threaded through all graph nodes."""

    query: str
    document_urls: list[str]
    top_k: int
    results: list[RetrievalResult]
    score: float
    iteration: int
    max_iterations: int


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
        self.last_iterations: int = 0
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public interface expected by coordinator
    # ------------------------------------------------------------------

    @property
    def entity_id(self) -> str:
        return self.inner.entity_id

    def index_document(self, document: ParsedDocument) -> list[Chunk]:
        """Delegate indexing to the inner retriever."""
        return self.inner.index_document(document)

    def retrieve(
        self,
        query: str,
        document_url: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Run the adaptive retrieval graph and return the best results.

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
        initial_state: _AgentState = {
            "query": query,
            "document_urls": [document_url],
            "top_k": effective_top_k,
            "results": [],
            "score": 0.0,
            "iteration": 0,
            "max_iterations": self.max_iterations,
        }
        final_state = self._graph.invoke(initial_state)
        self.last_iterations = final_state["iteration"]
        return final_state["results"]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        graph: StateGraph = StateGraph(_AgentState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("evaluate", self._evaluate_node)
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
    # Node implementations
    # ------------------------------------------------------------------

    def _retrieve_node(self, state: _AgentState) -> dict:
        """Retrieve chunks for every document URL in state."""
        accumulated: list[RetrievalResult] = []
        for url in state["document_urls"]:
            hits = self.inner.retrieve(
                state["query"],
                document_url=url,
                top_k=state["top_k"],
            )
            accumulated.extend(hits)
        return {"results": accumulated}

    def _evaluate_node(self, state: _AgentState) -> dict:
        """Score the current results and increment the iteration counter."""
        score = score_chunks(state["results"], state["query"])
        return {"score": score, "iteration": state["iteration"] + 1}

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
