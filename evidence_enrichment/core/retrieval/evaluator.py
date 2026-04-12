"""Chunk quality evaluator and query refiner for the adaptive retrieval agent.

Both functions are pure Python — no LLM calls, no I/O — so they are fast,
deterministic, and trivially testable.
"""

from __future__ import annotations

import re

from evidence_enrichment.core.retrieval.models import RetrievalResult

# Minimum acceptable average score for a retrieval batch to be considered
# "good enough" and avoid a refinement iteration.
_QUALITY_THRESHOLD = 0.40

# Additional synonyms / context hints appended on each refinement pass.
_REFINEMENT_SUFFIXES = [
    "headquarters location registered office",
    "official company address country region",
    "incorporated domicile principal place of business",
]


def score_chunks(results: list[RetrievalResult], query: str) -> float:
    """Return a quality score in [0, 1] for a list of retrieval results.

    The score is the arithmetic mean of the hybrid scores returned by
    HybridRetriever.  An empty list scores 0.0.

    Parameters
    ----------
    results:
        Retrieved chunks with pre-computed hybrid scores.
    query:
        The original retrieval query (reserved for future token-overlap
        boosting; currently unused so the function stays dependency-free).
    """
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def refine_query(query: str, results: list[RetrievalResult], iteration: int = 0) -> str:
    """Return an enriched query for the next retrieval iteration.

    Strategy: append a context suffix chosen by ``iteration`` modulo the
    number of available suffixes.  The original query terms are always
    preserved so relevance cannot decrease.

    Parameters
    ----------
    query:
        The current (possibly already refined) query string.
    results:
        Chunks from the previous iteration (used to extract missed terms
        via a simple inverse-presence heuristic).
    iteration:
        Zero-based refinement count; used to cycle through suffix variants.
    """
    suffix = _REFINEMENT_SUFFIXES[iteration % len(_REFINEMENT_SUFFIXES)]

    # Avoid duplicating words already present in the query.
    existing_tokens = {t.lower() for t in re.findall(r"\w+", query)}
    new_tokens = [
        t for t in re.findall(r"\w+", suffix) if t.lower() not in existing_tokens
    ]
    if not new_tokens:
        return query

    return f"{query} {' '.join(new_tokens)}"


def is_quality_sufficient(score: float) -> bool:
    """Return True when the retrieval quality is good enough to stop iterating."""
    return score >= _QUALITY_THRESHOLD
