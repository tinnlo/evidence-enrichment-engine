"""Entity matching helpers."""

from __future__ import annotations

import re

from evidence_enrichment.core.search.query_planner import company_name_tokens


def compute_entity_match_score(company_name: str, text: str) -> float:
    tokens = company_name_tokens(company_name)
    if not tokens:
        return 0.0
    corpus = text.lower()
    exact_pattern = r"\b" + r"\s+".join(re.escape(token) for token in tokens[:3]) + r"\b"
    if re.search(exact_pattern, corpus):
        return 1.0
    hits = sum(1 for token in tokens if token in corpus)
    return hits / len(tokens)

