"""Search planning helpers."""

from __future__ import annotations

import re


LEGAL_SUFFIXES = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "ltd",
    "ltd.",
    "limited",
    "llc",
    "ag",
    "se",
    "plc",
}


def company_name_tokens(name: str) -> list[str]:
    parts = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", name)]
    return [token for token in parts if token not in LEGAL_SUFFIXES and len(token) > 2]


def score_search_result(company_name: str, url: str, title: str, snippet: str) -> tuple[float, str]:
    tokens = company_name_tokens(company_name)
    corpus = f"{url} {title} {snippet}".lower()
    score = 0.0
    if any(token in url.lower() for token in tokens):
        score += 2.0
    token_hits = sum(1 for token in tokens if token in corpus)
    score += token_hits * 0.5
    if any(marker in corpus for marker in ["headquarters", "about", "company", "corporate"]):
        score += 0.5
    domain_tier = "other"
    if "sec.gov" in url or "companieshouse" in url:
        domain_tier = "regulatory"
        score += 0.6
    elif any(marker in url.lower() for marker in ["/about", "/company", "/corporate"]):
        domain_tier = "official"
        score += 0.4
    elif any(domain in url.lower() for domain in ["wikipedia.org", "linkedin.com", "reuters.com"]):
        domain_tier = "reference"
    return score, domain_tier

