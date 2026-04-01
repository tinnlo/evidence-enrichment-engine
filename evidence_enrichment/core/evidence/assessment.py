"""Evidence scoring helpers."""

from __future__ import annotations

from datetime import date


SOURCE_AUTHORITY = {
    "microsoft.com": 0.95,
    "sec.gov": 0.98,
    "reuters.com": 0.85,
    "wikipedia.org": 0.45,
    "_default": 0.30,
}


def compute_source_authority(domain: str) -> float:
    domain = domain.lower().replace("www.", "")
    for known_domain, score in SOURCE_AUTHORITY.items():
        if known_domain == "_default":
            continue
        # Exact match or proper subdomain (e.g. data.sec.gov) — prevent
        # lookalike abuse such as sec.gov.evil.example getting a high score.
        if domain == known_domain or domain.endswith("." + known_domain):
            return score
    return SOURCE_AUTHORITY["_default"]


def compute_freshness_score(reference_date: date | None = None) -> float:
    return 0.9 if reference_date else 0.7

