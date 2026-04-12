"""Hallucination check: verify claim source URLs are in fetched document set."""

from __future__ import annotations

from evidence_enrichment.guardrails.models import CheckResult


def check_hallucination(
    claim_urls: list[str],
    fetched_urls: set[str],
) -> CheckResult:
    """Flag claims that reference URLs not present in the fetched document set.

    Pure-function; no external dependencies.
    """
    if not claim_urls:
        return CheckResult(name="hallucination", passed=True)

    ungrounded = [url for url in claim_urls if url not in fetched_urls]
    if ungrounded:
        return CheckResult(
            name="hallucination",
            passed=False,
            reason=f"{len(ungrounded)} ungrounded claim(s)",
        )
    return CheckResult(name="hallucination", passed=True)
