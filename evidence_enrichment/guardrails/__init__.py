"""Guardrails package — post-synthesis safety checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evidence_enrichment.guardrails.confidence import check_confidence_floor
from evidence_enrichment.guardrails.hallucination import check_hallucination
from evidence_enrichment.guardrails.models import CheckResult, GuardrailsReport
from evidence_enrichment.guardrails.pii import check_pii

if TYPE_CHECKING:
    pass

__all__ = ["CheckResult", "GuardrailsReport", "run_guardrails"]


def run_guardrails(
    *,
    synthesis: Any,
    analysis_reports: list[Any],
    parsed_documents: list[Any],
    overall_confidence: float,
    floor: float = 0.4,
    entities: list[str] | None = None,
) -> GuardrailsReport:
    """Run all three guardrail checks and return a consolidated report."""
    # --- PII: scan synthesis text + each claim's supporting excerpt ---
    texts_to_scan = [synthesis.value or "", synthesis.reasoning or ""]
    for r in analysis_reports:
        for c in r.claims:
            texts_to_scan.append(c.supporting_excerpt or "")

    pii = check_pii(texts_to_scan, entities=entities)

    # --- Hallucination: claim URLs must be in fetched document set ---
    claim_urls = [
        c.source_url for r in analysis_reports for c in r.claims if c.source_url
    ]
    fetched_urls = {d.url for d in parsed_documents}
    hallucination = check_hallucination(claim_urls, fetched_urls)

    # --- Confidence floor ---
    confidence = check_confidence_floor(overall_confidence, floor)

    return GuardrailsReport(
        pii=pii,
        hallucination=hallucination,
        confidence=confidence,
        passed=all(c.passed for c in (pii, hallucination, confidence)),
    )
