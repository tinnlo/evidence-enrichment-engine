"""Confidence aggregation and review gating."""

from __future__ import annotations

from collections import Counter

from evidence_enrichment.core.models.contracts import FactClaim, SynthesisResult
from evidence_enrichment.core.models.enums import ReviewDecision


def compute_overall_confidence(claims: list[FactClaim], synthesis: SynthesisResult) -> float:
    if not claims:
        return min(synthesis.synthesis_confidence, 0.2)
    values = [claim.candidate_value for claim in claims if claim.candidate_value]
    agreement = 0.0
    if values:
        counts = Counter(values)
        agreement = counts.most_common(1)[0][1] / len(values)
    authority = sum(claim.source_authority_score for claim in claims) / len(claims)
    freshness = sum(claim.freshness_score for claim in claims) / len(claims)
    relevance = sum(claim.entity_match_score for claim in claims) / len(claims)
    avg_analysis_conf = sum(claim.analysis_confidence for claim in claims) / len(claims)
    raw = (
        authority * 20
        + freshness * 12
        + agreement * 20
        + relevance * 12
        + avg_analysis_conf * 16
        + synthesis.synthesis_confidence * 10
    ) / 90
    return round(raw, 4)


def gate_result(
    overall_confidence: float,
    claims: list[FactClaim],
    synthesis: SynthesisResult,
    *,
    auto_approve_min: float = 0.85,
    review_min: float = 0.50,
) -> tuple[ReviewDecision, str]:
    if not claims or synthesis.value is None:
        return ReviewDecision.AUTO_REJECT, "no_supported_value"
    if synthesis.conflicts:
        if overall_confidence >= auto_approve_min:
            return ReviewDecision.NEEDS_REVIEW, "conflicting_claims"
        if overall_confidence < review_min:
            return ReviewDecision.AUTO_REJECT, "conflicting_claims_low_confidence"
        return ReviewDecision.NEEDS_REVIEW, "conflicting_claims"
    if overall_confidence >= auto_approve_min:
        return ReviewDecision.AUTO_APPROVE, "meets_thresholds"
    if overall_confidence >= review_min:
        return ReviewDecision.NEEDS_REVIEW, "mid_confidence"
    return ReviewDecision.AUTO_REJECT, "below_review_band"
