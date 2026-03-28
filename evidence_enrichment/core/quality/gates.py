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
    raw = (
        authority * 25
        + freshness * 15
        + agreement * 25
        + relevance * 15
        + min(synthesis.synthesis_confidence, 1.0) * 10
    ) / 90
    return round(raw, 4)


def gate_result(overall_confidence: float, claims: list[FactClaim], synthesis: SynthesisResult) -> tuple[ReviewDecision, str]:
    if not claims or synthesis.value is None:
        return ReviewDecision.AUTO_REJECT, "no_supported_value"
    if synthesis.conflicts:
        if overall_confidence >= 0.85:
            return ReviewDecision.NEEDS_REVIEW, "conflicting_claims"
        if overall_confidence < 0.50:
            return ReviewDecision.AUTO_REJECT, "conflicting_claims_low_confidence"
        return ReviewDecision.NEEDS_REVIEW, "conflicting_claims"
    if overall_confidence >= 0.85:
        return ReviewDecision.AUTO_APPROVE, "meets_thresholds"
    if overall_confidence >= 0.50:
        return ReviewDecision.NEEDS_REVIEW, "mid_confidence"
    return ReviewDecision.AUTO_REJECT, "below_review_band"
