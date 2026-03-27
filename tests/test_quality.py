from evidence_enrichment.core.models.contracts import FactClaim, SynthesisResult
from evidence_enrichment.core.quality.gates import compute_overall_confidence, gate_result


def _claim(value: str, authority: float, freshness: float, relevance: float, confidence: float) -> FactClaim:
    return FactClaim(
        field_name="hq_country",
        candidate_value=value,
        supporting_excerpt="excerpt",
        source_url="https://example.com",
        source_title="Example",
        analysis_confidence=confidence,
        source_authority_score=authority,
        freshness_score=freshness,
        entity_match_score=relevance,
    )


def test_compute_overall_confidence_high_signal() -> None:
    claims = [
        _claim("USA", 0.95, 0.9, 1.0, 0.92),
        _claim("USA", 0.98, 0.9, 1.0, 0.95),
    ]
    synthesis = SynthesisResult(
        field_name="hq_country",
        value="USA",
        normalized_value="USA",
        reasoning="Strong agreement.",
        synthesis_confidence=0.93,
        supporting_urls=["https://example.com"],
        conflicts=[],
    )
    confidence = compute_overall_confidence(claims, synthesis)
    decision, reason = gate_result(confidence, claims, synthesis)
    assert confidence >= 0.85
    assert decision.value == "auto_approve"
    assert reason == "meets_thresholds"


def test_gate_result_needs_review_on_mid_confidence() -> None:
    claims = [_claim("USA", 0.45, 0.7, 0.66, 0.58)]
    synthesis = SynthesisResult(
        field_name="hq_country",
        value="USA",
        normalized_value="USA",
        reasoning="Weak support.",
        synthesis_confidence=0.58,
        supporting_urls=["https://example.com"],
        conflicts=[],
    )
    confidence = compute_overall_confidence(claims, synthesis)
    decision, reason = gate_result(confidence, claims, synthesis)
    assert 0.50 <= confidence < 0.85
    assert decision.value == "needs_review"
    assert reason == "mid_confidence"

