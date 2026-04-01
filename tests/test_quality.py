from evidence_enrichment.core.evidence.assessment import compute_source_authority
from evidence_enrichment.core.models.contracts import ConflictManifest, FactClaim, SynthesisResult
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


# ---------------------------------------------------------------------------
# Authority scoring — exact/subdomain match
# ---------------------------------------------------------------------------


def test_authority_known_domain_exact_match() -> None:
    score = compute_source_authority("sec.gov")
    assert score == 0.98


def test_authority_known_domain_subdomain_match() -> None:
    score = compute_source_authority("data.sec.gov")
    assert score == 0.98


def test_authority_lookalike_domain_not_elevated() -> None:
    """sec.gov.evil.example must not receive the sec.gov trust score."""
    score = compute_source_authority("sec.gov.evil.example")
    assert score == 0.30  # _default


def test_authority_www_prefix_stripped() -> None:
    score = compute_source_authority("www.reuters.com")
    assert score == 0.85


# ---------------------------------------------------------------------------
# Gate edge cases
# ---------------------------------------------------------------------------


def test_gate_result_auto_reject_no_claims() -> None:
    synthesis = SynthesisResult(
        field_name="hq_country",
        value="USA",
        normalized_value="USA",
        reasoning="No claims.",
        synthesis_confidence=0.9,
    )
    decision, reason = gate_result(0.0, [], synthesis)
    assert decision.value == "auto_reject"
    assert reason == "no_supported_value"


def test_gate_result_auto_reject_no_value() -> None:
    claims = [_claim("USA", 0.9, 0.9, 0.9, 0.9)]
    synthesis = SynthesisResult(
        field_name="hq_country",
        value=None,
        normalized_value=None,
        reasoning="Unclear.",
        synthesis_confidence=0.3,
    )
    decision, reason = gate_result(0.5, claims, synthesis)
    assert decision.value == "auto_reject"
    assert reason == "no_supported_value"


def test_gate_result_conflicting_claims_needs_review() -> None:
    claims = [_claim("USA", 0.9, 0.9, 0.9, 0.9), _claim("GBR", 0.9, 0.9, 0.9, 0.9)]
    synthesis = SynthesisResult(
        field_name="hq_country",
        value="USA",
        normalized_value="USA",
        reasoning="Conflicting.",
        synthesis_confidence=0.9,
        conflicts=[
            ConflictManifest(
                field_name="hq_country",
                candidate_values=["USA", "GBR"],
                source_urls=["https://a.com", "https://b.com"],
                reason="disagreement",
            )
        ],
    )
    confidence = compute_overall_confidence(claims, synthesis)
    decision, reason = gate_result(confidence, claims, synthesis)
    assert decision.value == "needs_review"
    assert "conflicting" in reason

