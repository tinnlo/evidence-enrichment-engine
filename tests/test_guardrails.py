"""Tests for the guardrails package."""

from __future__ import annotations

from unittest.mock import MagicMock


from evidence_enrichment.guardrails import GuardrailsReport, run_guardrails
from evidence_enrichment.guardrails.confidence import check_confidence_floor
from evidence_enrichment.guardrails.hallucination import check_hallucination
from evidence_enrichment.guardrails.pii import check_pii


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------


def test_pii_detection_flags_email_regex_fallback() -> None:
    """Email addresses should be caught by the regex fallback path."""
    result = check_pii(["Contact us at user@example.com for more info."])
    assert result.name == "pii"
    assert result.passed is False
    assert "pii" in result.reason.lower() or "match" in result.reason.lower()


def test_pii_passes_clean_text() -> None:
    result = check_pii(
        ["Microsoft Corporation is headquartered in Redmond, Washington."]
    )
    assert result.passed is True


def test_pii_empty_texts() -> None:
    result = check_pii([])
    assert result.passed is True


def test_pii_multiple_texts_any_match_fails() -> None:
    result = check_pii(["clean text", "reach me at bad@actor.io"])
    assert result.passed is False


# ---------------------------------------------------------------------------
# Hallucination
# ---------------------------------------------------------------------------


def test_hallucination_flag_on_unknown_url() -> None:
    result = check_hallucination(
        claim_urls=["https://unknown.example.com/report"],
        fetched_urls={"https://microsoft.com/ir", "https://sec.gov/filing"},
    )
    assert result.name == "hallucination"
    assert result.passed is False
    assert "ungrounded" in result.reason


def test_hallucination_passes_when_all_grounded() -> None:
    result = check_hallucination(
        claim_urls=["https://microsoft.com/ir", "https://sec.gov/filing"],
        fetched_urls={"https://microsoft.com/ir", "https://sec.gov/filing"},
    )
    assert result.passed is True


def test_hallucination_passes_with_empty_claim_urls() -> None:
    result = check_hallucination(claim_urls=[], fetched_urls={"https://microsoft.com"})
    assert result.passed is True


def test_hallucination_flag_counts_correctly() -> None:
    result = check_hallucination(
        claim_urls=["https://a.com", "https://b.com", "https://c.com"],
        fetched_urls={"https://a.com"},
    )
    assert result.passed is False
    assert "2" in result.reason


# ---------------------------------------------------------------------------
# Confidence floor
# ---------------------------------------------------------------------------


def test_confidence_floor_rejects_below_threshold() -> None:
    result = check_confidence_floor(0.31, floor=0.4)
    assert result.name == "confidence"
    assert result.passed is False
    assert "0.31" in result.reason
    assert "0.40" in result.reason


def test_confidence_floor_passes_above_threshold() -> None:
    result = check_confidence_floor(0.85, floor=0.4)
    assert result.passed is True


def test_confidence_floor_passes_at_exact_floor() -> None:
    result = check_confidence_floor(0.4, floor=0.4)
    assert result.passed is True


# ---------------------------------------------------------------------------
# run_guardrails integration
# ---------------------------------------------------------------------------


def _make_claim(source_url: str, excerpt: str = "Some excerpt.") -> MagicMock:
    c = MagicMock()
    c.source_url = source_url
    c.supporting_excerpt = excerpt
    return c


def _make_report(source_url: str, claims: list) -> MagicMock:
    r = MagicMock()
    r.claims = claims
    r.source_url = source_url
    return r


def _make_synthesis(
    value: str = "USA", reasoning: str = "Based on filings."
) -> MagicMock:
    s = MagicMock()
    s.value = value
    s.reasoning = reasoning
    return s


def _make_parsed_doc(url: str) -> MagicMock:
    d = MagicMock()
    d.url = url
    return d


def test_run_guardrails_all_pass() -> None:
    synthesis = _make_synthesis()
    report = _make_report(
        "https://microsoft.com/ir",
        [_make_claim("https://microsoft.com/ir")],
    )
    doc = _make_parsed_doc("https://microsoft.com/ir")

    result = run_guardrails(
        synthesis=synthesis,
        analysis_reports=[report],
        parsed_documents=[doc],
        overall_confidence=0.9,
        floor=0.4,
    )
    assert isinstance(result, GuardrailsReport)
    assert result.passed is True
    assert result.failure_summary() == ""


def test_run_guardrails_any_failure_marks_report_failed() -> None:
    """An ungrounded claim URL should cause the report to fail."""
    synthesis = _make_synthesis()
    report = _make_report(
        "https://microsoft.com/ir",
        [_make_claim("https://UNKNOWN-URL.example.com/report")],
    )
    doc = _make_parsed_doc("https://microsoft.com/ir")

    result = run_guardrails(
        synthesis=synthesis,
        analysis_reports=[report],
        parsed_documents=[doc],
        overall_confidence=0.9,
        floor=0.4,
    )
    assert result.passed is False
    assert result.hallucination.passed is False
    summary = result.failure_summary()
    assert "hallucination" in summary
    assert "guardrails failed" in summary


def test_run_guardrails_confidence_fail() -> None:
    synthesis = _make_synthesis()
    doc = _make_parsed_doc("https://microsoft.com/ir")
    report = _make_report(
        "https://microsoft.com/ir", [_make_claim("https://microsoft.com/ir")]
    )

    result = run_guardrails(
        synthesis=synthesis,
        analysis_reports=[report],
        parsed_documents=[doc],
        overall_confidence=0.2,
        floor=0.4,
    )
    assert result.passed is False
    assert result.confidence.passed is False
    assert "confidence" in result.failure_summary()


def test_guardrails_report_failure_summary_multiple_failures() -> None:
    synthesis = _make_synthesis()
    report = _make_report(
        "https://microsoft.com/ir",
        [_make_claim("https://UNKNOWN.example.com")],
    )
    doc = _make_parsed_doc("https://microsoft.com/ir")

    result = run_guardrails(
        synthesis=synthesis,
        analysis_reports=[report],
        parsed_documents=[doc],
        overall_confidence=0.1,
        floor=0.4,
    )
    summary = result.failure_summary()
    assert "hallucination" in summary
    assert "confidence" in summary
