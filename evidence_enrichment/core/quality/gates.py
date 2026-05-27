"""Confidence aggregation and review gating."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from evidence_enrichment.core.models.contracts import FactClaim, SynthesisResult
from evidence_enrichment.core.models.enums import ReviewDecision

if TYPE_CHECKING:
    from evidence_enrichment.core.extraction.models import ExtractionResult


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


# ── Stage C — Schema validation gate ─────────────────────────────────────────


@dataclass
class SchemaGateResult:
    """Result of running ``SchemaValidationGate.check()`` on an extraction.

    Attributes
    ----------
    passed:
        ``True`` when no hard-fail errors were found.
    confidence_after:
        Adjusted extraction confidence (penalties applied for soft-fail errors).
    hard_errors:
        Error tags from ``SchemaValidationGate.HARD_FAIL_ERRORS`` that were
        detected.  Non-empty → ``passed=False``.
    soft_errors:
        Error tags from ``SchemaValidationGate.SOFT_FAIL_ERRORS`` that were
        detected.  Each reduces confidence by ``CONFIDENCE_PENALTY``.
    annotations:
        Human-readable descriptions of detected issues.
    """

    passed: bool = True
    confidence_after: float = 0.0
    hard_errors: list[str] = field(default_factory=list)
    soft_errors: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)


class SchemaValidationGate:
    """Post-extraction gate that classifies validation failures as hard or soft.

    Hard failures (``HARD_FAIL_ERRORS``) set ``passed=False`` on the gate
    result — the extraction should not be used downstream without human review.

    Soft failures (``SOFT_FAIL_ERRORS``) apply a ``CONFIDENCE_PENALTY`` to
    the extraction confidence but do not block the result.

    This class operates on ``ExtractionResult`` objects only.  The existing
    ``QualityGate`` (``compute_overall_confidence`` / ``gate_result``) is not
    modified.

    Parameters
    ----------
    confidence_penalty:
        Per-soft-error confidence reduction.  Default ``0.15``.
    """

    HARD_FAIL_ERRORS: frozenset[str] = frozenset(
        {"row_sum_divergence", "missing_provenance"}
    )
    SOFT_FAIL_ERRORS: frozenset[str] = frozenset(
        {"percentage_sum_off", "missing_period", "unit_mismatch"}
    )
    CONFIDENCE_PENALTY: float = 0.15

    def __init__(self, confidence_penalty: float = 0.15) -> None:
        self._confidence_penalty = confidence_penalty

    def check(self, result: "ExtractionResult") -> SchemaGateResult:
        """Classify ``result.validation_errors`` and return a gate result.

        Parameters
        ----------
        result:
            An ``ExtractionResult`` — typically one produced by
            ``SchemaExtractor.extract()``.

        Returns
        -------
        SchemaGateResult
            ``passed=True`` when no hard-fail error tags are present.
            ``confidence_after`` is ``result.extraction_confidence`` minus
            ``confidence_penalty`` for each detected soft-fail error, clamped
            to ``[0.0, 1.0]``.
        """

        gate = SchemaGateResult(confidence_after=result.extraction_confidence)

        # When validation already passed cleanly, there is nothing to classify.
        if result.validation_passed and not result.validation_errors:
            return gate

        hard: list[str] = []
        soft: list[str] = []
        annotations: list[str] = []

        for error_msg in result.validation_errors:
            tag = self._classify(error_msg)
            if tag in self.HARD_FAIL_ERRORS:
                hard.append(tag)
                annotations.append(f"[HARD:{tag}] {error_msg}")
            elif tag in self.SOFT_FAIL_ERRORS:
                soft.append(tag)
                annotations.append(f"[SOFT:{tag}] {error_msg}")
            else:
                # Unknown error — treat as soft to avoid over-blocking.
                soft.append("unknown")
                annotations.append(f"[SOFT:unknown] {error_msg}")

        confidence = result.extraction_confidence
        confidence -= len(soft) * self._confidence_penalty
        confidence = max(0.0, min(1.0, confidence))

        gate.passed = len(hard) == 0
        gate.confidence_after = confidence
        gate.hard_errors = hard
        gate.soft_errors = soft
        gate.annotations = annotations
        return gate

    # ── Private helpers ──────────────────────────────────────────────────────

    def _classify(self, error_msg: str) -> str:
        """Map an error message to an error tag.

        Errors emitted by the extractor have the form ``"<loc>: <msg>"`` where
        ``<loc>`` is the dot-joined Pydantic field path and ``<msg>`` is the
        Pydantic ``error["msg"]`` string.  Matching is by substring so the
        format is tolerant of future message changes.

        Returns ``"unknown"`` when no tag matches.
        """
        lowered = error_msg.lower()
        # Hard fails
        if "row sum" in lowered or "row_sum" in lowered or "differs from reported total" in lowered:
            return "row_sum_divergence"
        # Provenance: loc contains "source_chunk_ids"; msg is "List should have at least 1 item"
        if (
            "source_chunk_ids" in lowered
            or "missing_provenance" in lowered
            or "at least 1 item" in lowered
            or "too_short" in lowered
        ):
            return "missing_provenance"
        # Soft fails
        if "percentage" in lowered or "pct" in lowered or "percent" in lowered:
            return "percentage_sum_off"
        if "period" in lowered:
            return "missing_period"
        if "unit" in lowered:
            return "unit_mismatch"
        return "unknown"
