"""Confidence floor check."""

from __future__ import annotations

from evidence_enrichment.guardrails.models import CheckResult

_DEFAULT_FLOOR = 0.4


def check_confidence_floor(
    overall_confidence: float,
    floor: float = _DEFAULT_FLOOR,
) -> CheckResult:
    """Fail when overall_confidence is strictly below the floor threshold."""
    if overall_confidence < floor:
        return CheckResult(
            name="confidence",
            passed=False,
            reason=f"{overall_confidence:.2f} < {floor:.2f}",
        )
    return CheckResult(name="confidence", passed=True)
