"""Guardrails models: GuardrailsReport and CheckResult."""

from __future__ import annotations

from pydantic import BaseModel


class CheckResult(BaseModel):
    name: str
    passed: bool
    reason: str = ""


class GuardrailsReport(BaseModel):
    pii: CheckResult
    hallucination: CheckResult
    confidence: CheckResult
    passed: bool

    def failure_summary(self) -> str:
        """Return a human-readable summary of failing checks for gate_reason."""
        failing = [
            c for c in (self.pii, self.hallucination, self.confidence) if not c.passed
        ]
        if not failing:
            return ""
        parts = ", ".join(f"{c.name} ({c.reason})" for c in failing)
        return f"guardrails failed: {parts}"
