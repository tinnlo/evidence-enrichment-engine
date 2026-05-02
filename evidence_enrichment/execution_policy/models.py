"""Domain models for the execution policy layer.

Execution policy governs *capability* (what actions the agent is permitted to
take at runtime).  It is intentionally separate from the FinOps layer, which
governs *cost*.  A run can be policy-allowed but budget-breaching, or
budget-safe but capability-blocked — both outcomes are independently visible
in their respective artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PolicyMode(str, Enum):
    """Operating mode for the execution policy engine.

    off     — policy is inactive; all actions pass through unrecorded.
    audit   — all actions are allowed, but violations (actions not in
              allowed_actions) are recorded in the policy report.
    enforce — actions not in allowed_actions are blocked; the pipeline
              returns a structured result with gate_reason set.
    """

    OFF = "off"
    AUDIT = "audit"
    ENFORCE = "enforce"


class ActionType(str, Enum):
    """Live-capability surfaces that the execution policy can govern."""

    SEARCH = "search"
    FETCH = "fetch"
    RETRIEVAL = "retrieval"
    REMOTE_TRACING = "remote_tracing"
    LIVE_PROVIDER_CALLS = "live_provider_calls"
    MCP_LIVE_RUNS = "mcp_live_runs"


class PolicyDecision(BaseModel):
    """A single policy decision for one action at a point in time."""

    action: ActionType
    allowed: bool
    mode: PolicyMode
    reason: str
    timestamp: datetime = Field(default_factory=_utc_now)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        d = super().model_dump(**kwargs)
        # Serialise enums to their string values for JSON artifact compatibility.
        d["action"] = self.action.value
        d["mode"] = self.mode.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


class ExecutionPolicyReport(BaseModel):
    """Aggregate policy report for a single pipeline run."""

    mode: PolicyMode
    # All decisions recorded during the run (audit + enforce modes only).
    decisions: list[PolicyDecision] = Field(default_factory=list)
    # Actions that were explicitly blocked (enforce mode only).
    blocked_actions: list[ActionType] = Field(default_factory=list)
    # Decisions where the action was not in allowed_actions (audit + enforce).
    violations: list[PolicyDecision] = Field(default_factory=list)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        d = super().model_dump(**kwargs)
        d["mode"] = self.mode.value
        d["decisions"] = [dec.model_dump() for dec in self.decisions]
        d["blocked_actions"] = [a.value for a in self.blocked_actions]
        d["violations"] = [v.model_dump() for v in self.violations]
        return d
