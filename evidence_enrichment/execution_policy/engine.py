"""Execution policy engine.

Single source of truth for allow / audit / block decisions on live-capability
surfaces.  Mirrors the BudgetPolicyEngine pattern from finops/policy.py but is
simpler: no cost cascade, just allow/record/block per action.

Instantiate one engine per pipeline run.  Call check_action() at each
live-capability gate.  Call build_report() at run end to get the artifact.
Call reset() between runs when reusing the same engine instance (e.g. tests).
"""

from __future__ import annotations

import logging

from evidence_enrichment.config.settings import ExecutionPolicySettings
from evidence_enrichment.execution_policy.models import (
    ActionType,
    ExecutionPolicyReport,
    PolicyDecision,
    PolicyMode,
)

_policy_logger = logging.getLogger(__name__)


class ExecutionPolicyEngine:
    """Evaluates and records execution policy decisions for a pipeline run.

    Thread-safety: instances are not thread-safe.  Create one per run
    (coordinator already does this via a run-scoped context).
    """

    def __init__(self, settings: ExecutionPolicySettings) -> None:
        self._enabled: bool = settings.enabled
        try:
            self._mode: PolicyMode = PolicyMode(settings.mode)
        except ValueError:
            _policy_logger.warning(
                "Unknown execution_policy.mode %r — falling back to 'off'.",
                settings.mode,
            )
            self._mode = PolicyMode.OFF
        self._allowed: frozenset[str] = frozenset(settings.allowed_actions)
        self._decisions: list[PolicyDecision] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mode(self) -> PolicyMode:
        return self._mode

    def check_action(self, action: ActionType) -> PolicyDecision:
        """Evaluate whether *action* is permitted under the current policy.

        off mode
            Always allowed; no decision is recorded.
        audit mode
            Always allowed; a decision is recorded, marked as a violation if
            the action is not in allowed_actions.
        enforce mode
            Allowed only if the action is in allowed_actions.  Disallowed
            actions are blocked and recorded.

        Returns the PolicyDecision for the caller to inspect.
        """
        if not self._enabled or self._mode is PolicyMode.OFF:
            return PolicyDecision(
                action=action,
                allowed=True,
                mode=self._mode,
                reason="policy_off",
            )

        action_permitted = action.value in self._allowed

        if self._mode is PolicyMode.AUDIT:
            reason = "audit_allowed" if action_permitted else "audit_violation"
            decision = PolicyDecision(
                action=action,
                allowed=True,
                mode=self._mode,
                reason=reason,
            )
            self._decisions.append(decision)
            if not action_permitted:
                _policy_logger.warning(
                    "Execution policy audit: action %r is not in allowed_actions "
                    "(mode=audit, execution continues).",
                    action.value,
                )
            return decision

        # enforce mode
        if action_permitted:
            decision = PolicyDecision(
                action=action,
                allowed=True,
                mode=self._mode,
                reason="enforce_allowed",
            )
        else:
            decision = PolicyDecision(
                action=action,
                allowed=False,
                mode=self._mode,
                reason=f"enforce_blocked:{action.value}_not_in_allowed_actions",
            )
            _policy_logger.warning(
                "Execution policy enforce: action %r blocked — not in allowed_actions.",
                action.value,
            )
        self._decisions.append(decision)
        return decision

    def build_report(self) -> ExecutionPolicyReport:
        """Build the aggregate policy report for the current run."""
        violations = [d for d in self._decisions if not d.allowed or "violation" in d.reason]
        blocked = [d.action for d in self._decisions if not d.allowed]
        return ExecutionPolicyReport(
            mode=self._mode,
            decisions=list(self._decisions),
            blocked_actions=blocked,
            violations=violations,
        )

    def reset(self) -> None:
        """Clear recorded decisions (for test isolation or run reuse)."""
        self._decisions.clear()
