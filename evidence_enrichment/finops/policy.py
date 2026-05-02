"""Budget policy engine for AI cost governance.

Implements three modes:
  off    — collect metrics only
  warn   — collect metrics and flag budget breaches
  strict — downgrade execution path, then block if still over budget

All policy decisions are deterministic and recorded in artifacts.
"""

from __future__ import annotations

from evidence_enrichment.finops.collector import FinOpsCollector
from evidence_enrichment.finops.models import (
    BudgetDecision,
    BudgetMode,
    BudgetStatus,
)
from evidence_enrichment.finops.pricing import PricingCatalog


class BudgetPolicyEngine:
    def __init__(
        self,
        *,
        mode: BudgetMode = BudgetMode.OFF,
        max_cost_per_run: float | None = None,
        max_cost_per_success: float | None = None,
        catalog: PricingCatalog,
    ) -> None:
        self.mode = mode
        self.max_cost_per_run = max_cost_per_run
        self.max_cost_per_success = max_cost_per_success
        self._catalog = catalog
        self._downgrade_exhausted = False

    @property
    def is_enforcing(self) -> bool:
        return self.mode != BudgetMode.OFF

    def check_before_stage(
        self,
        collector: FinOpsCollector,
        *,
        projected_marginal_cost: float = 0.0,
    ) -> BudgetDecision:
        if self.mode == BudgetMode.OFF:
            return BudgetDecision(status=BudgetStatus.NOMINAL)

        current = collector.accrued_cost_usd
        projected = current + projected_marginal_cost

        if self.max_cost_per_run is None or projected <= self.max_cost_per_run:
            return BudgetDecision(
                status=BudgetStatus.NOMINAL,
                budget_limit_usd=self.max_cost_per_run,
                projected_cost_usd=round(projected, 8),
                actual_cost_usd=round(current, 8),
            )

        if self.mode == BudgetMode.WARN:
            return BudgetDecision(
                status=BudgetStatus.WARN,
                budget_limit_usd=self.max_cost_per_run,
                projected_cost_usd=round(projected, 8),
                actual_cost_usd=round(current, 8),
                budget_reason=(
                    f"Projected cost ${projected:.6f} exceeds budget "
                    f"${self.max_cost_per_run:.6f}"
                ),
            )

        return self._apply_strict(collector, projected_marginal_cost)

    def check_post_run(
        self,
        collector: FinOpsCollector,
        *,
        succeeded: bool,
    ) -> BudgetDecision:
        actual = collector.accrued_cost_usd
        if self.mode == BudgetMode.OFF:
            return BudgetDecision(
                status=BudgetStatus.NOMINAL,
                actual_cost_usd=round(actual, 8),
            )

        over_run = (
            self.max_cost_per_run is not None and actual > self.max_cost_per_run
        )
        over_success = (
            succeeded
            and self.max_cost_per_success is not None
            and actual > self.max_cost_per_success
        )

        if over_run or over_success:
            status = BudgetStatus.EXCEEDED
            reasons: list[str] = []
            if over_run:
                reasons.append(
                    f"Actual cost ${actual:.6f} exceeds per-run budget "
                    f"${self.max_cost_per_run:.6f}"
                )
            if over_success:
                reasons.append(
                    f"Actual cost ${actual:.6f} exceeds per-success budget "
                    f"${self.max_cost_per_success:.6f}"
                )
            return BudgetDecision(
                status=status,
                budget_limit_usd=self.max_cost_per_run,
                actual_cost_usd=round(actual, 8),
                budget_reason="; ".join(reasons),
            )

        return BudgetDecision(
            status=BudgetStatus.NOMINAL,
            budget_limit_usd=self.max_cost_per_run,
            actual_cost_usd=round(actual, 8),
        )

    def mark_downgrade_exhausted(self) -> None:
        self._downgrade_exhausted = True

    def reset_downgrade_exhausted(self) -> None:
        """Reset the per-stage downgrade flag.

        Call this at the start of each stage's budget resolution so that every
        stage gets an independent downgrade opportunity.  ``reset_run_state``
        should still be called between full pipeline runs.
        """
        self._downgrade_exhausted = False

    def reset_run_state(self) -> None:
        self._downgrade_exhausted = False

    def _apply_strict(
        self,
        collector: FinOpsCollector,
        projected_marginal_cost: float,
    ) -> BudgetDecision:
        current = collector.accrued_cost_usd
        projected = current + projected_marginal_cost

        if not self._downgrade_exhausted:
            return BudgetDecision(
                status=BudgetStatus.WARN,
                budget_limit_usd=self.max_cost_per_run,
                projected_cost_usd=round(projected, 8),
                actual_cost_usd=round(current, 8),
                downgrade_actions=[],
                budget_reason=(
                    f"Strict budget: projected ${projected:.6f} exceeds "
                    f"${self.max_cost_per_run:.6f}. Applying downgrades."
                ),
            )

        return BudgetDecision(
            status=BudgetStatus.BLOCKED,
            budget_limit_usd=self.max_cost_per_run,
            projected_cost_usd=round(projected, 8),
            actual_cost_usd=round(current, 8),
            downgrade_actions=[],
            budget_reason=(
                f"Strict budget: projected ${projected:.6f} exceeds "
                f"${self.max_cost_per_run:.6f}. Downgrade path exhausted."
            ),
        )

    def should_disable_retrieval(self, decision: BudgetDecision) -> bool:
        if self.mode != BudgetMode.STRICT:
            return False
        return decision.status in (BudgetStatus.WARN, BudgetStatus.EXCEEDED, BudgetStatus.BLOCKED)

    def should_use_cheap_model(self, decision: BudgetDecision) -> bool:
        if self.mode != BudgetMode.STRICT:
            return False
        return decision.status in (BudgetStatus.WARN, BudgetStatus.EXCEEDED, BudgetStatus.BLOCKED)
