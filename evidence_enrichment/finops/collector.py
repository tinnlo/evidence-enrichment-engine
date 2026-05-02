"""FinOps cost collector that aggregates stage-level metrics into run summaries."""

from __future__ import annotations

from evidence_enrichment.finops.models import (
    BudgetDecision,
    DowngradeAction,
    RunFinOpsSummary,
    StageCostRecord,
)
from evidence_enrichment.finops.pricing import PricingCatalog


class FinOpsCollector:
    def __init__(self, catalog: PricingCatalog) -> None:
        self._catalog = catalog
        self._records: list[StageCostRecord] = []
        self._downgrade_actions: list[DowngradeAction] = []

    def record(self, record: StageCostRecord) -> None:
        self._records.append(record)

    def record_downgrade(self, action: DowngradeAction) -> None:
        self._downgrade_actions.append(action)

    @property
    def accrued_cost_usd(self) -> float:
        return round(sum(r.estimated_cost_usd for r in self._records), 8)

    def build_summary(
        self,
        *,
        budget_decision: BudgetDecision | None = None,
        total_latency_ms: float = 0.0,
    ) -> RunFinOpsSummary:
        cost_by_stage: dict[str, float] = {}
        cost_by_model: dict[str, float] = {}
        cost_by_provider: dict[str, float] = {}
        for rec in self._records:
            cost_by_stage[rec.stage] = round(
                cost_by_stage.get(rec.stage, 0.0) + rec.estimated_cost_usd, 8
            )
            cost_by_model[rec.model_name] = round(
                cost_by_model.get(rec.model_name, 0.0) + rec.estimated_cost_usd, 8
            )
            cost_by_provider[rec.provider] = round(
                cost_by_provider.get(rec.provider, 0.0) + rec.estimated_cost_usd, 8
            )
        return RunFinOpsSummary(
            total_estimated_cost_usd=self.accrued_cost_usd,
            cost_by_stage=cost_by_stage,
            cost_by_model=cost_by_model,
            cost_by_provider=cost_by_provider,
            stage_records=list(self._records),
            budget_decision=budget_decision or BudgetDecision(),
            downgrade_actions=list(self._downgrade_actions),
            pricing_catalog_version=self._catalog.version,
            total_latency_ms=round(total_latency_ms, 3),
        )
