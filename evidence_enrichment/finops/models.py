"""FinOps domain models for cost attribution, budget policy, and run summaries."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BudgetMode(str, Enum):
    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


class BudgetStatus(str, Enum):
    NOMINAL = "nominal"
    WARN = "warn"
    EXCEEDED = "exceeded"
    BLOCKED = "blocked"


class UsageSource(str, Enum):
    ESTIMATED = "estimated"
    PROVIDER_REPORTED = "provider_reported"


class LLMUsage(BaseModel):
    """Token usage captured at LLM call time.

    ``input_tokens`` and ``output_tokens`` are taken directly from the
    provider response when available (``usage_source=PROVIDER_REPORTED``),
    or estimated via the ``chars/4`` heuristic when the provider does not
    return usage data (``usage_source=ESTIMATED``).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    usage_source: UsageSource = UsageSource.ESTIMATED


class DowngradeAction(str, Enum):
    NONE = "none"
    RETRIEVAL_OFF = "retrieval_off"
    CHEAP_MODEL = "cheap_model"
    BLOCK = "block"


class StageCostRecord(BaseModel):
    stage: str
    provider: str
    model_name: str
    call_count: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    usage_source: UsageSource = UsageSource.ESTIMATED
    downgrade_applied: DowngradeAction = DowngradeAction.NONE


class BudgetDecision(BaseModel):
    status: BudgetStatus = BudgetStatus.NOMINAL
    budget_limit_usd: float | None = None
    projected_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    downgrade_actions: list[DowngradeAction] = Field(default_factory=list)
    budget_reason: str | None = None


class RunFinOpsSummary(BaseModel):
    total_estimated_cost_usd: float = 0.0
    cost_by_stage: dict[str, float] = Field(default_factory=dict)
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    cost_by_provider: dict[str, float] = Field(default_factory=dict)
    stage_records: list[StageCostRecord] = Field(default_factory=list)
    budget_decision: BudgetDecision = Field(default_factory=BudgetDecision)
    downgrade_actions: list[DowngradeAction] = Field(default_factory=list)
    pricing_catalog_version: str = ""
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
