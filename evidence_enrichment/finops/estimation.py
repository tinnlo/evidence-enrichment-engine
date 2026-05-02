"""Deterministic token and cost estimation.

Uses a simple chars/4 heuristic that matches the existing convention in
``context/resolver.py``. All estimates are intentionally conservative
and documented as approximations, not provider billing truth.
"""

from __future__ import annotations

import math

from evidence_enrichment.finops.models import StageCostRecord, UsageSource
from evidence_enrichment.finops.pricing import PricingCatalog


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def estimate_stage_cost(
    *,
    stage: str,
    provider: str,
    model_name: str,
    input_text: str,
    output_text: str,
    call_count: int = 1,
    catalog: PricingCatalog,
    usage_source: UsageSource = UsageSource.ESTIMATED,
) -> StageCostRecord:
    """Estimate the total cost for a pipeline stage.

    ``input_text`` and ``output_text`` represent a *single call's* worth of text.
    When a stage makes multiple LLM calls (e.g. one per document in analysis),
    pass ``call_count`` and the *average* per-call text; total cost is scaled
    by ``call_count``.  For single-call stages (synthesis, query-plan) leave
    ``call_count=1`` and pass the full text.
    """
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    cost_per_call = catalog.cost_for_tokens(model_name, input_tokens, output_tokens)
    total_cost = round(cost_per_call * call_count, 8)
    return StageCostRecord(
        stage=stage,
        provider=provider,
        model_name=model_name,
        call_count=call_count,
        estimated_input_tokens=input_tokens * call_count,
        estimated_output_tokens=output_tokens * call_count,
        estimated_total_tokens=(input_tokens + output_tokens) * call_count,
        estimated_cost_usd=total_cost,
        usage_source=usage_source,
    )


def stage_cost_from_tokens(
    *,
    stage: str,
    provider: str,
    model_name: str,
    total_input_tokens: int,
    total_output_tokens: int,
    call_count: int,
    catalog: PricingCatalog,
    usage_source: UsageSource = UsageSource.ESTIMATED,
) -> StageCostRecord:
    """Build a ``StageCostRecord`` directly from pre-summed token counts.

    Use this instead of :func:`estimate_stage_cost` when the caller has already
    computed exact per-document token sums (e.g. by summing ``ceil(chars/4)``
    for each document individually).  This avoids a second round-trip through
    the ``chars → ceil(chars/4)`` heuristic which can change totals when
    per-call sizes are uneven or not divisible by 4.

    ``call_count`` is stored as-is for observability; cost is computed directly
    from the supplied totals.
    """
    cost = catalog.cost_for_tokens(model_name, total_input_tokens, total_output_tokens)
    return StageCostRecord(
        stage=stage,
        provider=provider,
        model_name=model_name,
        call_count=call_count,
        estimated_input_tokens=total_input_tokens,
        estimated_output_tokens=total_output_tokens,
        estimated_total_tokens=total_input_tokens + total_output_tokens,
        estimated_cost_usd=round(cost, 8),
        usage_source=usage_source,
    )


def estimate_embedding_cost(
    *,
    stage: str,
    provider: str,
    model_name: str,
    text_count: int,
    total_chars: int,
    per_text_chars: list[int] | None = None,
    catalog: PricingCatalog,
) -> StageCostRecord:
    """Estimate cost for a batch of embedding calls.

    When ``per_text_chars`` is provided, tokens are summed per-text using
    ``ceil(chars/4)`` for each entry, which avoids the systematic undercount
    that results from tokenizing all chars as one blob (ceil is non-linear).
    Falls back to ``ceil(total_chars/4)`` when only the aggregate is known.
    """
    if per_text_chars:
        total_tokens = sum(estimate_tokens("x" * c) for c in per_text_chars)
    else:
        total_tokens = estimate_tokens("x" * total_chars)
    cost = catalog.cost_for_tokens(model_name, total_tokens, 0)
    return StageCostRecord(
        stage=stage,
        provider=provider,
        model_name=model_name,
        call_count=text_count,
        estimated_input_tokens=total_tokens,
        estimated_output_tokens=0,
        estimated_total_tokens=total_tokens,
        estimated_cost_usd=cost,
        usage_source=UsageSource.ESTIMATED,
    )
