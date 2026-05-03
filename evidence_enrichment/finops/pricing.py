"""Versioned pricing catalog for AI cost estimation.

Provides a static, deterministic mapping of model names to per-token prices.
The catalog is versioned so artifacts can record which version produced their
estimates. Operators can override prices via ``finops.pricing_override`` in
``evidence_enrichment.yaml``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

CATALOG_VERSION = "2025-01-01"

DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "GPT-5.5": {
        "input_per_1m": 5.00,
        "output_per_1m": 30.00,
    },
    "GPT-5.4": {
        "input_per_1m": 2.50,
        "output_per_1m": 15.00,
    },
    "gpt-5-mini": {
        "input_per_1m": 0.25,
        "output_per_1m": 2.00,
    },
    "claude-opus-4.7": {
        "input_per_1m": 5.00,
        "output_per_1m": 25.00,
    },
    "claude-sonnet-4.6": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    "text-embedding-3-small": {
        "input_per_1m": 0.02,
        "output_per_1m": 0.00,
    },
    "text-embedding-3-large": {
        "input_per_1m": 0.13,
        "output_per_1m": 0.00,
    },
}


class PricingCatalog(BaseModel):
    version: str = CATALOG_VERSION
    prices: dict[str, dict[str, float]] = DEFAULT_PRICES

    def cost_for_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        tier = self.prices.get(model)
        if tier is None:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * tier.get("input_per_1m", 0.0)
        output_cost = (output_tokens / 1_000_000) * tier.get("output_per_1m", 0.0)
        return round(input_cost + output_cost, 8)

    def lookup(self, model: str) -> dict[str, float]:
        return dict(self.prices.get(model, {}))


def build_catalog(overrides: dict[str, Any] | None = None) -> PricingCatalog:
    prices = dict(DEFAULT_PRICES)
    if overrides:
        for model, tier in overrides.items():
            if isinstance(tier, dict):
                prices[model] = {
                    "input_per_1m": float(tier.get("input_per_1m", 0.0)),
                    "output_per_1m": float(tier.get("output_per_1m", 0.0)),
                }
    return PricingCatalog(version=CATALOG_VERSION, prices=prices)
