"""Pydantic schemas for typed extraction fields.

Each schema class corresponds to one or more named pipeline fields.  The
``SCHEMA_REGISTRY`` maps field names to their schema class and version.

Design notes
------------
* Row-sum and percentage validators use ``mode="after"`` model_validators so
  they have access to all fields simultaneously.
* ``source_chunk_ids`` is required on every row (``min_length=1``) to enforce
  provenance tracing back to the retrieval layer.
* ``MoneyAmount.unit_multiplier`` normalises reported figures — a value of
  ``42`` with ``unit_multiplier=1_000_000`` means 42 million in the stated
  currency.
* ``SCHEMA_REGISTRY`` maps ``field_name → (schema_cls, schema_version)`` so
  callers can look up the right validator without inspecting class names.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ── Schema version ──────────────────────────────────────────────────────────

SCHEMA_VERSION = 1


# ── Shared primitives ────────────────────────────────────────────────────────


class MoneyAmount(BaseModel):
    """A monetary figure with currency and optional scale factor.

    Attributes
    ----------
    value:
        Numeric amount as reported (before applying ``unit_multiplier``).
    currency:
        ISO 4217 three-letter currency code (e.g. ``"USD"``).
    unit_multiplier:
        Scale factor.  ``1_000_000`` means *value* is expressed in millions.
    period:
        Reporting period label (e.g. ``"FY2024"`` or ``"Q3 2024"``).
    """

    value: Decimal
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    unit_multiplier: Literal[1, 1_000, 1_000_000, 1_000_000_000] = 1_000_000
    period: str | None = None


# ── Geographic revenue ───────────────────────────────────────────────────────


class GeographicRevenueRow(BaseModel):
    """One row of a geographic / segment revenue breakdown."""

    region: str
    region_type: Literal["country", "region", "segment"]
    amount: MoneyAmount
    percentage_of_total: float | None = Field(None, ge=0.0, le=100.0)
    source_chunk_ids: list[str] = Field(min_length=1)


class GeographicRevenueExtraction(BaseModel):
    """Full geographic revenue breakdown for one fiscal year.

    Cross-field validators
    ----------------------
    ``_check_sum_and_percentages``
        When ``total_revenue`` is provided, row amounts must sum to within 2%
        of the stated total.  When all rows carry a ``percentage_of_total``,
        they must sum to between 95% and 105%.
    """

    SCHEMA_VERSION: int = 1

    fiscal_year: int = Field(ge=1990, le=2100)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    rows: list[GeographicRevenueRow] = Field(min_length=1)
    total_revenue: MoneyAmount | None = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum_and_percentages(self) -> "GeographicRevenueExtraction":
        if self.total_revenue and self.total_revenue.value > 0:
            row_sum = sum(r.amount.value for r in self.rows)
            rel_err = abs(row_sum - self.total_revenue.value) / self.total_revenue.value
            if rel_err > 0.02:
                raise ValueError(
                    f"Row sum {row_sum} differs from reported total "
                    f"{self.total_revenue.value} by {rel_err:.1%} (threshold 2%)"
                )
        rows_with_pct = [r for r in self.rows if r.percentage_of_total is not None]
        if rows_with_pct and len(rows_with_pct) == len(self.rows):
            pct_sum = sum(r.percentage_of_total for r in rows_with_pct)  # type: ignore[arg-type]
            if not (95.0 <= pct_sum <= 105.0):
                raise ValueError(
                    f"Percentages sum to {pct_sum:.1f}%, expected 95–105%"
                )
        return self


# ── Emissions ────────────────────────────────────────────────────────────────


class EmissionsRow(BaseModel):
    """One row of a GHG emissions disclosure."""

    scope: Literal["scope_1", "scope_2_market", "scope_2_location", "scope_3"]
    value: Decimal
    unit: Literal["tCO2e", "ktCO2e", "MtCO2e"]
    year: int = Field(ge=1990, le=2100)
    boundary: str | None = None   # e.g. "operational control", "equity share"
    source_chunk_ids: list[str] = Field(min_length=1)


class EmissionsExtraction(BaseModel):
    """GHG emissions disclosure for one or more scopes.

    Cross-field validators
    ----------------------
    ``_check_scope_2_exclusivity``
        Scope 2 market-based and location-based figures may both appear in the
        same extraction, but they should not be double-counted.  This validator
        emits no error — it is a documentation reminder only.  Downstream
        ``SchemaValidationGate`` handles the unit-mismatch soft-fail.
    """

    SCHEMA_VERSION: int = 1

    fiscal_year: int = Field(ge=1990, le=2100)
    rows: list[EmissionsRow] = Field(min_length=1)
    reporting_standard: str | None = None   # e.g. "GHG Protocol", "ISO 14064"
    assurance_level: Literal["none", "limited", "reasonable"] = "none"
    extraction_confidence: float = Field(ge=0.0, le=1.0)


# ── Headcount ────────────────────────────────────────────────────────────────


class HeadcountRow(BaseModel):
    """One row of a workforce / headcount breakdown."""

    region: str
    region_type: Literal["country", "region", "global"]
    headcount: int = Field(ge=0)
    headcount_type: Literal["employees", "fte", "contractors", "total_workforce"] = (
        "employees"
    )
    year: int = Field(ge=1990, le=2100)
    source_chunk_ids: list[str] = Field(min_length=1)


class HeadcountExtraction(BaseModel):
    """Workforce / headcount breakdown for one reporting period.

    Cross-field validators
    ----------------------
    ``_check_total_consistency``
        If a "global" row is present, regional rows must not sum to more than
        the global total (they may sum to less if not all regions are reported).
    """

    SCHEMA_VERSION: int = 1

    fiscal_year: int = Field(ge=1990, le=2100)
    rows: list[HeadcountRow] = Field(min_length=1)
    extraction_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_total_consistency(self) -> "HeadcountExtraction":
        global_rows = [r for r in self.rows if r.region_type == "global"]
        regional_rows = [r for r in self.rows if r.region_type != "global"]
        if global_rows and regional_rows:
            global_total = max(r.headcount for r in global_rows)
            regional_sum = sum(r.headcount for r in regional_rows)
            if regional_sum > global_total * 1.05:
                raise ValueError(
                    f"Regional headcount sum {regional_sum} exceeds global "
                    f"total {global_total} by more than 5%"
                )
        return self


# ── Registry ─────────────────────────────────────────────────────────────────

#: Maps ``field_name → (schema_class, schema_version)``.
#: Add new fields here; no other module needs to change.
SCHEMA_REGISTRY: dict[str, tuple[type[BaseModel], int]] = {
    "geographic_revenue": (GeographicRevenueExtraction, SCHEMA_VERSION),
    "segment_revenue": (GeographicRevenueExtraction, SCHEMA_VERSION),
    "scope_1_emissions": (EmissionsExtraction, SCHEMA_VERSION),
    "scope_2_emissions": (EmissionsExtraction, SCHEMA_VERSION),
    "headcount_by_region": (HeadcountExtraction, SCHEMA_VERSION),
}
