"""Tests for Stage C — schema-driven typed extraction.

Coverage
--------
SC1  schemas.py — GeographicRevenueExtraction row-sum validator
SC2  schemas.py — GeographicRevenueExtraction percentage-sum validator
SC3  schemas.py — EmissionsExtraction basic construction
SC4  schemas.py — HeadcountExtraction total-consistency validator
SC5  schemas.py — SCHEMA_REGISTRY contains expected fields
SC6  models.py  — ExtractionResult construction
SC7  models.py  — ExtractionResult serialises without loss
SC8  repair.py  — _build_repair_prompt includes error messages
SC9  repair.py  — SchemaRepairHelper.attempt_repair returns clean dict on valid JSON
SC10 repair.py  — SchemaRepairHelper.attempt_repair strips markdown fencing
SC11 repair.py  — SchemaRepairHelper.attempt_repair returns errors on invalid JSON
SC12 extractor.py — SchemaExtractor.extract returns None for unregistered field
SC13 extractor.py — SchemaExtractor.extract returns ExtractionResult on valid LLM response
SC14 extractor.py — SchemaExtractor.extract triggers repair loop on validation failure
SC15 extractor.py — SchemaExtractor.extract returns failed result when all repairs exhausted
SC16 gates.py   — SchemaValidationGate.check passes clean result
SC17 gates.py   — SchemaValidationGate.check flags row_sum_divergence as hard fail
SC18 gates.py   — SchemaValidationGate.check flags percentage_sum_off as soft fail (penalty applied)
SC19 gates.py   — SchemaValidationGate.check applies penalty per soft error
SC20 contracts.py — PipelineRunResult.extraction_results defaults to empty list
SC21 contracts.py — PipelineRunResult.extraction_results roundtrips through model_dump / model_validate

Regression tests for the four findings (F1–F4 below)
F1   settings.py — RetrievalConfig exposes schema_validation and schema_repair_max_attempts
F2   models.py   — ExtractionResult.value round-trips through model_dump / model_validate
F3   extractor.py — _failed_result nested rows are typed GeographicRevenueRow instances
F4a  gates.py    — _classify maps real Pydantic "at least 1 item" msg to missing_provenance
F4b  gates.py    — _classify maps loc-prefixed error string ("source_chunk_ids: ...") correctly
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from evidence_enrichment.core.extraction.models import ExtractionResult
from evidence_enrichment.core.extraction.repair import (
    SchemaRepairHelper,
    _build_repair_prompt,
)
from evidence_enrichment.core.extraction.schemas import (
    EmissionsExtraction,
    EmissionsRow,
    GeographicRevenueExtraction,
    GeographicRevenueRow,
    HeadcountExtraction,
    HeadcountRow,
    MoneyAmount,
    SCHEMA_REGISTRY,
)
from evidence_enrichment.core.extraction.extractor import SchemaExtractor
from evidence_enrichment.core.quality.gates import SchemaGateResult, SchemaValidationGate


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _money(value: float, currency: str = "USD") -> MoneyAmount:
    return MoneyAmount(value=Decimal(str(value)), currency=currency)


def _geo_row(region: str, amount: float, chunk_ids: list[str] | None = None) -> GeographicRevenueRow:
    return GeographicRevenueRow(
        region=region,
        region_type="region",
        amount=_money(amount),
        source_chunk_ids=chunk_ids or ["chunk-1"],
    )


def _emissions_row(scope: str, value: float) -> EmissionsRow:
    return EmissionsRow(
        scope=scope,
        value=Decimal(str(value)),
        unit="tCO2e",
        year=2024,
        source_chunk_ids=["chunk-1"],
    )


def _headcount_row(region: str, count: int, region_type: str = "region") -> HeadcountRow:
    return HeadcountRow(
        region=region,
        region_type=region_type,
        headcount=count,
        year=2024,
        source_chunk_ids=["chunk-1"],
    )


# ── SC1 ───────────────────────────────────────────────────────────────────────


def test_sc1_geographic_revenue_row_sum_rejects_divergence():
    """Row sum diverging > 2% from total raises ValidationError."""
    rows = [_geo_row("Americas", 50.0), _geo_row("EMEA", 30.0)]  # sum = 80, total = 100
    with pytest.raises(ValidationError, match="differs from reported total"):
        GeographicRevenueExtraction(
            fiscal_year=2024,
            currency="USD",
            rows=rows,
            total_revenue=_money(100.0),
            extraction_confidence=0.9,
        )


def test_sc1_geographic_revenue_row_sum_accepts_within_tolerance():
    """Row sum within 2% of total is accepted."""
    rows = [_geo_row("Americas", 51.0), _geo_row("EMEA", 50.0)]  # sum = 101, total = 100 → 1% off
    obj = GeographicRevenueExtraction(
        fiscal_year=2024,
        currency="USD",
        rows=rows,
        total_revenue=_money(100.0),
        extraction_confidence=0.9,
    )
    assert len(obj.rows) == 2


def test_sc1_geographic_revenue_no_total_skips_sum_check():
    """When total_revenue is None the row-sum check is skipped."""
    rows = [_geo_row("Americas", 50.0), _geo_row("EMEA", 30.0)]
    obj = GeographicRevenueExtraction(
        fiscal_year=2024,
        currency="USD",
        rows=rows,
        total_revenue=None,
        extraction_confidence=0.8,
    )
    assert obj.total_revenue is None


# ── SC2 ───────────────────────────────────────────────────────────────────────


def test_sc2_geographic_revenue_pct_sum_rejects_far_off():
    """Percentage sum outside 95–105% raises ValidationError."""
    r1 = GeographicRevenueRow(
        region="Americas",
        region_type="region",
        amount=_money(60.0),
        percentage_of_total=60.0,
        source_chunk_ids=["c1"],
    )
    r2 = GeographicRevenueRow(
        region="EMEA",
        region_type="region",
        amount=_money(20.0),
        percentage_of_total=20.0,  # sum = 80, outside 95-105
        source_chunk_ids=["c1"],
    )
    with pytest.raises(ValidationError, match="Percentages sum"):
        GeographicRevenueExtraction(
            fiscal_year=2024,
            currency="USD",
            rows=[r1, r2],
            extraction_confidence=0.7,
        )


def test_sc2_geographic_revenue_partial_pct_skips_check():
    """When only some rows have percentages the sum check is skipped."""
    r1 = GeographicRevenueRow(
        region="Americas",
        region_type="region",
        amount=_money(60.0),
        percentage_of_total=60.0,
        source_chunk_ids=["c1"],
    )
    r2 = GeographicRevenueRow(
        region="EMEA",
        region_type="region",
        amount=_money(20.0),
        percentage_of_total=None,  # partial → skip sum check
        source_chunk_ids=["c1"],
    )
    obj = GeographicRevenueExtraction(
        fiscal_year=2024,
        currency="USD",
        rows=[r1, r2],
        extraction_confidence=0.7,
    )
    assert len(obj.rows) == 2


# ── SC3 ───────────────────────────────────────────────────────────────────────


def test_sc3_emissions_extraction_basic_construction():
    rows = [_emissions_row("scope_1", 12_000.0), _emissions_row("scope_2_market", 8_000.0)]
    obj = EmissionsExtraction(
        fiscal_year=2024,
        rows=rows,
        reporting_standard="GHG Protocol",
        assurance_level="limited",
        extraction_confidence=0.85,
    )
    assert obj.fiscal_year == 2024
    assert len(obj.rows) == 2
    assert obj.assurance_level == "limited"


def test_sc3_emissions_extraction_requires_at_least_one_row():
    with pytest.raises(ValidationError):
        EmissionsExtraction(fiscal_year=2024, rows=[], extraction_confidence=0.5)


# ── SC4 ───────────────────────────────────────────────────────────────────────


def test_sc4_headcount_rejects_regional_sum_exceeds_global():
    rows = [
        _headcount_row("Global", 1_000, "global"),
        _headcount_row("Americas", 600, "region"),
        _headcount_row("EMEA", 600, "region"),  # 1200 > 1000 * 1.05
    ]
    with pytest.raises(ValidationError, match="exceeds global total"):
        HeadcountExtraction(fiscal_year=2024, rows=rows, extraction_confidence=0.9)


def test_sc4_headcount_accepts_valid_distribution():
    rows = [
        _headcount_row("Global", 1_000, "global"),
        _headcount_row("Americas", 400, "region"),
        _headcount_row("EMEA", 400, "region"),  # 800 < 1000 ✓
    ]
    obj = HeadcountExtraction(fiscal_year=2024, rows=rows, extraction_confidence=0.9)
    assert len(obj.rows) == 3


# ── SC5 ───────────────────────────────────────────────────────────────────────


def test_sc5_schema_registry_contains_expected_fields():
    expected = {
        "geographic_revenue",
        "segment_revenue",
        "scope_1_emissions",
        "scope_2_emissions",
        "headcount_by_region",
    }
    assert expected == set(SCHEMA_REGISTRY)


def test_sc5_schema_registry_values_are_tuples_of_cls_and_version():
    for field_name, (cls, version) in SCHEMA_REGISTRY.items():
        assert issubclass(cls, __import__("pydantic").BaseModel), field_name
        assert isinstance(version, int) and version >= 1, field_name


# ── SC6 ───────────────────────────────────────────────────────────────────────


def test_sc6_extraction_result_construction():
    rows = [_geo_row("Americas", 100.0)]
    value = GeographicRevenueExtraction(
        fiscal_year=2024, currency="USD", rows=rows, extraction_confidence=0.9
    )
    result = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=["chunk-1", "chunk-2"],
        repair_count=0,
        validation_passed=True,
        validation_errors=[],
        extraction_confidence=0.9,
    )
    assert result.validation_passed is True
    assert result.repair_count == 0
    assert len(result.chunks_used) == 2


# ── SC7 ───────────────────────────────────────────────────────────────────────


def test_sc7_extraction_result_round_trips():
    rows = [_geo_row("Americas", 100.0)]
    value = GeographicRevenueExtraction(
        fiscal_year=2024, currency="USD", rows=rows, extraction_confidence=0.9
    )
    result = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=["c1"],
        extraction_confidence=0.9,
    )
    dumped = result.model_dump()
    assert dumped["field_name"] == "geographic_revenue"
    assert dumped["schema_version"] == 1
    assert dumped["validation_passed"] is True


# ── SC8 ───────────────────────────────────────────────────────────────────────


def test_sc8_build_repair_prompt_contains_errors():
    prompt = _build_repair_prompt(
        field_name="geographic_revenue",
        schema_cls=GeographicRevenueExtraction,
        raw_json='{"bad": true}',
        validation_errors=["Row sum 80 differs from total 100 by 20%"],
    )
    assert "geographic_revenue" in prompt
    assert "Row sum 80 differs from total 100 by 20%" in prompt
    assert "GeographicRevenueExtraction" in prompt  # schema class name appears in JSON schema


# ── SC9 ───────────────────────────────────────────────────────────────────────


def test_sc9_repair_helper_returns_clean_dict_on_valid_json():
    valid_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "100", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            }
        ],
        "extraction_confidence": 0.9,
    }

    async def _provider(prompt: str) -> str:
        return json.dumps(valid_payload)

    helper = SchemaRepairHelper(_provider)
    result_dict, errors = asyncio.run(
        helper.attempt_repair(
            field_name="geographic_revenue",
            schema_cls=GeographicRevenueExtraction,
            raw_json="{}",
            validation_errors=["missing fields"],
        )
    )
    assert errors == []
    assert result_dict is not None
    assert result_dict["fiscal_year"] == 2024


# ── SC10 ──────────────────────────────────────────────────────────────────────


def test_sc10_repair_helper_strips_markdown_fencing():
    valid_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "100", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            }
        ],
        "extraction_confidence": 0.9,
    }
    fenced = f"```json\n{json.dumps(valid_payload)}\n```"

    async def _provider(prompt: str) -> str:
        return fenced

    helper = SchemaRepairHelper(_provider)
    result_dict, errors = asyncio.run(
        helper.attempt_repair("geographic_revenue", GeographicRevenueExtraction, "{}", [])
    )
    assert errors == []
    assert result_dict is not None


# ── SC11 ──────────────────────────────────────────────────────────────────────


def test_sc11_repair_helper_returns_error_on_invalid_json():
    async def _provider(prompt: str) -> str:
        return "not valid json at all"

    helper = SchemaRepairHelper(_provider)
    result_dict, errors = asyncio.run(
        helper.attempt_repair("geographic_revenue", GeographicRevenueExtraction, "{}", [])
    )
    assert result_dict is None
    assert any("json_parse_error" in e for e in errors)


# ── SC12 ──────────────────────────────────────────────────────────────────────


def test_sc12_extractor_returns_none_for_unregistered_field():
    async def _provider(prompt: str) -> str:
        return "{}"

    extractor = SchemaExtractor(_provider)
    result = asyncio.run(extractor.extract("unregistered_field", []))
    assert result is None


# ── SC13 ──────────────────────────────────────────────────────────────────────


def test_sc13_extractor_returns_extraction_result_on_valid_response():
    valid_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "100", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            }
        ],
        "extraction_confidence": 0.88,
    }

    async def _provider(prompt: str) -> str:
        return json.dumps(valid_payload)

    extractor = SchemaExtractor(_provider, max_repair_attempts=0)
    result = asyncio.run(extractor.extract("geographic_revenue", [("c1", "Americas revenue 100M")]))
    assert result is not None
    assert result.validation_passed is True
    assert result.field_name == "geographic_revenue"
    assert result.extraction_confidence == pytest.approx(0.88, abs=1e-6)
    assert result.repair_count == 0


# ── SC14 ──────────────────────────────────────────────────────────────────────


def test_sc14_extractor_triggers_repair_on_validation_failure():
    """First call returns invalid JSON; repair returns valid JSON."""
    valid_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "100", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            }
        ],
        "extraction_confidence": 0.75,
    }
    call_count = {"n": 0}

    async def _provider(prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return '{"bad": true}'  # fails validation
        return json.dumps(valid_payload)

    extractor = SchemaExtractor(_provider, max_repair_attempts=1)
    result = asyncio.run(extractor.extract("geographic_revenue", [("c1", "text")]))
    assert result is not None
    assert result.validation_passed is True
    assert result.repair_count == 1
    assert call_count["n"] == 2


# ── SC15 ──────────────────────────────────────────────────────────────────────


def test_sc15_extractor_returns_failed_result_when_all_repairs_exhausted():
    call_count = {"n": 0}

    async def _provider(prompt: str) -> str:
        call_count["n"] += 1
        return '{"bad": true}'  # always invalid

    extractor = SchemaExtractor(_provider, max_repair_attempts=2)
    result = asyncio.run(extractor.extract("geographic_revenue", []))
    assert result is not None
    assert result.validation_passed is False
    assert len(result.validation_errors) > 0
    assert result.repair_count == 2
    assert call_count["n"] == 3  # 1 initial + 2 repairs


# ── SC16 ──────────────────────────────────────────────────────────────────────


def test_sc16_schema_gate_passes_clean_result():
    rows = [_geo_row("Americas", 100.0)]
    value = GeographicRevenueExtraction(
        fiscal_year=2024, currency="USD", rows=rows, extraction_confidence=0.9
    )
    result = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=["c1"],
        validation_passed=True,
        extraction_confidence=0.9,
    )
    gate = SchemaValidationGate()
    gate_result = gate.check(result)
    assert gate_result.passed is True
    assert gate_result.confidence_after == pytest.approx(0.9)
    assert gate_result.hard_errors == []
    assert gate_result.soft_errors == []


# ── SC17 ──────────────────────────────────────────────────────────────────────


def test_sc17_schema_gate_flags_row_sum_divergence_as_hard_fail():
    value = GeographicRevenueExtraction.model_construct(
        fiscal_year=2024, currency="USD", rows=[], extraction_confidence=0.5
    )
    result = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=[],
        validation_passed=False,
        validation_errors=["Row sum 80 differs from reported total 100 by 20.0% (threshold 2%)"],
        extraction_confidence=0.5,
    )
    gate = SchemaValidationGate()
    gate_result = gate.check(result)
    assert gate_result.passed is False
    assert "row_sum_divergence" in gate_result.hard_errors


# ── SC18 ──────────────────────────────────────────────────────────────────────


def test_sc18_schema_gate_flags_percentage_sum_off_as_soft():
    value = GeographicRevenueExtraction.model_construct(
        fiscal_year=2024, currency="USD", rows=[], extraction_confidence=0.7
    )
    result = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=[],
        validation_passed=False,
        validation_errors=["Percentages sum to 80.0%, expected 95–105%"],
        extraction_confidence=0.7,
    )
    gate = SchemaValidationGate()
    gate_result = gate.check(result)
    assert gate_result.passed is True  # soft fail → still passes
    assert "percentage_sum_off" in gate_result.soft_errors
    assert gate_result.confidence_after == pytest.approx(0.7 - 0.15, abs=1e-6)


# ── SC19 ──────────────────────────────────────────────────────────────────────


def test_sc19_schema_gate_applies_penalty_per_soft_error():
    value = GeographicRevenueExtraction.model_construct(
        fiscal_year=2024, currency="USD", rows=[], extraction_confidence=0.8
    )
    result = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=[],
        validation_passed=False,
        validation_errors=[
            "Percentages sum to 80.0%, expected 95–105%",
            "missing period field",
        ],
        extraction_confidence=0.8,
    )
    gate = SchemaValidationGate()
    gate_result = gate.check(result)
    assert gate_result.passed is True
    # Two soft errors → two penalties
    assert gate_result.confidence_after == pytest.approx(0.8 - 2 * 0.15, abs=1e-6)


# ── SC20 ──────────────────────────────────────────────────────────────────────


def test_sc20_pipeline_run_result_extraction_results_defaults_empty():
    from evidence_enrichment.core.models.contracts import (
        PipelineRunResult,
        SearchQueryPlan,
        SynthesisResult,
    )
    from evidence_enrichment.core.models.enums import ReviewDecision

    plan = SearchQueryPlan(
        field_name="hq_country",
        entity_id="test",
        primary_query="test query",
    )
    synth = SynthesisResult(
        field_name="hq_country",
        reasoning="test",
        synthesis_confidence=0.9,
    )
    result = PipelineRunResult(
        entity_id="test",
        field_name="hq_country",
        mode="replay",
        search_plan=plan,
        synthesis=synth,
        overall_confidence=0.9,
        decision=ReviewDecision.AUTO_APPROVE,
        gate_reason="meets_thresholds",
    )
    assert result.extraction_results == []


# ── SC21 ──────────────────────────────────────────────────────────────────────


def test_sc21_pipeline_run_result_extraction_results_roundtrips():
    from evidence_enrichment.core.models.contracts import (
        PipelineRunResult,
        SearchQueryPlan,
        SynthesisResult,
    )
    from evidence_enrichment.core.models.enums import ReviewDecision

    rows = [_geo_row("Americas", 100.0)]
    value = GeographicRevenueExtraction(
        fiscal_year=2024, currency="USD", rows=rows, extraction_confidence=0.9
    )
    extraction = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=["c1"],
        extraction_confidence=0.9,
    )
    plan = SearchQueryPlan(
        field_name="geographic_revenue",
        entity_id="test",
        primary_query="geographic revenue",
    )
    synth = SynthesisResult(
        field_name="geographic_revenue",
        reasoning="test",
        synthesis_confidence=0.9,
    )
    result = PipelineRunResult(
        entity_id="test",
        field_name="geographic_revenue",
        mode="live",
        search_plan=plan,
        synthesis=synth,
        overall_confidence=0.9,
        decision=ReviewDecision.AUTO_APPROVE,
        gate_reason="meets_thresholds",
        extraction_results=[extraction],
    )
    dumped = result.model_dump()
    assert len(dumped["extraction_results"]) == 1
    assert dumped["extraction_results"][0]["field_name"] == "geographic_revenue"
    assert dumped["extraction_results"][0]["validation_passed"] is True


# ── Finding F1 ────────────────────────────────────────────────────────────────


def test_f1_retrieval_config_has_schema_validation_field():
    """RetrievalConfig must expose schema_validation without AttributeError."""
    from evidence_enrichment.config.settings import RetrievalConfig

    rc = RetrievalConfig()
    assert rc.schema_validation is False  # default off


def test_f1_retrieval_config_has_schema_repair_max_attempts_field():
    """RetrievalConfig must expose schema_repair_max_attempts without AttributeError."""
    from evidence_enrichment.config.settings import RetrievalConfig

    rc = RetrievalConfig()
    assert rc.schema_repair_max_attempts == 2  # default 2


def test_f1_retrieval_config_schema_validation_can_be_enabled():
    from evidence_enrichment.config.settings import RetrievalConfig

    rc = RetrievalConfig(schema_validation=True, schema_repair_max_attempts=3)
    assert rc.schema_validation is True
    assert rc.schema_repair_max_attempts == 3


# ── Finding F2 ────────────────────────────────────────────────────────────────


def test_f2_extraction_result_value_roundtrips_model_validate():
    """model_validate(model_dump()) must restore the concrete value type."""
    rows = [_geo_row("Americas", 100.0)]
    value = GeographicRevenueExtraction(
        fiscal_year=2024, currency="USD", rows=rows, extraction_confidence=0.9
    )
    original = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=["c1"],
        extraction_confidence=0.9,
    )
    dumped = original.model_dump()
    restored = ExtractionResult.model_validate(dumped)

    assert type(restored.value).__name__ == "GeographicRevenueExtraction"
    assert isinstance(restored.value, GeographicRevenueExtraction)
    assert restored.value.fiscal_year == 2024
    assert restored.value.rows[0].region == "Americas"


def test_f2_extraction_result_value_roundtrips_emissions():
    """Round-trip also works for EmissionsExtraction."""
    rows = [_emissions_row("scope_1", 5_000.0)]
    value = EmissionsExtraction(
        fiscal_year=2024,
        rows=rows,
        extraction_confidence=0.8,
    )
    original = ExtractionResult(
        field_name="scope_1_emissions",
        schema_cls_name="EmissionsExtraction",
        schema_version=1,
        value=value,
        extraction_confidence=0.8,
    )
    restored = ExtractionResult.model_validate(original.model_dump())
    assert isinstance(restored.value, EmissionsExtraction)
    assert restored.value.rows[0].scope == "scope_1"


# ── Finding F3 ────────────────────────────────────────────────────────────────


def test_f3_failed_result_rows_are_typed_instances():
    """_failed_result must produce typed row instances, not raw dicts."""
    import asyncio
    import json

    valid_but_cross_field_fail = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "50", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            },
            {
                "region": "EMEA",
                "region_type": "region",
                "amount": {"value": "30", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            },
        ],
        # total 100, rows sum 80 → 20% divergence → row_sum_divergence
        "total_revenue": {"value": "100", "currency": "USD"},
        "extraction_confidence": 0.5,
    }

    async def _provider(prompt: str) -> str:
        return json.dumps(valid_but_cross_field_fail)

    from evidence_enrichment.core.extraction.extractor import SchemaExtractor

    extractor = SchemaExtractor(_provider, max_repair_attempts=0)
    result = asyncio.run(extractor.extract("geographic_revenue", []))
    assert result is not None
    assert result.validation_passed is False
    # Rows must be typed GeographicRevenueRow, not plain dicts
    assert isinstance(result.value, GeographicRevenueExtraction)
    for row in result.value.rows:
        assert isinstance(row, GeographicRevenueRow), (
            f"Expected GeographicRevenueRow, got {type(row)}"
        )


# ── Finding F4a ───────────────────────────────────────────────────────────────


def test_f4a_classify_real_pydantic_provenance_msg():
    """_classify must map real Pydantic 'at least 1 item' message to missing_provenance."""
    from evidence_enrichment.core.quality.gates import SchemaValidationGate

    gate = SchemaValidationGate()
    # Extractor now stores "rows.0.source_chunk_ids: List should have at least 1 item..."
    real_msg = "rows.0.source_chunk_ids: List should have at least 1 item after validation, not 0"
    tag = gate._classify(real_msg)
    assert tag == "missing_provenance", f"Expected missing_provenance, got {tag!r}"


def test_f4a_classify_bare_at_least_1_item_msg():
    """_classify must handle the bare Pydantic message without loc prefix."""
    from evidence_enrichment.core.quality.gates import SchemaValidationGate

    gate = SchemaValidationGate()
    bare_msg = "List should have at least 1 item after validation, not 0"
    assert gate._classify(bare_msg) == "missing_provenance"


# ── Finding F4b ───────────────────────────────────────────────────────────────


def test_f4b_extractor_error_format_includes_loc():
    """_parse_and_validate must store '<loc>: <msg>' strings."""
    import asyncio
    import json

    # Payload with empty source_chunk_ids → provenance error
    bad_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "100", "currency": "USD"},
                "source_chunk_ids": [],  # empty → validation error
            }
        ],
        "extraction_confidence": 0.5,
    }

    async def _provider(prompt: str) -> str:
        return json.dumps(bad_payload)

    from evidence_enrichment.core.extraction.extractor import SchemaExtractor

    extractor = SchemaExtractor(_provider, max_repair_attempts=0)
    result = asyncio.run(extractor.extract("geographic_revenue", []))
    assert result is not None
    assert result.validation_passed is False
    # At least one error string should mention source_chunk_ids or "at least 1"
    provenance_errors = [
        e for e in result.validation_errors
        if "source_chunk_ids" in e or "at least 1" in e
    ]
    assert provenance_errors, (
        f"No provenance error found in {result.validation_errors!r}"
    )


def test_f4b_gate_classifies_real_extractor_provenance_error():
    """End-to-end: extractor emits loc-prefixed error; gate classifies it as missing_provenance."""
    import asyncio
    import json

    from evidence_enrichment.core.extraction.extractor import SchemaExtractor
    from evidence_enrichment.core.quality.gates import SchemaValidationGate

    bad_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "100", "currency": "USD"},
                "source_chunk_ids": [],
            }
        ],
        "extraction_confidence": 0.5,
    }

    async def _provider(prompt: str) -> str:
        return json.dumps(bad_payload)

    extractor = SchemaExtractor(_provider, max_repair_attempts=0)
    result = asyncio.run(extractor.extract("geographic_revenue", []))
    assert result is not None

    gate = SchemaValidationGate()
    gate_result = gate.check(result)
    assert "missing_provenance" in gate_result.hard_errors, (
        f"hard_errors={gate_result.hard_errors!r}, errors={result.validation_errors!r}"
    )


# ── Finding G1 — failed result round-trip ────────────────────────────────────


def test_g1_failed_extraction_result_round_trips_provenance_error():
    """model_validate(model_dump()) must not raise for a failed ExtractionResult.

    Failed results store partial/invalid payloads (e.g. empty source_chunk_ids).
    The deserializer must skip cross-field validators on reload.
    """
    import asyncio
    import json

    from evidence_enrichment.core.extraction.extractor import SchemaExtractor

    bad_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "100", "currency": "USD"},
                "source_chunk_ids": [],  # invalid → ValidationError
            }
        ],
        "extraction_confidence": 0.4,
    }

    async def _provider(prompt: str) -> str:
        return json.dumps(bad_payload)

    extractor = SchemaExtractor(_provider, max_repair_attempts=0)
    result = asyncio.run(extractor.extract("geographic_revenue", []))
    assert result is not None
    assert result.validation_passed is False

    # Must not raise.
    dumped = result.model_dump()
    restored = ExtractionResult.model_validate(dumped)

    assert restored.validation_passed is False
    assert isinstance(restored.value, GeographicRevenueExtraction)


def test_g1_failed_extraction_result_round_trips_cross_field_error():
    """Row-sum-divergence (cross-field) result must also survive round-trip."""
    import asyncio
    import json

    from evidence_enrichment.core.extraction.extractor import SchemaExtractor

    cross_field_fail = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "50", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            }
        ],
        "total_revenue": {"value": "100", "currency": "USD"},
        "extraction_confidence": 0.5,
    }

    async def _provider(prompt: str) -> str:
        return json.dumps(cross_field_fail)

    extractor = SchemaExtractor(_provider, max_repair_attempts=0)
    result = asyncio.run(extractor.extract("geographic_revenue", []))
    assert result is not None
    assert result.validation_passed is False

    dumped = result.model_dump()
    restored = ExtractionResult.model_validate(dumped)

    assert restored.validation_passed is False
    assert isinstance(restored.value, GeographicRevenueExtraction)
    for row in restored.value.rows:
        assert isinstance(row, GeographicRevenueRow), (
            f"Expected GeographicRevenueRow after round-trip, got {type(row)}"
        )
    # Scalar nested model must also be typed after round-trip (Fix 1).
    assert isinstance(restored.value.total_revenue, MoneyAmount), (
        f"Expected MoneyAmount for total_revenue after round-trip, got "
        f"{type(restored.value.total_revenue)}"
    )


# ── Finding G1c — direct _failed_result scalar coercion in extractor ─────────


def test_g1c_failed_result_scalar_nested_model_is_typed():
    """_failed_result must coerce scalar BaseModel fields (e.g. total_revenue)
    to typed instances, not leave them as raw dicts.

    The cross-field failure path hits model_construct; without Fix 1 the
    total_revenue field stays as a raw dict because the old loop only handled
    list[BaseModel] fields.
    """
    import asyncio
    import json

    from evidence_enrichment.core.extraction.extractor import SchemaExtractor

    # Row sum (50) differs from total_revenue (100) by 50% — triggers cross-field.
    cross_field_fail = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "50", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            }
        ],
        "total_revenue": {"value": "100", "currency": "USD"},
        "extraction_confidence": 0.5,
    }

    async def _provider(prompt: str) -> str:
        return json.dumps(cross_field_fail)

    extractor = SchemaExtractor(_provider, max_repair_attempts=0)
    result = asyncio.run(extractor.extract("geographic_revenue", []))

    assert result is not None
    assert result.validation_passed is False
    assert isinstance(result.value.total_revenue, MoneyAmount), (
        f"total_revenue should be MoneyAmount before round-trip, got "
        f"{type(result.value.total_revenue)}"
    )
    assert isinstance(result.value.rows[0], GeographicRevenueRow), (
        f"rows[0] should be GeographicRevenueRow before round-trip, got "
        f"{type(result.value.rows[0])}"
    )


# ── Finding G2 — gate penalty propagates into value ──────────────────────────


def test_g2_gate_penalty_propagates_into_value_confidence():
    """The coordinator gate loop must update value.extraction_confidence to match
    the penalized outer extraction_confidence.
    """
    from evidence_enrichment.core.quality.gates import SchemaValidationGate

    value = GeographicRevenueExtraction.model_construct(
        fiscal_year=2024, currency="USD", rows=[], extraction_confidence=0.8
    )
    result = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=[],
        validation_passed=False,
        validation_errors=["Percentages sum to 80.0%, expected 95–105%"],
        extraction_confidence=0.8,
    )

    gate = SchemaValidationGate()
    gate_result_obj = gate.check(result)
    new_confidence = gate_result_obj.confidence_after

    # Simulate the coordinator loop.
    new_value = result.value
    if hasattr(result.value, "extraction_confidence"):
        new_value = result.value.model_copy(update={"extraction_confidence": new_confidence})
    penalized = result.model_copy(
        update={"extraction_confidence": new_confidence, "value": new_value}
    )

    expected = pytest.approx(0.8 - 0.15, abs=1e-6)
    assert penalized.extraction_confidence == expected
    assert penalized.value.extraction_confidence == expected, (
        "value.extraction_confidence was not updated by gate penalty"
    )


def test_g2_outer_and_inner_confidence_never_diverge():
    """After the coordinator gate loop, outer and inner confidence must be equal."""
    from evidence_enrichment.core.quality.gates import SchemaValidationGate

    rows = [_geo_row("Americas", 100.0)]
    value = GeographicRevenueExtraction(
        fiscal_year=2024, currency="USD", rows=rows, extraction_confidence=0.9
    )
    result = ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        value=value,
        chunks_used=["c1"],
        validation_passed=True,
        extraction_confidence=0.9,
    )
    gate = SchemaValidationGate()
    gr = gate.check(result)
    new_confidence = gr.confidence_after
    new_value = result.value.model_copy(update={"extraction_confidence": new_confidence})
    penalized = result.model_copy(
        update={"extraction_confidence": new_confidence, "value": new_value}
    )
    assert penalized.extraction_confidence == pytest.approx(
        penalized.value.extraction_confidence, abs=1e-9
    )


# ── Finding G3 — schema_extraction uses _resolve_stage_model_and_budget ──────


def test_g3_schema_extraction_respects_blocked_budget():
    """When _resolve_stage_model_and_budget returns BLOCKED, _run_schema_extraction
    must return [] without calling the provider.
    """
    import asyncio
    from unittest.mock import MagicMock

    from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator
    from evidence_enrichment.finops.models import BudgetDecision, BudgetStatus, DowngradeAction

    coord = EvidenceCoordinator.__new__(EvidenceCoordinator)
    coord.settings = MagicMock()
    coord.settings.finops.enabled = False

    mock_agent = MagicMock()
    mock_agent.provider_type.value = "openai"
    coord._analysis_agent = MagicMock(return_value=mock_agent)

    blocked = BudgetDecision(status=BudgetStatus.BLOCKED)
    coord._resolve_stage_model_and_budget = MagicMock(
        return_value=("gpt-4o-mini", False, blocked, DowngradeAction.NONE)
    )

    result = asyncio.run(
        coord._run_schema_extraction(
            field_name="geographic_revenue",
            retrieved_chunks_map={},
            assessed_documents=[],
        )
    )

    assert result == [], "Blocked budget must return empty list"
    coord._resolve_stage_model_and_budget.assert_called_once_with(
        "schema_extraction", "openai", []
    )


def test_g3_schema_extraction_uses_resolved_model_not_settings_model():
    """The provider callable must use the budget-resolved model, not
    self.settings.openai_model, so that cheap-model downgrades take effect.
    """
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator
    from evidence_enrichment.finops.models import BudgetDecision, BudgetStatus, DowngradeAction

    calls: list[str] = []

    coord = EvidenceCoordinator.__new__(EvidenceCoordinator)
    coord.settings = MagicMock()
    coord.settings.openai_api_key = "fake-key"
    coord.settings.openai_model = "gpt-4o"
    coord.settings.finops.enabled = False
    coord.settings.retrieval.schema_repair_max_attempts = 0

    mock_agent = MagicMock()
    mock_agent.provider_type.value = "openai"
    coord._analysis_agent = MagicMock(return_value=mock_agent)

    ok_decision = BudgetDecision(status=BudgetStatus.NOMINAL)
    coord._resolve_stage_model_and_budget = MagicMock(
        return_value=("gpt-4o-mini", False, ok_decision, DowngradeAction.CHEAP_MODEL)
    )

    valid_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "100", "currency": "USD"},
                "source_chunk_ids": ["c1"],
            }
        ],
        "extraction_confidence": 0.9,
    }

    async def _fake_create(*, model: str, input: str) -> MagicMock:
        calls.append(model)
        r = MagicMock()
        r.output_text = json.dumps(valid_payload)
        return r

    with patch("openai.AsyncOpenAI") as mock_cls:
        instance = MagicMock()
        instance.responses.create = _fake_create
        mock_cls.return_value = instance

        result = asyncio.run(
            coord._run_schema_extraction(
                field_name="geographic_revenue",
                retrieved_chunks_map={},
                assessed_documents=[],
            )
        )

    assert calls, "OpenAI was never called"
    assert calls[0] == "gpt-4o-mini", (
        f"Expected cheap model 'gpt-4o-mini', got {calls[0]!r}"
    )


# ── Finding G4 — FinOps repair loop call count ───────────────────────────────


def test_g4_finops_records_all_repair_calls_not_just_last():
    """_run_schema_extraction must record pre-summed token counts across the
    initial call and all repair attempts, without double-scaling.

    The old path concatenated prompts and passed call_count=N to
    _record_stage_finops, which then multiplied token counts by N — causing
    3× overcount on 3 calls.  The fix: accumulate estimate_tokens() per call
    and pass pre-summed totals to _record_stage_finops_from_tokens with
    call_count reflecting the actual number of calls.
    """
    import asyncio
    import json
    import math
    from unittest.mock import MagicMock, patch

    from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator
    from evidence_enrichment.finops.estimation import estimate_tokens
    from evidence_enrichment.finops.models import BudgetDecision, BudgetStatus, DowngradeAction

    # Bad payload → validation always fails → 1 initial + 2 repair = 3 calls.
    bad_payload = {
        "fiscal_year": 2024,
        "currency": "USD",
        "rows": [
            {
                "region": "Americas",
                "region_type": "region",
                "amount": {"value": "50", "currency": "USD"},
                "source_chunk_ids": [],   # always invalid
            }
        ],
        "extraction_confidence": 0.4,
    }
    response_text = json.dumps(bad_payload)
    response_tokens = estimate_tokens(response_text)

    call_log: list[str] = []

    coord = EvidenceCoordinator.__new__(EvidenceCoordinator)
    coord.settings = MagicMock()
    coord.settings.openai_api_key = "fake-key"
    coord.settings.openai_model = "gpt-4o"
    coord.settings.finops.enabled = True
    coord.settings.retrieval.schema_repair_max_attempts = 2

    mock_agent = MagicMock()
    mock_agent.provider_type.value = "openai"
    coord._analysis_agent = MagicMock(return_value=mock_agent)

    ok_decision = BudgetDecision(status=BudgetStatus.NOMINAL)
    coord._resolve_stage_model_and_budget = MagicMock(
        return_value=("gpt-4o", False, ok_decision, DowngradeAction.NONE)
    )

    recorded: list[dict] = []

    def _fake_record_from_tokens(span, *, stage, provider, model_name,
                                 total_input_tokens, total_output_tokens,
                                 call_count, downgrade_applied=None,
                                 usage_source=None):
        recorded.append({
            "call_count": call_count,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        })

    coord._record_stage_finops_from_tokens = _fake_record_from_tokens

    async def _fake_create(*, model: str, input: str) -> MagicMock:
        call_log.append(input)
        r = MagicMock()
        r.output_text = response_text
        return r

    with patch("openai.AsyncOpenAI") as mock_cls:
        instance = MagicMock()
        instance.responses.create = _fake_create
        mock_cls.return_value = instance

        asyncio.run(
            coord._run_schema_extraction(
                field_name="geographic_revenue",
                retrieved_chunks_map={},
                assessed_documents=[],
            )
        )

    assert len(call_log) == 3, (
        f"Expected 3 provider calls (1 initial + 2 repairs), got {len(call_log)}"
    )
    assert recorded, "_record_stage_finops_from_tokens was never called"

    rec = recorded[0]
    assert rec["call_count"] == 3, (
        f"Expected call_count=3, got {rec['call_count']}"
    )

    # Tokens must be the sum of per-call estimates, not 3× the concatenated total.
    expected_input = sum(estimate_tokens(p) for p in call_log)
    expected_output = response_tokens * 3
    assert rec["total_input_tokens"] == expected_input, (
        f"total_input_tokens {rec['total_input_tokens']} != expected {expected_input}; "
        "likely double-scaling via call_count multiplier"
    )
    assert rec["total_output_tokens"] == expected_output, (
        f"total_output_tokens {rec['total_output_tokens']} != expected {expected_output}"
    )
