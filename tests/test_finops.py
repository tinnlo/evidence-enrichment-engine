"""Tests for FinOps foundation: models, pricing, estimation, config, and contracts."""

from __future__ import annotations

import math

import pytest

from evidence_enrichment.config.settings import FinOpsSettings, Settings
from evidence_enrichment.core.models.contracts import PipelineRunResult
from evidence_enrichment.finops.collector import FinOpsCollector
from evidence_enrichment.finops.estimation import (
    estimate_embedding_cost,
    estimate_stage_cost,
    estimate_tokens,
)
from evidence_enrichment.finops.models import (
    BudgetDecision,
    BudgetMode,
    BudgetStatus,
    DowngradeAction,
    StageCostRecord,
    UsageSource,
)
from evidence_enrichment.finops.pricing import CATALOG_VERSION, PricingCatalog, build_catalog
from evidence_enrichment.finops.policy import BudgetPolicyEngine
from evidence_enrichment.observability.tracer import LocalTracer, SpanRecord, TraceSummary


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_basic(self):
        assert estimate_tokens("hello world") == math.ceil(11 / 4)

    def test_long_text(self):
        text = "a" * 1000
        assert estimate_tokens(text) == 250


class TestPricingCatalog:
    def test_default_catalog_has_version(self):
        catalog = PricingCatalog()
        assert catalog.version == CATALOG_VERSION

    def test_known_model_cost(self):
        catalog = PricingCatalog()
        cost = catalog.cost_for_tokens("GPT-5.4", 1_000_000, 0)
        assert cost == pytest.approx(2.50, abs=1e-6)

    def test_output_tokens(self):
        catalog = PricingCatalog()
        cost = catalog.cost_for_tokens("GPT-5.4", 0, 1_000_000)
        assert cost == pytest.approx(15.00, abs=1e-6)

    def test_unknown_model_zero_cost(self):
        catalog = PricingCatalog()
        assert catalog.cost_for_tokens("nonexistent-model", 1000, 1000) == 0.0

    def test_build_catalog_with_overrides(self):
        catalog = build_catalog({"custom-model": {"input_per_1m": 1.0, "output_per_1m": 2.0}})
        assert "custom-model" in catalog.prices
        cost = catalog.cost_for_tokens("custom-model", 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.0, abs=1e-6)

    def test_default_prices_preserved_with_overrides(self):
        catalog = build_catalog({"custom-model": {"input_per_1m": 1.0, "output_per_1m": 2.0}})
        assert "GPT-5.4" in catalog.prices


class TestStageCostRecord:
    def test_defaults(self):
        rec = StageCostRecord(stage="analysis", provider="openai", model_name="GPT-5.4")
        assert rec.call_count == 0
        assert rec.usage_source == UsageSource.ESTIMATED
        assert rec.downgrade_applied == DowngradeAction.NONE


class TestEstimateStageCost:
    def test_basic_estimation(self):
        catalog = PricingCatalog()
        rec = estimate_stage_cost(
            stage="analysis",
            provider="openai",
            model_name="GPT-5.4",
            input_text="a" * 4000,
            output_text="b" * 400,
            catalog=catalog,
        )
        assert rec.stage == "analysis"
        assert rec.estimated_input_tokens == 1000
        assert rec.estimated_output_tokens == 100
        assert rec.estimated_total_tokens == 1100
        assert rec.estimated_cost_usd > 0
        assert rec.usage_source == UsageSource.ESTIMATED

    def test_embedding_cost(self):
        catalog = PricingCatalog()
        rec = estimate_embedding_cost(
            stage="retrieval_indexing",
            provider="openai",
            model_name="text-embedding-3-small",
            text_count=5,
            total_chars=7500,
            catalog=catalog,
        )
        assert rec.call_count == 5
        assert rec.estimated_output_tokens == 0
        assert rec.estimated_cost_usd > 0


class TestFinOpsCollector:
    def test_empty_collector(self):
        catalog = PricingCatalog()
        collector = FinOpsCollector(catalog)
        assert collector.accrued_cost_usd == 0.0

    def test_record_and_summary(self):
        catalog = PricingCatalog()
        collector = FinOpsCollector(catalog)
        collector.record(
            StageCostRecord(
                stage="analysis",
                provider="openai",
                model_name="GPT-5.4",
                estimated_cost_usd=0.005,
                estimated_input_tokens=1000,
                estimated_output_tokens=200,
            )
        )
        collector.record(
            StageCostRecord(
                stage="synthesis",
                provider="openai",
                model_name="GPT-5.4",
                estimated_cost_usd=0.002,
                estimated_input_tokens=500,
                estimated_output_tokens=100,
            )
        )
        summary = collector.build_summary(total_latency_ms=150.0)
        assert summary.total_estimated_cost_usd == pytest.approx(0.007, abs=1e-6)
        assert summary.cost_by_stage["analysis"] == pytest.approx(0.005, abs=1e-6)
        assert summary.cost_by_model["GPT-5.4"] == pytest.approx(0.007, abs=1e-6)
        assert summary.total_latency_ms == 150.0
        assert len(summary.stage_records) == 2


class TestBudgetPolicyEngine:
    def test_off_mode_always_nominal(self):
        catalog = PricingCatalog()
        policy = BudgetPolicyEngine(mode=BudgetMode.OFF, catalog=catalog)
        collector = FinOpsCollector(catalog)
        collector.record(
            StageCostRecord(
                stage="analysis", provider="openai", model_name="x",
                estimated_cost_usd=999.0,
            )
        )
        decision = policy.check_before_stage(collector)
        assert decision.status == BudgetStatus.NOMINAL

    def test_warn_mode_does_not_block(self):
        catalog = PricingCatalog()
        policy = BudgetPolicyEngine(
            mode=BudgetMode.WARN, max_cost_per_run=0.001, catalog=catalog,
        )
        collector = FinOpsCollector(catalog)
        collector.record(
            StageCostRecord(
                stage="analysis", provider="openai", model_name="x",
                estimated_cost_usd=0.01,
            )
        )
        decision = policy.check_before_stage(collector)
        assert decision.status == BudgetStatus.WARN

    def test_strict_mode_blocks(self):
        catalog = PricingCatalog()
        policy = BudgetPolicyEngine(
            mode=BudgetMode.STRICT, max_cost_per_run=0.001, catalog=catalog,
        )
        collector = FinOpsCollector(catalog)
        collector.record(
            StageCostRecord(
                stage="analysis", provider="openai", model_name="x",
                estimated_cost_usd=0.01,
            )
        )
        decision = policy.check_before_stage(collector)
        assert decision.status == BudgetStatus.WARN
        policy.mark_downgrade_exhausted()
        decision = policy.check_before_stage(collector)
        assert decision.status == BudgetStatus.BLOCKED

    def test_post_run_exceeded(self):
        catalog = PricingCatalog()
        policy = BudgetPolicyEngine(
            mode=BudgetMode.WARN, max_cost_per_run=0.001, catalog=catalog,
        )
        collector = FinOpsCollector(catalog)
        collector.record(
            StageCostRecord(
                stage="analysis", provider="openai", model_name="x",
                estimated_cost_usd=0.01,
            )
        )
        decision = policy.check_post_run(collector, succeeded=True)
        assert decision.status == BudgetStatus.EXCEEDED

    def test_downgrade_helpers(self):
        catalog = PricingCatalog()
        policy = BudgetPolicyEngine(
            mode=BudgetMode.STRICT, max_cost_per_run=0.001, catalog=catalog,
        )
        collector = FinOpsCollector(catalog)
        collector.record(
            StageCostRecord(
                stage="analysis", provider="openai", model_name="x",
                estimated_cost_usd=0.01,
            )
        )
        decision = policy.check_before_stage(collector)
        assert policy.should_disable_retrieval(decision)
        assert policy.should_use_cheap_model(decision)

    def test_off_mode_no_downgrade(self):
        catalog = PricingCatalog()
        policy = BudgetPolicyEngine(mode=BudgetMode.OFF, catalog=catalog)
        decision = BudgetDecision(status=BudgetStatus.EXCEEDED)
        assert not policy.should_disable_retrieval(decision)
        assert not policy.should_use_cheap_model(decision)


class TestFinOpsSettings:
    def test_defaults(self):
        settings = FinOpsSettings()
        assert settings.enabled is True
        assert settings.budget_mode == "off"
        assert settings.max_cost_usd_per_run is None
        assert settings.openai_cheap_model == "gpt-5-mini"
        assert settings.anthropic_cheap_model == "claude-sonnet-4.6"

    def test_settings_has_finops(self):
        settings = Settings()
        assert settings.finops.enabled is True
        assert settings.finops.budget_mode == "off"


class TestSpanRecordFinOps:
    def test_finops_fields_default_none(self):
        span = SpanRecord(
            trace_id="t1", stage="s1", provider="p1", mode="replay",
            latency_ms=1.0, entity_id="e1", field="f1",
        )
        assert span.model_name is None
        assert span.estimated_cost_usd is None
        assert span.budget_status is None

    def test_finops_fields_set(self):
        span = SpanRecord(
            trace_id="t1", stage="analysis", provider="openai", mode="live",
            latency_ms=100.0, entity_id="e1", field="hq_country",
            model_name="GPT-5.4",
            estimated_input_tokens=1000,
            estimated_output_tokens=200,
            estimated_total_tokens=1200,
            estimated_cost_usd=0.00072,
            budget_status="nominal",
        )
        assert span.model_name == "GPT-5.4"
        assert span.estimated_cost_usd == pytest.approx(0.00072)


class TestTraceSummaryFinOps:
    def test_finops_fields_default_none(self):
        summary = TraceSummary(trace_id="t1", total_spans=1, total_latency_ms=1.0)
        assert summary.total_estimated_cost_usd is None
        assert summary.cost_by_stage is None
        assert summary.budget_status is None

    def test_finops_fields_set(self):
        summary = TraceSummary(
            trace_id="t1", total_spans=1, total_latency_ms=1.0,
            total_estimated_cost_usd=0.001,
            cost_by_stage={"analysis": 0.001},
            budget_status="nominal",
        )
        assert summary.total_estimated_cost_usd == pytest.approx(0.001)


class TestLocalTracerFinOps:
    def test_span_finops_passthrough(self):
        tracer = LocalTracer(mode="replay", entity_id="e1", field_name="hq_country")
        with tracer.span("analysis", provider="openai") as payload:
            payload["model_name"] = "GPT-5.4"
            payload["estimated_cost_usd"] = 0.001
        assert len(tracer.spans) == 1
        assert tracer.spans[0].model_name == "GPT-5.4"
        assert tracer.spans[0].estimated_cost_usd == pytest.approx(0.001)

    def test_write_with_finops_data(self, tmp_path):
        tracer = LocalTracer(mode="replay", entity_id="e1", field_name="hq_country")
        with tracer.span("analysis", provider="openai") as payload:
            payload["model_name"] = "GPT-5.4"
            payload["estimated_cost_usd"] = 0.001
        artifacts = tracer.write(
            tmp_path,
            finops_data={
                "total_estimated_cost_usd": 0.001,
                "budget_status": "nominal",
            },
        )
        assert artifacts.finops_summary_path is not None
        assert artifacts.finops_summary_path.exists()
        assert "finops_summary" in artifacts.as_refs()

    def test_write_without_finops_data(self, tmp_path):
        tracer = LocalTracer(mode="replay", entity_id="e1", field_name="hq_country")
        with tracer.span("analysis", provider="openai"):
            pass
        artifacts = tracer.write(tmp_path)
        assert artifacts.finops_summary_path is None

    def test_summary_populates_cost_fields(self, tmp_path):
        tracer = LocalTracer(mode="replay", entity_id="e1", field_name="hq_country")
        with tracer.span("analysis", provider="openai") as payload:
            payload["model_name"] = "GPT-5.4"
            payload["estimated_cost_usd"] = 0.001
        with tracer.span("synthesis", provider="openai") as payload:
            payload["model_name"] = "GPT-5.4"
            payload["estimated_cost_usd"] = 0.0005
        artifacts = tracer.write(tmp_path)
        import json
        summary = json.loads(artifacts.summary_path.read_text())
        assert summary["total_estimated_cost_usd"] == pytest.approx(0.0015)
        assert summary["cost_by_stage"]["analysis"] == pytest.approx(0.001)

    def test_timeline_includes_cost(self, tmp_path):
        tracer = LocalTracer(mode="replay", entity_id="e1", field_name="hq_country")
        with tracer.span("analysis", provider="openai") as payload:
            payload["model_name"] = "GPT-5.4"
            payload["estimated_cost_usd"] = 0.001
        artifacts = tracer.write(tmp_path)
        timeline = artifacts.timeline_path.read_text()
        assert "cost=$" in timeline
        assert "model=GPT-5.4" in timeline


class TestPipelineRunResultFinOps:
    def test_finops_summary_default_none(self):
        from evidence_enrichment.core.models.contracts import (
            SearchQueryPlan,
            SynthesisResult,
        )
        from evidence_enrichment.core.models.enums import ReviewDecision

        plan = SearchQueryPlan(
            field_name="hq_country",
            entity_id="microsoft",
            primary_query="Microsoft headquarters country",
        )
        synthesis = SynthesisResult(
            field_name="hq_country",
            value="US",
            reasoning="test",
            synthesis_confidence=0.9,
        )
        result = PipelineRunResult(
            entity_id="microsoft",
            field_name="hq_country",
            mode="replay",
            search_plan=plan,
            synthesis=synthesis,
            overall_confidence=0.9,
            decision=ReviewDecision.AUTO_APPROVE,
            gate_reason="test",
        )
        assert result.finops_summary is None

    def test_finops_summary_set(self):
        from evidence_enrichment.core.models.contracts import (
            SearchQueryPlan,
            SynthesisResult,
        )
        from evidence_enrichment.core.models.enums import ReviewDecision

        plan = SearchQueryPlan(
            field_name="hq_country",
            entity_id="microsoft",
            primary_query="Microsoft headquarters country",
        )
        synthesis = SynthesisResult(
            field_name="hq_country",
            value="US",
            reasoning="test",
            synthesis_confidence=0.9,
        )
        result = PipelineRunResult(
            entity_id="microsoft",
            field_name="hq_country",
            mode="replay",
            search_plan=plan,
            synthesis=synthesis,
            overall_confidence=0.9,
            decision=ReviewDecision.AUTO_APPROVE,
            gate_reason="test",
            finops_summary={"total_estimated_cost_usd": 0.001},
        )
        assert result.finops_summary["total_estimated_cost_usd"] == 0.001


class TestFinOpsPipelineIntegration:
    def test_replay_run_produces_finops_summary(self, tmp_path):
        from evidence_enrichment.config.settings import Settings
        from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
        from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

        settings = Settings(
            trace_output_dir=str(tmp_path / "traces"),
            replay_dir="examples/replay",
            finops=FinOpsSettings(enabled=True),
        )
        coordinator = EvidenceCoordinator(settings=settings)
        enricher = HeadquartersCountryEnricher()
        entity = {
            "entity_id": "microsoft",
            "name": "Microsoft Corporation",
            "website": "https://www.microsoft.com",
        }
        import asyncio
        result = asyncio.run(
            coordinator.run(
                entity, enricher, mode="replay",
                replay_bundle="examples/replay/microsoft_hq_country.json",
            )
        )
        assert result.finops_summary is not None
        assert result.finops_summary["total_estimated_cost_usd"] >= 0
        assert "cost_by_stage" in result.finops_summary
        # Replay agents return llm_usage=None — no LLM call is made, so no cost is
        # recorded for analysis or synthesis stages.  The summary is still emitted
        # (with zero cost) so downstream tooling can detect the replay run.
        assert result.finops_summary["total_estimated_cost_usd"] == 0.0
        assert result.artifact_refs.get("finops_summary") is not None
        finops_path = tmp_path / "traces" / result.trace_id / "finops_summary.json"
        assert finops_path.exists()

    def test_finops_disabled_no_summary(self, tmp_path):
        from evidence_enrichment.config.settings import Settings
        from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
        from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

        settings = Settings(
            trace_output_dir=str(tmp_path / "traces"),
            replay_dir="examples/replay",
            finops=FinOpsSettings(enabled=False),
        )
        coordinator = EvidenceCoordinator(settings=settings)
        enricher = HeadquartersCountryEnricher()
        entity = {
            "entity_id": "microsoft",
            "name": "Microsoft Corporation",
            "website": "https://www.microsoft.com",
        }
        import asyncio
        result = asyncio.run(
            coordinator.run(
                entity, enricher, mode="replay",
                replay_bundle="examples/replay/microsoft_hq_country.json",
            )
        )
        assert result.finops_summary is None


class TestBudgetPolicyPipeline:
    def test_strict_budget_blocks_replay_run(self, tmp_path):
        from evidence_enrichment.config.settings import Settings
        from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
        from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

        settings = Settings(
            trace_output_dir=str(tmp_path / "traces"),
            replay_dir="examples/replay",
            finops=FinOpsSettings(
                enabled=True,
                budget_mode="strict",
                max_cost_usd_per_run=0.0000001,
            ),
        )
        coordinator = EvidenceCoordinator(settings=settings)
        enricher = HeadquartersCountryEnricher()
        entity = {
            "entity_id": "microsoft",
            "name": "Microsoft Corporation",
            "website": "https://www.microsoft.com",
        }
        import asyncio
        result = asyncio.run(
            coordinator.run(
                entity, enricher, mode="replay",
                replay_bundle="examples/replay/microsoft_hq_country.json",
            )
        )
        assert result.decision.value == "auto_reject"
        assert "budget" in result.gate_reason.lower()
        assert result.finops_summary is not None
        assert result.finops_summary["budget_decision"]["status"] in ("blocked", "exceeded")

    def test_warn_budget_does_not_block(self, tmp_path):
        from evidence_enrichment.config.settings import Settings
        from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
        from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

        settings = Settings(
            trace_output_dir=str(tmp_path / "traces"),
            replay_dir="examples/replay",
            finops=FinOpsSettings(
                enabled=True,
                budget_mode="warn",
                max_cost_usd_per_run=0.0000001,
            ),
        )
        coordinator = EvidenceCoordinator(settings=settings)
        enricher = HeadquartersCountryEnricher()
        entity = {
            "entity_id": "microsoft",
            "name": "Microsoft Corporation",
            "website": "https://www.microsoft.com",
        }
        import asyncio
        result = asyncio.run(
            coordinator.run(
                entity, enricher, mode="replay",
                replay_bundle="examples/replay/microsoft_hq_country.json",
            )
        )
        assert result.decision.value != "auto_reject" or "budget" not in result.gate_reason.lower()
        assert result.finops_summary is not None

    def test_off_budget_no_blocking(self, tmp_path):
        from evidence_enrichment.config.settings import Settings
        from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
        from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

        settings = Settings(
            trace_output_dir=str(tmp_path / "traces"),
            replay_dir="examples/replay",
            finops=FinOpsSettings(
                enabled=True,
                budget_mode="off",
                max_cost_usd_per_run=0.0000001,
            ),
        )
        coordinator = EvidenceCoordinator(settings=settings)
        enricher = HeadquartersCountryEnricher()
        entity = {
            "entity_id": "microsoft",
            "name": "Microsoft Corporation",
            "website": "https://www.microsoft.com",
        }
        import asyncio
        result = asyncio.run(
            coordinator.run(
                entity, enricher, mode="replay",
                replay_bundle="examples/replay/microsoft_hq_country.json",
            )
        )
        assert "budget" not in (result.gate_reason or "").lower()
        assert result.output_value is not None
