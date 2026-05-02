"""Tests for the execution policy layer: models, engine, config, and integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_enrichment.config.settings import ExecutionPolicySettings, Settings
from evidence_enrichment.core.models.contracts import PipelineRunResult
from evidence_enrichment.execution_policy.engine import ExecutionPolicyEngine
from evidence_enrichment.execution_policy.models import (
    ActionType,
    ExecutionPolicyReport,
    PolicyDecision,
    PolicyMode,
)


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


class TestPolicyDecision:
    def test_allowed_decision_serialises_enums_as_strings(self):
        d = PolicyDecision(
            action=ActionType.SEARCH,
            allowed=True,
            mode=PolicyMode.AUDIT,
            reason="audit_allowed",
        )
        dumped = d.model_dump()
        assert dumped["action"] == "search"
        assert dumped["mode"] == "audit"
        assert isinstance(dumped["timestamp"], str)

    def test_blocked_decision_has_allowed_false(self):
        d = PolicyDecision(
            action=ActionType.FETCH,
            allowed=False,
            mode=PolicyMode.ENFORCE,
            reason="enforce_blocked:fetch_not_in_allowed_actions",
        )
        assert d.allowed is False


# ---------------------------------------------------------------------------
# ExecutionPolicyReport
# ---------------------------------------------------------------------------


class TestExecutionPolicyReport:
    def test_empty_report_serialises(self):
        report = ExecutionPolicyReport(mode=PolicyMode.OFF)
        dumped = report.model_dump()
        assert dumped["mode"] == "off"
        assert dumped["decisions"] == []
        assert dumped["blocked_actions"] == []
        assert dumped["violations"] == []

    def test_blocked_actions_serialised_as_strings(self):
        decision = PolicyDecision(
            action=ActionType.LIVE_PROVIDER_CALLS,
            allowed=False,
            mode=PolicyMode.ENFORCE,
            reason="enforce_blocked:live_provider_calls_not_in_allowed_actions",
        )
        report = ExecutionPolicyReport(
            mode=PolicyMode.ENFORCE,
            decisions=[decision],
            blocked_actions=[ActionType.LIVE_PROVIDER_CALLS],
            violations=[decision],
        )
        dumped = report.model_dump()
        assert dumped["blocked_actions"] == ["live_provider_calls"]


# ---------------------------------------------------------------------------
# ExecutionPolicySettings
# ---------------------------------------------------------------------------


class TestExecutionPolicySettings:
    def test_defaults(self):
        s = ExecutionPolicySettings()
        assert s.mode == "off"
        assert s.enabled is True
        assert isinstance(s.allowed_actions, list)

    def test_all_actions_allowed_by_default(self):
        s = ExecutionPolicySettings()
        all_actions = {a.value for a in ActionType}
        assert all_actions.issubset(set(s.allowed_actions))

    def test_settings_load_has_execution_policy(self):
        settings = Settings.load()
        assert hasattr(settings, "execution_policy")
        assert isinstance(settings.execution_policy, ExecutionPolicySettings)


# ---------------------------------------------------------------------------
# ExecutionPolicyEngine — off mode
# ---------------------------------------------------------------------------


class TestEngineOffMode:
    def _engine(self, **kwargs) -> ExecutionPolicyEngine:
        return ExecutionPolicyEngine(ExecutionPolicySettings(mode="off", **kwargs))

    def test_all_actions_allowed(self):
        engine = self._engine()
        for action in ActionType:
            decision = engine.check_action(action)
            assert decision.allowed is True

    def test_no_decisions_recorded(self):
        engine = self._engine()
        engine.check_action(ActionType.SEARCH)
        report = engine.build_report()
        assert report.decisions == []
        assert report.violations == []
        assert report.blocked_actions == []

    def test_reason_is_policy_off(self):
        engine = self._engine()
        decision = engine.check_action(ActionType.FETCH)
        assert decision.reason == "policy_off"


# ---------------------------------------------------------------------------
# ExecutionPolicyEngine — audit mode
# ---------------------------------------------------------------------------


class TestEngineAuditMode:
    def _engine(self, allowed_actions: list[str] | None = None) -> ExecutionPolicyEngine:
        actions = allowed_actions if allowed_actions is not None else [a.value for a in ActionType]
        return ExecutionPolicyEngine(
            ExecutionPolicySettings(mode="audit", allowed_actions=actions)
        )

    def test_allowed_action_passes_and_is_recorded(self):
        engine = self._engine(allowed_actions=["search"])
        decision = engine.check_action(ActionType.SEARCH)
        assert decision.allowed is True
        assert decision.reason == "audit_allowed"
        report = engine.build_report()
        assert len(report.decisions) == 1

    def test_disallowed_action_still_passes_but_is_a_violation(self):
        engine = self._engine(allowed_actions=["search"])
        decision = engine.check_action(ActionType.FETCH)
        assert decision.allowed is True  # audit never blocks
        assert decision.reason == "audit_violation"

    def test_violation_appears_in_report(self):
        engine = self._engine(allowed_actions=[])
        engine.check_action(ActionType.MCP_LIVE_RUNS)
        report = engine.build_report()
        assert len(report.violations) == 1
        assert report.violations[0].action == ActionType.MCP_LIVE_RUNS

    def test_blocked_actions_empty_in_audit(self):
        engine = self._engine(allowed_actions=[])
        engine.check_action(ActionType.REMOTE_TRACING)
        report = engine.build_report()
        assert report.blocked_actions == []

    def test_multiple_checks_all_recorded(self):
        engine = self._engine(allowed_actions=["search"])
        engine.check_action(ActionType.SEARCH)
        engine.check_action(ActionType.FETCH)
        report = engine.build_report()
        assert len(report.decisions) == 2


# ---------------------------------------------------------------------------
# ExecutionPolicyEngine — enforce mode
# ---------------------------------------------------------------------------


class TestEngineEnforceMode:
    def _engine(self, allowed_actions: list[str]) -> ExecutionPolicyEngine:
        return ExecutionPolicyEngine(
            ExecutionPolicySettings(mode="enforce", allowed_actions=allowed_actions)
        )

    def test_allowed_action_passes(self):
        engine = self._engine(["search", "fetch"])
        decision = engine.check_action(ActionType.SEARCH)
        assert decision.allowed is True
        assert decision.reason == "enforce_allowed"

    def test_blocked_action_returns_not_allowed(self):
        engine = self._engine(["search"])
        decision = engine.check_action(ActionType.LIVE_PROVIDER_CALLS)
        assert decision.allowed is False
        assert "enforce_blocked" in decision.reason

    def test_blocked_action_appears_in_report(self):
        engine = self._engine(["search"])
        engine.check_action(ActionType.FETCH)
        report = engine.build_report()
        assert ActionType.FETCH in report.blocked_actions
        assert len(report.violations) == 1

    def test_allowed_action_not_in_blocked(self):
        engine = self._engine(["search"])
        engine.check_action(ActionType.SEARCH)
        report = engine.build_report()
        assert report.blocked_actions == []

    def test_empty_allowed_actions_blocks_everything(self):
        engine = self._engine([])
        for action in ActionType:
            decision = engine.check_action(action)
            assert decision.allowed is False


# ---------------------------------------------------------------------------
# ExecutionPolicyEngine — reset
# ---------------------------------------------------------------------------


class TestEngineReset:
    def test_reset_clears_decisions(self):
        engine = ExecutionPolicyEngine(
            ExecutionPolicySettings(mode="audit", allowed_actions=[])
        )
        engine.check_action(ActionType.SEARCH)
        assert len(engine.build_report().decisions) == 1
        engine.reset()
        assert len(engine.build_report().decisions) == 0


# ---------------------------------------------------------------------------
# ExecutionPolicyEngine — unknown mode falls back to off
# ---------------------------------------------------------------------------


class TestEngineUnknownMode:
    def test_unknown_mode_falls_back_to_off(self):
        settings = ExecutionPolicySettings(mode="unknown_xyz", allowed_actions=[])
        engine = ExecutionPolicyEngine(settings)
        decision = engine.check_action(ActionType.SEARCH)
        assert decision.allowed is True
        assert decision.reason == "policy_off"


# ---------------------------------------------------------------------------
# PipelineRunResult has execution_policy_report field
# ---------------------------------------------------------------------------


class TestContractsIntegration:
    def test_pipeline_run_result_has_policy_report_field(self):
        # Check the field exists on the model class without constructing a full instance.
        assert "execution_policy_report" in PipelineRunResult.model_fields

    def test_policy_report_field_is_optional(self):
        field = PipelineRunResult.model_fields["execution_policy_report"]
        # Optional fields have a default of None.
        assert field.default is None


# ---------------------------------------------------------------------------
# Tracer artifact: execution_policy.json written when data provided
# ---------------------------------------------------------------------------


class TestTracerExecutionPolicyArtifact:
    def test_artifact_written_when_data_provided(self, tmp_path: Path):
        from evidence_enrichment.observability.tracer import LocalTracer

        tracer = LocalTracer(mode="replay", entity_id="test", field_name="hq_country")
        policy_data = {"mode": "audit", "decisions": [], "blocked_actions": [], "violations": []}
        artifacts = tracer.write(tmp_path, execution_policy_data=policy_data)

        assert artifacts.execution_policy_path is not None
        assert artifacts.execution_policy_path.exists()
        written = json.loads(artifacts.execution_policy_path.read_text())
        assert written["mode"] == "audit"

    def test_artifact_not_written_when_no_data(self, tmp_path: Path):
        from evidence_enrichment.observability.tracer import LocalTracer

        tracer = LocalTracer(mode="replay", entity_id="test", field_name="hq_country")
        artifacts = tracer.write(tmp_path)
        assert artifacts.execution_policy_path is None

    def test_artifact_in_as_refs(self, tmp_path: Path):
        from evidence_enrichment.observability.tracer import LocalTracer

        tracer = LocalTracer(mode="replay", entity_id="test", field_name="hq_country")
        policy_data = {"mode": "off", "decisions": [], "blocked_actions": [], "violations": []}
        artifacts = tracer.write(tmp_path, execution_policy_data=policy_data)
        refs = artifacts.as_refs()
        assert "execution_policy" in refs


# ---------------------------------------------------------------------------
# Replay pipeline smoke test — policy report always attached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_pipeline_attaches_policy_report():
    """Replay runs must always produce an execution_policy_report, even in off mode."""
    from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
    from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

    coordinator = EvidenceCoordinator()
    enricher = HeadquartersCountryEnricher()
    entity = {"entity_id": "microsoft", "name": "Microsoft Corporation"}

    result = await coordinator.run(entity, enricher, mode="replay")

    assert result.execution_policy_report is not None
    assert "mode" in result.execution_policy_report
    assert "decisions" in result.execution_policy_report
    # Replay never triggers live gates — no violations expected
    assert result.execution_policy_report["violations"] == []


@pytest.mark.asyncio
async def test_replay_pipeline_policy_report_has_no_violations():
    """Replay path must never produce policy violations regardless of mode."""
    from evidence_enrichment.config.settings import ExecutionPolicySettings
    from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
    from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

    # Use enforce mode with an empty allowed_actions list — replay should
    # still succeed because live gates are not reached in replay mode.
    coordinator = EvidenceCoordinator()
    # Override settings to enforce with nothing allowed
    coordinator.settings.execution_policy = ExecutionPolicySettings(
        mode="enforce", allowed_actions=[]
    )
    enricher = HeadquartersCountryEnricher()
    entity = {"entity_id": "microsoft", "name": "Microsoft Corporation"}

    result = await coordinator.run(entity, enricher, mode="replay")

    assert result.execution_policy_report is not None
    assert result.execution_policy_report["blocked_actions"] == []


# ---------------------------------------------------------------------------
# activate_remote_tracing_with_policy_check integration
# ---------------------------------------------------------------------------


class TestActivateRemoteTracingWithPolicyCheck:
    """Verify the policy-aware remote tracing helper is wired and behaves correctly."""

    def test_returns_token_when_no_run_context(self):
        """Without an active pipeline run the helper falls through and activates normally."""
        from evidence_enrichment.observability.runtime import (
            activate_remote_tracing_with_policy_check,
            reset_runtime_observability_config,
        )

        token = activate_remote_tracing_with_policy_check(backend="none")
        # Should return a ContextVar token (not None) when no policy run is active.
        assert token is not None
        reset_runtime_observability_config(token)

    def test_returns_none_when_remote_tracing_blocked(self):
        """In enforce mode with REMOTE_TRACING not allowed the helper must return None."""
        from evidence_enrichment.execution_policy.engine import ExecutionPolicyEngine
        from evidence_enrichment.observability.runtime import (
            activate_remote_tracing_with_policy_check,
        )
        from evidence_enrichment.pipeline.coordinator import (
            _PolicyRunContext,
            _prc_var,
        )

        engine = ExecutionPolicyEngine(
            ExecutionPolicySettings(mode="enforce", allowed_actions=[])
        )
        prc = _PolicyRunContext(engine=engine)
        token = _prc_var.set(prc)
        try:
            result = activate_remote_tracing_with_policy_check(backend="langfuse")
            assert result is None
        finally:
            _prc_var.reset(token)

    def test_returns_token_in_audit_mode(self):
        """Audit mode never blocks — helper must return a token even with REMOTE_TRACING absent."""
        from evidence_enrichment.execution_policy.engine import ExecutionPolicyEngine
        from evidence_enrichment.observability.runtime import (
            activate_remote_tracing_with_policy_check,
            reset_runtime_observability_config,
        )
        from evidence_enrichment.pipeline.coordinator import (
            _PolicyRunContext,
            _prc_var,
        )

        engine = ExecutionPolicyEngine(
            ExecutionPolicySettings(mode="audit", allowed_actions=[])
        )
        prc = _PolicyRunContext(engine=engine)
        token = _prc_var.set(prc)
        try:
            result = activate_remote_tracing_with_policy_check(backend="none")
            assert result is not None
            reset_runtime_observability_config(result)
            # Violation should be recorded
            report = engine.build_report()
            assert any(v.action.value == "remote_tracing" for v in report.violations)
        finally:
            _prc_var.reset(token)

    def test_returns_token_when_remote_tracing_in_allowed_actions(self):
        """Enforce mode with REMOTE_TRACING in allowed_actions must still return a token."""
        from evidence_enrichment.execution_policy.engine import ExecutionPolicyEngine
        from evidence_enrichment.observability.runtime import (
            activate_remote_tracing_with_policy_check,
            reset_runtime_observability_config,
        )
        from evidence_enrichment.pipeline.coordinator import (
            _PolicyRunContext,
            _prc_var,
        )

        engine = ExecutionPolicyEngine(
            ExecutionPolicySettings(mode="enforce", allowed_actions=["remote_tracing"])
        )
        prc = _PolicyRunContext(engine=engine)
        token = _prc_var.set(prc)
        try:
            result = activate_remote_tracing_with_policy_check(backend="none")
            assert result is not None
            reset_runtime_observability_config(result)
        finally:
            _prc_var.reset(token)

    def test_coordinator_run_does_not_crash_when_remote_tracing_blocked(self):
        """Full replay run must succeed even when REMOTE_TRACING is blocked (runtime_token None guard)."""
        import asyncio

        from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
        from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

        coordinator = EvidenceCoordinator()
        coordinator.settings.execution_policy = ExecutionPolicySettings(
            mode="enforce", allowed_actions=[]  # blocks everything including remote_tracing
        )
        enricher = HeadquartersCountryEnricher()
        entity = {"entity_id": "microsoft", "name": "Microsoft Corporation"}

        result = asyncio.get_event_loop().run_until_complete(
            coordinator.run(entity, enricher, mode="replay")
        )
        assert result is not None
        assert result.execution_policy_report is not None
