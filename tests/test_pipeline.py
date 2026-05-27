from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import json

import pytest
from typer.testing import CliRunner

from evidence_enrichment.cli import app
from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
from evidence_enrichment.core.models.contracts import ParsedDocument
from evidence_enrichment.core.models.enums import ProviderType, ReviewDecision
from evidence_enrichment.guardrails.models import CheckResult, GuardrailsReport
from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator


runner = CliRunner()


def test_replay_demo_pipeline_returns_usa() -> None:
    entity = {
        "entity_id": "microsoft",
        "name": "Microsoft Corporation",
        "website": "https://www.microsoft.com",
    }
    result = __import__("asyncio").run(
        EvidenceCoordinator().run(entity, HeadquartersCountryEnricher(), mode="replay")
    )
    assert result.output_value == "USA"
    assert result.decision.value == "auto_approve"
    assert result.overall_confidence >= 0.85
    assert result.resolved_context is not None
    assert result.trace_id is not None
    assert Path(result.artifact_refs["resolved_context"]).exists()
    assert Path(result.artifact_refs["trace_summary"]).exists()


def test_compare_cli_creates_output(tmp_path: Path) -> None:
    output = tmp_path / "compare.json"
    result = runner.invoke(app, ["compare", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    payload = output.read_text(encoding="utf-8")
    assert "baseline" in payload
    assert "assessed" in payload


def test_run_cli_replay_mode(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    result = runner.invoke(
        app,
        [
            "run",
            "--entity",
            "examples/microsoft.json",
            "--field",
            "hq_country",
            "--mode",
            "replay",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert output.exists()
    assert '"output_value": "USA"' in output.read_text(encoding="utf-8")


def test_context_pack_cli_creates_output(tmp_path: Path) -> None:
    output = tmp_path / "resolved_context.json"
    result = runner.invoke(app, ["context-pack", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    assert '"task_name": "hq_country_resolution"' in output.read_text(encoding="utf-8")


def test_eval_cli_creates_report(tmp_path: Path) -> None:
    output = tmp_path / "latest_report.json"
    finops_output = tmp_path / "latest_finops_report.json"
    result = runner.invoke(app, ["eval", "--output", str(output), "--finops-output", str(finops_output)])
    assert result.exit_code == 0
    assert output.exists()
    payload = output.read_text(encoding="utf-8")
    assert '"summary"' in payload
    assert '"total_cases": 7' in payload
    assert finops_output.exists()
    finops_payload = json.loads(finops_output.read_text(encoding="utf-8"))
    assert "summary" in finops_payload
    assert "cases" in finops_payload
    assert finops_payload["summary"]["total_cases"] == 7


def _make_accepted_doc(url: str, text: str = "some document text") -> ParsedDocument:
    return ParsedDocument(
        url=url,
        title="Test",
        content_type="text/html",
        text=text,
        excerpt=text[:200],
        accepted_for_analysis=True,
        entity_match_score=0.8,
        source_authority_score=0.7,
        freshness_score=0.9,
    )


class TestStageAnalysisAllFail:
    """_stage_analysis() must propagate a RuntimeError when every accepted
    document fails analysis so that auto mode can fall back to replay instead
    of silently returning empty claims."""

    def test_raises_when_all_accepted_documents_fail(self):
        """All-fail branch raises RuntimeError with an informative message."""
        failing_agent = MagicMock()
        failing_agent.analyze = AsyncMock(
            side_effect=RuntimeError("provider unavailable")
        )
        failing_agent.provider_type = ProviderType.OPENAI

        coordinator = EvidenceCoordinator()
        with patch.object(coordinator, "_analysis_agent", return_value=failing_agent):
            with pytest.raises(RuntimeError, match="analysis call"):
                asyncio.run(
                    coordinator._stage_analysis(
                        [
                            _make_accepted_doc("https://example.com/a"),
                            _make_accepted_doc("https://example.com/b"),
                        ],
                        field_name="hq_country",
                        company_name="Acme Corp",
                        bundle=None,
                        trace_payload={},
                    )
                )

    def test_partial_failure_does_not_raise(self):
        """A mix of passing and failing documents must not raise — partial
        claims are still useful and the run should continue."""
        from evidence_enrichment.core.models.contracts import AnalysisReport

        good_report = AnalysisReport(
            source_url="https://example.com/a",
            provider=ProviderType.OPENAI,
            claims=[],
            reasoning="ok",
        )

        def _side_effect(doc, *args, **kwargs):
            if "a" in doc.url:
                return good_report
            raise RuntimeError("provider unavailable")

        agent = MagicMock()
        agent.analyze = AsyncMock(side_effect=_side_effect)
        agent.provider_type = ProviderType.OPENAI

        coordinator = EvidenceCoordinator()
        with patch.object(coordinator, "_analysis_agent", return_value=agent):
            # Should complete without raising
            reports, claims, _ = asyncio.run(
                coordinator._stage_analysis(
                    [
                        _make_accepted_doc("https://example.com/a"),
                        _make_accepted_doc("https://example.com/b"),
                    ],
                    field_name="hq_country",
                    company_name="Acme Corp",
                    bundle=None,
                    trace_payload={},
                )
            )
        assert len(reports) == 2  # one good report + one error placeholder
        assert claims == []  # good_report had no claims

    def test_coordinator_short_circuits_on_guardrail_failure(self):
        """Guardrails failure must override AUTO_APPROVE to AUTO_REJECT.

        We run in replay mode (no live providers) so the gate would normally
        return AUTO_APPROVE for the microsoft fixture.  We then patch
        ``run_guardrails`` to return a failing report, proving that the
        coordinator's decision is driven by guardrails and not the review gate.
        """
        _failing_report = GuardrailsReport(
            pii=CheckResult(name="pii", passed=True),
            hallucination=CheckResult(
                name="hallucination",
                passed=False,
                reason="claim URL not in fetched document set",
            ),
            confidence=CheckResult(name="confidence", passed=True),
            passed=False,
        )

        entity = {
            "entity_id": "microsoft",
            "name": "Microsoft Corporation",
            "website": "https://www.microsoft.com",
        }

        with patch(
            "evidence_enrichment.pipeline.coordinator.run_guardrails",
            return_value=_failing_report,
        ):
            result = asyncio.run(
                EvidenceCoordinator().run(
                    entity, HeadquartersCountryEnricher(), mode="replay"
                )
            )

        assert result.decision == ReviewDecision.AUTO_REJECT
        assert result.guardrails_report is not None
        assert not result.guardrails_report.passed
        assert result.gate_reason == result.guardrails_report.failure_summary()

    def test_rejected_documents_do_not_count_as_accepted(self):
        """Documents with accepted_for_analysis=False must not trigger the
        all-fail guard — zero accepted docs is not an infrastructure failure."""
        coordinator = EvidenceCoordinator()
        rejected_doc = ParsedDocument(
            url="https://example.com/rejected",
            title="Rejected",
            content_type="text/html",
            text="irrelevant",
            excerpt="irrelevant",
            accepted_for_analysis=False,
            entity_match_score=0.1,
            source_authority_score=0.1,
            freshness_score=0.5,
        )
        # Should complete without raising even though no agent is ever called
        reports, claims, _ = asyncio.run(
            coordinator._stage_analysis(
                [rejected_doc],
                field_name="hq_country",
                company_name="Acme Corp",
                bundle={},  # use replay agent so no live provider needed
                trace_payload={},
            )
        )
        assert reports == []
        assert claims == []
