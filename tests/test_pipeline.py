from pathlib import Path

from typer.testing import CliRunner

from evidence_enrichment.cli import app
from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
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
    assert "\"output_value\": \"USA\"" in output.read_text(encoding="utf-8")

