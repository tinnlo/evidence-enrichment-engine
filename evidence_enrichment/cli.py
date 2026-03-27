"""CLI for evidence_enrichment."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evidence_enrichment.config.settings import get_settings
from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

app = typer.Typer(no_args_is_help=True)
console = Console()


def _write_result(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


async def _run_pipeline(entity: dict, *, mode: str, replay_bundle: str | None = None):
    coordinator = EvidenceCoordinator()
    enricher = HeadquartersCountryEnricher()
    return await coordinator.run(entity, enricher, mode=mode, replay_bundle=replay_bundle)


@app.command()
def run(
    entity: Path = typer.Option(..., exists=True, readable=True, help="Path to entity JSON."),
    field: str = typer.Option("hq_country", help="Supported field name."),
    mode: str = typer.Option("auto", help="live, replay, or auto"),
    replay_bundle: Path | None = typer.Option(None, exists=True, readable=True, help="Optional replay bundle path."),
    output: Path = typer.Option(Path("examples/output/run_result.json"), help="Output file."),
) -> None:
    if field != "hq_country":
        raise typer.BadParameter("Only hq_country is implemented in v0.1.")
    entity_payload = json.loads(entity.read_text(encoding="utf-8"))
    result = asyncio.run(_run_pipeline(entity_payload, mode=mode, replay_bundle=str(replay_bundle) if replay_bundle else None))
    _write_result(output, result)
    console.print(f"Saved run artifact to [bold]{output}[/bold]")
    console.print(f"Decision: [bold]{result.decision.value}[/bold] | Confidence: {result.overall_confidence:.2f}")


@app.command()
def demo(
    mode: str = typer.Option("auto", help="live, replay, or auto"),
    output: Path = typer.Option(Path("examples/output/demo_result.json"), help="Output file."),
) -> None:
    entity = {
        "entity_id": "microsoft",
        "name": "Microsoft Corporation",
        "website": "https://www.microsoft.com",
    }
    result = asyncio.run(_run_pipeline(entity, mode=mode))
    _write_result(output, result)
    console.print(f"Saved demo artifact to [bold]{output}[/bold]")
    console.print(f"HQ Country: [bold]{result.output_value}[/bold]")
    console.print(f"Decision: [bold]{result.decision.value}[/bold] | Confidence: {result.overall_confidence:.2f}")


@app.command()
def compare(
    output: Path = typer.Option(Path("examples/output/compare_result.json"), help="Combined output file."),
) -> None:
    entity = {
        "entity_id": "microsoft",
        "name": "Microsoft Corporation",
        "website": "https://www.microsoft.com",
    }
    baseline = asyncio.run(
        _run_pipeline(
            entity,
            mode="replay",
            replay_bundle="examples/replay/microsoft_hq_country_baseline.json",
        )
    )
    assessed = asyncio.run(
        _run_pipeline(
            entity,
            mode="replay",
            replay_bundle="examples/replay/microsoft_hq_country.json",
        )
    )
    payload = {
        "baseline": json.loads(baseline.model_dump_json()),
        "assessed": json.loads(assessed.model_dump_json()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = Table(title="Baseline vs Assessed")
    table.add_column("Path")
    table.add_column("Value")
    table.add_column("Decision")
    table.add_column("Confidence")
    table.add_row("Baseline", str(baseline.output_value), baseline.decision.value, f"{baseline.overall_confidence:.2f}")
    table.add_row("Assessed", str(assessed.output_value), assessed.decision.value, f"{assessed.overall_confidence:.2f}")
    console.print(table)
    console.print(f"Saved comparison artifact to [bold]{output}[/bold]")


@app.command("providers")
def providers_command() -> None:
    settings = get_settings()
    console.print(
        {
            "search": settings.search.provider_order,
            "analysis": settings.analysis.provider_order,
            "synthesis": settings.synthesis.provider_order,
            "replay_dir": settings.replay_dir,
        }
    )


if __name__ == "__main__":
    app()

