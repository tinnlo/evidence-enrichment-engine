"""CLI for evidence_enrichment."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from evidence_enrichment.config.settings import get_settings
from evidence_enrichment.context.resolver import ContextResolver
from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
from evidence_enrichment.evals.harness import run_eval_harness
from evidence_enrichment.observability.langsmith import flush_langsmith_traces
from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

app = typer.Typer(no_args_is_help=True)
console = Console()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump_json"):
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _demo_entity() -> dict[str, str]:
    return {
        "entity_id": "microsoft",
        "name": "Microsoft Corporation",
        "website": "https://www.microsoft.com",
    }


def _print_artifacts(result: Any) -> None:
    if not getattr(result, "artifact_refs", None):
        return
    console.print(f"Resolved context: [bold]{result.artifact_refs.get('resolved_context', 'n/a')}[/bold]")
    console.print(f"Trace directory: [bold]{result.artifact_refs.get('trace_dir', 'n/a')}[/bold]")
    console.print(f"Trace summary: [bold]{result.artifact_refs.get('trace_summary', 'n/a')}[/bold]")


def _print_named_artifacts(label: str, result: Any) -> None:
    if not getattr(result, "artifact_refs", None):
        return
    console.print(f"{label} context: [bold]{result.artifact_refs.get('resolved_context', 'n/a')}[/bold]")
    console.print(f"{label} trace dir: [bold]{result.artifact_refs.get('trace_dir', 'n/a')}[/bold]")
    console.print(f"{label} trace summary: [bold]{result.artifact_refs.get('trace_summary', 'n/a')}[/bold]")


async def _run_pipeline(entity: dict, *, mode: str, replay_bundle: str | None = None):
    coordinator = EvidenceCoordinator()
    enricher = HeadquartersCountryEnricher()
    return await coordinator.run(entity, enricher, mode=mode, replay_bundle=replay_bundle)


def _flush_observability() -> None:
    flush_langsmith_traces()


@app.command()
def run(
    entity: Path = typer.Option(..., exists=True, readable=True, help="Path to entity JSON."),
    field: str = typer.Option("hq_country", help="Supported field name."),
    mode: str = typer.Option("auto", help="live, replay, or auto"),
    replay_bundle: Path | None = typer.Option(None, exists=True, readable=True, help="Optional replay bundle path."),
    output: Path = typer.Option(Path("examples/output/run_result.json"), help="Output file."),
) -> None:
    try:
        if field != "hq_country":
            raise typer.BadParameter("Only hq_country is implemented in v0.1.")
        entity_payload = json.loads(entity.read_text(encoding="utf-8"))
        result = asyncio.run(_run_pipeline(entity_payload, mode=mode, replay_bundle=str(replay_bundle) if replay_bundle else None))
        _write_json(output, result)
        console.print(f"Saved run artifact to [bold]{output}[/bold]")
        console.print(f"Decision: [bold]{result.decision.value}[/bold] | Confidence: {result.overall_confidence:.2f}")
        _print_artifacts(result)
    finally:
        _flush_observability()


@app.command()
def demo(
    mode: str = typer.Option("auto", help="live, replay, or auto"),
    output: Path = typer.Option(Path("examples/output/demo_result.json"), help="Output file."),
) -> None:
    try:
        entity = _demo_entity()
        result = asyncio.run(_run_pipeline(entity, mode=mode))
        _write_json(output, result)
        console.print(f"Saved demo artifact to [bold]{output}[/bold]")
        console.print(f"HQ Country: [bold]{result.output_value}[/bold]")
        console.print(f"Decision: [bold]{result.decision.value}[/bold] | Confidence: {result.overall_confidence:.2f}")
        _print_artifacts(result)
    finally:
        _flush_observability()


@app.command("trace-demo")
def trace_demo(
    mode: str = typer.Option("auto", help="live, replay, or auto"),
    output: Path = typer.Option(Path("examples/output/trace_demo_result.json"), help="Output file."),
) -> None:
    try:
        entity = _demo_entity()
        result = asyncio.run(_run_pipeline(entity, mode=mode))
        _write_json(output, result)
        console.print(f"Saved traced demo artifact to [bold]{output}[/bold]")
        console.print(f"Trace ID: [bold]{result.trace_id}[/bold]")
        _print_artifacts(result)
    finally:
        _flush_observability()


@app.command()
def compare(
    output: Path = typer.Option(Path("examples/output/compare_result.json"), help="Combined output file."),
) -> None:
    try:
        entity = _demo_entity()
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
        _write_json(output, payload)

        table = Table(title="Baseline vs Assessed")
        table.add_column("Path")
        table.add_column("Value")
        table.add_column("Decision")
        table.add_column("Confidence")
        table.add_row("Baseline", str(baseline.output_value), baseline.decision.value, f"{baseline.overall_confidence:.2f}")
        table.add_row("Assessed", str(assessed.output_value), assessed.decision.value, f"{assessed.overall_confidence:.2f}")
        console.print(table)
        console.print(f"Saved comparison artifact to [bold]{output}[/bold]")
        _print_named_artifacts("Baseline", baseline)
        _print_named_artifacts("Assessed", assessed)
    finally:
        _flush_observability()


@app.command("context-pack")
def context_pack(
    entity_id: str = typer.Option("microsoft", help="Entity identifier for the context artifact."),
    field: str = typer.Option("hq_country", help="Supported field name."),
    output: Path = typer.Option(Path("examples/output/resolved_context.json"), help="Resolved context artifact path."),
) -> None:
    try:
        if field != "hq_country":
            raise typer.BadParameter("Only hq_country is implemented in v0.1.")
        settings = get_settings()
        resolver = ContextResolver(settings.context_path / "context_manifest.yaml")
        bundle = resolver.resolve(entity_id=entity_id, field_name=field)
        _write_json(output, bundle)
        console.print(f"Saved resolved context to [bold]{output}[/bold]")
    finally:
        _flush_observability()


@app.command("eval")
def eval_command(
    cases: Path = typer.Option(Path("evals/cases.yaml"), exists=True, readable=True, help="Path to eval case definitions."),
    output: Path = typer.Option(Path("evals/output/latest_report.json"), help="Eval report path."),
) -> None:
    try:
        report = run_eval_harness(cases_path=cases, output_path=output)
        summary = report["summary"]
        console.print(
            f"Eval summary: [bold]{summary['passed']}[/bold]/[bold]{summary['total_cases']}[/bold] passed | "
            f"value_match_rate={summary['value_match_rate']:.2f} | decision_match_rate={summary['decision_match_rate']:.2f}"
        )
        console.print(f"Saved eval report to [bold]{output}[/bold]")
    finally:
        _flush_observability()


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
