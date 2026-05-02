"""MCP (Model Context Protocol) server for the Evidence Enrichment Engine.

Exposes enriched and synthesized datasets as MCP resources and tools so that
any MCP-compatible AI agent (Claude Desktop, OpenCode, MCP Inspector, etc.)
can call into the pipeline directly.

Transports:
  - stdio   (default) — for agent integration (Claude Desktop, Claude Code)
  - streamable-http   — for browser clients and MCP Inspector

Usage:
    # Stdio (agent integration)
    python -m evidence_enrichment.mcp_server

    # HTTP (for MCP Inspector at http://localhost:8000/mcp)
    python -m evidence_enrichment.mcp_server --transport streamable-http

    # Via CLI
    evidence-enrich mcp
    evidence-enrich mcp --transport streamable-http

    # Via dedicated entrypoint
    evidence-enrich-mcp
    evidence-enrich-mcp --transport streamable-http
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from evidence_enrichment.config.settings import get_settings
from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
from evidence_enrichment.core.models.contracts import FactClaim, PipelineRunResult
from evidence_enrichment.core.models.enums import ReviewDecision
from evidence_enrichment.execution_policy.engine import ExecutionPolicyEngine
from evidence_enrichment.execution_policy.models import ActionType
from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as _mcp_import_error:  # pragma: no cover
    raise ImportError(
        "MCP SDK not installed. Install it with:\n\n"
        "    pip install 'evidence_enrichment[mcp]'\n\n"
        "or:\n\n"
        "    pip install mcp\n"
    ) from _mcp_import_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Evidence Enrichment Engine",
    instructions=(
        "This server exposes an evidence-backed entity enrichment pipeline. "
        "Use `list_replay_scenarios` to discover available datasets, "
        "`run_enrichment_pipeline` to run the full pipeline for an entity, "
        "`get_synthesis_result` for a concise summary, and "
        "`compare_scenarios` to compare two replay scenarios side-by-side. "
        "All tools default to `replay` mode and require no API keys."
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPPORTED_FIELDS = ("hq_country",)

_SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "microsoft_hq_country": "Primary scenario — two agreeing high-authority sources (auto_approve)",
    "microsoft_hq_country_baseline": "Weak secondary evidence only (needs_review)",
    "microsoft_hq_country_conflict": "Conflicting claims across sources (needs_review)",
    "microsoft_hq_country_no_support": "No accepted claims found (auto_reject)",
    "microsoft_hq_country_invalid_iso3": "Synthesis returns non-ISO3 value (auto_reject)",
    "microsoft_hq_country_low_signal": "Single low-signal claim (needs_review)",
}


def _replay_dir() -> Path:
    return Path(get_settings().replay_dir)


def _output_dir() -> Path:
    return Path("examples/output")


def _bundle_names() -> list[str]:
    """Return replay bundle stems (no extension) from the replay directory."""
    replay_dir = _replay_dir()
    if not replay_dir.exists():
        return []
    return sorted(p.stem for p in replay_dir.glob("*.json"))


def _bundle_path(name: str) -> Path:
    """Resolve a bundle file path from its stem name."""
    return _replay_dir() / f"{name}.json"


async def _run_pipeline(
    entity: dict,
    *,
    mode: str = "replay",
    replay_bundle: str | None = None,
) -> PipelineRunResult:
    """Run the enrichment pipeline and return a full PipelineRunResult."""
    settings = get_settings()

    # Gate live runs via execution policy before spinning up the coordinator.
    # Replay mode is always permitted — policy only applies to live network calls.
    if mode != "replay":
        policy_engine = ExecutionPolicyEngine(settings.execution_policy)
        decision = policy_engine.check_action(ActionType.MCP_LIVE_RUNS)
        if not decision.allowed:
            raise PermissionError(
                f"execution_policy: MCP_LIVE_RUNS is blocked in "
                f"'{settings.execution_policy.mode}' mode. "
                "Set execution_policy.mode to 'off' or 'audit' to allow live runs, "
                "or add 'mcp_live_runs' to the allowed_actions list."
            )

    coordinator = EvidenceCoordinator()
    enricher = HeadquartersCountryEnricher()
    return await coordinator.run(
        entity,
        enricher,
        mode=mode,
        replay_bundle=replay_bundle,
    )


def _entity_dict(entity_name: str, entity_id: str) -> dict:
    return {
        "entity_id": entity_id,
        "name": entity_name,
        "website": f"https://www.{entity_id}.com",
    }


# ---------------------------------------------------------------------------
# Pydantic models for structured tool outputs
# ---------------------------------------------------------------------------


class ScenarioInfo(BaseModel):
    """Metadata about a single replay scenario."""

    name: str = Field(description="Bundle file stem (use as replay_bundle argument)")
    description: str = Field(
        description="Human-readable description of what this scenario tests"
    )


class SynthesisSummary(BaseModel):
    """Concise synthesis + review outcome for a pipeline run."""

    entity_id: str
    field_name: str
    mode: str
    output_value: str | None = Field(
        description="The resolved field value, or null if rejected"
    )
    synthesis_reasoning: str
    synthesis_confidence: float
    overall_confidence: float
    decision: ReviewDecision
    gate_reason: str
    source_count: int
    claim_count: int
    supporting_urls: list[str] = Field(default_factory=list)
    conflicts_detected: bool


class ClaimsResult(BaseModel):
    """All fact claims extracted from a pipeline run."""

    entity_id: str
    field_name: str
    mode: str
    claim_count: int
    claims: list[FactClaim]


class ScenarioComparison(BaseModel):
    """Side-by-side comparison of two replay scenarios."""

    scenario_a: SynthesisSummary
    scenario_b: SynthesisSummary
    same_output: bool
    same_decision: bool


def _to_synthesis_summary(result: PipelineRunResult) -> SynthesisSummary:
    all_claims = [
        claim for report in result.analysis_reports for claim in report.claims
    ]
    return SynthesisSummary(
        entity_id=result.entity_id,
        field_name=result.field_name,
        mode=result.mode,
        output_value=result.output_value,
        synthesis_reasoning=result.synthesis.reasoning,
        synthesis_confidence=result.synthesis.synthesis_confidence,
        overall_confidence=result.overall_confidence,
        decision=result.decision,
        gate_reason=result.gate_reason,
        source_count=len(result.sources),
        claim_count=len(all_claims),
        supporting_urls=result.synthesis.supporting_urls,
        conflicts_detected=bool(result.synthesis.conflicts),
    )


# ---------------------------------------------------------------------------
# Resources — read-only data access
# ---------------------------------------------------------------------------


@mcp.resource("evidence://bundles")
def list_bundles() -> str:
    """List all available replay bundle names.

    Returns a JSON array of bundle stem names. Each name can be passed as the
    `replay_bundle` argument to `run_enrichment_pipeline` or used in
    `compare_scenarios`.
    """
    names = _bundle_names()
    return json.dumps(
        [
            {
                "name": n,
                "description": _SCENARIO_DESCRIPTIONS.get(n, "Replay bundle"),
            }
            for n in names
        ],
        indent=2,
    )


@mcp.resource("evidence://bundles/{name}")
def get_bundle(name: str) -> str:
    """Retrieve the raw contents of a named replay bundle.

    The bundle contains pre-recorded search results, parsed documents,
    analysis reports, and synthesis used for zero-API-key replay runs.

    Args:
        name: Bundle stem (e.g. 'microsoft_hq_country'). Use
              `evidence://bundles` to list available names.
    """
    path = _bundle_path(name)
    if not path.exists():
        available = ", ".join(_bundle_names()) or "none"
        return json.dumps(
            {
                "error": f"Bundle '{name}' not found.",
                "available_bundles": available,
            }
        )
    return path.read_text(encoding="utf-8")


@mcp.resource("evidence://results/latest")
def get_latest_result() -> str:
    """Retrieve the most recent pipeline run result artifact.

    Returns the full PipelineRunResult JSON from the last `demo` or `run`
    command execution, selecting by modification time so the newest file is
    always returned regardless of which CLI command produced it. Returns an
    informational message if no result exists yet.
    """
    candidates = [
        _output_dir() / "demo_result.json",
        _output_dir() / "trace_demo_result.json",
        _output_dir() / "run_result.json",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return json.dumps(
            {
                "message": (
                    "No pipeline result found. Run `evidence-enrich demo` first, "
                    "or call the `run_enrichment_pipeline` tool."
                )
            }
        )
    newest = max(existing, key=lambda p: p.stat().st_mtime)
    return newest.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tools — actions that invoke the pipeline
# ---------------------------------------------------------------------------


@mcp.tool()
def list_replay_scenarios() -> list[ScenarioInfo]:
    """List all available replay scenarios with their descriptions.

    Each scenario is a pre-recorded dataset that drives the enrichment
    pipeline without requiring any API keys. Use the returned `name`
    field as the `replay_bundle` argument in other tools.
    """
    return [
        ScenarioInfo(
            name=n,
            description=_SCENARIO_DESCRIPTIONS.get(n, "Replay bundle"),
        )
        for n in _bundle_names()
    ]


@mcp.tool()
async def run_enrichment_pipeline(
    entity_name: Annotated[
        str,
        Field(
            description="Full legal or common name of the entity, e.g. 'Microsoft Corporation'"
        ),
    ],
    entity_id: Annotated[
        str,
        Field(description="Slug-style identifier, e.g. 'microsoft'"),
    ] = "microsoft",
    field: Annotated[
        str,
        Field(description="Field to enrich. Currently only 'hq_country' is supported."),
    ] = "hq_country",
    mode: Annotated[
        str,
        Field(
            description="Execution mode: 'replay' (no API keys needed), 'live', or 'auto'"
        ),
    ] = "replay",
    replay_bundle: Annotated[
        str | None,
        Field(
            description="Replay bundle stem to use, e.g. 'microsoft_hq_country'. Leave null to auto-select."
        ),
    ] = None,
) -> PipelineRunResult:
    """Run the full evidence enrichment pipeline for a given entity and field.

    Orchestrates the full 8-stage pipeline:
      1. query_plan        — Build search queries
      2. search            — Find evidence sources
      3. fetch             — Download web documents
      4. parse             — Extract text from HTML
      5. evidence_assessment — Score and filter documents
      6. analysis          — Extract fact claims with LLM
      7. synthesis         — Resolve a final value from claims
      8. review_gate       — Apply confidence thresholds

    Returns the complete PipelineRunResult including all intermediate
    artifacts (search results, parsed documents, analysis reports,
    synthesis, confidence scores, and review decision).

    For a more concise result, use `get_synthesis_result` instead.
    """
    if field not in _SUPPORTED_FIELDS:
        raise ValueError(f"Unsupported field '{field}'. Supported: {_SUPPORTED_FIELDS}")
    entity = _entity_dict(entity_name, entity_id)
    bundle_path: str | None = None
    if replay_bundle:
        p = _bundle_path(replay_bundle)
        if not p.exists():
            raise FileNotFoundError(
                f"Replay bundle '{replay_bundle}' not found. "
                f"Use list_replay_scenarios to see available options."
            )
        bundle_path = str(p)
    return await _run_pipeline(entity, mode=mode, replay_bundle=bundle_path)


@mcp.tool()
async def get_synthesis_result(
    entity_name: Annotated[
        str,
        Field(
            description="Full legal or common name of the entity, e.g. 'Microsoft Corporation'"
        ),
    ],
    entity_id: Annotated[
        str,
        Field(description="Slug-style identifier, e.g. 'microsoft'"),
    ] = "microsoft",
    field: Annotated[
        str,
        Field(description="Field to enrich. Currently only 'hq_country' is supported."),
    ] = "hq_country",
    mode: Annotated[
        str,
        Field(
            description="Execution mode: 'replay' (no API keys needed), 'live', or 'auto'"
        ),
    ] = "replay",
    replay_bundle: Annotated[
        str | None,
        Field(
            description="Replay bundle stem, e.g. 'microsoft_hq_country'. Leave null to auto-select."
        ),
    ] = None,
) -> SynthesisSummary:
    """Run the pipeline and return a concise synthesis summary.

    A lightweight alternative to `run_enrichment_pipeline` that returns only
    the key decision fields: resolved value, confidence scores, review
    decision, and reasoning — without the full intermediate artifacts.

    Useful when you want a quick answer to "what is this entity's hq_country?"
    rather than the full evidence trail.
    """
    if field not in _SUPPORTED_FIELDS:
        raise ValueError(f"Unsupported field '{field}'. Supported: {_SUPPORTED_FIELDS}")
    entity = _entity_dict(entity_name, entity_id)
    bundle_path: str | None = None
    if replay_bundle:
        p = _bundle_path(replay_bundle)
        if not p.exists():
            raise FileNotFoundError(
                f"Replay bundle '{replay_bundle}' not found. "
                f"Use list_replay_scenarios to see available options."
            )
        bundle_path = str(p)
    result = await _run_pipeline(entity, mode=mode, replay_bundle=bundle_path)
    return _to_synthesis_summary(result)


@mcp.tool()
async def get_evidence_claims(
    entity_name: Annotated[
        str,
        Field(
            description="Full legal or common name of the entity, e.g. 'Microsoft Corporation'"
        ),
    ],
    entity_id: Annotated[
        str,
        Field(description="Slug-style identifier, e.g. 'microsoft'"),
    ] = "microsoft",
    field: Annotated[
        str,
        Field(description="Field to enrich. Currently only 'hq_country' is supported."),
    ] = "hq_country",
    mode: Annotated[
        str,
        Field(
            description="Execution mode: 'replay' (no API keys needed), 'live', or 'auto'"
        ),
    ] = "replay",
    replay_bundle: Annotated[
        str | None,
        Field(description="Replay bundle stem. Leave null to auto-select."),
    ] = None,
) -> ClaimsResult:
    """Run the pipeline and return all extracted fact claims.

    Each FactClaim includes the candidate value, supporting excerpt, source
    URL, and individual confidence scores (analysis_confidence,
    source_authority_score, freshness_score, entity_match_score).

    Useful for inspecting the raw evidence before synthesis resolves a
    final value, or for understanding why the pipeline chose a particular
    output.
    """
    if field not in _SUPPORTED_FIELDS:
        raise ValueError(f"Unsupported field '{field}'. Supported: {_SUPPORTED_FIELDS}")
    entity = _entity_dict(entity_name, entity_id)
    bundle_path: str | None = None
    if replay_bundle:
        p = _bundle_path(replay_bundle)
        if not p.exists():
            raise FileNotFoundError(
                f"Replay bundle '{replay_bundle}' not found. "
                f"Use list_replay_scenarios to see available options."
            )
        bundle_path = str(p)
    result = await _run_pipeline(entity, mode=mode, replay_bundle=bundle_path)
    all_claims = [
        claim for report in result.analysis_reports for claim in report.claims
    ]
    return ClaimsResult(
        entity_id=result.entity_id,
        field_name=result.field_name,
        mode=result.mode,
        claim_count=len(all_claims),
        claims=all_claims,
    )


@mcp.tool()
async def compare_scenarios(
    scenario_a: Annotated[
        str,
        Field(description="First replay bundle stem, e.g. 'microsoft_hq_country'"),
    ],
    scenario_b: Annotated[
        str,
        Field(
            description="Second replay bundle stem, e.g. 'microsoft_hq_country_conflict'"
        ),
    ],
    entity_name: Annotated[
        str,
        Field(description="Entity name (used for both scenarios)"),
    ] = "Microsoft Corporation",
    entity_id: Annotated[
        str,
        Field(description="Entity ID slug (used for both scenarios)"),
    ] = "microsoft",
) -> ScenarioComparison:
    """Run two replay scenarios and return a side-by-side comparison.

    Runs both scenarios through the full pipeline and compares the synthesis
    outcomes, confidence scores, and review decisions. Useful for
    understanding how evidence quality and conflict affect pipeline output.

    Example: compare 'microsoft_hq_country' (auto_approve) vs
    'microsoft_hq_country_conflict' (needs_review) to see how conflicting
    evidence lowers confidence and changes the review gate decision.
    """
    entity = _entity_dict(entity_name, entity_id)

    for name in (scenario_a, scenario_b):
        p = _bundle_path(name)
        if not p.exists():
            raise FileNotFoundError(
                f"Replay bundle '{name}' not found. "
                f"Use list_replay_scenarios to see available options."
            )

    result_a, result_b = await asyncio.gather(
        _run_pipeline(
            entity, mode="replay", replay_bundle=str(_bundle_path(scenario_a))
        ),
        _run_pipeline(
            entity, mode="replay", replay_bundle=str(_bundle_path(scenario_b))
        ),
    )

    summary_a = _to_synthesis_summary(result_a)
    summary_b = _to_synthesis_summary(result_b)

    return ScenarioComparison(
        scenario_a=summary_a,
        scenario_b=summary_b,
        same_output=summary_a.output_value == summary_b.output_value,
        same_decision=summary_a.decision == summary_b.decision,
    )


# ---------------------------------------------------------------------------
# Prompt — reusable agent interaction template
# ---------------------------------------------------------------------------


@mcp.prompt()
def analyze_entity(
    entity_name: str,
    field_name: str = "hq_country",
    scenario: str = "microsoft_hq_country",
) -> str:
    """Generate a structured analysis prompt for an entity enrichment task.

    Guides an AI agent through interpreting evidence enrichment pipeline
    output, evaluating claim quality, and explaining the review decision.

    Args:
        entity_name:  The entity to analyse, e.g. 'Microsoft Corporation'.
        field_name:   The field being enriched, e.g. 'hq_country'.
        scenario:     The replay scenario to use as evidence context.
    """
    return (
        f"You are an evidence analyst reviewing the output of an automated "
        f"entity enrichment pipeline.\n\n"
        f"**Task**: Enrich the `{field_name}` field for **{entity_name}**.\n\n"
        f"**Instructions**:\n"
        f"1. Call `list_replay_scenarios` to see available evidence datasets.\n"
        f"2. Call `get_synthesis_result` with entity_name='{entity_name}' and "
        f"   replay_bundle='{scenario}' to get the pipeline's synthesis.\n"
        f"3. Call `get_evidence_claims` with the same arguments to inspect the "
        f"   individual fact claims that informed the synthesis.\n"
        f"4. Explain:\n"
        f"   - What value was resolved and why\n"
        f"   - Which sources were most authoritative\n"
        f"   - Why the pipeline issued its review decision "
        f"     (auto_approve / needs_review / auto_reject)\n"
        f"   - Any conflicts or low-confidence signals in the evidence\n"
        f"5. If you want to see how a different evidence scenario changes the "
        f"   outcome, call `compare_scenarios` to run two bundles side-by-side.\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the MCP server.

    Reads transport from ``--transport`` / ``-t`` flag.
    Defaults to stdio (for agent integration).

    Examples::

        evidence-enrich-mcp
        evidence-enrich-mcp --transport streamable-http
    """
    import sys

    transport = "stdio"
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ("--transport", "-t") and i + 1 < len(args):
            transport = args[i + 1]
        elif arg.startswith("--transport="):
            transport = arg.split("=", 1)[1]

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
