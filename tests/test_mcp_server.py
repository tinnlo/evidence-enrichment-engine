"""Tests for the MCP server (evidence_enrichment/mcp_server.py).

These tests exercise the MCP tools and resources directly as Python functions
— no MCP transport connection required.  The pipeline runs in `replay` mode
throughout so no API keys are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Guard: skip the whole module if `mcp` is not installed
# ---------------------------------------------------------------------------
pytest.importorskip(
    "mcp", reason="mcp package not installed — skipping MCP server tests"
)

from evidence_enrichment.mcp_server import (  # noqa: E402
    ClaimsResult,
    ScenarioComparison,
    ScenarioInfo,
    SynthesisSummary,
    _bundle_names,
    _bundle_path,
    _entity_dict,
    compare_scenarios,
    get_bundle,
    get_evidence_claims,
    get_latest_result,
    get_synthesis_result,
    list_bundles,
    list_replay_scenarios,
    run_enrichment_pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

KNOWN_SCENARIOS = [
    "microsoft_hq_country",
    "microsoft_hq_country_baseline",
    "microsoft_hq_country_conflict",
    "microsoft_hq_country_no_support",
    "microsoft_hq_country_invalid_iso3",
    "microsoft_hq_country_low_signal",
]


# ---------------------------------------------------------------------------
# Resource: evidence://bundles
# ---------------------------------------------------------------------------


def test_list_bundles_returns_json_array():
    raw = list_bundles()
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) >= 6


def test_list_bundles_includes_known_scenarios():
    raw = list_bundles()
    data = json.loads(raw)
    names = {item["name"] for item in data}
    for scenario in KNOWN_SCENARIOS:
        assert scenario in names, f"Expected scenario '{scenario}' in bundle list"


def test_list_bundles_has_description_field():
    raw = list_bundles()
    data = json.loads(raw)
    for item in data:
        assert "name" in item
        assert "description" in item
        assert len(item["description"]) > 0


# ---------------------------------------------------------------------------
# Resource: evidence://bundles/{name}
# ---------------------------------------------------------------------------


def test_get_bundle_returns_valid_json_for_primary_scenario():
    raw = get_bundle("microsoft_hq_country")
    data = json.loads(raw)
    assert "search_results" in data
    assert "parsed_documents" in data
    assert "analysis_reports" in data
    assert "synthesis" in data


def test_get_bundle_contains_synthesis_value():
    raw = get_bundle("microsoft_hq_country")
    data = json.loads(raw)
    assert data["synthesis"]["value"] == "USA"


def test_get_bundle_missing_returns_error_json():
    raw = get_bundle("nonexistent_bundle_xyz")
    data = json.loads(raw)
    assert "error" in data
    assert "available_bundles" in data


# ---------------------------------------------------------------------------
# Resource: evidence://results/latest
# ---------------------------------------------------------------------------


def test_get_latest_result_returns_json():
    raw = get_latest_result()
    # Either a valid result or the informational message — both are JSON
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_get_latest_result_message_when_no_output(tmp_path):
    """When output dir has no artifacts, return the informational message."""
    with patch("evidence_enrichment.mcp_server._output_dir", return_value=tmp_path):
        raw = get_latest_result()
    data = json.loads(raw)
    assert "message" in data


def test_get_latest_result_returns_newest_by_mtime(tmp_path):
    """When multiple artifacts exist, return the one with the latest mtime."""
    import time

    older = tmp_path / "demo_result.json"
    newer = tmp_path / "run_result.json"
    older.write_text(json.dumps({"source": "older"}), encoding="utf-8")
    time.sleep(0.01)  # ensure distinct mtimes
    newer.write_text(json.dumps({"source": "newer"}), encoding="utf-8")

    with patch("evidence_enrichment.mcp_server._output_dir", return_value=tmp_path):
        raw = get_latest_result()
    data = json.loads(raw)
    assert data["source"] == "newer"


# ---------------------------------------------------------------------------
# Tool: list_replay_scenarios
# ---------------------------------------------------------------------------


def test_list_replay_scenarios_returns_list_of_scenario_info():
    result = list_replay_scenarios()
    assert isinstance(result, list)
    assert len(result) >= 6
    for item in result:
        assert isinstance(item, ScenarioInfo)
        assert item.name
        assert item.description


def test_list_replay_scenarios_includes_all_known():
    result = list_replay_scenarios()
    names = {s.name for s in result}
    for scenario in KNOWN_SCENARIOS:
        assert scenario in names


# ---------------------------------------------------------------------------
# Tool: run_enrichment_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_enrichment_pipeline_default_scenario():
    from evidence_enrichment.core.models.contracts import PipelineRunResult

    result = await run_enrichment_pipeline(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
        field="hq_country",
        mode="replay",
    )
    assert isinstance(result, PipelineRunResult)
    assert result.entity_id == "microsoft"
    assert result.field_name == "hq_country"
    assert result.output_value == "USA"


@pytest.mark.asyncio
async def test_run_enrichment_pipeline_explicit_bundle():
    from evidence_enrichment.core.models.contracts import PipelineRunResult

    result = await run_enrichment_pipeline(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
        field="hq_country",
        mode="replay",
        replay_bundle="microsoft_hq_country",
    )
    assert isinstance(result, PipelineRunResult)
    assert result.output_value == "USA"


@pytest.mark.asyncio
async def test_run_enrichment_pipeline_conflict_scenario():
    from evidence_enrichment.core.models.enums import ReviewDecision

    result = await run_enrichment_pipeline(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
        field="hq_country",
        mode="replay",
        replay_bundle="microsoft_hq_country_conflict",
    )
    assert result.decision == ReviewDecision.NEEDS_REVIEW


@pytest.mark.asyncio
async def test_run_enrichment_pipeline_no_support_auto_reject():
    from evidence_enrichment.core.models.enums import ReviewDecision

    result = await run_enrichment_pipeline(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
        field="hq_country",
        mode="replay",
        replay_bundle="microsoft_hq_country_no_support",
    )
    assert result.decision == ReviewDecision.AUTO_REJECT


@pytest.mark.asyncio
async def test_run_enrichment_pipeline_unsupported_field_raises():
    with pytest.raises(ValueError, match="Unsupported field"):
        await run_enrichment_pipeline(
            entity_name="Acme Corp",
            entity_id="acme",
            field="revenue",
            mode="replay",
        )


@pytest.mark.asyncio
async def test_run_enrichment_pipeline_missing_bundle_raises():
    with pytest.raises(FileNotFoundError):
        await run_enrichment_pipeline(
            entity_name="Acme Corp",
            entity_id="acme",
            field="hq_country",
            mode="replay",
            replay_bundle="nonexistent_bundle_xyz",
        )


# ---------------------------------------------------------------------------
# Tool: get_synthesis_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_synthesis_result_returns_summary():
    result = await get_synthesis_result(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
    )
    assert isinstance(result, SynthesisSummary)
    assert result.output_value == "USA"
    assert result.entity_id == "microsoft"
    assert result.field_name == "hq_country"


@pytest.mark.asyncio
async def test_get_synthesis_result_auto_approve_confidence():
    from evidence_enrichment.core.models.enums import ReviewDecision

    result = await get_synthesis_result(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
        replay_bundle="microsoft_hq_country",
    )
    assert result.decision == ReviewDecision.AUTO_APPROVE
    assert result.overall_confidence >= 0.85


@pytest.mark.asyncio
async def test_get_synthesis_result_has_required_fields():
    result = await get_synthesis_result(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
    )
    assert result.synthesis_reasoning
    assert result.synthesis_confidence > 0
    assert result.claim_count > 0
    assert isinstance(result.supporting_urls, list)
    assert isinstance(result.conflicts_detected, bool)


# ---------------------------------------------------------------------------
# Tool: get_evidence_claims
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_evidence_claims_returns_claims_result():
    result = await get_evidence_claims(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
    )
    assert isinstance(result, ClaimsResult)
    assert result.claim_count > 0
    assert len(result.claims) == result.claim_count


@pytest.mark.asyncio
async def test_get_evidence_claims_field_values():
    result = await get_evidence_claims(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
    )
    for claim in result.claims:
        assert claim.field_name == "hq_country"
        assert claim.candidate_value
        assert 0.0 <= claim.analysis_confidence <= 1.0
        assert 0.0 <= claim.source_authority_score <= 1.0


@pytest.mark.asyncio
async def test_get_evidence_claims_no_support_empty():
    result = await get_evidence_claims(
        entity_name="Microsoft Corporation",
        entity_id="microsoft",
        replay_bundle="microsoft_hq_country_no_support",
    )
    # no-support scenario has no accepted claims → no claims extracted
    assert result.claim_count == 0


# ---------------------------------------------------------------------------
# Tool: compare_scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_scenarios_returns_comparison():
    result = await compare_scenarios(
        scenario_a="microsoft_hq_country",
        scenario_b="microsoft_hq_country_conflict",
    )
    assert isinstance(result, ScenarioComparison)
    assert isinstance(result.scenario_a, SynthesisSummary)
    assert isinstance(result.scenario_b, SynthesisSummary)


@pytest.mark.asyncio
async def test_compare_scenarios_same_output_different_decision():
    from evidence_enrichment.core.models.enums import ReviewDecision

    result = await compare_scenarios(
        scenario_a="microsoft_hq_country",
        scenario_b="microsoft_hq_country_conflict",
    )
    # Both resolve to USA but with different decisions
    assert result.scenario_a.decision == ReviewDecision.AUTO_APPROVE
    assert result.scenario_b.decision == ReviewDecision.NEEDS_REVIEW
    # same_output depends on both returning USA
    assert result.same_output is True
    assert result.same_decision is False


@pytest.mark.asyncio
async def test_compare_scenarios_approve_vs_reject():
    from evidence_enrichment.core.models.enums import ReviewDecision

    result = await compare_scenarios(
        scenario_a="microsoft_hq_country",
        scenario_b="microsoft_hq_country_no_support",
    )
    assert result.scenario_a.decision == ReviewDecision.AUTO_APPROVE
    assert result.scenario_b.decision == ReviewDecision.AUTO_REJECT
    assert result.same_decision is False


@pytest.mark.asyncio
async def test_compare_scenarios_missing_bundle_raises():
    with pytest.raises(FileNotFoundError):
        await compare_scenarios(
            scenario_a="microsoft_hq_country",
            scenario_b="nonexistent_xyz",
        )


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------


def test_bundle_names_returns_sorted_list():
    names = _bundle_names()
    assert names == sorted(names)
    assert len(names) >= 6


def test_entity_dict_structure():
    d = _entity_dict("Acme Corp", "acme")
    assert d["entity_id"] == "acme"
    assert d["name"] == "Acme Corp"
    assert "website" in d


def test_bundle_path_resolves_correctly():
    p = _bundle_path("microsoft_hq_country")
    assert p.suffix == ".json"
    assert p.exists()


# ---------------------------------------------------------------------------
# Guard: _mcp_entry console-script wrapper
# ---------------------------------------------------------------------------


def test_mcp_entry_exits_with_guidance_when_mcp_missing(capsys):
    """_mcp_entry.main() must print installation guidance and exit 1 when mcp
    is not importable — not raise an uncaught ImportError."""
    import sys
    import types
    from unittest.mock import patch as _patch

    # Simulate mcp being absent by injecting a broken module into sys.modules.
    broken = types.ModuleType("mcp")
    broken.__spec__ = None  # type: ignore[attr-defined]

    # Remove any cached mcp_server and _mcp_entry so imports are re-evaluated.
    for key in list(sys.modules):
        if "mcp_server" in key or "_mcp_entry" in key:
            del sys.modules[key]

    with _patch.dict(
        sys.modules,
        {
            "mcp": None,  # type: ignore[dict-item]
            "mcp.server": None,  # type: ignore[dict-item]
            "mcp.server.fastmcp": None,  # type: ignore[dict-item]
        },
    ):
        from evidence_enrichment._mcp_entry import main as _entry_main

        with pytest.raises(SystemExit) as exc_info:
            _entry_main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "pip install" in captured.err
