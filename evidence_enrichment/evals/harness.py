"""Replay eval harness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_case(entity: dict[str, Any], replay_bundle: str) -> Any:
    coordinator = EvidenceCoordinator()
    enricher = HeadquartersCountryEnricher()
    return asyncio.run(coordinator.run(entity, enricher, mode="replay", replay_bundle=replay_bundle))


def _build_finops_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    finops_cases: list[dict[str, Any]] = []
    total_cost = 0.0
    total_latency = 0.0
    cost_by_decision: dict[str, float] = {}
    passing_costs: list[float] = []

    for result in results:
        finops = result.get("finops_summary") or {}
        cost = finops.get("total_estimated_cost_usd", 0.0)
        latency = finops.get("total_latency_ms", 0.0)
        budget = finops.get("budget_decision", {})
        total_cost += cost
        total_latency += latency

        decision = result.get("actual_decision") or "unknown"
        cost_by_decision[decision] = round(cost_by_decision.get(decision, 0.0) + cost, 8)

        case_entry: dict[str, Any] = {
            "id": result["id"],
            "estimated_cost_usd": cost,
            "latency_ms": latency,
            "decision": decision,
            "confidence": result.get("actual_confidence", 0.0),
            "budget_status": budget.get("status", "unknown"),
            "budget_reason": budget.get("budget_reason"),
            "cost_by_stage": finops.get("cost_by_stage", {}),
            "cost_by_model": finops.get("cost_by_model", {}),
        }
        finops_cases.append(case_entry)
        if result.get("pass"):
            passing_costs.append(cost)

    total_cases = len(results)
    avg_cost_per_pass = round(sum(passing_costs) / len(passing_costs), 8) if passing_costs else 0.0

    return {
        "summary": {
            "total_cases": total_cases,
            "total_estimated_cost_usd": round(total_cost, 8),
            "average_latency_ms": round(total_latency / total_cases, 3) if total_cases else 0.0,
            "average_cost_per_passing_case": avg_cost_per_pass,
            "cost_by_decision": cost_by_decision,
        },
        "cases": finops_cases,
    }


def run_eval_harness(
    *,
    cases_path: Path,
    output_path: Path,
    finops_output_path: Path | None = None,
) -> dict[str, Any]:
    root_dir = cases_path.resolve().parent.parent
    raw = _load_yaml(cases_path)
    cases = list(raw.get("cases") or [])
    results: list[dict[str, Any]] = []

    for case in cases:
        entity_path = root_dir / str(case["entity_file"])
        replay_path = root_dir / str(case["replay_bundle"])
        entity = _load_json(entity_path)
        expected_value = case["expected_value"]
        expected_decision = case["expected_decision"]
        min_confidence = float(case["min_confidence"])

        try:
            run_result = _run_case(entity, str(replay_path))
            actual_value = run_result.output_value
            actual_decision = run_result.decision.value
            actual_confidence = run_result.overall_confidence
            mismatch_reason = None
            passed = True
            if actual_value != expected_value:
                mismatch_reason = "value_mismatch"
                passed = False
            elif actual_decision != expected_decision:
                mismatch_reason = "decision_mismatch"
                passed = False
            elif actual_confidence < min_confidence:
                mismatch_reason = "below_confidence_threshold"
                passed = False
            results.append(
                {
                    "id": case["id"],
                    "description": case["description"],
                    "expected_value": expected_value,
                    "expected_decision": expected_decision,
                    "minimum_confidence": min_confidence,
                    "actual_value": actual_value,
                    "actual_decision": actual_decision,
                    "actual_confidence": actual_confidence,
                    "pass": passed,
                    "mismatch_reason": mismatch_reason,
                    "artifact_refs": run_result.artifact_refs,
                    "finops_summary": (
                        json.loads(run_result.finops_summary.model_dump_json())
                        if hasattr(run_result.finops_summary, "model_dump_json")
                        else run_result.finops_summary
                    ),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": case["id"],
                    "description": case["description"],
                    "expected_value": expected_value,
                    "expected_decision": expected_decision,
                    "minimum_confidence": min_confidence,
                    "actual_value": None,
                    "actual_decision": None,
                    "actual_confidence": 0.0,
                    "pass": False,
                    "mismatch_reason": "run_error",
                    "error": str(exc),
                    "artifact_refs": {},
                }
            )

    total_cases = len(results)
    passed_count = sum(1 for result in results if result["pass"])
    failed_count = total_cases - passed_count
    value_matches = sum(1 for result in results if result["actual_value"] == result["expected_value"])
    decision_matches = sum(1 for result in results if result["actual_decision"] == result["expected_decision"])
    mismatch_counts: dict[str, int] = {}
    for result in results:
        if result["mismatch_reason"] is None:
            continue
        mismatch_counts[result["mismatch_reason"]] = mismatch_counts.get(result["mismatch_reason"], 0) + 1
    summary = {
        "total_cases": total_cases,
        "passed": passed_count,
        "failed": failed_count,
        "value_match_rate": round(value_matches / total_cases, 4) if total_cases else 0.0,
        "decision_match_rate": round(decision_matches / total_cases, 4) if total_cases else 0.0,
        "average_confidence": round(sum(result["actual_confidence"] for result in results) / total_cases, 4) if total_cases else 0.0,
        "mismatch_counts": mismatch_counts,
    }
    report = {
        "harness": {
            "task_name": raw.get("task_name", "hq_country_resolution"),
            "field_name": raw.get("field_name", "hq_country"),
            "mode": "replay",
            "case_count": total_cases,
        },
        "summary": summary,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    effective_finops_path = finops_output_path or output_path.parent / "latest_finops_report.json"
    finops_report = _build_finops_report(results)
    effective_finops_path.parent.mkdir(parents=True, exist_ok=True)
    effective_finops_path.write_text(
        json.dumps(finops_report, indent=2), encoding="utf-8"
    )

    return report
