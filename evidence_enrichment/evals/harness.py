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


def run_eval_harness(*, cases_path: Path, output_path: Path) -> dict[str, Any]:
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
    return report
