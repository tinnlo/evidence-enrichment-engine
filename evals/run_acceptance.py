"""Stage D acceptance harness — schema extraction and gate behavior.

Acceptance criteria (from docs/hierarchical_retrieval_upgrade.md §D):
  - schema_valid_rate  >= 0.80   (fraction of *expected-valid* cases where
                                   validation_passed=True; deliberately-invalid
                                   negative fixtures are excluded from this rate)
  - gate_confidence_mean >= 0.75  (mean confidence_after for expected-valid cases
                                   that produced validation_passed=True)
  - no_hard_fail_passes_gate      (every expected-fail case must have
                                   gate_passed=False; confidence value is
                                   irrelevant for this criterion)
  - all_acceptance_cases_pass     (every per-case assertion passes)

Note on negative cases: ``geo_revenue_sum_mismatch`` and
``geo_revenue_missing_provenance`` are deliberately-invalid fixtures.  They are
excluded from ``schema_valid_rate`` and ``gate_confidence_mean`` because those
metrics measure extraction quality, not gate behavior.  Gate behavior for
invalid inputs is covered by ``no_hard_fail_passes_gate``.

This harness does NOT require live LLM calls.  It constructs ``ExtractionResult``
objects from fixture data, runs them through ``SchemaValidationGate``, and
checks acceptance criteria.  This is the deterministic acceptance gate that CI
can run without provider credentials.

Exit codes
----------
    0  All acceptance criteria met.
    1  One or more criteria failed.
    2  Import / setup error.

Usage
-----
    python evals/run_acceptance.py
    python evals/run_acceptance.py --output evals/output/latest_acceptance_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


def _check_deps() -> None:
    try:
        import pydantic  # noqa: F401
    except ImportError:
        print("ERROR: pydantic not installed", file=sys.stderr)
        sys.exit(2)
    try:
        import evidence_enrichment  # noqa: F401
    except ImportError:
        print(
            "ERROR: evidence_enrichment not installed. Run: pip install -e '.[dev]'",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_geo_revenue_valid() -> "ExtractionResult":
    """Microsoft FY2024 geographic revenue — passes all validators."""
    from evidence_enrichment.core.extraction.models import ExtractionResult
    from evidence_enrichment.core.extraction.schemas import (
        GeographicRevenueExtraction,
        GeographicRevenueRow,
        MoneyAmount,
    )

    value = GeographicRevenueExtraction(
        fiscal_year=2024,
        currency="USD",
        rows=[
            GeographicRevenueRow(
                region="United States",
                region_type="region",
                amount=MoneyAmount(value=Decimal("133141"), currency="USD"),
                percentage_of_total=54.3,
                source_chunk_ids=["chunk_geo_us_001"],
            ),
            GeographicRevenueRow(
                region="Other regions",
                region_type="region",
                amount=MoneyAmount(value=Decimal("111945"), currency="USD"),
                percentage_of_total=45.7,
                source_chunk_ids=["chunk_geo_int_001"],
            ),
        ],
        total_revenue=MoneyAmount(value=Decimal("245086"), currency="USD"),
        extraction_confidence=0.91,
    )
    return ExtractionResult(
        field_name="geographic_revenue",
        schema_cls_name="GeographicRevenueExtraction",
        schema_version=1,
        validation_passed=True,
        validation_errors=[],
        value=value,
        extraction_confidence=0.91,
        chunks_used=["chunk_geo_us_001", "chunk_geo_int_001"],
        repair_count=0,
    )


def _make_geo_revenue_sum_mismatch() -> "ExtractionResult":
    """Row sum diverges from total by >2% — should be a hard fail in gate."""
    from evidence_enrichment.core.extraction.models import ExtractionResult
    from evidence_enrichment.core.extraction.schemas import (
        GeographicRevenueExtraction,
        GeographicRevenueRow,
        MoneyAmount,
    )
    from pydantic import ValidationError

    try:
        GeographicRevenueExtraction(
            fiscal_year=2024,
            currency="USD",
            rows=[
                GeographicRevenueRow(
                    region="US",
                    region_type="region",
                    amount=MoneyAmount(value=Decimal("100000"), currency="USD"),
                    source_chunk_ids=["chunk_001"],
                ),
                GeographicRevenueRow(
                    region="Rest of world",
                    region_type="region",
                    amount=MoneyAmount(value=Decimal("50000"), currency="USD"),
                    source_chunk_ids=["chunk_002"],
                ),
            ],
            total_revenue=MoneyAmount(value=Decimal("200000"), currency="USD"),
            extraction_confidence=0.70,
        )
    except ValidationError as exc:
        errors = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        # Build a failed ExtractionResult via model_construct
        from evidence_enrichment.core.extraction.models import _coerce_nested_fields

        raw = {
            "fiscal_year": 2024,
            "currency": "USD",
            "rows": [
                {"region": "US", "region_type": "region",
                 "amount": {"value": 100000, "currency": "USD"}, "source_chunk_ids": ["chunk_001"]},
                {"region": "Rest of world", "region_type": "region",
                 "amount": {"value": 50000, "currency": "USD"}, "source_chunk_ids": ["chunk_002"]},
            ],
            "total_revenue": {"value": 200000, "currency": "USD"},
            "extraction_confidence": 0.70,
        }
        coerced = _coerce_nested_fields(GeographicRevenueExtraction, raw)
        value = GeographicRevenueExtraction.model_construct(**coerced)
        return ExtractionResult(
            field_name="geographic_revenue",
            schema_cls_name="GeographicRevenueExtraction",
            schema_version=1,
            validation_passed=False,
            validation_errors=errors,
            value=value,
            extraction_confidence=0.70,
            chunks_used=["chunk_001", "chunk_002"],
            repair_count=2,
        )
    # Should not reach here — if validator changed, treat as valid
    return _make_geo_revenue_valid()


def _make_scope1_valid() -> "ExtractionResult":
    """Scope 1 emissions fixture — passes all validators."""
    from decimal import Decimal

    from evidence_enrichment.core.extraction.models import ExtractionResult
    from evidence_enrichment.core.extraction.schemas import EmissionsExtraction, EmissionsRow

    value = EmissionsExtraction(
        fiscal_year=2024,
        rows=[
            EmissionsRow(
                scope="scope_1",
                value=Decimal("7100"),
                unit="tCO2e",
                year=2024,
                boundary="operational control",
                source_chunk_ids=["chunk_scope1_001"],
            ),
        ],
        reporting_standard="GHG Protocol",
        assurance_level="limited",
        extraction_confidence=0.88,
    )
    return ExtractionResult(
        field_name="scope_1_emissions",
        schema_cls_name="EmissionsExtraction",
        schema_version=1,
        validation_passed=True,
        validation_errors=[],
        value=value,
        extraction_confidence=0.88,
        chunks_used=["chunk_scope1_001"],
        repair_count=0,
    )


def _make_headcount_valid() -> "ExtractionResult":
    """Headcount fixture — passes total-consistency validator."""
    from evidence_enrichment.core.extraction.models import ExtractionResult
    from evidence_enrichment.core.extraction.schemas import HeadcountExtraction, HeadcountRow

    value = HeadcountExtraction(
        fiscal_year=2024,
        rows=[
            HeadcountRow(
                region="Global",
                region_type="global",
                headcount=228000,
                headcount_type="employees",
                year=2024,
                source_chunk_ids=["chunk_hc_global_001"],
            ),
            HeadcountRow(
                region="United States",
                region_type="region",
                headcount=123000,
                headcount_type="employees",
                year=2024,
                source_chunk_ids=["chunk_hc_us_001"],
            ),
        ],
        extraction_confidence=0.85,
    )
    return ExtractionResult(
        field_name="headcount_by_region",
        schema_cls_name="HeadcountExtraction",
        schema_version=1,
        validation_passed=True,
        validation_errors=[],
        value=value,
        extraction_confidence=0.85,
        chunks_used=["chunk_hc_global_001", "chunk_hc_us_001"],
        repair_count=0,
    )


def _make_geo_revenue_missing_provenance() -> "ExtractionResult":
    """Row with empty source_chunk_ids — provenance hard fail."""
    from evidence_enrichment.core.extraction.models import ExtractionResult
    from evidence_enrichment.core.extraction.schemas import (
        GeographicRevenueExtraction,
        GeographicRevenueRow,
        MoneyAmount,
    )
    from pydantic import ValidationError

    try:
        GeographicRevenueExtraction(
            fiscal_year=2024,
            currency="USD",
            rows=[
                GeographicRevenueRow(
                    region="US",
                    region_type="region",
                    amount=MoneyAmount(value=Decimal("100000"), currency="USD"),
                    source_chunk_ids=[],  # empty — provenance missing
                ),
            ],
            extraction_confidence=0.60,
        )
    except ValidationError as exc:
        errors = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        # Minimal ExtractionResult — value is raw dict (uncoercible provenance row)
        raw_value = {
            "fiscal_year": 2024,
            "currency": "USD",
            "rows": [{"region": "US", "region_type": "region",
                       "amount": {"value": 100000, "currency": "USD"}, "source_chunk_ids": []}],
            "extraction_confidence": 0.60,
            "__schema_cls__": "geographic_revenue",
        }
        return ExtractionResult(
            field_name="geographic_revenue",
            schema_cls_name="GeographicRevenueExtraction",
            schema_version=1,
            validation_passed=False,
            validation_errors=errors,
            value=raw_value,  # type: ignore[arg-type]
            extraction_confidence=0.60,
            chunks_used=[],
            repair_count=2,
        )
    return _make_geo_revenue_valid()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

ACCEPTANCE_CASES: list[dict[str, Any]] = [
    {
        "id": "geo_revenue_msft_fy24_valid",
        "description": "Microsoft FY2024 geographic revenue — valid row sums and provenance.",
        "field_name": "geographic_revenue",
        "fixture": _make_geo_revenue_valid,
        "expect_validation_passed": True,
        "expect_confidence_ge": 0.80,
    },
    {
        "id": "geo_revenue_sum_mismatch",
        "description": "Row sum diverges from total by >2% — gate must emit a hard-fail tag.",
        "field_name": "geographic_revenue",
        "fixture": _make_geo_revenue_sum_mismatch,
        "expect_validation_passed": False,
        "expect_gate_tag": "row_sum_divergence",
    },
    {
        "id": "scope1_msft_fy24_valid",
        "description": "Scope 1 GHG emissions disclosure — valid scope + unit.",
        "field_name": "scope_1_emissions",
        "fixture": _make_scope1_valid,
        "expect_validation_passed": True,
        "expect_confidence_ge": 0.80,
    },
    {
        "id": "headcount_valid",
        "description": "Headcount breakdown — global total consistent with regional rows.",
        "field_name": "headcount_by_region",
        "fixture": _make_headcount_valid,
        "expect_validation_passed": True,
        "expect_confidence_ge": 0.80,
    },
    {
        "id": "geo_revenue_missing_provenance",
        "description": "Row with empty source_chunk_ids — gate must emit missing_provenance tag.",
        "field_name": "geographic_revenue",
        "fixture": _make_geo_revenue_missing_provenance,
        "expect_validation_passed": False,
        "expect_gate_tag": "missing_provenance",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    from evidence_enrichment.core.quality.gates import SchemaValidationGate

    er = case["fixture"]()
    gate = SchemaValidationGate()
    gate_result = gate.check(er)

    passed = True
    reasons: list[str] = []

    if "expect_validation_passed" in case:
        if er.validation_passed != case["expect_validation_passed"]:
            passed = False
            reasons.append(
                f"validation_passed: expected {case['expect_validation_passed']}, "
                f"got {er.validation_passed}"
            )

    if "expect_confidence_ge" in case:
        if gate_result.confidence_after < case["expect_confidence_ge"]:
            passed = False
            reasons.append(
                f"confidence_after {gate_result.confidence_after:.3f} < "
                f"{case['expect_confidence_ge']}"
            )

    if "expect_gate_tag" in case:
        all_tags = gate_result.hard_errors + gate_result.soft_errors
        expected_tag = case["expect_gate_tag"]
        if expected_tag not in all_tags:
            passed = False
            reasons.append(
                f"gate_tag: expected '{expected_tag}' in errors, "
                f"got hard={gate_result.hard_errors} soft={gate_result.soft_errors}"
            )

    return {
        "id": case["id"],
        "description": case["description"],
        "field_name": case["field_name"],
        "validation_passed": er.validation_passed,
        "confidence_after": gate_result.confidence_after,
        "hard_errors": gate_result.hard_errors,
        "soft_errors": gate_result.soft_errors,
        "gate_passed": gate_result.passed,
        "repair_count": er.repair_count,
        "pass": passed,
        "failure_reasons": reasons,
    }


def run_acceptance_harness(output_path: Path) -> dict[str, Any]:
    results = [_run_case(c) for c in ACCEPTANCE_CASES]

    total = len(results)
    passed_count = sum(1 for r in results if r["pass"])

    # schema_valid_rate: only count cases that are *expected* to produce a valid extraction.
    # Negative cases (deliberately invalid fixtures) are excluded — they test gate behavior,
    # not extraction success rate.
    expected_valid_cases = [c for c in ACCEPTANCE_CASES if c.get("expect_validation_passed") is True]
    expected_valid_ids = {c["id"] for c in expected_valid_cases}
    valid_results = [r for r in results if r["id"] in expected_valid_ids]
    actual_valid_count = sum(1 for r in valid_results if r["validation_passed"])
    schema_valid_rate = actual_valid_count / len(valid_results) if valid_results else 0.0

    confidences = [r["confidence_after"] for r in valid_results if r["validation_passed"]]
    gate_confidence_mean = sum(confidences) / len(confidences) if confidences else 0.0

    # Hard fail check: any case that was expected to fail but arrived with high confidence
    # indicates the gate penalty is not being applied.
    expected_fail_ids = {
        c["id"] for c in ACCEPTANCE_CASES if c.get("expect_validation_passed") is False
    }
    fail_results = [r for r in results if r["id"] in expected_fail_ids]

    criteria: dict[str, dict[str, Any]] = {
        "schema_valid_rate_ge_0.80": {
            "threshold": 0.80,
            "actual": round(schema_valid_rate, 4),
            "passed": schema_valid_rate >= 0.80,
            "note": "Excludes deliberately-invalid negative test cases.",
        },
        "gate_confidence_mean_ge_0.75": {
            "threshold": 0.75,
            "actual": round(gate_confidence_mean, 4),
            "passed": gate_confidence_mean >= 0.75,
        },
        "no_hard_fail_passes_gate": {
            "threshold": "gate_passed=False for all expected-fail cases",
            "actual": sum(1 for r in fail_results if not r["gate_passed"]),
            "passed": all(not r["gate_passed"] for r in fail_results),
        },
        "all_acceptance_cases_pass": {
            "threshold": total,
            "actual": passed_count,
            "passed": passed_count == total,
        },
    }

    overall_pass = all(c["passed"] for c in criteria.values())

    report = {
        "harness": {
            "name": "stage_d_acceptance",
            "case_count": total,
            "overall_pass": overall_pass,
        },
        "acceptance_criteria": criteria,
        "summary": {
            "total_cases": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "expected_valid_cases": len(valid_results),
            "schema_valid_rate": round(schema_valid_rate, 4),
            "gate_confidence_mean": round(gate_confidence_mean, 4),
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage D acceptance harness for schema extraction and gate behavior."
    )
    p.add_argument(
        "--output",
        default="evals/output/latest_acceptance_report.json",
        help="Path to write the acceptance report JSON.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case output; only print summary.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _check_deps()

    report = run_acceptance_harness(Path(args.output))
    summary = report["summary"]
    criteria = report["acceptance_criteria"]

    if not args.quiet:
        print(f"\nStage D Acceptance Harness")
        print(f"{'─' * 50}")
        for result in report["results"]:
            status = "PASS" if result["pass"] else "FAIL"
            print(f"  [{status}] {result['id']}")
            if result["failure_reasons"]:
                for reason in result["failure_reasons"]:
                    print(f"         {reason}")

        print(f"\nAcceptance criteria:")
        for name, c in criteria.items():
            status = "PASS" if c["passed"] else "FAIL"
            print(f"  [{status}] {name}: {c['actual']}")

        print(f"\nSummary: {summary['passed']}/{summary['total_cases']} cases passed")
        print(f"  schema_valid_rate:      {summary['schema_valid_rate']:.2%}")
        print(f"  gate_confidence_mean:   {summary['gate_confidence_mean']:.3f}")
        print(f"\nReport written to: {args.output}")

    overall_pass = report["harness"]["overall_pass"]
    if not overall_pass:
        print("\nACCEPTANCE FAILED — see report for details.", file=sys.stderr)
        return 1

    print("\nACCEPTANCE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
