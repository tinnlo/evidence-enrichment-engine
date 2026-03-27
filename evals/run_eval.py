"""CLI entrypoint for replay evals."""

from __future__ import annotations

from pathlib import Path

from evidence_enrichment.evals.harness import run_eval_harness


def main() -> None:
    run_eval_harness(
        cases_path=Path("evals/cases.yaml"),
        output_path=Path("evals/output/latest_report.json"),
    )


if __name__ == "__main__":
    main()
