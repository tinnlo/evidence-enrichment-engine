# Evidence Enrichment Engine

`evidence-enrichment-engine` is a narrow public demo for context engineering, replay-backed evaluation, and local agent observability around one evidence-backed enrichment workflow.

This public demo rebuilds a production architectural pattern in a public-safe form. It preserves system design, module boundaries, and execution flow while removing proprietary business logic and internal data.

## Why This Repo Exists

- What this demo shows: how to structure agent context, inspect agent workflow stages, and evaluate replay-backed behavior against expected outcomes.
- Which production capability it mirrors: a search-grounded enrichment flow that resolves one structured field from public evidence instead of prompting directly from raw entity inputs.
- What was intentionally generalized or removed: proprietary business rules, internal datasets, internal URLs, organization-specific runbooks, and production infrastructure assumptions.
- Why it exists in the portfolio: to show disciplined workflow design and instrumentation without pretending this repo is a broad agent platform.

The repo stays intentionally narrow:

- one entity anchor: `Microsoft Corporation`
- one field: `hq_country`
- one coordinator
- one replay-safe pipeline

## Context Engineering

The `context/` directory defines the context pack used by the workflow:

- `system_role.md`
- `task_spec.md`
- `data_contracts.md`
- `failure_modes.md`
- `decision_rubric.md`
- `context_manifest.yaml`

`context_manifest.yaml` controls:

- load order
- per-stage context usage
- priority labels
- max character budgets

Each run resolves the manifest into a `resolved_context.json` artifact so the workflow shows exactly what context bundle was available to each stage.

## Eval Harness

The repo includes a replay eval harness under `evals/`:

- `cases.yaml` defines replay-backed cases for the same entity under different evidence conditions
- `run_eval.py` runs the harness
- `report_schema.json` documents the report shape
- `output/latest_report.json` stores the latest report

The eval surface is intentionally small. It checks whether the workflow returns the expected value, expected decision, and minimum confidence threshold for a fixed task.

## Observability

The coordinator emits local trace artifacts for these spans:

- `query_plan`
- `search`
- `fetch`
- `parse`
- `evidence_assessment`
- `analysis`
- `synthesis`
- `review_gate`

Every span records:

- `trace_id`
- `stage`
- `provider`
- `mode`
- `latency_ms`
- `entity_id`
- `field`
- `input_count`
- `output_count`
- `decision` where relevant

Artifacts are written to `examples/output/traces/<trace_id>/`:

- `spans.jsonl`
- `trace_summary.json`
- `trace_timeline.md`
- `openinference_trace.json`
- `resolved_context.json`

The OpenInference-style JSON is a compatibility export, not a claim of deployed production telemetry infrastructure.

## Architecture

```text
context pack
    ->
coordinator
    ->
query_plan -> search -> fetch -> parse -> evidence_assessment -> analysis -> synthesis -> review_gate
    ->
result artifact + resolved_context.json + trace artifacts + eval report
```

## Replay-First Flow

The public-safe default is replay mode. The coordinator can also run in `auto` or `live` mode when credentials are present, but the repository is designed to remain fully runnable without external services.

```bash
pip install -e ".[dev]"
```

Run the core demo:

```bash
evidence-enrich demo --mode replay
```

Resolve the context pack without running the pipeline:

```bash
evidence-enrich context-pack
```

Run the trace-focused demo:

```bash
evidence-enrich trace-demo --mode replay
```

Run the comparison artifact:

```bash
evidence-enrich compare
```

Run the eval harness:

```bash
evidence-enrich eval
```

Run directly on the entity fixture:

```bash
evidence-enrich run --entity examples/microsoft.json --field hq_country --mode replay
```

## Results Snapshot

| Path | Expected Outcome | Decision | Confidence |
| --- | --- | --- | --- |
| Baseline replay | `USA` from weak secondary evidence | `needs_review` | `0.75` |
| Assessed replay | `USA` from agreeing primary sources | `auto_approve` | `0.97` |
| Eval harness | replay cases pass against expectations | `6/6 pass` | case-dependent |

## Repo Layout

```text
context/
evals/
examples/replay/
examples/output/
evidence_enrichment/
tests/
```

## Honesty Note

This is a public-safe demo of patterns used to structure and inspect agentic workflows. It is deliberately small, local, and replay-driven. It is not presented as a full production observability stack or a general-purpose agent framework.

## Development

```bash
pytest
python evals/run_eval.py
```

The test suite also checks that the repository stays free of banned internal identifiers and company-specific references.
