# Goals and Features — Evidence Enrichment Engine

## Goal

Demonstrate disciplined context engineering, replay-backed evaluation, and local agent observability for a structured field enrichment workflow.

The repo shows how to resolve a structured fact from public evidence through a staged pipeline — rather than prompting an LLM directly from raw entity inputs. Every design decision (context scoping, stage boundaries, replay safety, confidence gating) is made explicit and inspectable.

## What This Solves

Naive LLM enrichment is brittle: prompts are opaque, results are unrepeatable, and failures are hard to diagnose. This repo shows the architectural layer that makes enrichment workflows reliable: structured context packs, stage-level tracing, replay-backed evaluation, and explicit review gates that block low-confidence outputs from passing through.

---

## Features

### 8-Stage Pipeline

Each stage has a defined input contract, output contract, and trace span:

| Stage | Role |
|---|---|
| `query_plan` | Generates search queries scoped to the target field |
| `search` | Executes queries via configured providers |
| `fetch` | Retrieves and normalises page content |
| `parse` | Extracts field-relevant evidence fragments |
| `evidence_assessment` | Scores source reliability and evidence agreement |
| `analysis` | Reasons over assessed evidence |
| `synthesis` | Produces a structured field value with confidence |
| `review_gate` | Blocks low-confidence outputs; flags for human review |

### Structured Context Pack

The `context/` directory defines the full context bundle used by the workflow:

| File | Purpose |
|---|---|
| `system_role.md` | Agent role definition |
| `task_spec.md` | Field-level task specification |
| `data_contracts.md` | Input/output schemas |
| `failure_modes.md` | Known failure patterns and handling rules |
| `decision_rubric.md` | Confidence band definitions and decision criteria |
| `context_manifest.yaml` | Load order, per-stage scoping, priority labels, character budgets |

Every run resolves the manifest into a `resolved_context.json` artifact — the exact context available to each stage is preserved with the run outputs.

### Replay-First Execution

The default run mode is replay: pre-recorded search and fetch responses substitute for live providers. The full pipeline runs, traces, and evaluates with no external credentials or network access required.

Live mode (`--mode live`) and auto-fallback mode (`--mode auto`) are available when provider keys are present.

### Eval Harness

`evals/cases.yaml` defines replay-backed cases covering different evidence conditions for the same entity and field:

- Baseline replay (weak secondary evidence) → `needs_review`, confidence 0.75
- Assessed replay (agreeing primary sources) → `auto_approve`, confidence 0.97

`evidence-enrich eval` runs all cases and writes a structured report. Current result: **6/6 pass**.

### Local Trace Artifacts

Every run writes trace artifacts to `examples/output/traces/<trace_id>/`:

| File | Content |
|---|---|
| `spans.jsonl` | One JSON line per stage span |
| `trace_summary.json` | Pipeline-level summary with decision and confidence |
| `trace_timeline.md` | Human-readable stage timeline |
| `openinference_trace.json` | OpenInference-compatible compatibility export |
| `resolved_context.json` | Context bundle snapshot for the run |

Each span records: `trace_id`, `stage`, `provider`, `mode`, `latency_ms`, `input_count`, `output_count`, and `decision` where applicable.

### LangSmith Integration (Opt-In)

Set `LANGSMITH_TRACING=true` to enable LangSmith tracing alongside local artifacts. Stage-level spans are recorded without changing pipeline logic. See `docs/observability.md` for setup and dashboard flow.

### Docker + CI

Full Docker and Docker Compose setup. GitHub Actions CI runs the test suite and eval harness on every push.

---

## What This Repo Does Not Cover

- Bulk enrichment across entity universes (this demo is intentionally scoped to one entity and one field)
- Document acquisition or PDF retrieval (see `document-acquisition-workbench`)
- Lakehouse storage of enriched outputs (see `entity-data-lakehouse`)
