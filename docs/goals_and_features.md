# Goals and Features — Evidence Enrichment Engine

## Goal

Demonstrate disciplined context engineering, replay-backed evaluation, and local agent observability for a structured field enrichment workflow.

The repo shows how to resolve a structured fact from public evidence through a staged pipeline — rather than prompting an LLM directly from raw entity inputs. Every design decision (context scoping, stage boundaries, replay safety, confidence gating) is made explicit and inspectable.

## What This Solves

Naive LLM enrichment is brittle: prompts are opaque, results are unrepeatable, and failures are hard to diagnose. This repo shows the architectural layer that makes enrichment workflows reliable: structured context packs, stage-level tracing, replay-backed evaluation, and explicit review gates that block low-confidence outputs from passing through.

---

## Features

### 8-Stage Pipeline (10-Stage with Retrieval)

Each stage has a defined input contract, output contract, and trace span:

| Stage | Role |
|---|---|
| `query_plan` | Generates search queries scoped to the target field |
| `search` | Executes queries via configured providers |
| `fetch` | Retrieves and normalises page content |
| `parse` | Extracts field-relevant evidence fragments (structured mode: block-level HTML extraction) |
| `evidence_assessment` | Scores source reliability and evidence agreement |
| `retrieval_indexing` | *(optional)* Chunks accepted documents; embeds and upserts into Chroma |
| `retrieval_query` | *(optional)* Retrieves top-k chunks per document via hybrid scoring |
| `analysis` | Reasons over assessed evidence (uses retrieved chunks when available) |
| `synthesis` | Produces a structured field value with confidence |
| `review_gate` | Blocks low-confidence outputs; flags for human review |

The two retrieval stages are active only when `retrieval.mode: local` is set in `evidence_enrichment.yaml`. When `mode: off` (default), the pipeline is unchanged.

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

The default run mode is `auto`: the pipeline attempts to run live when provider credentials are present.  If credentials are missing **and** a replay bundle exists for the entity, it falls back to replay automatically.  If no replay bundle is found, the pipeline proceeds live and will error if credentials are also absent.  To force replay mode unconditionally (no network access, no credentials required), pass `--mode replay`.

Live mode (`--mode live`) is also available when provider keys are present.

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
| `trace_summary.json` | Pipeline-level summary: `trace_id`, `total_spans`, `total_latency_ms`, `stages`, `decision`, `overall_confidence` |
| `trace_timeline.md` | Human-readable stage timeline |
| `openinference_trace.json` | OpenInference-compatible compatibility export |
| `resolved_context.json` | Context bundle snapshot for the run |

Each span records: `trace_id`, `stage`, `provider`, `mode`, `latency_ms`, `input_count`, `output_count`, and `decision` where applicable.

### Retrieval-Augmented Analysis (Optional)

An optional RAG layer can be enabled to replace the default `text[:6000]` truncation with semantically ranked evidence.

**Key design decisions:**

- **Table-aware chunking:** HTML `<table>` elements and plain-text tables (pipe/tab/Markdown-delimited) are kept as atomic chunks when they fit within `max_table_size`. Large tables that exceed the limit are split on whitespace boundaries without overlap to avoid duplicating numeric data (note: this may break row boundaries for very large tables).
- **Hybrid scoring:** `0.7 × vector_score + 0.2 × keyword_score + 0.1 × table_boost`. The keyword component anchors exact numeric matches that embeddings compress. The table boost prioritises structured data for financial/numeric queries.
- **Document-scoped retrieval:** Every query filters by `document_url`. Each document's evidence is retrieved independently, preserving per-document claim attribution through to `FactClaim.source_url`.
- **Replay safety:** Retrieval is skipped entirely in replay mode. No embedding API calls, no Chroma initialisation, bundles unchanged.

See [docs/retrieval.md](retrieval.md) for full architecture, config reference, and deferred v2 items.

### LangSmith Integration (Opt-In)

Set `LANGSMITH_TRACING=true` to enable LangSmith tracing alongside local artifacts. Stage-level spans are recorded without changing pipeline logic. See `docs/observability.md` for setup and dashboard flow.

### Docker + CI

Full Docker and Docker Compose setup. GitHub Actions CI runs the test suite and eval harness on every push.

---

## What This Repo Does Not Cover

- Bulk enrichment across entity universes (this demo is intentionally scoped to one entity and one field)
- Document acquisition or PDF retrieval (see `document-acquisition-workbench`)
- Lakehouse storage of enriched outputs (see `entity-data-lakehouse`)
