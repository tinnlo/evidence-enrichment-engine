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
- `retrieval_indexing` *(optional)*
- `retrieval_query` *(optional)*
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

## Retrieval-Augmented Analysis (Optional)

An optional RAG layer is available. When enabled (`retrieval.mode: local` in `evidence_enrichment.yaml`), the pipeline:

1. Uses `parse_with_structure` to extract block-level HTML content (headings, tables, text paragraphs)
2. Chunks accepted documents with a table-aware chunker (tables kept atomic up to `max_table_size`; larger tables split on whitespace boundaries; character-based overlap for text)
3. Embeds chunks via OpenAI `text-embedding-3-small` and stores them in a local Chroma vector store
4. Retrieves the top-k most relevant chunks per document using hybrid scoring (vector + keyword + table boost)
5. Passes retrieved chunks to the analysis agent instead of the default `text[:6000]` truncation

Retrieval is **document-scoped**: each query filters by `document_url` to preserve per-document claim attribution. Replay mode skips retrieval entirely — no embedding API calls, bundles unchanged.

To enable:

```yaml
# evidence_enrichment.yaml
retrieval:
  mode: "local"
```

Requires `OPENAI_API_KEY` and the `[retrieval]` dependency group:

```bash
pip install -e ".[retrieval]"
```

See [docs/retrieval.md](docs/retrieval.md) for the full architecture, chunking rationale, hybrid scoring formula, config reference, and deferred v2 items.

**Privacy note:** Chroma storage (`examples/output/chroma/`) is covered by `.gitignore` and never committed.

## Replay-First Flow

The default run mode is `auto`: the pipeline attempts to run live when provider credentials are present.  If credentials are missing **and** a replay bundle exists for the entity, it falls back to replay automatically.  If no replay bundle is found, the pipeline proceeds live and will error if credentials are also absent.  To force replay mode unconditionally (no network access, no credentials required), pass `--mode replay`.

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

## MCP Server

The engine exposes an MCP (Model Context Protocol) server so any MCP-compatible AI agent can call into the pipeline directly — no UI required.

### Install

```bash
pip install -e ".[mcp]"
```

### Run

```bash
# stdio transport (default) — for Claude Desktop and OpenCode
evidence-enrich mcp

# HTTP transport — for MCP Inspector or browser clients
evidence-enrich mcp --transport streamable-http
# MCP Inspector URL: http://localhost:8000/mcp

# Dedicated entrypoint (same as above)
evidence-enrich-mcp
evidence-enrich-mcp --transport streamable-http
```

The dedicated `evidence-enrich-mcp` entrypoint requires the `[mcp]` extra. In a base install without that extra, it exits with installation guidance instead of raising a traceback.

All tools default to `replay` mode and require no API keys.

### Connect

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "evidence-enrichment": {
      "command": "evidence-enrich-mcp"
    }
  }
}
```

**OpenCode** — add to `.opencode/config.json`:

```json
{
  "mcp": {
    "evidence-enrichment": {
      "type": "local",
      "command": ["evidence-enrich-mcp"]
    }
  }
}
```

**MCP Inspector** — run with `--transport streamable-http` and point the Inspector at `http://localhost:8000/mcp`.

### Resources (read-only)

| URI | Description |
| --- | --- |
| `evidence://bundles` | JSON list of all 6 replay bundle names and descriptions |
| `evidence://bundles/{name}` | Raw JSON of a named replay bundle |
| `evidence://results/latest` | Newest pipeline result artifact by modification time from a CLI run |

### Tools (actions)

| Tool | Description |
| --- | --- |
| `list_replay_scenarios` | List all replay bundles with descriptions |
| `run_enrichment_pipeline` | Run the full 8-stage pipeline; returns complete `PipelineRunResult` |
| `get_synthesis_result` | Run the pipeline; returns a concise `SynthesisSummary` (value, decision, confidence) |
| `get_evidence_claims` | Run the pipeline; returns all extracted `FactClaim` objects |
| `compare_scenarios` | Run two bundles concurrently and return a side-by-side `ScenarioComparison` |

### Prompt

`analyze_entity(entity_name, field_name, scenario)` — guides an agent through interpreting pipeline output, evaluating claim quality, and explaining the review decision.

### Replay Scenarios

| Bundle name | Expected decision | Notes |
| --- | --- | --- |
| `microsoft_hq_country` | `auto_approve` | Two agreeing high-authority sources |
| `microsoft_hq_country_baseline` | `needs_review` | Weak secondary evidence only |
| `microsoft_hq_country_conflict` | `needs_review` | Conflicting claims across sources |
| `microsoft_hq_country_no_support` | `auto_reject` | No accepted claims found |
| `microsoft_hq_country_invalid_iso3` | `auto_reject` | Non-ISO3 synthesis output |
| `microsoft_hq_country_low_signal` | `needs_review` | Single low-confidence claim |

---

## Quick Start with Docker

The default container path runs the replay demo, so you can keep provider keys blank unless you want live mode or LangSmith tracing.

```bash
cp .env.example .env
docker compose up
```

Run tests in the same image:

```bash
docker compose run --rm pipeline pytest tests/
```

Run the eval harness in the container:

```bash
docker compose run --rm pipeline evidence-enrich eval
```

If you want live providers instead of replay, fill in the relevant keys in `.env` and override the command:

```bash
docker compose run --rm pipeline evidence-enrich demo --mode auto
```

## Observability

Every run still writes local trace artifacts under `examples/output/traces/<trace_id>/`. If you also enable LangSmith with `LANGSMITH_TRACING=true`, the pipeline records compact stage-level traces for query planning, search, fetch, parsing, evidence assessment, analysis, synthesis, and review gating without changing the pipeline logic. See [docs/observability.md](docs/observability.md) for env setup, stage payloads, and the LangSmith dashboard flow.

**Privacy note:** When LangSmith tracing is enabled, the wrapped OpenAI and Anthropic clients also capture raw prompts and full LLM responses — which may include document text and retrieved chunks — in your LangSmith project. Avoid enabling LangSmith when processing sensitive documents.

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
ruff check .
```

The test suite also checks that the repository stays free of banned internal identifiers and company-specific references.
