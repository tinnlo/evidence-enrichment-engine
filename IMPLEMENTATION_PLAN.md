# IMPLEMENTATION PLAN — AI FinOps Layer

## Locked Decisions

- Eval output: sidecar `evals/output/latest_finops_report.json`, keep `latest_report.json` stable.
- `budget_mode=strict`: downgrade first, then return structured budget-rejected result (no exception).
- Cost scope: LLM calls + embedding calls (analysis, synthesis, retrieval_indexing, retrieval_query).
- Usage source: provider-reported when available, deterministic estimate otherwise.
- Same-provider tiering: model downgrades stay within the selected provider family.
- Architecture: small `evidence_enrichment/finops/` package, not sprinkled logic.

## Stage 1: FinOps Foundation

**Goal**: Introduce durable FinOps contracts, config, and defaults without changing runtime behavior.

**Success Criteria**: FinOps config loads with safe defaults. Existing pipeline and eval code unchanged. Trace contracts remain backward-compatible.

**Tests**: Settings/config tests for FinOps defaults. Model serialization tests for FinOps fields.

**Status**: Complete

## Stage 2: Cost Capture And Aggregation

**Goal**: Collect stage-level and run-level AI cost data deterministically.

**Success Criteria**: Replay and live runs both emit consistent FinOps summaries. All four model-backed stages represented. Provider usage preferred but never required.

**Tests**: Token estimation and pricing unit tests. Provider normalization tests. Retrieval embedder cost tests.

**Status**: Complete

## Stage 3: Budget Policy And Same-Provider Tiered Routing

**Goal**: Make FinOps operational by controlling execution, not just observing it.

**Success Criteria**: `warn` never changes execution. `strict` applies downgrade before blocking. Budget-blocked runs produce normal artifacts. No provider switching.

**Tests**: Policy projection, downgrade, and block tests. Pipeline mode tests. Structured result tests.

**Status**: Complete

## Stage 4: Artifacts, Eval Sidecar, And CLI Surfacing

**Goal**: Make cost and budget outcomes inspectable locally and in eval output.

**Success Criteria**: Trace directories show cost/route/budget. Eval harness emits dedicated FinOps artifact. Local-only replay generates FinOps artifacts without credentials.

**Tests**: Artifact ref assertions. FinOps report schema conformance. CLI output checks.

**Status**: Complete

## Stage 5: Docs, Hardening, And End-To-End Verification

**Goal**: Present the repo as a credible AI FinOps production prototype.

**Success Criteria**: Docs tell a coherent quality/latency/cost governance story. All verification commands pass.

**Tests**: `pytest tests/`, `evidence-enrich eval`, `evidence-enrich demo --mode replay`, `ruff check .`

**Status**: Complete
