# IMPLEMENTATION PLAN — Execution Policy Layer

All 5 stages complete.

## Stage 1: Config surface + domain models
**Status**: Complete

- `evidence_enrichment/execution_policy/models.py`: `PolicyMode`, `ActionType`, `PolicyDecision`, `ExecutionPolicyReport`
- `evidence_enrichment/execution_policy/__init__.py`: package exports
- `evidence_enrichment/config/settings.py`: `ExecutionPolicySettings`, `Settings.execution_policy`
- `evidence_enrichment.yaml`: `execution_policy:` block with replay-safe defaults
- `evidence_enrichment/core/models/contracts.py`: `execution_policy_report` field on `PipelineRunResult`

## Stage 2: Policy engine
**Status**: Complete

- `evidence_enrichment/execution_policy/engine.py`: `ExecutionPolicyEngine` with `check_action()`, `build_report()`, `reset()`
- off / audit / enforce logic; mirrors `BudgetPolicyEngine` pattern

## Stage 3: Pipeline integration
**Status**: Complete

- `evidence_enrichment/pipeline/coordinator.py`:
  - `_PolicyRunContext` + `_prc_var` (mirrors `_FinOpsRunContext`)
  - `LIVE_PROVIDER_CALLS` gate at top of `_run_live()`
  - `SEARCH`, `FETCH`, `RETRIEVAL` gates at each live surface
  - `execution_policy_report` attached to `PipelineRunResult` after every run
  - `execution_policy_data` passed to `tracer.write()`

## Stage 4: Observability + MCP integration
**Status**: Complete

- `evidence_enrichment/observability/runtime.py`: `activate_remote_tracing_with_policy_check()` — policy-aware wrapper for remote backend activation
- `evidence_enrichment/observability/tracer.py`: `execution_policy.json` artifact written per run; `TraceArtifacts.execution_policy_path` and `as_refs()` updated
- `evidence_enrichment/mcp_server.py`: `MCP_LIVE_RUNS` gate in `_run_pipeline()` before coordinator is spun up

## Stage 5: Tests + documentation
**Status**: Complete

- `tests/test_execution_policy.py`: 29 tests covering models, engine (all modes), config, tracer artifact, contract field, and replay pipeline smoke tests
- `docs/goals_and_features.md`: Execution Policy row added to feature table
- `docs/observability.md`: Full "Execution Policy" section added (modes, governed actions, replay safety, artifact schema, YAML config)
- This file: replaced FinOps plan with completed Execution Policy plan

## Locked Decisions

- `RETRIEVAL` gate: enforce mode disables retrieval silently (pipeline continues without context) — retrieval is optional context, not a hard dependency
- `REMOTE_TRACING` gate: uses `activate_remote_tracing_with_policy_check()` which returns `None` on block; callers must handle `None` token
- `off` mode: `execution_policy.json` still written with empty `decisions` list — "policy off" is distinguishable from "not configured"
- `audit` mode: violations are recorded but never block execution
- Block results use `gate_reason="policy_blocked:<action>"` mirroring `budget_blocked:` pattern
- Replay path is unconditionally exempt — no live gates are reached
