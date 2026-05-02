# HANDOVER

Repo: `evidence-enrichment-engine`

## Objective

Build on the shipped replay-first enrichment demo and make this repo the portfolio's clearest **agent safety control plane** example:

- AI FinOps for agent workflows
- budget-aware execution
- permissioned agent actions
- policy-governed live execution

This is not a generic multi-agent supervisor framework. It remains a narrow, inspectable evidence-enrichment workflow with stronger execution controls.

## Current shipped baseline

The following are already shipped and must be treated as baseline, not redesign scope:

- replay-first eval harness with passing bundled cases
- Langfuse-first observability routing with `OBSERVABILITY_BACKEND=langfuse|langsmith|dual|none`
- local trace artifacts that remain on regardless of remote backend choice
- LangGraph adaptive retrieval path as an optional retrieval mode
- guardrails between synthesis and review gate
- MCP server with replay-safe defaults
- AI FinOps layer with cost attribution, budget-aware reporting, and FinOps artifacts

This handover must extend that baseline rather than replacing it.

## Required shipped outcome

A reviewer should be able to see that the repo governs both:

- **cost** via the existing FinOps layer
- **capability** via a new execution-policy layer

Minimum shipped outcome:

1. Add a distinct `execution_policy` layer separate from FinOps config.
2. Support `execution_policy.mode=off|audit|enforce`.
3. Enforce action-level allowlisting for live-capability surfaces.
4. Emit machine-readable execution-policy artifacts per run.
5. Preserve replay-first behavior and current output contracts.
6. Update docs so the repo clearly presents safety policy and FinOps as separate controls.

## In scope

- execution-policy config surface
- structured policy evaluation and decision recording
- action-level allowlisting for at least:
  - `search`
  - `fetch`
  - `retrieval`
  - `remote_tracing`
  - `live_provider_calls`
  - `mcp_live_runs`
- blocked-run behavior that returns structured results instead of crashing
- policy visibility in run summaries, traces, or both
- MCP policy enforcement for live execution paths
- docs and tests

## Out of scope

- redesigning the replay harness
- replacing the current FinOps design
- converting the repo into a generic agent platform
- cloud IAM integrations
- enterprise policy engines
- multi-tenant authorization

## Required public framing

Use wording like:

- AI FinOps for agent workflows
- budget-aware execution
- permissioned agent actions
- policy-governed live execution
- replay-first safety evaluation

Do not describe the repo as:

- a general multi-agent supervisor framework
- a cloud sandbox product
- a LangChain-based repo unless the code actually adds LangChain

## Required interfaces, artifacts, and config surfaces

Add or expose:

- `execution_policy` config surface
- `execution_policy.mode=off|audit|enforce`
- run artifact: `execution_policy.json`
- policy decision fields in summaries and/or spans

Preserve:

- existing local trace artifact contract
- existing FinOps artifacts such as cost summaries and benchmark reports
- existing replay outputs
- existing MCP replay-safe defaults

Budget and execution policy must remain inspectable separately:

- FinOps governs cost
- execution policy governs capabilities

## Implementation guidance

### 1. Add a distinct execution policy layer

Do not overload the existing FinOps config.

Suggested structure:

- `execution_policy.enabled`
- `execution_policy.mode`
- `execution_policy.allowed_actions`
- optional live/replay policy defaults

### 2. Gate live-capability surfaces explicitly

At minimum, policy decisions must cover:

- search execution
- remote fetches
- retrieval mode escalation
- remote tracing egress
- live provider calls
- MCP-triggered live runs

Replay mode must remain the safest path and should continue to work without credentials.

### 3. Block safely, not noisily

When policy blocks an action:

- do not raise an uncaught exception
- return a normal pipeline result
- include a policy-specific gate reason
- record the block in `execution_policy.json`

### 4. Keep policy and budget separate

The repo already has cost governance.

Do not merge the concepts:

- a run can be policy-allowed but budget-breaching
- a run can be budget-safe but capability-blocked

Both outcomes should be visible in artifacts.

### 5. Integrate with traces and docs

The reviewer should be able to follow:

- what was attempted
- what was allowed
- what was blocked
- whether the block was due to policy or budget

Update docs so the public story is explicit and technically honest.

## Likely files to modify

- `README.md`
- `docs/observability.md`
- `docs/goals_and_features.md`
- `evidence_enrichment/config/...`
- `evidence_enrichment/observability/...`
- `evidence_enrichment/core/...`
- `evidence_enrichment/pipeline/...`
- `evals/...`
- `tests/...`

## Verification commands

Run the repo's real test and demo paths after implementation. At minimum:

```bash
pytest tests/
python evals/run_eval.py
evidence-enrich run --mode replay
```

Also verify policy-specific paths, for example:

```bash
OBSERVABILITY_BACKEND=langfuse evidence-enrich run --mode replay
OBSERVABILITY_BACKEND=langfuse evidence-enrich run --mode auto
```

Add implementation-specific checks that prove:

- `audit` records violations without blocking
- `enforce` blocks disallowed live actions
- remote tracing can be disabled by policy even when credentials exist
- MCP live execution respects policy restrictions
- existing FinOps tests still pass

## Guardrails

- Do not break replay-first local usability.
- Do not require credentials for the default demo path.
- Do not remove or weaken current FinOps artifacts.
- Do not silently downgrade behavior without recording it in artifacts.
- Do not claim LangChain unless it is actually present in code and docs.
- Do not turn policy enforcement into an unstructured exception path.

## Acceptance standard

Accept the implementation only if all of the following are true:

1. Replay path still passes unchanged.
2. `execution_policy.mode=off|audit|enforce` is implemented and tested.
3. Live actions are permissioned explicitly and auditable.
4. Policy blocks return structured outcomes with clear gate reasons.
5. Remote tracing can be disabled by policy even when credentials are configured.
6. MCP live execution follows the same policy model.
7. Existing FinOps reporting still works and remains separate from policy reporting.
8. README and supporting docs describe the shipped behavior accurately.
