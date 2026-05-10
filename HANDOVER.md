# HANDOVER: Redis Cache-Aside Implementation

**Project:** `evidence-enrichment-engine`  
**Priority:** HIGH  
**Purpose:** Track A for Deriv interview prep  
**Target outcome:** one clean, bounded Redis implementation that is easy to demo and defend
**Estimated time:** 8 hours + 2 hour buffer  
**Target completion:** Sunday, May 11, 2026

---

## Why This Task Exists

This repo is already the strongest public proof point for:
- replay-first evals
- Langfuse-first observability
- context engineering
- guardrails and review gating
- AI FinOps style trace visibility

What it does **not** currently show clearly enough is:
- Redis usage in a live agent workflow
- explicit freshness/staleness handling
- cache behaviour that is visible in traces
- a concrete explanation of why Redis fits a bounded hot path better than PostgreSQL

The goal is to close that gap quickly, without turning the repo into a general coordination system.

---

## Scope: What To Build

Implement a **bounded Redis cache-aside layer** for two stages only:

1. **Fetch-stage caching**
   - Cache fetched document results
   - TTL: 24 hours
   - **CRITICAL:** Key MUST include execution mode (`live`/`replay`/`auto`) to prevent replay/live cross-contamination
   - Key format: `fetch:v1:{mode}:{url_hash}`

2. **Evidence-assessment caching**
   - Cache evidence assessment outputs
   - TTL: 7 days
   - **CRITICAL:** Key MUST include execution mode to isolate replay from live
   - Key format: `assess:v1:{mode}:{content_hash}:{company_hash}`

3. **Connection pooling**
   - Use a pooled Redis client path
   - Keep the configuration small and explainable

4. **Staleness metadata**
   - Store fetched-at / age metadata with cached values
   - Surface `hit`, `miss`, and `stale` states explicitly

5. **Trace-visible cache behaviour**
   - Add cache metadata into the existing trace or span outputs
   - The interview story depends on being able to show observable behaviour, not just say "I added a cache"

6. **Graceful fallback**
   - If Redis is absent or disabled, current replay/demo behaviour must continue to work

7. **Replay mode isolation (CRITICAL)**
   - Replay mode MUST bypass cache entirely OR use mode-isolated cache keys
   - Replay runs must NEVER read live cache entries (and vice versa)
   - This preserves the repo's deterministic replay-first contract

8. **Execution policy integration (CRITICAL)**
   - Cache reads/writes only occur when `ActionType.FETCH` is permitted
   - Cache respects existing `execution_policy` layer
   - Policy violations must be tested in `off`, `audit`, and `enforce` modes

---

## Why This Scope Is Intentionally Narrow

This task is not trying to make the repo look like a miniature distributed systems platform.

The interview value comes from being able to explain, in detail:
- why cache-aside is the right pattern here
- why Redis fits bounded hot-path reuse better than PostgreSQL
- how freshness becomes visible rather than implicit
- how traces make cache behaviour auditable

If a feature does not strengthen one of those answers directly, it is out of scope.

---

## Explicit Non-Goals

Do **not** implement:
- a Redis queue
- a worker coordination system
- broad runtime state management
- write-through or write-back persistence
- a large architectural rewrite

This is intentionally **cache-aside only**.

---

## Current Repo Touchpoints

These are the most likely places to inspect before editing:
- `docker-compose.yml`
- `README.md`
- `tests/test_fetcher.py`
- `tests/test_pipeline.py`
- `tests/test_observability.py`
- `tests/test_finops.py`
- `tests/test_execution_policy.py`
- fetch / assessment / observability code under `evidence_enrichment/` package:
  - `evidence_enrichment/core/fetch/fetcher.py`
  - `evidence_enrichment/core/evidence/assessor.py`
  - `evidence_enrichment/pipeline/coordinator.py`
  - `evidence_enrichment/observability/tracer.py`
  - `evidence_enrichment/config/settings.py`

Do not assume the implementation must create a brand-new architecture module if a smaller integration into the current package shape is cleaner.

---

## Implementation Shape

### Docker / config

- Add a local Redis service to `docker-compose.yml`
- Add a small config surface for Redis connection settings and TTLs
- Keep Redis optional for the base replay/demo path

### Code

- Add or update a cache module responsible for:
  - key generation
  - pooled Redis access
  - serialization/deserialization
  - age/staleness metadata
  - graceful fallback on Redis errors

- Integrate cache-aside behaviour into:
  - fetch path
  - evidence-assessment path

- Keep the review gate and overall workflow unchanged unless integration requires a minimal change

### Cache design expectations

- Use **cache-aside**, not write-through or write-back
- Keep keys deterministic and easy to reason about
- Include age metadata with cached payloads
- Handle Redis read/write errors without breaking the main pipeline
- Make stale handling explicit instead of silently reusing old data

### Suggested shape

- one small cache helper or module for Redis access
- one place for key-generation helpers
- lightweight integration points in fetch and assessment stages
- trace metadata added close to the stage that owns the cache interaction

### Observability

Each cached stage should expose enough metadata to answer:
- was this a hit, miss, or stale result?
- how old was the cached value?
- did this avoid a network fetch or expensive reasoning step?

If there is a cost/latency summary surface already, prefer enriching that rather than inventing a parallel reporting path.

---

## Step-by-Step Delivery Plan

### Step 1: Inspect current fetch, assessment, and observability hooks

Goal:
- identify the smallest insertion points for cache-aside behaviour
- confirm where stage metadata is currently emitted

Expected outcome:
- clear edit list for code, tests, and README

### Step 2: Add Redis service and config surface

Goal:
- make Redis runnable locally
- keep it optional

Expected outcome:
- local Redis path exists
- base replay flow still works without Redis enabled

### Step 3: Implement fetch-stage cache-aside

Goal:
- repeated fetches can reuse cached outputs within the freshness window

Expected outcome:
- miss path fetches and caches
- hit path reuses cached value
- stale path is visible and refreshes the entry

### Step 4: Implement evidence-assessment cache-aside

Goal:
- repeated reasoning on unchanged evidence can be reused selectively

Expected outcome:
- miss, hit, and stale behaviour mirrors fetch-stage logic

### Step 5: Add trace-visible cache metadata

Goal:
- the demo can show observable behaviour, not only internal behaviour

Expected outcome:
- `hit`, `miss`, `stale`, and age/freshness metadata appear in stage outputs or traces

### Step 6: Add tests

Minimum coverage:
- Redis-enabled miss path
- Redis-enabled hit path
- stale path
- no-Redis fallback path
- **replay/live isolation tests** (verify replay never reads live cache)
- **execution policy tests** (off/audit/enforce modes)
- stage metadata assertions where practical

### Step 7: Update README

README changes should explain:
- what Redis adds
- what it does not add
- how to run it locally
- what demo sequence to use in the interview prep flow

---

## Success Criteria

- [ ] Redis runs locally with the repo
- [ ] Fetch-stage cache hit/miss path works
- [ ] Evidence-assessment cache hit/miss path works
- [ ] Stale cache entries are explicitly marked stale
- [ ] Cache metadata appears in trace-visible outputs
- [ ] Replay/demo mode still works when Redis is disabled or unavailable
- [ ] README explains what Redis adds and what it does not add
- [ ] Demo path is easy to show: miss -> hit -> stale
- [ ] The design can be defended as **cache-aside**, not write-through or write-back
- [ ] The answer to "why Redis here instead of PostgreSQL?" is obvious from the implementation

---

## Likely File Areas to Change

Use the live tree to choose exact files, but expect changes in some subset of:
- `docker-compose.yml`
- `README.md`
- fetch-related code under `evidence_enrichment/`
- evidence-assessment or pipeline code under `evidence_enrichment/`
- observability-related code under `evidence_enrichment/`
- `tests/test_fetcher.py`
- `tests/test_pipeline.py`
- `tests/test_observability.py`
- `tests/test_finops.py`

---

## Interview Translation

When this task is complete, the repo should support this exact class of explanation:

> I took my strongest public evals and observability repo and added one bounded Redis pattern I was missing in recent demos: cache-aside for fetch and evidence-assessment stages, with explicit freshness metadata and trace-visible cache behaviour. I did not try to turn it into a queueing system. The point was to show that I can add hot-path reuse, staleness handling, and connection pooling cleanly in a workflow that already had guardrails and replay evals.

---

## Verification Expectations

At minimum, verify:
- miss path
- hit path
- stale path
- no-Redis fallback path

Useful local verification commands will likely include some subset of:
- targeted pytest for fetch / pipeline / observability tests
- the repo's existing broader pytest command
- a local run that produces trace artifacts you can inspect manually

The handoff is not complete unless the demo path is both working and simple to explain.

---

## Demo Expectations

The desired interview-practice demo is:

1. first run -> cache miss
2. immediate second run -> cache hit
3. forced or simulated aged entry -> stale path

The demo does not need a UI. It only needs to be:
- real
- reproducible
- visible in outputs or traces
- explainable in under 2 minutes

---

## Delivery Notes for the Implementer

- Preserve the repo's current identity: evidence-backed enrichment with evals and observability.
- Prefer the smallest implementation that proves the point clearly.
- If a design choice increases explanation cost without increasing interview value, do not take it.
- If you need to choose between broader feature scope and stronger trace visibility, choose trace visibility.
- Do not expand into queueing, orchestration, or generic Redis platform patterns.
