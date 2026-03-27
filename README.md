# Evidence Enrichment Engine

`evidence-enrichment-engine` is a public demo repository for an evidence-first enrichment pipeline.

The repo shows a single explainable path from `fact search` to `text parsing` to `analysis agent` to `synthesis agent`, with provenance, computed confidence, and explicit review gates.

## What It Demonstrates

- Search-first enrichment instead of prompting an LLM directly from raw entity metadata
- Provider routing and fallback between live providers and replay bundles
- Text parsing and evidence assessment before analysis
- Per-document fact extraction with supporting excerpts
- Final synthesis with computed confidence and review gating

## V1 Scope

The first public flow enriches one field:

- `hq_country_iso3`

The main demo entity is `Microsoft Corporation`, and the repo also includes a comparison example that contrasts:

- a baseline path with weak snippet-only evidence
- an assessed path with parsed primary-source evidence

## Architecture

The coordinator runs fixed stages:

1. Build a `SearchQueryPlan`
2. Search for evidence candidates
3. Fetch and parse top documents
4. Score entity match, authority, and freshness
5. Run the analysis agent on accepted documents
6. Run the synthesis agent over the extracted claims
7. Compute confidence and apply the review gate

The replay path uses the same stage contracts, but reads recorded artifacts from `examples/replay/` so the pipeline is still runnable without external credentials.

## Quick Start

Install the package:

```bash
pip install -e ".[dev]"
```

Run the replay-backed demo:

```bash
evidence-enrich demo
```

Run the comparison example:

```bash
evidence-enrich compare
```

Run the field pipeline directly against a JSON entity:

```bash
evidence-enrich run --entity examples/microsoft.json --field hq_country --mode replay
```

## Live Providers

Live mode is optional. Configure credentials in `.env` based on `.env.example`.

Supported live adapters:

- Search: `serper`, `tavily`
- Agents: `openai`, `anthropic`

If you run `--mode auto`, the coordinator tries live providers first and falls back to replay when a matching replay bundle exists.

## Output Artifacts

The CLI writes JSON artifacts under `examples/output/`. Each result includes:

- the search plan
- ranked search results
- parsed documents
- analysis reports
- synthesis output
- computed confidence
- final review decision

## Development

Run tests:

```bash
pytest
```

The test suite also verifies that the public repo does not contain banned internal identifiers or proprietary references.

