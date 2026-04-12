# Observability

This repo keeps its local trace artifacts and can optionally emit LangSmith and/or Langfuse traces for the same workflow stages.

## What Is Traced

The pipeline always writes local artifacts under `examples/output/traces/<trace_id>/`:

- `spans.jsonl`
- `trace_summary.json`
- `trace_timeline.md`
- `openinference_trace.json`
- `resolved_context.json`

When LangSmith or Langfuse tracing is enabled, the coordinator also emits compact stage traces for:

- `query_plan`
- `search`
- `fetch`
- `parse`
- `evidence_assessment`
- `retrieval_indexing` *(when `retrieval.mode` is `"local"` or `"agent"`)*
- `retrieval_query` *(when `retrieval.mode` is `"local"` or `"agent"`)*
- `analysis`
- `synthesis`
- `review_gate`

Both backends capture summarized inputs and outputs rather than raw full-document payloads:

- `query_plan`: entity id, field, company name, resolved context entry ids, primary query, query variant count
- `search`: query text, provider order, result count, top URLs
- `fetch`: requested URLs, fetched count, success count
- `parse`: document URLs, titles, and text-length summaries
- `evidence_assessment`: acceptance decisions and score summaries
- `retrieval_indexing`: document count indexed, total chunk count
- `retrieval_query`: chunk count returned, top chunk score, `agent_iterations` (number of LangGraph retrieve→evaluate cycles; `null` in `"local"` mode)
- `analysis`: accepted document URLs, claim count, candidate values
- `synthesis`: selected value, supporting URLs, conflict count, synthesis confidence
- `review_gate`: overall confidence, final decision, gate reason

The `review_gate` stage is the confidence-scoring surface for the current pipeline, so confidence stays attached to that stage instead of being split into a separate run.

## Enabling LangSmith

1. Copy `.env.example` to `.env`.
2. Set `LANGSMITH_TRACING=true`.
3. Set `LANGSMITH_API_KEY`.
4. Optionally change `LANGSMITH_PROJECT` from the default `evidence-enrichment-engine`.
5. Run any CLI command, for example:

```bash
evidence-enrich trace-demo --mode replay
```

Replay mode works with blank provider keys. For live mode, also fill in the relevant provider credentials in `.env`.

Legacy `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` values are still honored for compatibility, but `LANGSMITH_*` is the preferred naming.

## Enabling Langfuse

Langfuse is an open-source observability platform. Install the optional dependency first:

```bash
pip install 'evidence_enrichment[observability]'
# or: pip install 'langfuse>=4.0.0,<5'
```

Then set in `.env`:

```
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

The Langfuse backend is feature-flagged by `LANGFUSE_SECRET_KEY`. When the key is absent (or the `langfuse` package is not installed), the pipeline operates unchanged — LangSmith and LocalTracer remain unaffected.

Each stage function is decorated with `@observe(capture_input=False, capture_output=False)` — full IO capture is disabled to prevent exfiltrating large parsed documents. Instead, the same compact summarized payload used by LangSmith is written via `update_current_span(input=..., output=...)`.

Legacy `LANGFUSE_HOST` is accepted as an alias for `LANGFUSE_BASE_URL` for back-compat and will also be kept in sync.

## Privacy Note

When LangSmith tracing is enabled, the wrapped OpenAI and Anthropic clients (`wrap_openai`, `wrap_anthropic`) also capture raw prompts and full LLM responses in your LangSmith project. These prompts may include full document text and retrieved chunks from the RAG layer. Avoid enabling LangSmith tracing when processing sensitive or confidential documents.

Langfuse's `capture_input=False, capture_output=False` flags prevent automatic full IO capture, but the compact summarized payloads (claim values, URLs, confidence scores) are still sent.

## Viewing Traces

1. Run a traced command locally or in Docker.
2. Open the LangSmith/Langfuse project.
3. Open the latest run tree.
4. Inspect each stage span to compare the summarized stage inputs and outputs with the local trace artifacts written to disk.

![LangSmith trace view](../docs/assets/langsmith-trace.png)
