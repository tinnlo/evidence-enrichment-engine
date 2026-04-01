# Observability

This repo keeps its local trace artifacts and can optionally emit LangSmith traces for the same workflow stages.

## What Is Traced

The pipeline always writes local artifacts under `examples/output/traces/<trace_id>/`:

- `spans.jsonl`
- `trace_summary.json`
- `trace_timeline.md`
- `openinference_trace.json`
- `resolved_context.json`

When LangSmith tracing is enabled, the coordinator also emits compact stage traces for:

- `query_plan`
- `search`
- `fetch`
- `parse`
- `evidence_assessment`
- `analysis`
- `synthesis`
- `review_gate`

LangSmith captures summarized inputs and outputs rather than raw full-document payloads:

- `query_plan`: entity id, field, company name, resolved context entry ids, primary query, query variant count
- `search`: query text, provider order, result count, top URLs
- `fetch`: requested URLs, fetched count, success count
- `parse`: document URLs, titles, and text-length summaries
- `evidence_assessment`: acceptance decisions and score summaries
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

## Privacy Note

When LangSmith tracing is enabled, the wrapped OpenAI and Anthropic clients (`wrap_openai`, `wrap_anthropic`) also capture raw prompts and full LLM responses in your LangSmith project. These prompts may include full document text and retrieved chunks from the RAG layer. Avoid enabling LangSmith tracing when processing sensitive or confidential documents.

## Viewing Traces

1. Run a traced command locally or in Docker.
2. Open the LangSmith project named by `LANGSMITH_PROJECT`.
3. Open the latest run tree.
4. Inspect each stage span to compare the summarized stage inputs and outputs with the local trace artifacts written to disk.

![LangSmith trace view](../docs/assets/langsmith-trace.png)
