# Retrieval-Augmented Analysis (Optional)

This document covers the optional RAG layer added to the evidence enrichment pipeline.

When enabled, the pipeline indexes accepted documents into a local Chroma vector store after evidence assessment and retrieves the most relevant chunks for each document before the analysis stage. Retrieved chunks replace the default `text[:6000]` truncation with semantically ranked evidence.

---

## Architecture

The retrieval layer sits between `evidence_assessment` and `analysis`:

```text
query_plan
  -> search
  -> fetch
  -> parse  (structured: HTML table/heading detection active)
  -> evidence_assessment
  -> retrieval_indexing  (NEW: chunk + embed + upsert accepted docs)
  -> retrieval_query     (NEW: retrieve top_k chunks per doc before analysis)
  -> analysis  (uses retrieved chunks instead of text[:6000])
  -> synthesis
  -> review_gate
```

When retrieval mode is `"off"` (default), the pipeline is unchanged. No new stages appear, the parser uses the shallow path, and `parse_with_structure` is never called.

---

## Table-Aware Chunking

The `TableAwareChunker` (adapted from a production KB ingestion pattern) treats HTML tables and plain-text tables as atomic units rather than splitting them at character boundaries.

**Motivation:** LLM analysis of structured numeric evidence (revenue, employee counts, geographic breakdowns) degrades significantly when tables are split mid-row. An atomic table chunk keeps row/column relationships intact for the embedding model and the downstream LLM prompt.

**Logic:**

1. The parser (`parse_with_structure`) extracts block-level structure from HTML:
   - `<table>` tags → `ContentBlock(block_type="table")`
   - `<h1>`–`<h6>` tags → `ContentBlock(block_type="heading")`
   - All remaining content → `ContentBlock(block_type="text")`
   - Plain-text table heuristics detect pipe-delimited, tab-delimited, and Markdown tables in the text remainder

2. The chunker processes each block:
   - Tables within `max_table_size` characters → single atomic chunk (no overlap)
   - Tables exceeding `max_table_size` → split on whitespace boundaries **without overlap** to avoid duplicating numeric rows
   - Text blocks → character-based chunking with configurable `chunk_size` and `overlap`
   - Blocks below `min_size` chars are dropped

**Defaults:** `chunk_size=1500`, `overlap=200`, `min_size=80`, `max_table_size=4000`

---

## Hybrid Scoring Formula

The `HybridRetriever` combines three signals:

```
score = 0.7 × vector_score + 0.2 × keyword_score + 0.1 × table_boost
```

| Signal | Source | Range |
|--------|--------|-------|
| `vector_score` | Cosine similarity from Chroma (distance → similarity) | [0, 1] |
| `keyword_score` | Term-overlap fraction: `|query_terms ∩ chunk_terms| / |query_terms|` | [0, 1] |
| `table_boost` | 0.1 if chunk is a table AND query contains numeric/financial keywords | {0, 0.1} |

**Rationale:** The vector score captures semantic similarity but misses exact numeric matches (e.g., "$1.4B", "Q3 2024") because embeddings compress token-level detail. The keyword score provides a lexical anchor. The table boost prioritises structured numeric evidence for financial-type queries.

Weights are configurable in `evidence_enrichment.yaml` under `retrieval.weights`.

---

## Document-Scoped Retrieval

All retrieval queries are filtered by `document_url`:

```python
results = retriever.retrieve(query=field_name, document_url=document.url)
```

This is a deliberate architectural constraint. The pipeline produces one `AnalysisReport` per source document, with claims attributed to that document's URL. If retrieval pulled chunks across documents, source attribution on `FactClaim.source_url` would break — the claim would appear to come from document A but use evidence from document B.

**Effect:** Each document's chunks are retrieved independently. Cross-document synthesis happens downstream at the `synthesis` stage where all claims are pooled.

---

## Configuration Reference

All retrieval settings live under the `retrieval` key in `evidence_enrichment.yaml`:

```yaml
retrieval:
  mode: "off"               # "off" | "local"
  persist_path: examples/output/chroma
  chunk_size: 1500
  overlap: 200
  max_table_size: 4000
  top_k: 5
  embedding_model: text-embedding-3-small
  min_doc_chars: 2000
```

| Key | Description | Default |
|-----|-------------|---------|
| `mode` | `"off"` disables retrieval entirely. `"local"` enables Chroma-backed RAG. | `"off"` |
| `persist_path` | Path for Chroma's persistent storage. Covered by `.gitignore`. | `examples/output/chroma` |
| `chunk_size` | Target character count per text chunk. | `1500` |
| `overlap` | Overlap in characters between consecutive text chunks. | `200` |
| `max_table_size` | Max chars before a table block is split (split has no overlap). | `4000` |
| `top_k` | Number of retrieved chunks returned to the analysis prompt. | `5` |
| `embedding_model` | OpenAI embedding model name. | `text-embedding-3-small` |
| `min_doc_chars` | Minimum document character count to index. Shorter docs skip indexing. | `2000` |

### Enabling Retrieval

```yaml
# evidence_enrichment.yaml
retrieval:
  mode: "local"
```

Requires `OPENAI_API_KEY` in `.env` (the same key used for live analysis providers).

Install the retrieval dependency group:

```bash
pip install -e ".[retrieval]"
```

Run with retrieval active:

```bash
evidence-enrich run --entity examples/microsoft.json --field hq_country --mode live
```

---

## Replay Mode Behaviour

Retrieval is skipped entirely in replay mode. No embedding API calls are made, Chroma is not initialised, and replay bundles are loaded unchanged. This preserves the zero-credential, zero-network-access guarantee of the default replay flow.

The `ReplayAnalysisAgent.analyze()` method accepts the `retrieved_chunks` parameter for interface compatibility but ignores it.

---

## Privacy Note

The `examples/output/chroma/` path is covered by `.gitignore`. Chroma embeddings and chunks derived from public web content are never committed to the repository.

---

## Deferred (v2)

- `text-embedding-3-large` benchmarking against `text-embedding-3-small`
- Embedding-based entity matching (semantic similarity to company name)
- Cross-document retrieval with evidence re-attribution
- PDF parsing support (currently HTML-only)
- LangSmith `@traceable` decorators on retrieval stages
- Replay fixtures with pre-computed embeddings for offline retrieval tests
