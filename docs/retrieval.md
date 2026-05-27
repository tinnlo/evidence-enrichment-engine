# Retrieval

The retrieval layer is optional. When enabled, it inserts a bounded RAG step between `evidence_assessment` and `analysis` so the analysis prompt can work from semantically ranked document chunks instead of a fixed `text[:6000]` truncation.

The design goal is not "add more AI." The design goal is to make evidence selection explicit while preserving the replay-first artifact contract.

## Retrieval System View

```mermaid
graph TD

subgraph INPUTS["Inputs and controls"]
  direction TB
  D["Accepted documents"]:::bronze
  Q["Field query"]:::external
  C["Retrieval config"]:::external
end

subgraph RET["Retrieval pipeline"]
  direction TB
  P["Structured parse"]:::bronze
  CH["Chunker<br/>(TableAware or Hierarchical)"]:::silver
  VS["Chroma store<br/>(schema_version=1 or 2)"]:::external
  HR["Retriever<br/>(Hybrid or Hierarchical)"]:::silver
  AG["LangGraph loop<br/>(optional)"]:::golden
end

subgraph OUTPUTS["Outputs and telemetry"]
  direction TB
  O1["Ranked chunks for analysis"]:::artifact
  O2["Local spans<br/>retrieval_indexing | retrieval_query"]:::artifact
  O3["Persisted vectors<br/>examples/output/chroma"]:::external
end

D --> P --> CH --> VS
Q --> HR
VS --> HR
C -. "chunking rules" .-> CH
C -. "weights + top_k" .-> HR
HR --> O1
HR --> O2
VS --> O3
HR -. "agent mode" .-> AG
AG --> HR

classDef bronze fill:#ffe6e6,stroke:#b30000,stroke-width:1px
classDef silver fill:#e6f0ff,stroke:#003399,stroke-width:1px
classDef golden fill:#e6ffe6,stroke:#006400,stroke-width:1px
classDef external fill:#fce8ff,stroke:#7b2fa8,stroke-width:1.5px,stroke-dasharray:5 3
classDef artifact fill:#fff8e1,stroke:#e65100,stroke-width:1px,stroke-dasharray:5 5

style INPUTS fill:transparent,stroke:#7b2fa8,stroke-width:1px,stroke-dasharray:4 4
style RET fill:transparent,stroke:#003399,stroke-width:1px,stroke-dasharray:4 4
style OUTPUTS fill:transparent,stroke:#e65100,stroke-width:1px,stroke-dasharray:4 4
```

## Mode Comparison

| `retrieval.mode` | Behavior | Operational notes |
|---|---|---|
| `off` | No retrieval stages are added. `analysis` works from the parsed document text directly. | Default behavior. Replay bundles and artifacts stay unchanged. |
| `local` | Accepted documents are chunked, embedded, stored in Chroma, and queried before `analysis`. | Requires the `[retrieval]` extra and `OPENAI_API_KEY` for embeddings in live mode. |
| `agent` | Runs the same retrieval stack, but query refinement is managed by a LangGraph `StateGraph`. | Adds `agent_iterations` to the `retrieval_query` span so loop depth is inspectable. |

Replay mode skips retrieval entirely even if retrieval is configured. That preserves the zero-network demo path and avoids changing stored replay bundles.

## Pipeline Placement

Base path:

```text
query_plan -> search -> fetch -> parse -> evidence_assessment -> analysis -> synthesis -> review_gate
```

Retrieval-enabled path:

```text
query_plan -> search -> fetch -> parse -> evidence_assessment -> retrieval_indexing -> retrieval_query -> analysis -> synthesis -> review_gate
```

The retrieval stages are additive. They do not rename, remove, or replace the rest of the pipeline contract.

## Chunker and Retriever Comparison

Two implementations are available, selected by `retrieval.chunker` and `retrieval.retriever`:

| `retrieval.chunker` | `retrieval.retriever` | Behavior |
|---|---|---|
| `flat` (default) | `hybrid` (default) | `TableAwareChunker` produces fixed-size character chunks. `HybridRetriever` runs a single-stage vector + keyword + table scan across all chunks for the document. |
| `hierarchical` | `hierarchical` | `HierarchicalChunker` parses the document into a `SectionNode` tree and emits `section_summary`, `content`, and `table` chunks tagged with `section_id` and `section_path_str`. `HierarchicalRetriever` runs a two-stage query: Stage 1 selects the top-scoring sections via `section_summary` chunks (~50 comparisons); Stage 2 retrieves content/table chunks within matched sections and their descendants (~30 comparisons). Falls back to a full-document scan for legacy flat-indexed documents. |

Setting `retriever="hierarchical"` with `chunker="flat"` automatically promotes `chunker` to `"hierarchical"` with a warning, since `section_summary` chunks are only produced by `HierarchicalChunker`.

The `schema_version` field controls which Chroma collection is addressed. `schema_version=1` (default) uses the legacy flat collection; `schema_version=2` uses the `_v2` collection written by `HierarchicalChunker`. Use `scripts/migrate_to_hierarchical.py` to re-index an existing collection.

## Chunking Strategy

### `flat` — `TableAwareChunker`

The `TableAwareChunker` is intentionally conservative because the workflow is evidence-oriented rather than chatbot-oriented.

| Chunk type | Rule | Why it exists |
|---|---|---|
| HTML tables and plain-text tables | Keep as one chunk when the table fits within `max_table_size` | Row and column relationships matter for numeric evidence |
| Large tables | Split on whitespace boundaries without overlap | Avoid duplicate numeric rows across adjacent chunks |
| Normal text blocks | Character-based chunking with overlap | Preserve local context while keeping embedding payloads bounded |
| Very small blocks | Drop below `min_size` | Avoid noisy low-information chunks |

Defaults: `chunk_size=1500`, `overlap=200`, `min_size=80`, `max_table_size=4000`.

### `hierarchical` — `HierarchicalChunker`

`HierarchicalChunker` requires a `ParsedDocument` with a populated `SectionNode` tree (produced by `HTMLStructuredParser` or `PDFStructuredParser` when `use_structured=True`). For documents without a section tree it falls back to `TableAwareChunker` flat chunking.

| Chunk role | Source | Metadata |
|---|---|---|
| `section_summary` | Heading text + first content sentence | `section_id`, `section_path_str`, `chunk_role=section_summary` — used by Stage 1 to route the query to the right section |
| `content` | Text blocks within the section | `section_id`, `section_path_str`, `parent_section_id`, `chunk_role=content` |
| `table` | Table blocks within the section | `section_id`, `section_path_str`, `chunk_role=table`, `table_data` |

When retrieval is active, parsing uses the structured path so headings, tables, and plain-text table heuristics can be preserved before chunking.

## Scoring Formula

### `hybrid` — `HybridRetriever`

Single-stage hybrid score:

```text
score = 0.7 * vector_score + 0.2 * keyword_score + 0.1 * table_boost
```

| Signal | Source | Range | Purpose |
|---|---|---|---|
| `vector_score` | Chroma cosine similarity transformed into similarity space | `[0, 1]` | Captures semantic relevance |
| `keyword_score` | `|query_terms ∩ chunk_terms| / |query_terms|` | `[0, 1]` | Preserves exact lexical anchors for numeric or named evidence |
| `table_boost` | Applied when the chunk is table-like and the query is numeric/financial | `{0, 0.1}` | Prioritizes structured evidence when tables are likely to matter |

### `hierarchical` — `HierarchicalRetriever`

Five-component hybrid score applied in Stage 2:

```text
score = 0.5 * vector_score + 0.15 * keyword_score + 0.1 * table_boost
      + 0.15 * section_summary_boost + 0.1 * role_table_boost
```

Stage 1 uses vector score alone to select sections. Stage 2 reranks content/table chunks within selected sections using the five-component formula.

The weights are configurable in `evidence_enrichment.yaml` under `retrieval.weights`.

## Document-Scoped Retrieval

Retrieval is filtered by `document_url`, not run across the full accepted-document pool.

That constraint is deliberate:

1. `analysis` produces claims that must stay attributable to the source document.
2. `FactClaim.source_url` would become misleading if a claim nominally attributed to document A used chunks from document B.
3. Cross-document synthesis already happens later at the `synthesis` stage where claims are pooled explicitly.

## LangGraph Adaptive Loop

`retrieval.mode=agent` wraps the retriever in a LangGraph loop that can iterate before handing chunks to `analysis`.

```mermaid
graph TD

subgraph LOOP["Adaptive retrieval loop"]
  direction TB
  R["Retrieve"]:::silver
  E["Evaluate chunk quality"]:::silver
  F["Refine query"]:::golden
  ENDNODE["Return chunks"]:::artifact
end

R --> E
E -. "quality low" .-> F
F --> R
E -. "quality OK or cap hit" .-> ENDNODE

classDef silver fill:#e6f0ff,stroke:#003399,stroke-width:1px
classDef golden fill:#e6ffe6,stroke:#006400,stroke-width:1px
classDef artifact fill:#fff8e1,stroke:#e65100,stroke-width:1px,stroke-dasharray:5 5

style LOOP fill:transparent,stroke:#003399,stroke-width:1px,stroke-dasharray:4 4
```

Operational details:

- The loop stops when the average chunk score clears the quality threshold.
- The loop also stops when the configured iteration cap is reached.
- The number of retrieve-evaluate cycles is recorded as `agent_iterations` on the `retrieval_query` span.

## Config Reference

```yaml
retrieval:
  mode: "off"               # "off" | "local" | "agent"
  chunker: "flat"           # "flat" | "hierarchical"
  retriever: "hybrid"       # "hybrid" | "hierarchical"
  schema_version: 1         # 1 = legacy flat collection; 2 = _v2 hierarchical collection
  persist_path: examples/output/chroma
  chunk_size: 1500
  overlap: 200
  max_table_size: 4000
  top_k: 5
  embedding_model: text-embedding-3-small
  min_doc_chars: 2000
```

| Key | Meaning | Default |
|---|---|---|
| `mode` | Retrieval execution mode | `off` |
| `chunker` | Chunking strategy | `flat` |
| `retriever` | Retrieval strategy | `hybrid` |
| `schema_version` | Chroma collection schema version (`1` = flat, `2` = hierarchical `_v2`) | `1` |
| `persist_path` | Chroma persistence directory | `examples/output/chroma` |
| `chunk_size` | Target text chunk size in characters | `1500` |
| `overlap` | Character overlap between text chunks | `200` |
| `max_table_size` | Maximum chars before a table is split | `4000` |
| `top_k` | Number of ranked chunks returned per document | `5` |
| `embedding_model` | Embedding model used for indexing and retrieval | `text-embedding-3-small` |
| `min_doc_chars` | Minimum document length to index | `2000` |

## Install And Run

```bash
python -m pip install -e ".[retrieval]"
```

Example config (hierarchical):

```yaml
# evidence_enrichment.yaml
retrieval:
  mode: "local"
  chunker: "hierarchical"
  retriever: "hierarchical"
  schema_version: 2
```

Example run:

```bash
evidence-enrich run --entity examples/microsoft.json --field hq_country --mode live
```

`OPENAI_API_KEY` is required for live retrieval because chunk embeddings are created with the OpenAI embeddings API.

## Operational Boundaries

- Replay mode skips retrieval entirely.
- Chroma data lives under `examples/output/chroma/` and is covered by `.gitignore`.
- Retrieval does not change the local trace artifact contract; it only adds `retrieval_indexing` and `retrieval_query` spans when active.
- The replay analysis agent accepts `retrieved_chunks` for interface compatibility but ignores them in replay mode.
- v1 and v2 Chroma collections co-exist in the same directory. Use `scripts/migrate_to_hierarchical.py --dry-run` to preview a migration; `--delete-v1` is only safe after `schema_version=2` is deployed and confirmed.

