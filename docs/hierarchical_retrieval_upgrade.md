# Hierarchical Retrieval & Schema Extraction Upgrade Plan

*Created: 2026-05-25 · Revised: 2026-05-26 (r3 — post-implementation sync)*

Addresses three production-quality gaps not covered by the existing [upgrade-plan](../.opencode/plans/upgrade-plan.md):

1. **Flat chunking loses section hierarchy** — annual reports and sustainability reports are navigated by section; a flat sequence of 1,500-char chunks discards that signal.
2. **Flat namespace is slow at scale** — a 200-page report yields ~1,500 chunks; every query scans all of them.
3. **No typed extraction schema** — `FactClaim.candidate_value` is a free-form string; fields like geographic revenue require structured validation (row sums, currency consistency, completeness).

This plan is **staged and backward-compatible**. The existing `TableAwareChunker` / `HybridRetriever` remain alive behind a feature flag until evals confirm the upgrade.

---

## Implementation Status (as of 2026-05-26)

Stages A, B, and C are **fully implemented and passing** (`444 passed, 7 skipped`).
Stage D (migration, evals, acceptance) is the remaining deliverable.

| Stage | Status | Test file | Tests |
|---|---|---|---|
| A — Hierarchical doc model + parsers | Complete | `tests/test_hierarchical.py` | 42 |
| B — Two-stage retrieval + coordinator wiring | Complete | `tests/test_hierarchical.py` | (included above) |
| C — Schema extraction package | Complete | `tests/test_extraction.py` | 45 |
| D — Migration, evals, acceptance | Not started | — | — |

### Key implementation decisions that diverged from this plan

**`_coerce_nested_fields` helper (`extraction/models.py`)**
The plan described per-list-field coercion in the failure path. The implementation adds a shared `_coerce_nested_fields(schema_cls, raw_dict)` helper that handles both `list[SomeModel]` fields *and* scalar `SomeModel | None` fields (e.g. `total_revenue: MoneyAmount | None`). Python 3.12's `X | Y` union syntax requires checking `isinstance(annotation, types.UnionType)` in addition to `get_origin(...) is typing.Union` — the old-style `Optional[X]` form uses the latter. Without this the `total_revenue` field on the cross-field failure path was left as a raw `dict` after round-trip.

**FinOps for repair loops (`coordinator.py:_run_schema_extraction`)**
The plan did not specify how to record FinOps when `SchemaExtractor` makes multiple provider calls (initial + repair attempts). The implemented approach:
- Accumulate `estimate_tokens(prompt)` / `estimate_tokens(response)` per call into integer lists.
- In the `finally` block call `_record_stage_finops_from_tokens` with pre-summed totals and `call_count=len(calls)`.
- **Do not** use `_record_stage_finops` with concatenated text and `call_count=N`: that function treats its `input_text` as a single-call average and multiplies by `call_count`, which would cause N× overcount on repaired extractions.

**`_to_chroma_where()` in `store.py`**
Multi-key `where` dicts must be rewritten to `{"$and": [...]}` form for Chroma ≥ 0.6.0. The plan noted the Chroma metadata constraints (no `$contains` on plain string fields) but did not include this rewrite utility explicitly.

**`_ensure_descendant_map()` lazy rebuild**
Legacy collections (pre-Stage-A, no `parent_section_id` in metadata) fall back to a path-prefix heuristic on fresh-instance rebuild. Precision is approximate until re-indexed with `HierarchicalChunker`.

**`ExtractionResult` serialiser/deserialiser**
`model_serializer(mode="wrap")` injects `__schema_cls__` into the dumped `value` dict. `model_validator(mode="before")` dispatches on that tag via `SCHEMA_REGISTRY` to reconstruct the concrete type. On the failure path (`validation_passed=False`) `model_construct` is used with per-field coercion so cross-field validators do not re-run and raise again.

### Resolved findings

| Finding | Resolution |
|---|---|
| Config: `schema_validation` and `schema_repair_max_attempts` not exposed | Added to `RetrievalConfig` in `settings.py` |
| Confidence propagation: gate penalty not written into `value.extraction_confidence` | Coordinator gate loop `model_copy`s both wrapper and inner `value` |
| Budget gating: `_run_schema_extraction` not checking budget before calling provider | `_resolve_stage_model_and_budget("schema_extraction", ...)` called; `BLOCKED` returns `[]` immediately |
| Failed round-trip: nested rows/scalar models stayed as raw `dict` after `model_validate(model_dump())` | `_coerce_nested_fields` handles both list and scalar `BaseModel` fields; `types.UnionType` detection added for Python 3.12 `X \| Y` unions |
| FinOps repair loop overcount: concatenated-text + `call_count=N` caused N× scaling | Switched to `_record_stage_finops_from_tokens` with pre-summed per-call token counts |

---

## Scope decision

**Stage C (schema extraction) is a scope extension, not an in-place replacement.**

The repo's current stated goal is *single-field enrichment* (`hq_country` is the only implemented enricher; see `goals_and_features.md:7`). Stage C introduces a general multi-field typed extraction framework. That is a meaningful scope change. It is kept in this plan because the retrieval infrastructure in Stages A/B is only fully motivated when there are structured fields to extract — but Stage C must be treated as its own deliverable and should not be started before Stages A/B are validated.

Stage C also does **not** replace `FactClaim.candidate_value: str`. It adds a parallel typed artifact (`ExtractionResult`) that sits alongside the existing claim pipeline without touching `FactClaim`, synthesis, review gate, or replay bundles. See §C-compatibility below.

---

## Current State Audit

| Component | File | Gap |
|---|---|---|
| `TableAwareChunker` | `core/retrieval/chunker.py:16` | Fixed `chunk_size=1500`, no section context, char-based sizing |
| `ContentBlock` | `core/models/contracts.py:36` | No `section_id`, no `page`, no `bbox`, no structured `table_data` |
| `ParsedDocument` | `core/models/contracts.py:79` | `blocks: list[ContentBlock]` is flat — heading→body parent/child relationship is lost at parse time |
| `ChromaVectorStore.upsert` | `core/retrieval/store.py:115` | Metadata only carries `document_url, chunk_type, index, char_count, content_hash` — no section path, no page |
| `ChromaVectorStore.query` | `core/retrieval/store.py:226` | Already forwards arbitrary `where` to Chroma correctly; the missing work is writing new fields in `upsert` and surfacing them in `_build_analysis_context` |
| `HybridRetriever.retrieve` | `core/retrieval/retriever.py:235` | Single-stage vector+keyword+table scan over the full document chunk namespace |
| `_build_analysis_context` | `core/providers/agents.py:71` | Prompt formatter only exposes `chunk_type` and `score`; section path and page are not shown to the LLM |
| `_get_retriever` | `pipeline/coordinator.py:234` | Hard-wires `TableAwareChunker → HybridRetriever`; feature flag not wired |
| Budget estimation path | `pipeline/coordinator.py:1424` | Second hard-wired `TableAwareChunker` instantiation inside FinOps budget gate; must be updated in the same pass as `_get_retriever` |
| `EvidenceCoordinator.__init__` | `pipeline/coordinator.py:149` | Assigns `self.parser = TextParser()` directly; `ParserRegistry` has no injection point yet |
| `_parse_documents` | `pipeline/coordinator.py:943` | Calls `self.parser.parse_with_structure()` / `self.parser.parse()` directly; PDF documents fall through to HTML path silently |
| `RetrievalConfig` | `config/settings.py:302` | No `chunker` / `retriever` discriminator fields |
| `DocumentFetcher` | `core/fetch/fetcher.py:24` | `_ALLOWED_CONTENT_PREFIXES` excludes `application/pdf`; `body` is always decoded to `str`; PDF bytes are unreachable |
| `RetrievedDocument` | `core/models/contracts.py:68` | `body: str` only — no raw bytes field |
| `FactClaim` | `core/models/contracts.py:105` | `candidate_value: str` — used by agents, gates, summarizers, replay; must not be replaced |

---

## Stage A — Hierarchical Document Model

**Goal:** Replace the flat `ContentBlock` list with a tree that preserves section hierarchy and page numbers. Includes the fetch-layer changes required before any PDF parser is reachable.

**Success criteria:**
- `ParsedDocument.sections` carries a `list[SectionNode]` with correct parent/child links
- `ContentBlock` has `section_id`, `page`, and optional `table_data: list[list[str]]`
- `HTMLStructuredParser` and `PDFStructuredParser` both populate the tree
- Existing `TextParser.parse_with_structure` aliased to `HTMLStructuredParser` — no callers broken
- Unit tests verify: heading nesting depth, `section_path` propagation, page attribution on ≥1 PDF fixture

### A0. Fetch layer — PDF support prerequisite

Before any PDF parser can be reached, two changes are required:

**A0a. `RetrievedDocument` — add raw bytes field**

Extend `evidence_enrichment/core/models/contracts.py`:

```python
class RetrievedDocument(BaseModel):
    url: str
    final_url: str
    title: str
    content_type: str
    body: str                       # unchanged — HTML/text responses
    body_bytes: bytes | None = None # NEW — populated for binary content types (PDF)
    provider: str
    fetch_success: bool = True
    error: str | None = None
```

`body_bytes` is `None` for all existing text responses; no downstream code breaks.

**A0b. `DocumentFetcher` — conditionally allow `application/pdf`**

PDF fetching must only be enabled when the **complete** PDF parser stack is available. The parser uses both `pdfplumber` (text/bbox extraction) and `pymupdf` (TOC/bookmark extraction), so probing only `pdfplumber` would allow the fetch in partial environments and then fail at parse time when `pymupdf` is absent.

The fetch gate must probe the same capability check that parser registration uses — i.e. whether `PDFStructuredParser` itself can be imported — rather than probing individual dependency packages. This way the two gates are guaranteed to stay in sync:

In `evidence_enrichment/core/fetch/fetcher.py`:

1. Keep `_ALLOWED_CONTENT_PREFIXES` unchanged for the base install. Add `"application/pdf"` to a separate `_ALLOWED_BINARY_PREFIXES` set that is **conditionally populated at module load using the same import guard as A2b**:

```python
_ALLOWED_BINARY_PREFIXES: frozenset[str] = frozenset()

try:
    # Probe the parser module — not individual packages — so the fetch gate
    # and the parser registration gate use an identical capability check.
    # If either pdfplumber or pymupdf is absent, this import fails and PDFs
    # are blocked at fetch time exactly as today.
    from evidence_enrichment.core.parse.pdf_structured import PDFStructuredParser  # noqa: F401
    _ALLOWED_BINARY_PREFIXES = frozenset({"application/pdf"})
except ImportError:
    pass  # PDF fetching disabled in base install
```

2. In the fetch method, check `_ALLOWED_BINARY_PREFIXES` alongside `_ALLOWED_CONTENT_PREFIXES`. For binary content types, skip the `raw_bytes.decode()` step and populate `body_bytes` instead of `body` (leave `body=""` for binary responses).
3. Raise the size cap for PDF to a separate `_MAX_PDF_BYTES = 50 MB` constant; the existing `_MAX_BODY_BYTES = 5 MB` cap applies to text responses only.
4. Keep the SSRF guard (`_validate_url`, `_validate_host`) unchanged.

**A0c. `ParserRegistry` dispatch** (see A2b) routes `application/pdf` to `PDFStructuredParser`, which reads `doc.body_bytes`. Passing `body_bytes=None` to `PDFStructuredParser` raises `ValueError` early rather than silently producing an empty parse. Because A0b already blocks the fetch when `pdfplumber` is absent, `body_bytes=None` on a PDF document is a bug, not a missing-optional-dep condition.

### A1. New contract types

Extend `evidence_enrichment/core/models/contracts.py`:

```python
class SectionNode(BaseModel):
    section_id: str                # stable hash of (doc_url + heading_path_joined)
    level: int                     # 0 = doc root, 1 = H1/Part, 2 = H2, ...
    heading: str                   # "" for implicit root
    path: list[str]                # ["Financial Review", "Segment Information", "Revenue by Geography"]
    page_start: int | None
    page_end: int | None
    parent_id: str | None
    children_ids: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)    # IDs of leaf ContentBlocks


class ContentBlock(BaseModel):    # extended — all new fields have defaults
    block_id: str = ""             # NEW — stable hash; "" for legacy blocks
    block_type: Literal["text", "table", "heading", "figure_caption", "list", "kpi"]
    content: str
    section_id: str = ""           # NEW
    page: int | None = None        # NEW
    bbox: tuple[float, float, float, float] | None = None   # NEW — PDF visual grounding
    table_data: list[list[str]] | None = None               # NEW — raw rows
    char_count: int = 0


class ParsedDocument(BaseModel):  # extended — all new fields have defaults
    # ... all existing fields unchanged ...
    sections: list[SectionNode] = Field(default_factory=list)    # NEW
    section_tree_root: str | None = None                          # NEW
    page_count: int = 0                                           # NEW
```

All additions use default values so existing `ParsedDocument` construction sites are unaffected.

### A2. Pluggable parser interface

Create `evidence_enrichment/core/parse/base.py`:

```python
class UnsupportedContentTypeError(ValueError):
    """Raised by ParserRegistry when no parser is registered for a content type
    and the content type is not a text type (binary without a registered parser)."""

class DocumentParser(Protocol):
    def can_parse(self, doc: RetrievedDocument) -> bool: ...
    def parse(self, doc: RetrievedDocument) -> ParsedDocument: ...

class ParserRegistry:
    """Dispatch by MIME type.

    Resolution order:
    1. Try each registered parser in registration order (first-registered wins).
    2. If none match and the content type is a text type (`text/*`,
       `application/json`, `application/xml`, `application/xhtml`), fall back
       to `GenericTextParser` — this preserves current TextParser behaviour for
       all non-HTML, non-PDF text documents.
    3. If none match and the content type is binary (e.g. `application/pdf`
       without PDFStructuredParser registered), raise `UnsupportedContentTypeError`
       so missing-optional-dep conditions surface as a clear error, not a silent
       empty parse.
    """
    _TEXT_FALLBACK_PREFIXES = ("text/", "application/json", "application/xml", "application/xhtml")

    def register(self, parser: DocumentParser) -> None: ...

    def parse(self, doc: RetrievedDocument) -> ParsedDocument:
        for parser in self._parsers:
            if parser.can_parse(doc):
                return parser.parse(doc)
        ct = doc.content_type.split(";")[0].strip().lower()
        if ct.startswith(self._TEXT_FALLBACK_PREFIXES):
            return GenericTextParser().parse(doc)
        raise UnsupportedContentTypeError(
            f"No parser registered for content_type={doc.content_type!r}. "
            "Install the .[retrieval] extra to enable PDF parsing."
        )
```

Implementations:

| Class | File | Handles |
|---|---|---|
| `GenericTextParser` | `core/parse/generic_text.py` | `text/*`, `application/json`, `application/xml`, `application/xhtml` — wraps current `TextParser.parse_with_structure()` (not `parse()`), preserving block extraction, `full_text`, `content_hash`, `mime_type`, and plain-text table detection for non-HTML text docs; always available, no optional deps; **registered implicitly as the text fallback in `ParserRegistry`** |
| `HTMLStructuredParser` | `core/parse/html_structured.py` | `text/html` — wraps current `TextParser.parse_with_structure`, emits `SectionNode` tree |
| `PDFStructuredParser` | `core/parse/pdf_structured.py` | `application/pdf` — reads `doc.body_bytes`; raises `ValueError` if `None` |
| `XBRLFinancialParser` | `core/parse/xbrl.py` | `application/xml` (SEC/EDGAR) — optional, Stage D |

`GenericTextParser.can_parse()` returns `True` for all `_TEXT_FALLBACK_PREFIXES` types, but because `ParserRegistry` tries registered parsers first, `HTMLStructuredParser` (registered at coordinator init) takes precedence for `text/html`. `GenericTextParser` is never explicitly registered — it is the implicit text fallback inside `ParserRegistry.parse()` — so it cannot accidentally shadow a more specific registered parser.

**PDF heading detection strategy** (annual reports rarely have semantic H-tags in PDF):

1. Extract per-character font sizes via `pdfplumber.page.chars`.
2. Cluster sizes into `{body, subheading, heading, title}` tiers using the top-3 most frequent sizes as anchors (no external ML dependency needed).
3. Lines whose average char size falls in `heading` or `title` tier become `SectionNode` entries.
4. If bookmarks/TOC are available (`doc.get_toc()` via `pymupdf`), use those directly — they are more reliable and cheaper than font clustering.

### A2b. Wire `ParserRegistry` into `EvidenceCoordinator`

`EvidenceCoordinator.__init__` (`coordinator.py:149`) currently does:

```python
self.parser = TextParser()
```

Replace with a **lazy, import-guarded** construction so that the optional `pdfplumber`/`pymupdf` stack is only imported when it is actually installed:

```python
from evidence_enrichment.core.parse.base import ParserRegistry
from evidence_enrichment.core.parse.html_structured import HTMLStructuredParser

self.parser = ParserRegistry()
self.parser.register(HTMLStructuredParser())   # always available — no optional deps

try:
    from evidence_enrichment.core.parse.pdf_structured import PDFStructuredParser
    self.parser.register(PDFStructuredParser())
except ImportError:
    # pdfplumber / pymupdf not installed (base install without .[retrieval]).
    # PDFs are already blocked at fetch time by _ALLOWED_BINARY_PREFIXES (see A0b),
    # so no PDF RetrievedDocument will ever reach this registry.
    # If one somehow did arrive (e.g. injected in tests), ParserRegistry.parse()
    # should raise UnsupportedContentTypeError rather than silently delegating to
    # HTMLStructuredParser — add that guard to ParserRegistry.parse().
    pass
```

This preserves the optional-dependency contract: a `replay` or `retrieval.mode=off` install without `.[retrieval]` continues to behave exactly as today — PDFs are rejected at the fetch boundary, never at parse time.

The two `_parse_documents` call sites at `coordinator.py:943` and `coordinator.py:945` then become:

```python
# was: self.parser.parse_with_structure(doc)  /  self.parser.parse(doc)
# becomes (unified):
parsed = self.parser.parse(doc)   # ParserRegistry.parse() dispatches by content_type
```

`HTMLStructuredParser` wraps the existing `TextParser.parse_with_structure` logic — no behaviour change for HTML documents.

**Files to modify:**
- `evidence_enrichment/pipeline/coordinator.py:149` — replace `TextParser()` with guarded `ParserRegistry` construction above
- `evidence_enrichment/pipeline/coordinator.py:943–945` — unify to single `self.parser.parse(doc)` call

### A3. Hierarchical chunker

Create `evidence_enrichment/core/retrieval/hierarchical_chunker.py`:

```python
class HierarchicalChunker:
    """Section-aware chunker. Chunks never cross section boundaries.

    Parameters
    ----------
    target_chunk_tokens:
        Target token count per content chunk sized via tiktoken (cl100k_base).
        Default 400 ≈ ~1,600 chars for financial prose.
    max_chunk_tokens:
        Hard ceiling. Default 800.
    overlap_tokens:
        Overlap between consecutive text chunks within one section. Default 80.
    keep_tables_atomic:
        Tables never split unless they exceed max_chunk_tokens * 3.
    emit_section_summaries:
        Emit one "section_summary" chunk per section: heading path + first ~200
        tokens. These are the Stage-1 routing targets in HierarchicalRetriever.
    """
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        # Walk section tree depth-first.
        # For each leaf section:
        #   1. Emit one "section_summary" chunk (heading path + first ~200 tokens)
        #   2. Emit "content" chunks that do NOT cross section boundaries
        #   3. Emit "table" chunks for each table block (atomic)
        # For each non-leaf section:
        #   1. Emit a "navigation" chunk: heading path + concatenated child headings
```

**Why section summaries:** `HierarchicalRetriever` queries only the ~50 summary chunks in Stage 1 instead of ~1,500 content chunks, then does a second focused query inside the matched sections.

### A4. Extended `Chunk` model and store metadata

Extend `evidence_enrichment/core/retrieval/models.py`:

```python
class Chunk(BaseModel):
    chunk_id: str
    document_url: str
    content_hash: str
    index: int
    content: str
    chunk_type: str = "text"
    char_count: int = 0
    # New fields — all defaulted for backward compatibility
    section_id: str = ""
    section_path_str: str = ""      # CANONICAL field — pipe-joined scalar stored in Chroma
                                    # e.g. "Financial Review|Segment Information|Revenue by Geography"
                                    # derive list on read: section_path_str.split("|") if section_path_str else []
    section_level: int = 0
    page: int | None = None
    chunk_role: Literal["section_summary", "content", "table", "navigation"] = "content"
    token_count: int = 0
    table_data: list[list[str]] | None = None
```

**There is no separate `section_path: list[str]` field on `Chunk`.** The list form is always derived in-memory via `chunk.section_path_str.split("|")` where needed. This eliminates any risk of the two representations drifting out of sync.

The formatter snippet (`_build_analysis_context`) therefore uses:

```python
section_parts = result.chunk.section_path_str.split("|") if result.chunk.section_path_str else []
section = " > ".join(section_parts) or "—"
```

**Chroma version note — `section_path` must be a scalar; no partial-match filter is available.**
Chroma's metadata filter operators for plain string fields are limited to equality comparisons (`$eq`, `$ne`) and ordering (`$gt`, `$gte`, `$lt`, `$lte`). There is **no** `$contains` operator for plain string metadata on Chroma `>=0.6.0` — `$contains` only applies to array metadata which requires Chroma ≥ 1.5.0 (not pinned here).

`section_path` is therefore stored as a **pipe-joined scalar** (`"Financial Review|Segment Information|Revenue by Geography"`) for human readability and exact-match filtering only. The in-memory `Chunk.section_path_str: str` field is both the source of truth and the Chroma metadata field. The corresponding Python list representation is derived on read via `section_path_str.split("|")`.

Practical consequence: **Chroma cannot filter by partial section path.** Stage 1 of `HierarchicalRetriever` relies on vector cosine similarity to surface the right section summaries; it does *not* push down a substring filter. Section hints in `extraction_fields.yaml` are used as reranking signals (see B4) and as exact `$eq` match when the full pipe-joined path is known, but not as a `LIKE`-style filter.

If the minimum Chroma version is bumped to ≥ 1.5.0 in a future cycle, `section_path_str` can be migrated to a native list column and a `$contains` filter can then be used for partial-path matching.

Write all new fields in `ChromaVectorStore.upsert` (`store.py:115`). The `query` method at `store.py:198` already passes `where` through to Chroma unchanged — no changes needed there.

**Analysis context formatter — surface new metadata to the LLM**

Update `_build_analysis_context` in `core/providers/agents.py:71`:

```python
# Before (shows type + score only)
chunk_label = f"[Chunk {i} | type={result.chunk.chunk_type} | score={result.score:.3f}]"

# After (shows section path and page as well)
section_parts = result.chunk.section_path_str.split("|") if result.chunk.section_path_str else []
section = " > ".join(section_parts) or "—"
page = f"p.{result.chunk.page}" if result.chunk.page else "—"
chunk_label = (
    f"[Chunk {i} | role={result.chunk.chunk_role} | "
    f"section={section} | page={page} | score={result.score:.3f}]"
)
```

This is required for the analysis LLM to reason about section provenance, not just vector scores.

**Files to create:**
- `evidence_enrichment/core/parse/base.py`
- `evidence_enrichment/core/parse/generic_text.py`
- `evidence_enrichment/core/parse/html_structured.py`
- `evidence_enrichment/core/parse/pdf_structured.py`
- `evidence_enrichment/core/retrieval/hierarchical_chunker.py`

**Files to modify:**
- `evidence_enrichment/core/models/contracts.py` — add `SectionNode`; extend `ContentBlock`, `ParsedDocument`, `RetrievedDocument`
- `evidence_enrichment/core/retrieval/models.py` — extend `Chunk`
- `evidence_enrichment/core/retrieval/store.py:115` — write new metadata in `upsert`
- `evidence_enrichment/core/providers/agents.py:71` — extend `_build_analysis_context`
- `evidence_enrichment/core/fetch/fetcher.py:24` — allow PDF, add `_MAX_PDF_BYTES`, populate `body_bytes`
- `pyproject.toml` — add `pdfplumber>=0.11`, `pymupdf>=1.24`, `tiktoken>=0.7` to optional `[retrieval]` extra

---

## Stage B — Two-Stage Section-Pruning Retrieval

**Goal:** Stop scanning every chunk. Route each query first to the right *section* (~50 comparisons), then retrieve within that section (~30 comparisons). Net: ~80 vs. ~1,500 today.

**Success criteria:**
- `HierarchicalRetriever.retrieve` issues two Chroma queries with `where` filters
- Field routing config is YAML, not code constants
- Hybrid scoring includes section-path-match and page-position terms
- Feature flag in `RetrievalConfig` is wired all the way through `coordinator._get_retriever`
- `RetrievalAgent` wrapping still works with `HierarchicalRetriever` (parity test in `tests/test_retrieval_agent.py`)
- `evals/cases.yaml` shows recall@5 ≥ current for all existing cases, plus ≥3 new section-routing cases

### B1. `HierarchicalRetriever`

Create `evidence_enrichment/core/retrieval/hierarchical_retriever.py`. The interface must match `HybridRetriever` exactly (`entity_id`, `index_document`, `retrieve`, `evict_document`) so `coordinator._get_retriever` and `RetrievalAgent` can substitute it without changes to callers.

```python
class HierarchicalRetriever:
    """Drop-in replacement for HybridRetriever with two-stage section routing.

    Interface is intentionally identical to HybridRetriever so RetrievalAgent
    and coordinator._get_retriever require no changes beyond the flag dispatch.
    """

    def retrieve(
        self,
        query: str,
        document_url: str,
        top_k: int | None = None,
        *,
        section_hint: list[str] | None = None,
        target_field: "ExtractionFieldConfig | None" = None,
    ) -> list[RetrievalResult]:

        k = top_k or self.top_k
        q_emb = self.embedder.embed_query(query)

        # ── Stage 1: section routing (~50 comparisons) ───────────────────
        section_hits = self.store.query(
            entity_id=self.entity_id,
            query_embedding=q_emb,
            top_k=8,
            where={
                "document_url": document_url,
                "chunk_role": {"$in": ["section_summary", "navigation"]},
            },
        )
        target_section_ids = self._select_sections(
            section_hits,
            section_hint=section_hint,
            field_hint=target_field.section_hints if target_field else None,
        )

        # ── Stage 2: focused content retrieval (~30 comparisons) ─────────
        if target_section_ids:
            content_hits = self.store.query(
                entity_id=self.entity_id,
                query_embedding=q_emb,
                top_k=k * 3,
                where={
                    "document_url": document_url,
                    "chunk_role": {"$in": ["content", "table"]},
                    "section_id": {"$in": target_section_ids},
                },
            )
        else:
            # Stage-1 returned no usable section ids. Two distinct root causes:
            #
            # (a) Document indexed with HierarchicalChunker but section routing
            #     scored too low — _select_sections guarantees this returns at
            #     least the top-1 hit, so this branch means zero hits came back
            #     from Stage 1 entirely.
            #
            # (b) Document indexed BEFORE hierarchical chunking — the legacy
            #     collection has no chunk_role metadata at all (store.py:115
            #     only writes document_url, chunk_type, index, char_count,
            #     content_hash), so the Stage-1 query matched nothing.
            #
            # In both cases: fall back to a full-document scan with NO chunk_role
            # filter so that legacy chunks (which have no chunk_role field) are
            # included. This is identical in behaviour to HybridRetriever, satisfying
            # recall >= current for all pre-migration documents.
            content_hits = self.store.query(
                entity_id=self.entity_id,
                query_embedding=q_emb,
                top_k=k * 3,
                where={"document_url": document_url},
                # Intentionally no chunk_role filter — must reach legacy chunks
                # that predate hierarchical indexing.
            )
        return self._hybrid_rerank(query, content_hits, k, target_field)
```

**`_select_sections` policy — never return empty when hits exist.**

```python
def _select_sections(
    self,
    section_hits: list[RetrievalResult],
    *,
    section_hint: list[str] | None,
    field_hint: list[str] | None,
    min_score: float = 0.30,
    top_n: int = 4,
) -> list[str]:
    """Return section_ids to filter Stage 2.

    Guarantee: always returns at least the top-scoring section id when any
    hit exists, regardless of score threshold, so Stage 2 is never forced
    into the full-document fallback by an arbitrarily low min_score alone.
    """
    candidates = [r for r in section_hits if r.score >= min_score]
    if not candidates and section_hits:
        # No hit clears the threshold — take the single best hit anyway
        candidates = [section_hits[0]]
    # Boost candidates whose section_path_str matches a hint
    hints = set((section_hint or []) + (field_hint or []))
    if hints:
        candidates.sort(
            key=lambda r: (
                any(h in r.chunk.section_path_str for h in hints),
                r.score,
            ),
            reverse=True,
        )
    return [r.chunk.section_id for r in candidates[:top_n] if r.chunk.section_id]
```

This means Stage 2 falls back to the full-document scan only when Chroma returns zero section-summary hits (e.g. a document that was indexed before `HierarchicalChunker` ran and contains no `chunk_role="section_summary"` entries). In that case the behaviour is identical to `HybridRetriever`, satisfying the recall ≥ current acceptance criterion.

### B2. Coordinator wiring — the feature flag must be real

Update `evidence_enrichment/pipeline/coordinator.py:234` (`_get_retriever`):

```python
def _get_retriever(self, entity_id: str) -> "HybridRetriever | HierarchicalRetriever | RetrievalAgent | None":
    rc = self.settings.retrieval
    if rc.mode not in ("local", "agent"):
        return None
    # ... embedder / store / chunker setup ...

    if rc.chunker == "hierarchical":
        from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker
        chunker = HierarchicalChunker(
            target_chunk_tokens=rc.target_chunk_tokens,
            overlap_tokens=rc.overlap_tokens,
        )
    else:
        from evidence_enrichment.core.retrieval.chunker import TableAwareChunker
        chunker = TableAwareChunker(chunk_size=rc.chunk_size, overlap=rc.overlap, ...)

    if rc.retriever == "hierarchical":
        from evidence_enrichment.core.retrieval.hierarchical_retriever import HierarchicalRetriever
        base = HierarchicalRetriever(entity_id=entity_id, store=store, embedder=embedder,
                                     chunker=chunker, top_k=rc.top_k,
                                     weights=rc.hierarchical_weights)
    else:
        from evidence_enrichment.core.retrieval.retriever import HybridRetriever
        base = HybridRetriever(entity_id=entity_id, store=store, embedder=embedder,
                               chunker=chunker, top_k=rc.top_k,
                               weights=rc.hybrid_weights)

    if rc.mode == "agent":
        from evidence_enrichment.core.retrieval.agent import RetrievalAgent
        return RetrievalAgent(base)    # RetrievalAgent wraps the base; no interface change
    return base
```

**Second dispatch site — FinOps budget gate (`coordinator.py:1416`).**
There is a second hard-wired `TableAwareChunker` instantiation inside the retrieval budget projection block at `coordinator.py:1416–1429`. This block estimates token consumption for the FinOps gate and must use the same chunker the live path will use. Apply the same flag dispatch there:

```python
# coordinator.py ~1416 — budget projection block
rc = self.settings.retrieval
if rc.chunker == "hierarchical":
    from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker
    _budget_chunker = HierarchicalChunker(target_chunk_tokens=rc.target_chunk_tokens)
else:
    from evidence_enrichment.core.retrieval.chunker import TableAwareChunker
    _budget_chunker = TableAwareChunker(chunk_size=rc.chunk_size, overlap=rc.overlap)
# rest of budget estimation unchanged — uses _budget_chunker.chunk(...)
```

`RetrievalAgent` already accepts any object with a `.retrieve()` signature (`agent.py:143`) — no changes to `RetrievalAgent` are needed.

### B3. Extended `RetrievalConfig`

Extend `evidence_enrichment/config/settings.py:302`:

```python
from pydantic import model_validator

class RetrievalConfig(BaseModel):
    # --- existing fields unchanged ---
    mode: str = "off"
    persist_path: str = "examples/output/chroma"
    chunk_size: int = 1500
    overlap: int = 200
    max_table_size: int = 4000
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    min_doc_chars: int = 2000
    # --- DEPRECATED: kept as a no-op alias so existing YAML/env configs don't error ---
    weights: tuple[float, float, float] = (0.7, 0.2, 0.1)
    # --- new fields ---
    chunker: Literal["flat", "hierarchical"] = "flat"
    retriever: Literal["hybrid", "hierarchical"] = "hybrid"
    target_chunk_tokens: int = 400
    overlap_tokens: int = 80
    schema_validation: bool = False
    schema_repair_max_attempts: int = 2
    hybrid_weights: tuple[float, float, float] = (0.7, 0.2, 0.1)
    hierarchical_weights: tuple[float, float, float, float, float] = (0.60, 0.15, 0.10, 0.10, 0.05)

    @model_validator(mode="after")
    def _coerce_chunker_retriever_consistency(self) -> "RetrievalConfig":
        """HierarchicalRetriever requires section_summary/navigation chunks that only
        HierarchicalChunker emits.  Silently promote chunker to match retriever rather
        than letting the pipeline fail at query time with an empty Stage-1 result set."""
        if self.retriever == "hierarchical" and self.chunker == "flat":
            import warnings
            warnings.warn(
                "retriever='hierarchical' requires chunker='hierarchical'; "
                "coercing chunker to 'hierarchical'.",
                stacklevel=2,
            )
            self.chunker = "hierarchical"
        return self

    @model_validator(mode="after")
    def _migrate_deprecated_weights(self) -> "RetrievalConfig":
        """If the caller set the deprecated `weights` field but left `hybrid_weights`
        at its default, copy `weights` → `hybrid_weights` and warn.  This preserves
        the effect of any existing configs that tuned `retrieval.weights` — they will
        not silently revert to defaults after the upgrade."""
        _default_hybrid = (0.7, 0.2, 0.1)
        _default_legacy  = (0.7, 0.2, 0.1)
        if self.weights != _default_legacy and self.hybrid_weights == _default_hybrid:
            import warnings
            warnings.warn(
                "retrieval.weights is deprecated; please rename to hybrid_weights. "
                f"Copying weights={self.weights} → hybrid_weights for this run.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.hybrid_weights = self.weights
        return self
```

The validator coerces `chunker="flat"` → `"hierarchical"` when `retriever="hierarchical"` is set, and emits a `UserWarning` so the promotion is visible in logs. The reverse combination (`chunker="hierarchical"`, `retriever="hybrid"`) is valid — `HybridRetriever` ignores `chunk_role` and will score all chunks normally, at the cost of slightly higher latency from the extra summary/navigation chunks.

### B4. Updated hybrid scoring

`HierarchicalRetriever` uses a 5-term scoring function (read from `rc.hierarchical_weights`):

```
score = w_vec  * vector_cosine_similarity
      + w_kw   * bm25_term_overlap
      + w_sec  * section_path_match_score    # fraction of query tokens appearing in section_path_str
      + w_tbl  * table_boost_if_numeric      # existing table boost logic
      + w_pos  * page_position_prior         # sigmoid centred on page 60% of doc.page_count
```

Default `hierarchical_weights = (0.60, 0.15, 0.10, 0.10, 0.05)`.

`HybridRetriever` scoring is unchanged — still a 3-tuple (`w_vec, w_kw, w_tbl`) read from `rc.hybrid_weights`. The `len(weights) == 3` guard at `retriever.py:162` is not modified; `HierarchicalRetriever` simply does not share that guard.

### B5. Field routing configuration

Create `evidence_enrichment/config/extraction_fields.yaml`:

```yaml
geographic_revenue:
  retrieval_query: "revenue by geography region country segment"
  section_hints:
    - "Financial Review|Segment Information"
    - "Notes to Financial Statements|Revenue by Geography"
    - "Segment Reporting"
  preferred_block_types: [table]
  top_k: 8

scope_1_emissions:
  retrieval_query: "scope 1 direct GHG greenhouse gas emissions tCO2e"
  section_hints:
    - "Sustainability|Climate|GHG Emissions"
    - "Environmental Data|Emissions Performance"
  preferred_block_types: [table, kpi]
  top_k: 6

scope_2_emissions:
  retrieval_query: "scope 2 indirect GHG electricity market location based"
  section_hints:
    - "Sustainability|Climate|GHG Emissions"
    - "Environmental Data|Emissions Performance"
  preferred_block_types: [table, kpi]
  top_k: 6

headcount_by_region:
  retrieval_query: "employees headcount workforce by country region"
  section_hints:
    - "Human Capital|Workforce|People"
    - "Sustainability|Social|Employees"
  preferred_block_types: [table]
  top_k: 6
```

Note: hints use the pipe-joined scalar format matching `section_path_str` in Chroma metadata.

**Files to create:**
- `evidence_enrichment/core/retrieval/hierarchical_retriever.py`
- `evidence_enrichment/config/extraction_fields.yaml`

**Files to modify:**
- `evidence_enrichment/config/settings.py:302` — extend `RetrievalConfig`
- `evidence_enrichment/pipeline/coordinator.py:234` — wire `chunker` / `retriever` flags in `_get_retriever`

---

## Stage C — Schema-Driven Extraction (additive, parallel artifact)

**Scope boundary:** This stage extends the repo from single-field enrichment to a general typed extraction framework. It must be validated as a separate deliverable after Stages A/B are stable.

### C-compatibility: `FactClaim` is not modified

`FactClaim.candidate_value: str` is used in:
- `core/providers/agents.py:148` — LLM output parsing
- `core/quality/gates.py:11` — confidence gate
- `observability/summarizers.py:88` — trace summarisation
- `core/analysis/replay.py:30` — replay bundle loading

**None of these are touched.** Stage C introduces a parallel `ExtractionResult` object that is produced by a new `SchemaExtractor` code path and stored as a separate artifact alongside `PipelineRunResult`. The existing claim→synthesis→review flow is unchanged.

The connection point is `PipelineRunResult`: add an optional `extraction_results: list[ExtractionResult] = Field(default_factory=list)` field. When `schema_validation=True` and the field has a registered schema, `SchemaExtractor` runs *in addition to* the existing analysis path; its output is attached to the run result but does not replace `synthesis.value`.

**Goal:** Produce a typed, validated extraction artifact for schema-registered fields. The artifact can be used by downstream consumers (dashboards, databases, API responses) that need structured data, while the existing free-form synthesis path remains the authoritative pipeline output.

**Success criteria:**
- `SCHEMA_REGISTRY` covers: `geographic_revenue`, `segment_revenue`, `scope_1_emissions`, `scope_2_emissions`, `headcount_by_region`
- `GeographicRevenueExtraction` validator rejects row sums diverging from total by >2%
- `SchemaExtractor` retries up to `schema_repair_max_attempts` times on `ValidationError`
- `FactClaim`, `SynthesisResult`, `PipelineRunResult` (existing fields) are unchanged
- All existing `evals/cases.yaml` cases pass — `schema_validation=False` by default

### C1. Schema registry

Create `evidence_enrichment/core/extraction/schemas.py`:

```python
from decimal import Decimal
from pydantic import BaseModel, Field, model_validator

class MoneyAmount(BaseModel):
    value: Decimal
    currency: str = Field(pattern=r"^[A-Z]{3}$")        # ISO 4217
    unit_multiplier: Literal[1, 1_000, 1_000_000, 1_000_000_000] = 1_000_000
    period: str | None = None                             # "FY2024", "Q3 2024"

class GeographicRevenueRow(BaseModel):
    region: str
    region_type: Literal["country", "region", "segment"]
    amount: MoneyAmount
    percentage_of_total: float | None = Field(None, ge=0.0, le=100.0)
    source_chunk_ids: list[str] = Field(min_length=1)    # provenance required

class GeographicRevenueExtraction(BaseModel):
    fiscal_year: int = Field(ge=1990, le=2100)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    rows: list[GeographicRevenueRow] = Field(min_length=1)
    total_revenue: MoneyAmount | None = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum_and_percentages(self) -> "GeographicRevenueExtraction":
        if self.total_revenue and self.total_revenue.value > 0:
            row_sum = sum(r.amount.value for r in self.rows)
            rel_err = abs(row_sum - self.total_revenue.value) / self.total_revenue.value
            if rel_err > 0.02:
                raise ValueError(
                    f"Row sum {row_sum} differs from reported total "
                    f"{self.total_revenue.value} by {rel_err:.1%} (threshold 2%)"
                )
        rows_with_pct = [r for r in self.rows if r.percentage_of_total is not None]
        if len(rows_with_pct) == len(self.rows) and self.rows:
            pct_sum = sum(r.percentage_of_total for r in rows_with_pct)  # type: ignore[arg-type]
            if not 95.0 <= pct_sum <= 105.0:
                raise ValueError(f"Percentages sum to {pct_sum:.1f}%, expected ~100%")
        return self

class EmissionsRow(BaseModel):
    scope: Literal["scope_1", "scope_2_market", "scope_2_location", "scope_3"]
    value: Decimal
    unit: Literal["tCO2e", "ktCO2e", "MtCO2e"]
    year: int = Field(ge=1990, le=2100)
    boundary: str | None = None
    source_chunk_ids: list[str] = Field(min_length=1)

class EmissionsExtraction(BaseModel):
    fiscal_year: int = Field(ge=1990, le=2100)
    rows: list[EmissionsRow] = Field(min_length=1)
    reporting_standard: str | None = None
    assurance_level: Literal["none", "limited", "reasonable"] = "none"
    extraction_confidence: float = Field(ge=0.0, le=1.0)

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "geographic_revenue": GeographicRevenueExtraction,
    "scope_1_emissions": EmissionsExtraction,
    "scope_2_emissions": EmissionsExtraction,
    # extend as fields are added
}
```

### C2. `ExtractionResult` model and `PipelineRunResult` wiring

Create `evidence_enrichment/core/extraction/models.py`:

```python
class ExtractionResult(BaseModel):
    field_name: str
    schema_cls_name: str
    schema_version: int
    value: BaseModel                      # typed extraction; not a str
    chunks_used: list[str]                # chunk_ids providing provenance
    repair_count: int = 0
    validation_passed: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    extraction_confidence: float = 0.0
```

**`PipelineRunResult` — Pydantic v2 forward-reference requirement.**

`PipelineRunResult` lives in `contracts.py`; `ExtractionResult` lives in `core/extraction/models.py`. In Pydantic v2 a string forward reference in a field annotation is not automatically resolved across module boundaries — the class must either be imported or `model_rebuild()` called after it is importable, otherwise Pydantic raises `PydanticUserError: class not fully defined`.

The correct pattern for `contracts.py`:

```python
# contracts.py — top of file
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evidence_enrichment.core.extraction.models import ExtractionResult

# ... rest of contracts.py unchanged ...

class PipelineRunResult(BaseModel):
    # ... all existing fields unchanged ...
    extraction_results: list[ExtractionResult] = Field(default_factory=list)  # NEW

# ── bottom of contracts.py ────────────────────────────────────────────────────
# Resolve forward references for models that depend on types defined outside
# this module. Must be called after all dependent modules are importable.
# schema_validation=False by default so this import is always safe (no optional deps).
from evidence_enrichment.core.extraction.models import ExtractionResult  # noqa: E402
PipelineRunResult.model_rebuild()
```

The `from __future__ import annotations` at the top makes all annotations strings by default (deferred evaluation), so the `if TYPE_CHECKING` block satisfies type checkers without triggering a circular import at load time. The bare `from ... import ExtractionResult` at the bottom of the file resolves the reference at module load, and `model_rebuild()` finalises the schema. This is the standard Pydantic v2 pattern for cross-module forward references.

**Circular import risk:** `extraction/models.py` must not import from `contracts.py` (it only uses `BaseModel` from pydantic directly). Verify this before implementing.

### C3. `SchemaExtractor`

Create `evidence_enrichment/core/extraction/extractor.py`. It is invoked from the coordinator *after* the existing analysis step, not instead of it. `schema_validation=False` by default — callers that have not opted in never instantiate it.

### C4. `SchemaValidationGate`

Extend `evidence_enrichment/core/quality/gates.py` with an additive class. It operates on `ExtractionResult`, not on `FactClaim`:

```python
class SchemaValidationGate:
    HARD_FAIL_ERRORS = {"row_sum_divergence", "missing_provenance"}
    SOFT_FAIL_ERRORS = {"percentage_sum_off", "missing_period", "unit_mismatch"}
    CONFIDENCE_PENALTY = 0.15

    def check(self, result: ExtractionResult) -> GateResult:
        # Hard fail → mark validation_passed=False, emit warning
        # Soft fail → apply confidence penalty, annotate validation_errors
        # Pass → return unchanged
```

The existing `QualityGate` at `gates.py:11` is not modified.

**Files to create:**
- `evidence_enrichment/core/extraction/__init__.py`
- `evidence_enrichment/core/extraction/schemas.py`
- `evidence_enrichment/core/extraction/extractor.py`
- `evidence_enrichment/core/extraction/repair.py`
- `evidence_enrichment/core/extraction/models.py`

**Files to modify:**
- `evidence_enrichment/core/models/contracts.py` — add `extraction_results` to `PipelineRunResult`
- `evidence_enrichment/core/quality/gates.py` — add `SchemaValidationGate` class
- `evidence_enrichment/pipeline/coordinator.py` — invoke `SchemaExtractor` when `schema_validation=True`

---

## Stage D — Migration, Evals & Acceptance

**Goal:** Keep the existing pipeline running while the new stack is validated. Flip defaults only after evals confirm improvement.

### D1. Collection versioning

`store.py` already has `_SCHEMA_VERSION = 1`. Bump to `2` for the hierarchical schema. The new collection name is `entity_{id}__{model}_v2`. Old `_v1` collections remain untouched until migration is confirmed.

Migration steps:
1. Re-parse all accepted documents using `PDFStructuredParser` / `HTMLStructuredParser`.
2. Re-chunk with `HierarchicalChunker`. Write to `_v2` collection.
3. Run side-by-side on `evals/cases.yaml`:
   - Measure recall@5, MRR, schema-valid extraction rate, p95 latency.
   - Acceptance threshold: recall@5 ≥ current, schema-valid rate ≥ 80%, latency ≤ 1.5× current.
4. Flip `RetrievalConfig` defaults to `chunker="hierarchical"`, `retriever="hierarchical"`.
5. Delete `_v1` collections and remove the `"flat"` code path.

### D2. New eval cases

Add to `evals/cases.yaml`:

```yaml
- id: geo_revenue_msft_fy24
  entity_id: microsoft
  field_name: geographic_revenue
  document_url: "<annual-report-pdf-url>"
  expected_schema: GeographicRevenueExtraction
  expected_rows_min: 3
  expected_total_currency: USD
  note: "Tests fetch→PDF→section routing→schema row-sum validation"

- id: scope1_msft_fy24
  entity_id: microsoft
  field_name: scope_1_emissions
  document_url: "<sustainability-report-pdf-url>"
  expected_schema: EmissionsExtraction
  expected_unit: tCO2e
  note: "Tests PDF heading detection and emissions section routing"

- id: section_routing_html
  entity_id: microsoft
  field_name: hq_country
  document_url: "<10-K-html-url>"
  note: "Regression: existing field must still work under hierarchical retriever"
```

### D3. `RetrievalAgent` parity

`RetrievalAgent` (`core/retrieval/agent.py:143`) wraps any object with `.retrieve()`. Because `HierarchicalRetriever` exposes the same interface as `HybridRetriever`, no changes to `RetrievalAgent` are needed. The parity test in `tests/test_retrieval_agent.py` should be parameterised to run against both retriever implementations.

---

## Effort Summary

| Stage | Deliverable | Effort |
|---|---|---|
| A0 | Fetch layer: PDF bytes, `RetrievedDocument.body_bytes`, size cap | ~2 hrs |
| A1–A4 | Hierarchical doc model, parsers, chunker, store+formatter wiring | ~10 hrs |
| B | Two-stage retriever, coordinator wiring, `RetrievalConfig` extension | ~6 hrs |
| C | Schema registry, extractor, repair, additive `ExtractionResult` artifact | ~8 hrs |
| D | Migration, eval cases, `RetrievalAgent` parity tests | ~4 hrs |
| **Total** | | **~30 hrs** |

---

## Dependency additions

| Package | Why | Optional extra |
|---|---|---|
| `pdfplumber>=0.11` | PDF text+bbox+page extraction | `[retrieval]` |
| `pymupdf>=1.24` | PDF bookmark/TOC extraction (heading source) | `[retrieval]` |
| `tiktoken>=0.7` | Token-based chunk sizing (cl100k_base) | `[retrieval]` |

All are optional. The existing HTML retrieval path is fully functional without them. `chromadb` minimum version stays at `>=0.6.0` (scalar metadata only; array filter upgrade deferred).
