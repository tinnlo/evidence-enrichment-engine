"""Tests for hierarchical chunking/retrieval paths.

Covers:
- F1: ChromaVectorStore round-trip preserves all hierarchical Chunk fields
- F2: HTMLStructuredParser stamps block_id on ContentBlocks and wires block_ids / children_ids on SectionNodes
- F2: PDFStructuredParser (heading-fallback path) wires block_ids / children_ids
- F3: Repeated headings produce distinct section_ids (no collision / overwrite)
- F4: Non-leaf section block_ids are emitted as content chunks (intro content not dropped)
- F5/F8-F9: HierarchicalRetriever Stage-2 expands selected section_ids to all descendants;
            multi-key Chroma where filters use $and; section-scoped path confirmed end-to-end
- F6: ChromaVectorStore round-trips token_count and table_data (JSON-serialised)
- F10: _descendant_map rebuilt lazily from persisted chunk metadata on fresh retriever instances
- F11: parent_section_id persisted in Chroma; fresh-instance rebuild uses exact parent edges,
       not path-prefix heuristic — fixes repeated-heading cross-contamination and root ancestry
- HierarchicalChunker: produces correct chunk roles and section metadata from a section tree
- HierarchicalChunker: flat fallback when no section tree present
- HierarchicalRetriever._select_sections: always includes top-1 regardless of min_score
"""

from __future__ import annotations

import hashlib
import tempfile

import pytest

from evidence_enrichment.core.models.contracts import (
    ContentBlock,
    ParsedDocument,
    RetrievedDocument,
    SectionNode,
)
from evidence_enrichment.core.parse.html_structured import HTMLStructuredParser
from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker
from evidence_enrichment.core.retrieval.models import Chunk
from evidence_enrichment.core.retrieval.store import ChromaVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embed(n: int, dim: int = 8) -> list[list[float]]:
    vecs = []
    for i in range(n):
        v = [0.0] * dim
        v[i % dim] = 1.0
        vecs.append(v)
    return vecs


def _make_hierarchical_chunk(index: int, **overrides) -> Chunk:
    content = overrides.pop("content", f"content for chunk {index}")
    ch = hashlib.sha256(content.encode()).hexdigest()[:16]
    defaults = dict(
        chunk_id=Chunk.make_id("http://ex.com/doc", index, ch),
        document_url="http://ex.com/doc",
        content_hash=ch,
        index=index,
        content=content,
        section_id="sec1",
        section_path_str="Financial Review|Revenue",
        section_level=2,
        chunk_role="section_summary",
        page=5,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def _make_section_doc() -> ParsedDocument:
    """Return a ParsedDocument with a two-level section tree, fully wired."""
    root = SectionNode(section_id="root", heading="", level=0, path=[], children_ids=["s1", "s2"])
    s1 = SectionNode(
        section_id="s1", heading="Revenue", level=1, path=["Revenue"],
        parent_id="root", block_ids=["b1", "b2"],
    )
    s2 = SectionNode(
        section_id="s2", heading="Risk Factors", level=1, path=["Risk Factors"],
        parent_id="root", block_ids=["b3"],
    )
    b1 = ContentBlock(block_id="b1", block_type="text", content="Revenue grew 12% YoY.", section_id="s1")
    b2 = ContentBlock(block_id="b2", block_type="table", content="Q1 | Q2\n1.2 | 1.5",
                      section_id="s1", table_data=[["Q1", "Q2"], ["1.2", "1.5"]])
    b3 = ContentBlock(block_id="b3", block_type="text", content="Market risk may affect results.", section_id="s2")
    return ParsedDocument(
        url="http://ex.com/ar.html",
        title="AR",
        content_type="text/html",
        text="",
        excerpt="",
        sections=[root, s1, s2],
        section_tree_root="root",
        blocks=[b1, b2, b3],
    )


# ---------------------------------------------------------------------------
# F1: Store round-trip
# ---------------------------------------------------------------------------

class TestStoreRoundTrip:
    def test_hierarchical_fields_persisted_and_hydrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            entity = "test_entity"
            chunk = _make_hierarchical_chunk(0)
            store.upsert(entity, [chunk], _fake_embed(1))

            hits = store.query(entity, _fake_embed(1)[0], top_k=1,
                               where={"document_url": "http://ex.com/doc"})
            assert hits, "query returned no hits"
            c = hits[0].chunk
            assert c.section_id == "sec1"
            assert c.section_path_str == "Financial Review|Revenue"
            assert c.section_level == 2
            assert c.chunk_role == "section_summary"
            assert c.page == 5

    def test_legacy_chunk_missing_hierarchical_fields_defaults_gracefully(self):
        """A chunk stored without hierarchical metadata must still deserialize."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            entity = "legacy_entity"
            # Build a legacy chunk (no hierarchical fields in metadata)
            # by manually upserting with minimal metadata via the collection directly
            col = store._get_collection(entity)
            col.upsert(
                ids=["leg_001"],
                embeddings=[_fake_embed(1)[0]],
                documents=["some legacy content"],
                metadatas=[{
                    "document_url": "http://ex.com/old",
                    "chunk_type": "text",
                    "index": 0,
                    "char_count": 20,
                    "content_hash": "abc",
                }],
            )
            hits = store.query(entity, _fake_embed(1)[0], top_k=1,
                               where={"document_url": "http://ex.com/old"})
            assert hits
            c = hits[0].chunk
            assert c.section_id == ""
            assert c.section_path_str == ""
            assert c.chunk_role == "content"
            assert c.page is None

    def test_page_none_not_stored_as_metadata_key(self):
        """page=None chunks must not write 'page' into Chroma metadata (Chroma rejects None values)."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            ch = hashlib.sha256(b"x").hexdigest()[:16]
            chunk = Chunk(
                chunk_id=Chunk.make_id("http://ex.com/doc", 0, ch),
                document_url="http://ex.com/doc",
                content_hash=ch, index=0, content="hello",
                page=None,  # must be omitted from metadata dict
            )
            # Should not raise
            store.upsert("ent", [chunk], _fake_embed(1))


# ---------------------------------------------------------------------------
# F2: HTMLStructuredParser block / section wiring
# ---------------------------------------------------------------------------

class TestHTMLStructuredParserWiring:
    _HTML = """
    <h1>Revenue</h1>
    <p>Revenue grew 12% YoY.</p>
    <p>EMEA contributed 40%.</p>
    <h1>Risk Factors</h1>
    <p>Market volatility may affect results.</p>
    """

    def _parse(self) -> ParsedDocument:
        doc = RetrievedDocument(
            url="http://ex.com/ar.html", final_url="http://ex.com/ar.html",
            title="AR", content_type="text/html", body=self._HTML, provider="test",
        )
        return HTMLStructuredParser().parse(doc)

    def test_leaf_sections_have_block_ids(self):
        parsed = self._parse()
        leaf_sections = [s for s in parsed.sections if s.level == 1]
        assert leaf_sections, "no level-1 sections produced"
        for s in leaf_sections:
            assert s.block_ids, f"section {s.heading!r} has empty block_ids"

    def test_root_has_children_ids(self):
        parsed = self._parse()
        root = next(s for s in parsed.sections if s.level == 0)
        assert root.children_ids, "root.children_ids is empty"
        assert len(root.children_ids) == 2

    def test_non_heading_blocks_have_block_id(self):
        parsed = self._parse()
        for b in parsed.blocks:
            if b.block_type != "heading":
                assert b.block_id, f"block {b.content[:30]!r} missing block_id"

    def test_hierarchical_chunker_uses_section_tree(self):
        parsed = self._parse()
        chunks = HierarchicalChunker().chunk(parsed)
        roles = {c.chunk_role for c in chunks}
        assert "section_summary" in roles, f"no section_summary; roles={roles}"
        assert "content" in roles, f"no content chunks; roles={roles}"

    def test_chunks_carry_section_path(self):
        parsed = self._parse()
        chunks = HierarchicalChunker().chunk(parsed)
        content_chunks = [c for c in chunks if c.chunk_role == "content"]
        for c in content_chunks:
            assert c.section_path_str, "content chunk missing section_path_str"


# ---------------------------------------------------------------------------
# F2: PDFStructuredParser heading-fallback wiring (no pdfplumber needed)
# ---------------------------------------------------------------------------

_PDF_SKIP = pytest.mark.skipif(
    __import__("importlib").util.find_spec("fitz") is None,
    reason="pymupdf not installed (optional dep)",
)


@_PDF_SKIP
class TestPDFHeadingFallbackWiring:
    """Test _build_sections_from_headings directly (no PDF bytes required)."""

    def test_children_ids_wired(self):
        from evidence_enrichment.core.parse.pdf_structured import _build_sections_from_headings
        blocks = [
            ContentBlock(block_id="b0", block_type="heading", content="Revenue", page=1),
            ContentBlock(block_id="b1", block_type="text", content="Revenue detail.", page=1),
            ContentBlock(block_id="b2", block_type="heading", content="Risk", page=2),
            ContentBlock(block_id="b3", block_type="text", content="Risk detail.", page=2),
        ]
        sections = _build_sections_from_headings("http://ex.com/doc", blocks)
        root = next(s for s in sections if s.level == 0)
        assert len(root.children_ids) == 2, f"expected 2 children, got {root.children_ids}"

    def test_block_ids_assigned_to_sections(self):
        from evidence_enrichment.core.parse.pdf_structured import _build_sections_from_headings
        blocks = [
            ContentBlock(block_id="b0", block_type="heading", content="Revenue", page=1),
            ContentBlock(block_id="b1", block_type="text", content="Revenue detail.", page=1),
            ContentBlock(block_id="b2", block_type="table", content="Q1|Q2", page=1),
            ContentBlock(block_id="b3", block_type="heading", content="Risk", page=2),
            ContentBlock(block_id="b4", block_type="text", content="Risk detail.", page=2),
        ]
        sections = _build_sections_from_headings("http://ex.com/doc", blocks)
        rev = next(s for s in sections if s.heading == "Revenue")
        risk = next(s for s in sections if s.heading == "Risk")
        assert "b1" in rev.block_ids
        assert "b2" in rev.block_ids
        assert "b4" in risk.block_ids


@_PDF_SKIP
class TestPDFTOCWiring:
    def test_toc_children_ids_wired(self):
        from evidence_enrichment.core.parse.pdf_structured import _build_sections_from_toc
        toc = [(1, "Financial Highlights", 1), (2, "Revenue", 2), (1, "Risk Factors", 5)]
        sections = _build_sections_from_toc("http://ex.com/doc", toc, page_count=10)
        root = next(s for s in sections if s.level == 0)
        assert len(root.children_ids) == 2, f"expected 2 top-level children, got {root.children_ids}"
        fin = next(s for s in sections if s.heading == "Financial Highlights")
        assert any(
            s.heading == "Revenue" for s in sections if s.section_id in fin.children_ids
        ), "Revenue should be child of Financial Highlights"

    def test_toc_block_ids_assigned_by_page(self):
        from evidence_enrichment.core.parse.pdf_structured import _build_sections_from_toc
        blocks = [
            ContentBlock(block_id="b0", block_type="text", content="Revenue detail.", page=2),
            ContentBlock(block_id="b1", block_type="text", content="Risk detail.", page=5),
        ]
        toc = [(1, "Revenue", 2), (1, "Risk Factors", 5)]
        sections = _build_sections_from_toc("http://ex.com/doc", toc, page_count=10, blocks=blocks)
        rev = next(s for s in sections if s.heading == "Revenue")
        risk = next(s for s in sections if s.heading == "Risk Factors")
        assert "b0" in rev.block_ids, f"b0 should be in Revenue, got {rev.block_ids}"
        assert "b1" in risk.block_ids, f"b1 should be in Risk Factors, got {risk.block_ids}"


# ---------------------------------------------------------------------------
# HierarchicalChunker
# ---------------------------------------------------------------------------

class TestHierarchicalChunker:
    def test_produces_section_summary_and_content_chunks(self):
        doc = _make_section_doc()
        chunks = HierarchicalChunker().chunk(doc)
        roles = {c.chunk_role for c in chunks}
        assert "section_summary" in roles
        assert "content" in roles

    def test_table_gets_table_role(self):
        doc = _make_section_doc()
        chunks = HierarchicalChunker().chunk(doc)
        table_chunks = [c for c in chunks if c.chunk_role == "table"]
        assert table_chunks, "no table chunks produced"
        assert all(c.chunk_type == "table" for c in table_chunks)

    def test_section_path_str_populated(self):
        doc = _make_section_doc()
        chunks = HierarchicalChunker().chunk(doc)
        for c in chunks:
            if c.chunk_role in ("section_summary", "content", "table"):
                assert c.section_path_str, f"chunk role={c.chunk_role} missing section_path_str"

    def test_chunk_ids_are_unique(self):
        doc = _make_section_doc()
        chunks = HierarchicalChunker().chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "duplicate chunk_ids produced"

    def test_flat_fallback_when_no_sections(self):
        """No sections → falls back to TableAwareChunker, still returns Chunks."""
        doc = ParsedDocument(
            url="http://ex.com/doc", title="T", content_type="text/plain",
            text="hello world " * 200, excerpt="",
            blocks=[ContentBlock(block_type="text", content="hello world " * 200)],
        )
        chunks = HierarchicalChunker().chunk(doc)
        assert chunks, "flat fallback returned no chunks"
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_section_aware_chunks_carry_section_id_from_structured_html(self):
        """HierarchicalChunker uses the section tree: every chunk must have a
        section_id that matches one of the sections in the ParsedDocument.

        This guards against regressions where the chunker silently falls back to
        _flat_chunk() even when a section tree is present (sections != []).
        """
        doc = _make_section_doc()
        assert doc.sections, "fixture must have sections"

        chunks = HierarchicalChunker().chunk(doc)
        valid_ids = {s.section_id for s in doc.sections}

        # Every chunk must carry a section_id referencing a known section.
        for c in chunks:
            assert c.section_id in valid_ids, (
                f"chunk role={c.chunk_role!r} has section_id={c.section_id!r} "
                f"which is not in the section tree {valid_ids}; "
                "HierarchicalChunker may have fallen back to _flat_chunk() unexpectedly."
            )

        # Both leaf sections must be represented.
        chunk_sids = {c.section_id for c in chunks}
        assert "s1" in chunk_sids, "section 's1' (Revenue) produced no chunks"
        assert "s2" in chunk_sids, "section 's2' (Risk Factors) produced no chunks"


# ---------------------------------------------------------------------------
# HierarchicalRetriever._select_sections
# ---------------------------------------------------------------------------

class TestSelectSections:
    def _make_retriever(self):
        from unittest.mock import MagicMock
        from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever
        return HierarchicalRetriever("ent", MagicMock(), MagicMock(), section_min_score=0.5)

    def test_always_includes_top1_even_below_threshold(self):
        r = self._make_retriever()
        from unittest.mock import MagicMock
        h1 = MagicMock()
        h1.vector_score = 0.1
        h1.chunk.section_id = "s_low"
        h2 = MagicMock()
        h2.vector_score = 0.9
        h2.chunk.section_id = "s_high"
        # h1 is top-1 (first in list), below min_score=0.5
        selected = r._select_sections([h1, h2])
        assert "s_low" in selected, "top-1 must be included regardless of min_score"

    def test_high_score_hits_included(self):
        r = self._make_retriever()
        from unittest.mock import MagicMock
        h1 = MagicMock()
        h1.vector_score = 0.9
        h1.chunk.section_id = "s1"
        h2 = MagicMock()
        h2.vector_score = 0.8
        h2.chunk.section_id = "s2"
        selected = r._select_sections([h1, h2])
        assert "s1" in selected
        assert "s2" in selected

    def test_below_threshold_hits_excluded_except_top1(self):
        r = self._make_retriever()
        from unittest.mock import MagicMock
        h1 = MagicMock()
        h1.vector_score = 0.9
        h1.chunk.section_id = "s1"
        h2 = MagicMock()
        h2.vector_score = 0.1
        h2.chunk.section_id = "s2"
        selected = r._select_sections([h1, h2])
        assert "s1" in selected
        assert "s2" not in selected

    def test_empty_hits_returns_empty(self):
        r = self._make_retriever()
        assert r._select_sections([]) == []

    def test_empty_section_id_filtered_out(self):
        r = self._make_retriever()
        from unittest.mock import MagicMock
        h1 = MagicMock()
        h1.vector_score = 0.9
        h1.chunk.section_id = ""
        selected = r._select_sections([h1])
        assert selected == [], f"empty section_id should be filtered: {selected}"


# ---------------------------------------------------------------------------
# F3: Repeated headings must produce distinct section_ids
# ---------------------------------------------------------------------------

class TestRepeatedHeadingIds:
    _HTML_REPEATED = """
    <h1>Overview</h1>
    <p>First overview paragraph.</p>
    <h1>Revenue</h1>
    <p>Revenue detail.</p>
    <h1>Overview</h1>
    <p>Second overview paragraph.</p>
    """

    def _parse(self) -> ParsedDocument:
        doc = RetrievedDocument(
            url="http://ex.com/ar.html", final_url="http://ex.com/ar.html",
            title="AR", content_type="text/html", body=self._HTML_REPEATED, provider="test",
        )
        return HTMLStructuredParser().parse(doc)

    def test_all_section_ids_are_distinct(self):
        parsed = self._parse()
        ids = [s.section_id for s in parsed.sections]
        assert len(ids) == len(set(ids)), f"duplicate section_ids: {ids}"

    def test_two_overview_sections_both_present(self):
        parsed = self._parse()
        overviews = [s for s in parsed.sections if s.heading == "Overview"]
        assert len(overviews) == 2, f"expected 2 Overview sections, got {len(overviews)}"

    def test_each_overview_has_its_own_block(self):
        parsed = self._parse()
        overviews = [s for s in parsed.sections if s.heading == "Overview"]
        assert overviews[0].block_ids != [] and overviews[1].block_ids != []
        # They must reference different blocks
        assert set(overviews[0].block_ids).isdisjoint(set(overviews[1].block_ids)), \
            "both Overview sections share block_ids — content will be duplicated"

    def test_hierarchical_chunker_emits_both_overview_chunks(self):
        parsed = self._parse()
        chunks = HierarchicalChunker().chunk(parsed)
        # Each Overview section should produce at least a section_summary chunk
        overview_summaries = [
            c for c in chunks
            if "Overview" in c.section_path_str and c.chunk_role == "section_summary"
        ]
        assert len(overview_summaries) == 2, \
            f"expected 2 section_summary chunks for Overview, got {len(overview_summaries)}"

    @_PDF_SKIP
    def test_pdf_heading_fallback_ids_distinct(self):
        from evidence_enrichment.core.parse.pdf_structured import _build_sections_from_headings
        blocks = [
            ContentBlock(block_id="b0", block_type="heading", content="Overview", page=1),
            ContentBlock(block_id="b1", block_type="text", content="First.", page=1),
            ContentBlock(block_id="b2", block_type="heading", content="Overview", page=3),
            ContentBlock(block_id="b3", block_type="text", content="Second.", page=3),
        ]
        sections = _build_sections_from_headings("http://ex.com/doc", blocks)
        ids = [s.section_id for s in sections]
        assert len(ids) == len(set(ids)), f"PDF heading-fallback produced duplicate ids: {ids}"
        overviews = [s for s in sections if s.heading == "Overview"]
        assert len(overviews) == 2

    @_PDF_SKIP
    def test_pdf_toc_ids_distinct(self):
        from evidence_enrichment.core.parse.pdf_structured import _build_sections_from_toc
        toc = [(1, "Overview", 1), (1, "Revenue", 5), (1, "Overview", 10)]
        sections = _build_sections_from_toc("http://ex.com/doc", toc, page_count=15)
        ids = [s.section_id for s in sections]
        assert len(ids) == len(set(ids)), f"PDF TOC produced duplicate ids: {ids}"
        overviews = [s for s in sections if s.heading == "Overview"]
        assert len(overviews) == 2


# ---------------------------------------------------------------------------
# F4: Non-leaf section block_ids must be emitted (intro content not dropped)
# ---------------------------------------------------------------------------

class TestNonLeafBlocksEmitted:
    _HTML_INTRO = """
    <p>This is the document introduction.</p>
    <h1>Revenue</h1>
    <p>Revenue grew 12% YoY.</p>
    """

    def test_intro_before_first_heading_appears_in_chunks(self):
        """Content before the first heading is owned by root (level 0). It should
        be emitted as content chunks, not silently dropped."""
        doc = RetrievedDocument(
            url="http://ex.com/ar.html", final_url="http://ex.com/ar.html",
            title="AR", content_type="text/html", body=self._HTML_INTRO, provider="test",
        )
        parsed = HTMLStructuredParser().parse(doc)
        chunks = HierarchicalChunker().chunk(parsed)
        all_content = "\n".join(c.content for c in chunks)
        assert "document introduction" in all_content, \
            f"intro paragraph missing from chunks. Chunk content:\n{all_content}"

    def test_parent_section_blocks_emitted_alongside_children(self):
        """A non-leaf SectionNode with both block_ids and children_ids must
        produce content chunks for its own blocks as well as recursing into children."""
        root = SectionNode(
            section_id="root", heading="", level=0, path=[],
            children_ids=["parent"],
        )
        parent = SectionNode(
            section_id="parent", heading="Financial Review", level=1,
            path=["Financial Review"], parent_id="root",
            children_ids=["child"],
            block_ids=["intro_block"],   # intro text on the parent section itself
        )
        child = SectionNode(
            section_id="child", heading="Revenue", level=2,
            path=["Financial Review", "Revenue"], parent_id="parent",
            block_ids=["child_block"],
        )
        intro = ContentBlock(
            block_id="intro_block", block_type="text",
            content="This section covers all financial results.",
        )
        detail = ContentBlock(
            block_id="child_block", block_type="text",
            content="Revenue grew 12% YoY.",
        )
        doc = ParsedDocument(
            url="http://ex.com/doc", title="T", content_type="text/html",
            text="", excerpt="",
            sections=[root, parent, child],
            section_tree_root="root",
            blocks=[intro, detail],
        )
        chunks = HierarchicalChunker().chunk(doc)
        all_content = "\n".join(c.content for c in chunks)
        assert "all financial results" in all_content, \
            "parent-section intro block missing from chunks"
        assert "Revenue grew" in all_content, \
            "child-section block missing from chunks"

    def test_html_root_blocks_not_lost_when_sections_present(self):
        """Blocks assigned to root (level=0) when sections exist must still be emitted."""
        parsed = HTMLStructuredParser().parse(
            RetrievedDocument(
                url="http://ex.com/ar.html", final_url="http://ex.com/ar.html",
                title="AR", content_type="text/html", body=self._HTML_INTRO, provider="test",
            )
        )
        # Confirm root has blocks
        root = next(s for s in parsed.sections if s.level == 0)
        assert root.block_ids, "root should own the intro block"


# ---------------------------------------------------------------------------
# F6: ChromaVectorStore must round-trip token_count and table_data
# ---------------------------------------------------------------------------

class TestTokenCountTableDataRoundTrip:
    """token_count and table_data must survive upsert → query unchanged."""

    def _make_chunk(self, idx: int, content: str, **kwargs) -> Chunk:
        ch = hashlib.sha256(content.encode()).hexdigest()[:16]
        return Chunk(
            chunk_id=Chunk.make_id("http://ex.com/r", idx, ch),
            document_url="http://ex.com/r",
            content_hash=ch,
            index=idx,
            content=content,
            **kwargs,
        )

    def test_token_count_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            chunk = self._make_chunk(0, "hello world", token_count=42)
            store.upsert("e", [chunk], [[0.1] * 8])
            results = store.query("e", [0.1] * 8, top_k=1)
            assert results, "expected at least one result"
            assert results[0].chunk.token_count == 42, \
                f"token_count not round-tripped: got {results[0].chunk.token_count}"

    def test_table_data_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            table = [["Header A", "Header B"], ["row1a", "row1b"], ["row2a", "row2b"]]
            chunk = self._make_chunk(0, "table content", chunk_type="table", table_data=table)
            store.upsert("e", [chunk], [[0.2] * 8])
            results = store.query("e", [0.2] * 8, top_k=1)
            assert results, "expected at least one result"
            assert results[0].chunk.table_data == table, \
                f"table_data not round-tripped: got {results[0].chunk.table_data}"

    def test_table_data_none_round_trips(self):
        """Chunks without table_data should come back with table_data=None."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            chunk = self._make_chunk(0, "plain text", chunk_type="text")
            store.upsert("e", [chunk], [[0.3] * 8])
            results = store.query("e", [0.3] * 8, top_k=1)
            assert results, "expected at least one result"
            assert results[0].chunk.table_data is None, \
                f"table_data should be None for non-table chunk, got {results[0].chunk.table_data}"

    def test_token_count_zero_default_for_legacy_chunks(self):
        """Legacy chunks without token_count in metadata should hydrate as 0."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            # Manually insert a chunk with no token_count key in metadata
            col = store._get_collection("e")
            col.upsert(
                ids=["legacy1"],
                embeddings=[[0.4] * 8],
                documents=["legacy content"],
                metadatas=[{
                    "document_url": "http://ex.com/r",
                    "chunk_type": "text",
                    "index": 0,
                    "char_count": 14,
                    "content_hash": "abc123",
                }],
            )
            results = store.query("e", [0.4] * 8, top_k=1)
            assert results, "expected at least one result"
            assert results[0].chunk.token_count == 0, \
                f"legacy chunk token_count should default to 0, got {results[0].chunk.token_count}"
            assert results[0].chunk.table_data is None, \
                f"legacy chunk table_data should be None, got {results[0].chunk.table_data}"


# ---------------------------------------------------------------------------
# F5: Stage-2 must expand selected section_ids to include all descendants
# ---------------------------------------------------------------------------

class TestDescendantMap:
    def test_build_descendant_map_leaf(self):
        from evidence_enrichment.core.retrieval.retriever import _build_descendant_map
        root = SectionNode(section_id="root", heading="", level=0, path=[], children_ids=["parent"])
        parent = SectionNode(section_id="parent", heading="Fin", level=1, path=["Fin"],
                             parent_id="root", children_ids=["child"])
        child = SectionNode(section_id="child", heading="Revenue", level=2,
                            path=["Fin", "Revenue"], parent_id="parent")
        dm = _build_descendant_map([root, parent, child])
        assert dm["root"] == frozenset({"root", "parent", "child"})
        assert dm["parent"] == frozenset({"parent", "child"})
        assert dm["child"] == frozenset({"child"})

    def test_build_descendant_map_flat(self):
        from evidence_enrichment.core.retrieval.retriever import _build_descendant_map
        # Flat tree: root with two independent children
        root = SectionNode(section_id="root", heading="", level=0, path=[],
                           children_ids=["s1", "s2"])
        s1 = SectionNode(section_id="s1", heading="A", level=1, path=["A"], parent_id="root")
        s2 = SectionNode(section_id="s2", heading="B", level=1, path=["B"], parent_id="root")
        dm = _build_descendant_map([root, s1, s2])
        assert dm["root"] == frozenset({"root", "s1", "s2"})
        assert dm["s1"] == frozenset({"s1"})
        assert dm["s2"] == frozenset({"s2"})

    def test_build_descendant_map_from_chunks_prefix(self):
        from evidence_enrichment.core.retrieval.retriever import _build_descendant_map_from_chunks
        import hashlib
        def _chunk(sid, path_str):
            ch = hashlib.sha256(sid.encode()).hexdigest()[:16]
            return Chunk(chunk_id=sid, document_url="u", content_hash=ch,
                         index=0, content="x", section_id=sid, section_path_str=path_str)
        chunks = [
            _chunk("p", "Financial Review"),
            _chunk("c1", "Financial Review|Revenue"),
            _chunk("c2", "Financial Review|Costs"),
            _chunk("u", "Unrelated"),
        ]
        dm = _build_descendant_map_from_chunks(chunks)
        assert dm["p"] == frozenset({"p", "c1", "c2"})
        assert dm["c1"] == frozenset({"c1"})
        assert dm["u"] == frozenset({"u"})

    def test_stage2_expands_parent_to_children(self):
        """When Stage 1 selects a parent section_id, Stage 2 must query all descendant
        section_ids so leaf content is not missed — and the section-scoped path must
        actually have executed (no silent fallback to whole-document retrieval).

        Protocol:
        1. Index a 3-node tree (root → parent → child).
        2. Replace store.query with a combined spy+stub:
           - Stage-1 call (chunk_role filter): returns only the parent summary.
           - Stage-2 calls (section_id filter): pass through to real Chroma AND
             record every section_id that was requested.
           - Whole-document fallback call (document_url only, no section_id):
             records that the fallback fired.
        3. Call retrieve() end-to-end.
        4. Assert Stage-1 ran (spy saw a chunk_role filter).
        5. Assert the whole-document fallback did NOT fire (section-scoped path worked).
        6. Assert Stage-2 queried both 'parent' AND 'child' section_ids.
        7. Assert child leaf content appears in the final results.
        """
        import tempfile
        from unittest.mock import MagicMock
        from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever
        from evidence_enrichment.core.retrieval.models import RetrievalResult

        # --- Build doc with root → parent → child ---
        root = SectionNode(section_id="root", heading="", level=0, path=[],
                           children_ids=["parent"])
        parent = SectionNode(section_id="parent", heading="Financial Review", level=1,
                             path=["Financial Review"], parent_id="root",
                             children_ids=["child"], block_ids=["intro_b"])
        child = SectionNode(section_id="child", heading="Revenue", level=2,
                            path=["Financial Review", "Revenue"], parent_id="parent",
                            block_ids=["rev_b"])
        intro = ContentBlock(block_id="intro_b", block_type="text",
                             content="This section covers financials.")
        detail = ContentBlock(block_id="rev_b", block_type="text",
                              content="Revenue grew 12% YoY.")
        doc = ParsedDocument(
            url="http://ex.com/doc", title="T", content_type="text/html",
            text="", excerpt="",
            sections=[root, parent, child],
            section_tree_root="root",
            blocks=[intro, detail],
        )

        with tempfile.TemporaryDirectory() as tmp:
            from evidence_enrichment.core.retrieval.store import ChromaVectorStore
            from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker

            store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")

            import hashlib as _hs
            def _det_embed(texts):
                vecs = []
                for t in texts:
                    h = int(_hs.sha256(t.encode()).hexdigest(), 16)
                    v = [float((h >> (i * 4)) & 0xF) / 16.0 for i in range(8)]
                    vecs.append(v)
                return vecs

            embedder = MagicMock()
            embedder.embed_texts.side_effect = _det_embed
            embedder.embed_query.return_value = _det_embed(["Revenue grew"])[0]

            retriever = HierarchicalRetriever(
                entity_id="ent",
                store=store,
                embedder=embedder,
                chunker=HierarchicalChunker(),
                top_k=5,
                section_top_k=3,
                section_min_score=0.0,
            )
            indexed_chunks = retriever.index_document(doc)

            # Verify descendant map was populated at index time.
            dm = retriever._descendant_map.get("http://ex.com/doc", {})
            assert "parent" in dm, "parent not in descendant map after index_document"
            assert "child" in dm["parent"], \
                f"child not a descendant of parent: {dm['parent']}"

            # Locate the parent's section_summary chunk (needed for stub).
            parent_summary = next(
                (c for c in indexed_chunks
                 if c.section_id == "parent" and c.chunk_role == "section_summary"),
                None,
            )
            assert parent_summary is not None, \
                "Expected a section_summary chunk for 'parent' — HierarchicalChunker must emit one"

            # --- Single combined spy that stubs Stage 1 and observes Stage 2 ---
            original_query = store.query
            stage1_fired: list[bool] = []
            stage2_queried_sids: list[str] = []
            fallback_fired: list[bool] = []

            def _combined_spy(entity_id, query_embedding, top_k, where=None):
                if where is None:
                    return original_query(entity_id, query_embedding, top_k, where=where)

                has_role   = "chunk_role" in where
                has_sid    = "section_id" in where
                has_docurl = "document_url" in where

                if has_role:
                    # Stage-1: stub to return only the parent summary
                    stage1_fired.append(True)
                    return [RetrievalResult(chunk=parent_summary, score=0.9, vector_score=0.9)]

                if has_sid:
                    # Stage-2: record which section_id was requested, then pass through
                    stage2_queried_sids.append(where["section_id"])
                    return original_query(entity_id, query_embedding, top_k, where=where)

                if has_docurl and not has_sid and not has_role:
                    # Whole-document fallback — should NOT fire when pruning works
                    fallback_fired.append(True)
                    return original_query(entity_id, query_embedding, top_k, where=where)

                return original_query(entity_id, query_embedding, top_k, where=where)

            store.query = _combined_spy  # type: ignore[method-assign]

            results = retriever.retrieve("Revenue grew", document_url="http://ex.com/doc")

            # 4. Stage 1 must have run.
            assert stage1_fired, "Stage 1 (section_summary query) never fired"

            # 5. Whole-document fallback must NOT have fired.
            assert not fallback_fired, (
                "Whole-document fallback fired — section-scoped Stage-2 queries failed "
                "(likely a Chroma where-filter error). Stage-2 queried: "
                f"{stage2_queried_sids}"
            )

            # 6. Stage-2 must have queried both parent and child.
            stage2_sids_set = set(stage2_queried_sids)
            assert "child" in stage2_sids_set, (
                f"Stage 2 did not query 'child' section_id; queried: {stage2_sids_set}. "
                "HierarchicalRetriever must expand selected sections to all descendants."
            )
            assert "parent" in stage2_sids_set, \
                f"Stage 2 must also query the selected parent itself; got: {stage2_sids_set}"

            # 7. Child leaf content must appear in results.
            result_contents = [r.chunk.content for r in results]
            assert any("Revenue grew" in c for c in result_contents), (
                f"Child leaf chunk ('Revenue grew 12% YoY.') missing from retrieve() results. "
                f"Got: {result_contents}"
            )

    def test_descendant_map_rebuilt_from_persisted_store(self):
        """A fresh HierarchicalRetriever instance (empty _descendant_map) must rebuild
        the descendant map lazily from the persisted Chroma collection so that
        Stage-2 expansion works across process boundaries.

        Protocol:
        1. Instance A indexes the doc (populates _descendant_map in memory).
        2. Instance B is created against the same persist_path with _descendant_map={}.
        3. Instance B's retrieve() is called; it must rebuild the map and query
           both 'parent' and 'child' section_ids in Stage 2.
        """
        import tempfile
        from unittest.mock import MagicMock
        from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever
        from evidence_enrichment.core.retrieval.models import RetrievalResult

        root = SectionNode(section_id="root", heading="", level=0, path=[],
                           children_ids=["parent"])
        parent = SectionNode(section_id="parent", heading="Financial Review", level=1,
                             path=["Financial Review"], parent_id="root",
                             children_ids=["child"], block_ids=["intro_b"])
        child = SectionNode(section_id="child", heading="Revenue", level=2,
                            path=["Financial Review", "Revenue"], parent_id="parent",
                            block_ids=["rev_b"])
        intro = ContentBlock(block_id="intro_b", block_type="text",
                             content="This section covers financials.")
        detail = ContentBlock(block_id="rev_b", block_type="text",
                              content="Revenue grew 12% YoY.")
        doc = ParsedDocument(
            url="http://ex.com/doc", title="T", content_type="text/html",
            text="", excerpt="",
            sections=[root, parent, child],
            section_tree_root="root",
            blocks=[intro, detail],
        )

        import hashlib as _hs
        def _det_embed(texts):
            vecs = []
            for t in texts:
                h = int(_hs.sha256(t.encode()).hexdigest(), 16)
                v = [float((h >> (i * 4)) & 0xF) / 16.0 for i in range(8)]
                vecs.append(v)
            return vecs

        def _make_embedder():
            em = MagicMock()
            em.embed_texts.side_effect = _det_embed
            em.embed_query.return_value = _det_embed(["Revenue grew"])[0]
            return em

        with tempfile.TemporaryDirectory() as tmp:
            from evidence_enrichment.core.retrieval.store import ChromaVectorStore
            from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker

            # --- Instance A: index the document ---
            store_a = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            retriever_a = HierarchicalRetriever(
                entity_id="ent", store=store_a, embedder=_make_embedder(),
                chunker=HierarchicalChunker(), top_k=5,
            )
            indexed_chunks = retriever_a.index_document(doc)

            parent_summary = next(
                (c for c in indexed_chunks
                 if c.section_id == "parent" and c.chunk_role == "section_summary"),
                None,
            )
            assert parent_summary is not None

            # --- Instance B: fresh retriever, same persist_path, empty _descendant_map ---
            store_b = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
            retriever_b = HierarchicalRetriever(
                entity_id="ent", store=store_b, embedder=_make_embedder(),
                chunker=HierarchicalChunker(), top_k=5, section_min_score=0.0,
            )
            assert retriever_b._descendant_map == {}, \
                "Fresh instance must start with empty _descendant_map"

            # Spy + Stage-1 stub on instance B's store
            original_query_b = store_b.query
            stage2_sids: list[str] = []
            fallback_fired: list[bool] = []

            def _spy_b(entity_id, query_embedding, top_k, where=None):
                if where and "chunk_role" in where:
                    return [RetrievalResult(chunk=parent_summary, score=0.9, vector_score=0.9)]
                if where and "section_id" in where:
                    stage2_sids.append(where["section_id"])
                    return original_query_b(entity_id, query_embedding, top_k, where=where)
                if where and "document_url" in where and "section_id" not in where and "chunk_role" not in where:
                    fallback_fired.append(True)
                    return original_query_b(entity_id, query_embedding, top_k, where=where)
                return original_query_b(entity_id, query_embedding, top_k, where=where)

            store_b.query = _spy_b  # type: ignore[method-assign]

            results = retriever_b.retrieve("Revenue grew", document_url="http://ex.com/doc")

            # The map must have been rebuilt lazily.
            assert "http://ex.com/doc" in retriever_b._descendant_map, \
                "_descendant_map not rebuilt from persisted store"

            assert not fallback_fired, (
                "Whole-document fallback fired on fresh instance — "
                f"lazy map rebuild failed. Stage-2 queried: {stage2_sids}"
            )
            assert "child" in set(stage2_sids), (
                f"Fresh instance Stage-2 did not query 'child'; queried: {stage2_sids}"
            )
            result_contents = [r.chunk.content for r in results]
            assert any("Revenue grew" in c for c in result_contents), (
                f"Child content missing from fresh-instance results: {result_contents}"
            )

    def _make_retriever_and_store(self, tmp: str):
        """Helper: create a HierarchicalRetriever backed by a real ChromaVectorStore."""
        import hashlib as _hs
        from unittest.mock import MagicMock
        from evidence_enrichment.core.retrieval.store import ChromaVectorStore
        from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker
        from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever

        def _det_embed(texts):
            vecs = []
            for t in texts:
                h = int(_hs.sha256(t.encode()).hexdigest(), 16)
                v = [float((h >> (i * 4)) & 0xF) / 16.0 for i in range(8)]
                vecs.append(v)
            return vecs

        em = MagicMock()
        em.embed_texts.side_effect = _det_embed
        em.embed_query.return_value = _det_embed(["query"])[0]

        store = ChromaVectorStore(persist_path=tmp, embedding_model="text-embedding-3-small")
        retriever = HierarchicalRetriever(
            entity_id="ent", store=store, embedder=em,
            chunker=HierarchicalChunker(), top_k=5, section_min_score=0.0,
        )
        return retriever, store

    def test_fresh_instance_repeated_headings_no_cross_contamination(self):
        """Fresh-instance rebuild must not expand p1 to c2 when p1 and p2 share
        the same heading (and thus the same section_path_str).

        Tree:  root
               ├── p1 (Overview)  ──── c1 (Details)
               └── p2 (Overview)  ──── c2 (Details)   ← different p1 IDs via F3

        A fresh retriever selecting p1 must query c1 but NOT c2.
        """
        import tempfile
        from evidence_enrichment.core.retrieval.models import RetrievalResult

        root = SectionNode(section_id="root", heading="", level=0, path=[],
                           children_ids=["p1", "p2"])
        p1 = SectionNode(section_id="p1", heading="Overview", level=1,
                         path=["Overview"], parent_id="root", children_ids=["c1"],
                         block_ids=["p1b"])
        p2 = SectionNode(section_id="p2", heading="Overview", level=1,
                         path=["Overview"], parent_id="root", children_ids=["c2"],
                         block_ids=["p2b"])
        c1 = SectionNode(section_id="c1", heading="Details", level=2,
                         path=["Overview", "Details"], parent_id="p1", block_ids=["c1b"])
        c2 = SectionNode(section_id="c2", heading="Details", level=2,
                         path=["Overview", "Details"], parent_id="p2", block_ids=["c2b"])

        blocks = [
            ContentBlock(block_id="p1b", block_type="text", content="p1 intro"),
            ContentBlock(block_id="p2b", block_type="text", content="p2 intro"),
            ContentBlock(block_id="c1b", block_type="text", content="c1 detail content"),
            ContentBlock(block_id="c2b", block_type="text", content="c2 detail content"),
        ]
        doc = ParsedDocument(
            url="http://ex.com/repeated", title="T", content_type="text/html",
            text="", excerpt="",
            sections=[root, p1, p2, c1, c2],
            section_tree_root="root",
            blocks=blocks,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ret_a, _ = self._make_retriever_and_store(tmp)
            indexed = ret_a.index_document(doc)

            p1_summary = next(
                (c for c in indexed if c.section_id == "p1" and c.chunk_role == "section_summary"),
                None,
            )
            assert p1_summary is not None

            # Instance B: fresh, must rebuild from parent edges
            ret_b, store_b = self._make_retriever_and_store(tmp)
            assert ret_b._descendant_map == {}

            original_q = store_b.query
            stage2_sids: list[str] = []

            def _spy(entity_id, query_embedding, top_k, where=None):
                if where and "chunk_role" in where:
                    return [RetrievalResult(chunk=p1_summary, score=0.9, vector_score=0.9)]
                if where and "section_id" in where:
                    stage2_sids.append(where["section_id"])
                    return original_q(entity_id, query_embedding, top_k, where=where)
                return original_q(entity_id, query_embedding, top_k, where=where)

            store_b.query = _spy  # type: ignore[method-assign]

            ret_b.retrieve("query", document_url="http://ex.com/repeated")

            queried = set(stage2_sids)
            assert "p1" in queried, f"p1 itself should be queried; got {queried}"
            assert "c1" in queried, f"c1 (child of p1) should be queried; got {queried}"
            assert "c2" not in queried, (
                f"c2 (child of p2) must NOT be queried when only p1 is selected; "
                f"got {queried} — repeated-heading cross-contamination detected"
            )

    def test_fresh_instance_root_section_expands_to_children(self):
        """Fresh-instance rebuild must treat root-owned chunks (parent_section_id='')
        correctly so that selecting root expands to all its children.

        Tree:  root  ──── child

        A fresh retriever selecting root in Stage 1 must also query child in Stage 2.
        """
        import tempfile
        from evidence_enrichment.core.retrieval.models import RetrievalResult

        root = SectionNode(section_id="root", heading="", level=0, path=[],
                           children_ids=["child"], block_ids=["root_b"])
        child = SectionNode(section_id="child", heading="Results", level=1,
                            path=["Results"], parent_id="root", block_ids=["child_b"])
        blocks = [
            ContentBlock(block_id="root_b", block_type="text", content="Root intro text."),
            ContentBlock(block_id="child_b", block_type="text", content="Child results text."),
        ]
        doc = ParsedDocument(
            url="http://ex.com/root_test", title="T", content_type="text/html",
            text="", excerpt="",
            sections=[root, child],
            section_tree_root="root",
            blocks=blocks,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ret_a, _ = self._make_retriever_and_store(tmp)
            indexed = ret_a.index_document(doc)

            root_summary = next(
                (c for c in indexed if c.section_id == "root" and c.chunk_role == "section_summary"),
                None,
            )
            # root may not emit a section_summary if it has no path text;
            # fall back to any root-owned chunk as the Stage-1 stub.
            if root_summary is None:
                root_summary = next(
                    (c for c in indexed if c.section_id == "root"), None
                )
            assert root_summary is not None, "root must have at least one indexed chunk"

            ret_b, store_b = self._make_retriever_and_store(tmp)
            assert ret_b._descendant_map == {}

            original_q = store_b.query
            stage2_sids: list[str] = []

            def _spy(entity_id, query_embedding, top_k, where=None):
                if where and "chunk_role" in where:
                    return [RetrievalResult(chunk=root_summary, score=0.9, vector_score=0.9)]
                if where and "section_id" in where:
                    stage2_sids.append(where["section_id"])
                    return original_q(entity_id, query_embedding, top_k, where=where)
                return original_q(entity_id, query_embedding, top_k, where=where)

            store_b.query = _spy  # type: ignore[method-assign]

            ret_b.retrieve("query", document_url="http://ex.com/root_test")

            queried = set(stage2_sids)
            assert "child" in queried, (
                f"'child' must be queried when root is selected in Stage 1; got {queried}. "
                "Root ancestry must be reconstructed from persisted parent edges."
            )


# ---------------------------------------------------------------------------
# Stage E — End-to-end hierarchical retrieval eval (fixture-based, no LLM)
#
# Success criteria (from docs/hierarchical_retrieval_upgrade.md §Stage E):
#   E1: retrieve() returns results with a non-None retrieved_chunks_map
#       (i.e. hits are non-empty and keyed by document_url).
#   E2: Every returned chunk carries a non-empty section_id and
#       section_path_str (proving the hierarchical path was taken, not the
#       flat fallback).
#   E3: The content of the top-scoring chunk is drawn from the correct section
#       of the source document (analogous to "correct answer" for a simple
#       structured field).
#
# Implementation: uses a deterministic stub embedder so the test is fully
# reproducible in CI without any provider credentials.
# ---------------------------------------------------------------------------

class TestEndToEndHierarchicalRetrieval:
    """Fixture-based end-to-end proof that the hierarchical retrieval path
    (use_structured=True analogue) surfaces section-aware chunks.

    We bypass the coordinator entirely and exercise:
        ParsedDocument (with sections+blocks)
        → HierarchicalChunker.chunk()
        → HierarchicalRetriever.index_document()
        → HierarchicalRetriever.retrieve()
        → retrieved_chunks_map construction

    This mirrors what coordinator._stage_retrieval does when
    chunker="hierarchical" and retriever="hierarchical".
    """

    # Shared deterministic embedder: SHA-256 hash → 8-dim float vector.
    # Identical texts always produce identical vectors; different texts produce
    # different vectors, so cosine similarity is meaningful within a test.
    @staticmethod
    def _det_embed(texts: list[str]) -> list[list[float]]:
        import hashlib
        vecs = []
        for t in texts:
            h = int(hashlib.sha256(t.encode()).hexdigest(), 16)
            v = [float((h >> (i * 4)) & 0xF) / 16.0 for i in range(8)]
            # Normalise so cosine == dot product (makes scores predictable).
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])
        return vecs

    def _make_embedder(self, query_text: str = ""):
        from unittest.mock import MagicMock
        em = MagicMock()
        em.embed_texts.side_effect = self._det_embed
        em.embed_query.return_value = (
            self._det_embed([query_text])[0] if query_text
            else self._det_embed(["query"])[0]
        )
        return em

    def _make_doc(self) -> "ParsedDocument":
        """Two-level section tree: root → [hq_section, risk_section].

        hq_section contains the headquarters country fact; risk_section
        contains an unrelated risk paragraph.  The query targets hq_section.
        """
        root = SectionNode(
            section_id="root", heading="", level=0, path=[],
            children_ids=["hq_section", "risk_section"],
        )
        hq_section = SectionNode(
            section_id="hq_section",
            heading="Corporate Headquarters",
            level=1,
            path=["Corporate Headquarters"],
            parent_id="root",
            block_ids=["hq_b"],
        )
        risk_section = SectionNode(
            section_id="risk_section",
            heading="Risk Factors",
            level=1,
            path=["Risk Factors"],
            parent_id="root",
            block_ids=["risk_b"],
        )
        hq_block = ContentBlock(
            block_id="hq_b", block_type="text", section_id="hq_section",
            content=(
                "Microsoft Corporation is headquartered in Redmond, Washington, "
                "United States of America."
            ),
        )
        risk_block = ContentBlock(
            block_id="risk_b", block_type="text", section_id="risk_section",
            content="Adverse macroeconomic conditions may reduce demand for our products.",
        )
        return ParsedDocument(
            url="http://ex.com/10k.html",
            title="Microsoft 10-K",
            content_type="text/html",
            text="",
            excerpt="",
            sections=[root, hq_section, risk_section],
            section_tree_root="root",
            blocks=[hq_block, risk_block],
        )

    def test_e1_retrieved_chunks_map_is_populated(self):
        """E1: retrieve() returns non-empty hits that can form a retrieved_chunks_map."""
        import tempfile
        from evidence_enrichment.core.retrieval.store import ChromaVectorStore
        from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever
        from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker

        doc = self._make_doc()
        query = "headquarters country location"

        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="stub")
            retriever = HierarchicalRetriever(
                entity_id="msft",
                store=store,
                embedder=self._make_embedder(query),
                chunker=HierarchicalChunker(),
            )
            retriever.index_document(doc)

            hits = retriever.retrieve(query, document_url=doc.url)

        # Build the map the same way coordinator does.
        retrieved_chunks_map = {doc.url: hits} if hits else {}

        assert retrieved_chunks_map, (
            "retrieved_chunks_map must be non-empty after indexing and querying "
            "a document with a populated section tree."
        )
        assert doc.url in retrieved_chunks_map, (
            f"map must be keyed by document_url={doc.url!r}"
        )
        assert len(retrieved_chunks_map[doc.url]) > 0, (
            "hits list for the document must be non-empty"
        )

    def test_e2_chunks_carry_section_id_and_section_path_str(self):
        """E2: Every returned chunk has non-empty section_id and section_path_str."""
        import tempfile
        from evidence_enrichment.core.retrieval.store import ChromaVectorStore
        from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever
        from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker

        doc = self._make_doc()
        query = "headquarters country location"

        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="stub")
            retriever = HierarchicalRetriever(
                entity_id="msft",
                store=store,
                embedder=self._make_embedder(query),
                chunker=HierarchicalChunker(),
            )
            retriever.index_document(doc)
            hits = retriever.retrieve(query, document_url=doc.url)

        assert hits, "retrieve() must return at least one hit"

        for result in hits:
            assert result.chunk.section_id, (
                f"chunk {result.chunk.chunk_id!r} has empty section_id — "
                "hierarchical path not taken; flat fallback may have fired."
            )
            assert result.chunk.section_path_str, (
                f"chunk {result.chunk.chunk_id!r} has empty section_path_str — "
                "section path metadata not written during indexing."
            )

    def test_e3_top_chunk_content_from_correct_section(self):
        """E3: The top-ranked hit comes from hq_section, not risk_section.

        The query targets headquarters location; the document has exactly two
        leaf sections — one about headquarters, one about risk factors.
        hits[0] must be from hq_section and must contain the expected fact.
        This is the fixture-based analogue of 'correct answer' verification:
        the most relevant section must win, not merely appear somewhere in
        the result list.
        """
        import tempfile
        from evidence_enrichment.core.retrieval.store import ChromaVectorStore
        from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever
        from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker

        doc = self._make_doc()
        query = "headquarters country location"

        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_path=tmp, embedding_model="stub")
            retriever = HierarchicalRetriever(
                entity_id="msft",
                store=store,
                embedder=self._make_embedder(query),
                chunker=HierarchicalChunker(),
            )
            retriever.index_document(doc)
            hits = retriever.retrieve(query, document_url=doc.url)

        assert hits, "retrieve() must return at least one hit"

        top = hits[0]
        hq_section_ids = {"hq_section", "root"}  # root may emit a covering summary

        assert top.chunk.section_id in hq_section_ids, (
            f"hits[0] must come from hq_section (or root summary), "
            f"got section_id={top.chunk.section_id!r} score={top.score:.4f}. "
            f"Full ranking: {[(r.chunk.section_id, round(r.score, 4)) for r in hits]}. "
            "The most relevant section must rank first, not merely appear in the list."
        )

        assert (
            "united states" in top.chunk.content.lower()
            or "redmond" in top.chunk.content.lower()
            or "washington" in top.chunk.content.lower()
            or "headquarters" in top.chunk.content.lower()
        ), (
            f"hits[0] content does not mention expected headquarters text; "
            f"content={top.chunk.content!r}"
        )
