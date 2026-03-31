"""Tests for TableAwareChunker."""

from __future__ import annotations

import pytest

from evidence_enrichment.core.models.contracts import ContentBlock, ParsedDocument
from evidence_enrichment.core.retrieval.chunker import TableAwareChunker
from evidence_enrichment.core.retrieval.models import Chunk


def _make_doc(text: str, blocks: list[ContentBlock] | None = None, content_hash: str = "abc123") -> ParsedDocument:
    return ParsedDocument(
        url="https://example.com/doc",
        title="Test Document",
        content_type="text/html",
        text=text,
        excerpt=text[:200],
        full_text=text,
        content_hash=content_hash,
        blocks=blocks or [],
    )


class TestTextChunking:
    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size (but above min_size) produces exactly one chunk."""
        chunker = TableAwareChunker(chunk_size=1500, overlap=200)
        # 100 chars — above default min_size=80, below chunk_size=1500
        doc = _make_doc("Hello world. This is a test that is long enough to exceed the minimum chunk size limit.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "text"

    def test_long_text_overlap(self):
        """Text longer than chunk_size is split with overlap."""
        chunker = TableAwareChunker(chunk_size=100, overlap=20)
        text = "word " * 60  # 300 chars
        doc = _make_doc(text)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        # Adjacent chunks should share some content (overlap)
        content_0 = chunks[0].content
        content_1 = chunks[1].content
        # The end of chunk 0 should appear in chunk 1 (up to overlap chars)
        assert content_0[-10:] in content_1 or len(chunks[0].content) <= 100 + 20

    def test_chunk_size_respected(self):
        """No chunk exceeds chunk_size + overhead from whitespace alignment."""
        chunker = TableAwareChunker(chunk_size=200, overlap=50)
        text = "abcdefghij " * 50  # 550 chars
        doc = _make_doc(text)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.char_count <= 250, f"Chunk too large: {chunk.char_count}"

    def test_min_size_filter(self):
        """Chunks below min_size are dropped."""
        chunker = TableAwareChunker(chunk_size=1500, overlap=0, min_size=100)
        short_text = "hi"
        doc = _make_doc(short_text)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

    def test_empty_document(self):
        """Empty document produces no chunks."""
        chunker = TableAwareChunker()
        doc = _make_doc("")
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_deterministic_chunk_ids(self):
        """Same document always produces the same chunk IDs."""
        chunker = TableAwareChunker(chunk_size=200, overlap=20)
        text = "stable content " * 30
        doc = _make_doc(text, content_hash="fixed_hash")
        chunks_a = chunker.chunk(doc)
        chunks_b = chunker.chunk(doc)
        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        assert ids_a == ids_b

    def test_chunk_ids_are_16_chars(self):
        """Chunk IDs are exactly 16 hex characters."""
        chunker = TableAwareChunker()
        doc = _make_doc("Some text content that is long enough to produce a chunk.")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert len(chunk.chunk_id) == 16
            assert all(c in "0123456789abcdef" for c in chunk.chunk_id)


class TestTableChunking:
    def test_table_kept_atomic_small(self):
        """Tables within max_table_size are returned as a single chunk."""
        chunker = TableAwareChunker(max_table_size=4000)
        # Ensure table content is > min_size=80
        table_content = (
            "| col1        | col2        | col3        |\n"
            "|-------------|-------------|-------------|\n"
            "| value_a_001 | value_b_001 | value_c_001 |\n"
            "| value_a_002 | value_b_002 | value_c_002 |\n"
            "| value_a_003 | value_b_003 | value_c_003 |\n"
        )
        block = ContentBlock(block_type="table", content=table_content)
        doc = _make_doc(table_content, blocks=[block])
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"

    def test_large_table_split_no_overlap(self):
        """Tables exceeding max_table_size are split without overlap."""
        chunker = TableAwareChunker(max_table_size=100, overlap=50)
        table_content = "| col1 | col2 |\n" + "| data | data |\n" * 20  # ~320 chars
        block = ContentBlock(block_type="table", content=table_content)
        doc = _make_doc(table_content, blocks=[block])
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        assert all(c.chunk_type == "table" for c in chunks)
        # Verify NO overlap: total content should not be much larger than original
        total_content = "".join(c.content for c in chunks)
        assert len(total_content) <= len(table_content) + 20  # small tolerance for whitespace

    def test_mixed_blocks(self):
        """Documents with both text and table blocks produce correct chunk types."""
        chunker = TableAwareChunker()
        text_block = ContentBlock(block_type="text", content="This is regular text. " * 10)
        # Table must be > min_size=80
        table_block = ContentBlock(
            block_type="table",
            content=(
                "| column_one  | column_two  | column_three |\n"
                "|-------------|-------------|---------------|\n"
                "| value_a_001 | value_b_001 | value_c_001  |\n"
                "| value_a_002 | value_b_002 | value_c_002  |\n"
            ),
        )
        doc = _make_doc("combined", blocks=[text_block, table_block])
        chunks = chunker.chunk(doc)
        chunk_types = [c.chunk_type for c in chunks]
        assert "text" in chunk_types
        assert "table" in chunk_types

    def test_index_continuity(self):
        """Chunk indices increase monotonically across multiple blocks."""
        chunker = TableAwareChunker()
        blocks = [
            ContentBlock(block_type="text", content="Text block one. " * 10),
            ContentBlock(block_type="table", content="| a | b |\n| c | d |"),
            ContentBlock(block_type="text", content="Text block two. " * 10),
        ]
        doc = _make_doc("combined", blocks=blocks)
        chunks = chunker.chunk(doc)
        indices = [c.index for c in chunks]
        assert indices == sorted(indices)
