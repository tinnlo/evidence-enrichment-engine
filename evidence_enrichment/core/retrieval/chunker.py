"""Table-aware document chunker.

Adapted from production MVP KB script patterns. Tables are kept as atomic
chunks to preserve row/column relationships when they fit within
``max_table_size``. Large tables that exceed this limit are split on
whitespace boundaries without overlap, which may split mid-row. Plain text
uses character-based chunking with overlap.
"""

from __future__ import annotations

from evidence_enrichment.core.models.contracts import ContentBlock, ParsedDocument
from evidence_enrichment.core.retrieval.models import Chunk


class TableAwareChunker:
    """Chunk a ParsedDocument into Chunk objects for embedding and retrieval.

    Parameters
    ----------
    chunk_size:
        Target character count for text chunks (default 1500).
    overlap:
        Overlap in characters between consecutive text chunks (default 200).
    min_size:
        Minimum character count for a chunk to be emitted (default 80).
    max_table_size:
        Maximum character count before a table block is split (default 4000).
        Splits are performed on whitespace boundaries, WITHOUT overlap, to
        avoid duplicating numeric data.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 200,
        min_size: int = 80,
        max_table_size: int = 4000,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size
        self.max_table_size = max_table_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Return a list of Chunk objects from a ParsedDocument.

        If the document has structured blocks (from parse_with_structure), those
        are used directly.  Otherwise the plain ``text`` field is treated as a
        single text block.
        """
        if document.blocks:
            return self._chunk_blocks(document)
        return self._chunk_plain_text(document)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_blocks(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 0
        content_hash = document.content_hash or ""

        for block in document.blocks:
            block_chunks = self._chunk_block(
                block=block,
                document_url=document.url,
                content_hash=content_hash,
                start_idx=idx,
            )
            chunks.extend(block_chunks)
            idx += len(block_chunks)

        return chunks

    def _chunk_plain_text(self, document: ParsedDocument) -> list[Chunk]:
        """Fallback: treat the whole plain text as one text block."""
        import hashlib

        content_hash = document.content_hash or hashlib.sha256(
            document.text.encode()
        ).hexdigest()
        block = ContentBlock(block_type="text", content=document.text)
        return self._chunk_block(
            block=block,
            document_url=document.url,
            content_hash=content_hash,
            start_idx=0,
        )

    def _chunk_block(
        self,
        block: ContentBlock,
        document_url: str,
        content_hash: str,
        start_idx: int,
    ) -> list[Chunk]:
        if block.block_type == "table":
            return self._chunk_table(block.content, document_url, content_hash, start_idx)
        return self._chunk_text(block.content, document_url, content_hash, start_idx)

    def _chunk_table(
        self,
        content: str,
        document_url: str,
        content_hash: str,
        start_idx: int,
    ) -> list[Chunk]:
        """Keep tables atomic; split only if they exceed max_table_size.

        Large table splits use NO overlap to avoid duplicating numeric rows.
        """
        if len(content) <= self.max_table_size:
            return self._make_chunk(content, document_url, content_hash, start_idx, "table")

        # Split large table on whitespace boundaries without overlap
        parts = self._split_no_overlap(content, self.max_table_size)
        chunks: list[Chunk] = []
        for i, part in enumerate(parts):
            chunks.extend(
                self._make_chunk(part, document_url, content_hash, start_idx + i, "table")
            )
        return chunks

    def _chunk_text(
        self,
        content: str,
        document_url: str,
        content_hash: str,
        start_idx: int,
    ) -> list[Chunk]:
        """Character-based text chunking with overlap."""
        if len(content) <= self.chunk_size:
            return self._make_chunk(content, document_url, content_hash, start_idx, "text")

        chunks: list[Chunk] = []
        pos = 0
        local_idx = 0
        while pos < len(content):
            end = pos + self.chunk_size
            # Try to break on a whitespace boundary to avoid mid-word splits
            if end < len(content):
                ws = content.rfind(" ", pos, end)
                if ws > pos:
                    end = ws
            piece = content[pos:end].strip()
            if len(piece) >= self.min_size:
                chunks.extend(
                    self._make_chunk(
                        piece, document_url, content_hash, start_idx + local_idx, "text"
                    )
                )
                local_idx += 1
            # Advance with overlap
            pos = end - self.overlap if end < len(content) else len(content)

        return chunks

    def _make_chunk(
        self,
        content: str,
        document_url: str,
        content_hash: str,
        index: int,
        chunk_type: str,
    ) -> list[Chunk]:
        """Create a single Chunk if it meets the minimum size threshold."""
        stripped = content.strip()
        if len(stripped) < self.min_size:
            return []
        chunk_id = Chunk.make_id(document_url, index, content_hash)
        return [
            Chunk(
                chunk_id=chunk_id,
                document_url=document_url,
                content_hash=content_hash,
                index=index,
                content=stripped,
                chunk_type=chunk_type,
            )
        ]

    @staticmethod
    def _split_no_overlap(text: str, max_size: int) -> list[str]:
        """Split text into parts of at most max_size chars on whitespace boundaries."""
        parts: list[str] = []
        pos = 0
        while pos < len(text):
            end = pos + max_size
            if end < len(text):
                ws = text.rfind(" ", pos, end)
                if ws > pos:
                    end = ws
            parts.append(text[pos:end].strip())
            pos = end
        return [p for p in parts if p]
