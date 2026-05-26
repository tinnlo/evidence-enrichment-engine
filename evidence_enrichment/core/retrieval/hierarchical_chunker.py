"""HierarchicalChunker — section-aware chunker that never crosses section boundaries.

Chunks are sized by token count (tiktoken cl100k_base) and follow the
section tree produced by HTMLStructuredParser / PDFStructuredParser.

For each leaf section the chunker emits:
  1. One ``section_summary`` chunk: heading path + first ~summary_tokens tokens.
  2. One or more ``content`` chunks that do NOT cross section boundaries.
  3. One ``table`` chunk per table block in the section (atomic unless huge).

For each non-leaf section it emits:
  1. One ``navigation`` chunk: heading path + concatenated child headings.

Falls back gracefully to flat chunking when the document has no section tree
(sections == []) — this handles documents parsed by GenericTextParser or
legacy TextParser paths.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

try:
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_TOKENIZER.encode(text, disallowed_special=()))

except ImportError:  # tiktoken absent — degrade to char/4 approximation
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4


if TYPE_CHECKING:
    from evidence_enrichment.core.models.contracts import ParsedDocument, SectionNode

from evidence_enrichment.core.retrieval.models import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_TOKENS = 400
_DEFAULT_MAX_TOKENS = 800
_DEFAULT_OVERLAP_TOKENS = 80
_DEFAULT_SUMMARY_TOKENS = 200
_TABLE_SPLIT_MULTIPLIER = 3  # tables > max_tokens*3 are split


class HierarchicalChunker:
    """Section-aware chunker.

    Parameters
    ----------
    target_chunk_tokens:
        Target token count per content chunk. Default 400 ≈ ~1,600 chars.
    max_chunk_tokens:
        Hard ceiling. Default 800.
    overlap_tokens:
        Overlap between consecutive text chunks within one section. Default 80.
    summary_tokens:
        Token budget for section_summary chunks. Default 200.
    keep_tables_atomic:
        Tables never split unless they exceed max_chunk_tokens * _TABLE_SPLIT_MULTIPLIER.
    """

    def __init__(
        self,
        target_chunk_tokens: int = _DEFAULT_TARGET_TOKENS,
        max_chunk_tokens: int = _DEFAULT_MAX_TOKENS,
        overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
        summary_tokens: int = _DEFAULT_SUMMARY_TOKENS,
        keep_tables_atomic: bool = True,
    ) -> None:
        self.target = target_chunk_tokens
        self.max_tokens = max_chunk_tokens
        self.overlap = overlap_tokens
        self.summary_tokens = summary_tokens
        self.keep_tables_atomic = keep_tables_atomic

    def chunk(self, doc: "ParsedDocument") -> list[Chunk]:
        """Chunk a ParsedDocument into Chunks, respecting section boundaries."""
        if not doc.sections:
            # No section tree — fall back to flat chunking (preserves backward compat)
            return self._flat_chunk(doc)

        # Build a section_id -> SectionNode lookup
        section_map = {s.section_id: s for s in doc.sections}
        # Build a block_id -> block lookup (block_id may be "" for legacy blocks)
        block_map = {b.block_id: b for b in doc.blocks if b.block_id}

        chunks: list[Chunk] = []
        index = [0]  # mutable counter shared across helpers

        def _emit(
            content: str,
            section: "SectionNode",
            chunk_role: str,
            chunk_type: str = "text",
            table_data: list[list[str]] | None = None,
        ) -> None:
            if not content.strip():
                return
            ch = hashlib.sha256(content.encode()).hexdigest()[:16]
            path_str = "|".join(section.path)
            page: int | None = section.page_start
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(doc.url, index[0], ch),
                    document_url=doc.url,
                    content_hash=ch,
                    index=index[0],
                    content=content,
                    chunk_type=chunk_type,
                    section_id=section.section_id,
                    section_path_str=path_str,
                    section_level=section.level,
                    parent_section_id=section.parent_id or "",
                    page=page,
                    chunk_role=chunk_role,  # type: ignore[arg-type]
                    token_count=_count_tokens(content),
                    table_data=table_data,
                )
            )
            index[0] += 1

        def _emit_blocks_for_node(node: "SectionNode") -> None:
            """Emit section_summary + content/table chunks for a node's own block_ids.

            Called for both leaf and non-leaf nodes so that intro content
            attached directly to a parent section is never silently dropped.
            """
            own_blocks = [
                block_map[bid]
                for bid in node.block_ids
                if bid in block_map
            ]
            if not own_blocks:
                return

            heading_prefix = " > ".join(node.path) + "\n\n" if node.path else ""

            # Section summary: heading + first summary_tokens of content
            all_text = "\n\n".join(b.content for b in own_blocks)
            summary_text = _truncate_to_tokens(all_text, self.summary_tokens)
            _emit(heading_prefix + summary_text, node, "section_summary")

            # Content / table chunks
            pending_text: list[str] = []
            pending_tokens = 0

            def _flush_text() -> None:
                if pending_text:
                    _emit(heading_prefix + "\n\n".join(pending_text), node, "content")
                    pending_text.clear()
                    nonlocal pending_tokens
                    pending_tokens = 0

            for block in own_blocks:
                if block.block_type == "table":
                    _flush_text()
                    toks = _count_tokens(block.content)
                    hard_limit = self.max_tokens * _TABLE_SPLIT_MULTIPLIER
                    if self.keep_tables_atomic or toks <= hard_limit:
                        _emit(block.content, node, "table", "table", block.table_data)
                    else:
                        for row_chunk in _split_table(block, self.max_tokens, node.page_start):
                            _emit(row_chunk, node, "table", "table")
                else:
                    block_toks = _count_tokens(block.content)
                    if pending_tokens + block_toks > self.max_tokens and pending_text:
                        _flush_text()
                    pending_text.append(block.content)
                    pending_tokens += block_toks
                    if pending_tokens >= self.target:
                        _flush_text()

            _flush_text()

        def _process_section(node: "SectionNode") -> None:
            has_children = bool(node.children_ids)
            heading_prefix = " > ".join(node.path) + "\n\n" if node.path else ""

            if has_children:
                # Navigation chunk: heading + child headings
                child_headings = [
                    section_map[cid].heading
                    for cid in node.children_ids
                    if cid in section_map
                ]
                nav_content = heading_prefix + "\n".join(f"• {h}" for h in child_headings)
                _emit(nav_content, node, "navigation")
                # Emit any blocks directly owned by this non-leaf node
                # (e.g. intro paragraphs before the first sub-heading)
                _emit_blocks_for_node(node)
                for cid in node.children_ids:
                    if cid in section_map:
                        _process_section(section_map[cid])
            else:
                # Leaf section — delegate entirely to shared helper
                _emit_blocks_for_node(node)

        # Process all root-level sections (children of the doc root or level-0 node)
        root_nodes = [s for s in doc.sections if s.level == 0]
        if root_nodes:
            root_node = root_nodes[0]
            # Root's own block_ids hold content before the first heading (e.g. intro text).
            # Emit them directly — root is not a named section so no navigation chunk.
            _emit_blocks_for_node(root_node)
            for cid in root_node.children_ids:
                if cid in section_map:
                    _process_section(section_map[cid])
        else:
            for s in doc.sections:
                if s.parent_id is None:
                    _process_section(s)

        if not chunks:
            return self._flat_chunk(doc)
        return chunks

    # ------------------------------------------------------------------
    # Flat fallback (no section tree)
    # ------------------------------------------------------------------

    def _flat_chunk(self, doc: "ParsedDocument") -> list[Chunk]:
        """Simple token-based chunking — used when no section tree is present."""
        from evidence_enrichment.core.retrieval.chunker import TableAwareChunker

        # Delegate to the existing TableAwareChunker with char-based sizing,
        # converting token target to approximate char count (1 token ≈ 4 chars).
        flat = TableAwareChunker(
            chunk_size=self.target * 4,
            overlap=self.overlap * 4,
            max_table_size=self.max_tokens * 4 * _TABLE_SPLIT_MULTIPLIER,
        )
        return flat.chunk(doc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return a prefix of text that fits within max_tokens."""
    if _count_tokens(text) <= max_tokens:
        return text
    # Binary-search approximate cut point
    lo, hi = 0, len(text)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if _count_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid
    return text[:lo]


def _split_table(block: object, max_tokens: int, page: int | None) -> list[str]:
    """Split an oversized table block into row-group sub-chunks."""
    lines = getattr(block, "content", "").splitlines()
    groups: list[str] = []
    current: list[str] = []
    current_toks = 0
    for line in lines:
        toks = _count_tokens(line)
        if current_toks + toks > max_tokens and current:
            groups.append("\n".join(current))
            current = []
            current_toks = 0
        current.append(line)
        current_toks += toks
    if current:
        groups.append("\n".join(current))
    return groups
