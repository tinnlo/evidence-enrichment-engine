"""HTMLStructuredParser — structured HTML parsing with SectionNode tree emission.

Wraps TextParser.parse_with_structure and additionally builds a SectionNode
hierarchy from the heading blocks it extracts.
"""

from __future__ import annotations

import hashlib

from evidence_enrichment.core.models.contracts import (
    ParsedDocument,
    RetrievedDocument,
    SectionNode,
)
from evidence_enrichment.core.parse.parser import TextParser


class HTMLStructuredParser:
    """Handles text/html — wraps TextParser.parse_with_structure and emits a
    SectionNode tree from extracted heading blocks."""

    def can_parse(self, doc: RetrievedDocument) -> bool:
        ct = doc.content_type.split(";")[0].strip().lower()
        return ct in ("text/html", "application/xhtml+xml")

    def parse(self, doc: RetrievedDocument) -> ParsedDocument:
        parsed = TextParser().parse_with_structure(doc)
        sections, root_id, updated_blocks = _build_section_tree(doc.final_url, parsed)
        return parsed.model_copy(
            update={
                "sections": sections,
                "section_tree_root": root_id,
                "blocks": updated_blocks,
            }
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_section_id(doc_url: str, path: list[str], index: int = 0) -> str:
    """Stable, positionally-unique section ID.

    ``index`` is the ordinal of the section in document order.  Including it
    in the hash means repeated headings (same path) produce distinct IDs while
    remaining deterministic across re-parses of the same document.
    """
    key = f"{doc_url}|{index}|" + "|".join(path)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _build_section_tree(
    doc_url: str, parsed: ParsedDocument
) -> tuple[list[SectionNode], str | None, list]:
    """Build a flat list of SectionNode objects from heading blocks.

    Headings are detected from blocks with block_type == "heading".  Each
    heading becomes a SectionNode at depth 1 (we don't have H-level
    information at this stage from plain heading text, so all are level=1).
    Non-heading blocks between headings are associated with the preceding
    SectionNode via block_ids.

    Also stamps each ContentBlock with a stable block_id so HierarchicalChunker
    can build its block_map correctly.

    Returns (sections, root_id, updated_blocks).  root_id is the section_id of
    the implicit document root node.
    """
    from evidence_enrichment.core.models.contracts import ContentBlock

    root_path: list[str] = []
    root_id = _make_section_id(doc_url, root_path, index=0)
    root = SectionNode(
        section_id=root_id,
        level=0,
        heading="",
        path=root_path,
    )

    sections: list[SectionNode] = [root]
    current: SectionNode = root
    updated_blocks: list[ContentBlock] = []

    for i, block in enumerate(parsed.blocks):
        # Assign a stable block_id if the block doesn't already have one
        block_id = block.block_id or f"block_{i}"
        stamped = block.model_copy(update={"block_id": block_id}) if not block.block_id else block

        if block.block_type == "heading":
            path = [block.content[:120]]  # truncate very long headings
            # Use len(sections) as the positional index so repeated headings
            # get distinct IDs while remaining deterministic on re-parse.
            sid = _make_section_id(doc_url, path, index=len(sections))
            node = SectionNode(
                section_id=sid,
                level=1,
                heading=block.content,
                path=path,
                parent_id=root_id,
            )
            root.children_ids.append(sid)
            sections.append(node)
            current = node
        else:
            current.block_ids.append(block_id)

        updated_blocks.append(stamped)

    return sections, root_id, updated_blocks
