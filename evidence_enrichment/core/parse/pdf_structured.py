"""PDFStructuredParser — PDF parsing with SectionNode tree emission.

Requires optional extras: pdfplumber>=0.11, pymupdf>=1.24.
Import of this module will fail with ImportError if either dep is absent,
which is the intended capability probe used by both the fetch gate
(_ALLOWED_BINARY_PREFIXES in fetcher.py) and the coordinator registry guard.
"""

from __future__ import annotations

import hashlib
import io
import logging

import fitz  # pymupdf — intentional: triggers ImportError if absent
import pdfplumber  # intentional: triggers ImportError if absent

from evidence_enrichment.core.models.contracts import (
    ContentBlock,
    ParsedDocument,
    RetrievedDocument,
    SectionNode,
)

logger = logging.getLogger(__name__)

# Font-size tier names
_TIER_TITLE = "title"
_TIER_HEADING = "heading"
_TIER_SUBHEADING = "subheading"
_TIER_BODY = "body"


class PDFStructuredParser:
    """Handles application/pdf — reads doc.body_bytes, raises ValueError if None."""

    def can_parse(self, doc: RetrievedDocument) -> bool:
        ct = doc.content_type.split(";")[0].strip().lower()
        return ct == "application/pdf"

    def parse(self, doc: RetrievedDocument) -> ParsedDocument:
        if doc.body_bytes is None:
            raise ValueError(
                f"PDFStructuredParser received body_bytes=None for {doc.url}. "
                "This is a bug: PDF documents must have body_bytes populated by the fetcher."
            )
        blocks, sections, page_count = _extract_pdf(doc.url, doc.body_bytes)
        full_text = "\n\n".join(b.content for b in blocks)
        content_hash = hashlib.sha256(full_text.encode()).hexdigest()
        excerpt = full_text[:500]
        root_id = sections[0].section_id if sections else None
        return ParsedDocument(
            url=doc.final_url,
            title=doc.title,
            content_type=doc.content_type,
            text=full_text,
            excerpt=excerpt,
            full_text=full_text,
            content_hash=content_hash,
            blocks=blocks,
            mime_type="application/pdf",
            sections=sections,
            section_tree_root=root_id,
            page_count=page_count,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_section_id(doc_url: str, path: list[str], index: int = 0) -> str:
    """Stable, positionally-unique section ID.

    ``index`` is the ordinal of the section in document order so that repeated
    headings with the same path produce distinct IDs while remaining
    deterministic across re-parses of the same document.
    """
    key = f"{doc_url}|{index}|" + "|".join(path)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _extract_pdf(
    doc_url: str, pdf_bytes: bytes
) -> tuple[list[ContentBlock], list[SectionNode], int]:
    """Extract blocks and section tree from PDF bytes.

    Strategy (in priority order):
    1. Use pymupdf TOC/bookmarks if available — most reliable for annual reports.
    2. Fall back to font-size clustering via pdfplumber chars to detect headings.
    """
    # --- TOC via pymupdf ---
    toc_entries: list[tuple[int, str, int]] = []  # (level, title, page_1indexed)
    try:
        fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        toc_entries = fitz_doc.get_toc(simple=True)  # [(level, title, page), ...]
        fitz_doc.close()
    except Exception:  # noqa: BLE001
        logger.debug("pymupdf TOC extraction failed for %s; falling back to font sizing", doc_url)

    blocks: list[ContentBlock] = []
    page_count = 0

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        font_sizes: list[float] = []
        # Collect all char font sizes for tier clustering
        for page in pdf.pages:
            for char in page.chars:
                if char.get("size"):
                    font_sizes.append(float(char["size"]))

        size_tiers = _cluster_font_sizes(font_sizes)

        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            # Extract tables — assign stable block_ids
            for table in page.extract_tables():
                rows = [[cell or "" for cell in row] for row in table if row]
                if rows:
                    serialized = "\n".join(" | ".join(row) for row in rows)
                    bid = f"block_{len(blocks)}"
                    blocks.append(
                        ContentBlock(
                            block_id=bid,
                            block_type="table",
                            content=serialized,
                            table_data=rows,
                            page=page_num,
                        )
                    )
            # Extract text lines, classify headings by font size
            if not toc_entries and page_text:
                for line in page_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Approximate line font size as median of its chars
                    line_sizes = [
                        float(c["size"])
                        for c in page.chars
                        if c.get("size") and c.get("text", "").strip() and c["text"] in line
                    ]
                    median_size = _median(line_sizes) if line_sizes else 0.0
                    tier = size_tiers.get(median_size)
                    btype: str = "heading" if tier in (_TIER_TITLE, _TIER_HEADING) else "text"
                    bid = f"block_{len(blocks)}"
                    blocks.append(ContentBlock(block_id=bid, block_type=btype, content=line, page=page_num))  # type: ignore[arg-type]
            elif page_text:
                # TOC available — emit all lines as text; headings come from TOC
                for line in page_text.splitlines():
                    line = line.strip()
                    if line:
                        bid = f"block_{len(blocks)}"
                        blocks.append(ContentBlock(block_id=bid, block_type="text", content=line, page=page_num))

    sections = _build_sections_from_toc(doc_url, toc_entries, page_count, blocks) if toc_entries else _build_sections_from_headings(doc_url, blocks)
    return blocks, sections, page_count


def _build_sections_from_toc(
    doc_url: str,
    toc: list[tuple[int, str, int]],
    page_count: int,
    blocks: list[ContentBlock] | None = None,
) -> list[SectionNode]:
    """Build SectionNode list from pymupdf TOC entries.

    Wires children_ids (parent→child) and block_ids (section→blocks by page
    range) so HierarchicalChunker can traverse the tree without falling back.
    """
    root_id = _make_section_id(doc_url, [], index=0)
    root = SectionNode(section_id=root_id, level=0, heading="", path=[])
    nodes: list[SectionNode] = [root]
    # Stack tracks (level, SectionNode) for parent resolution
    ancestor_stack: list[tuple[int, SectionNode]] = [(0, root)]

    for level, title, page in toc:
        # Pop stack until we find the parent (strictly lower level)
        while len(ancestor_stack) > 1 and ancestor_stack[-1][0] >= level:
            ancestor_stack.pop()
        parent_node = ancestor_stack[-1][1]

        path = [s.heading for _, s in ancestor_stack if s.heading] + [title]
        # Use len(nodes) as positional index so repeated headings are unique
        sid = _make_section_id(doc_url, path, index=len(nodes))
        node = SectionNode(
            section_id=sid,
            level=level,
            heading=title,
            path=path,
            page_start=page,
            parent_id=parent_node.section_id,
        )
        parent_node.children_ids.append(sid)
        nodes.append(node)
        ancestor_stack.append((level, node))

    # Assign block_ids to leaf sections by page range
    if blocks:
        _assign_block_ids_by_page(nodes, blocks, page_count)

    return nodes


def _build_sections_from_headings(
    doc_url: str, blocks: list[ContentBlock]
) -> list[SectionNode]:
    """Build SectionNode list from heading blocks (font-size fallback).

    Wires root.children_ids and each section's block_ids so
    HierarchicalChunker can traverse the tree.
    """
    root_id = _make_section_id(doc_url, [], index=0)
    root = SectionNode(section_id=root_id, level=0, heading="", path=[])
    sections: list[SectionNode] = [root]
    current: SectionNode = root

    for block in blocks:
        if block.block_type == "heading":
            path = [block.content[:120]]
            # Use len(sections) as positional index so repeated headings are unique
            sid = _make_section_id(doc_url, path, index=len(sections))
            node = SectionNode(
                section_id=sid,
                level=1,
                heading=block.content,
                path=path,
                page_start=block.page,
                parent_id=root_id,
            )
            root.children_ids.append(sid)
            sections.append(node)
            current = node
        else:
            if block.block_id:
                current.block_ids.append(block.block_id)
    return sections


def _assign_block_ids_by_page(
    nodes: list[SectionNode],
    blocks: list[ContentBlock],
    page_count: int,
) -> None:
    """Assign block_ids to leaf sections based on page-range overlap.

    For each leaf SectionNode, determine its page range as
    [page_start, next_sibling_page_start) and assign all blocks whose page
    falls in that range.  Blocks without a page (page=None or page=0) go to
    the root node.
    """
    # Build ordered list of (page_start, section) for non-root nodes
    ordered = [(n.page_start or 0, n) for n in nodes if n.level > 0]
    ordered.sort(key=lambda x: x[0])

    root = next((n for n in nodes if n.level == 0), None)

    for block in blocks:
        if not block.block_id:
            continue
        bp = block.page or 0
        if bp == 0 or not ordered:
            if root is not None:
                root.block_ids.append(block.block_id)
            continue
        # Find the last section whose page_start <= block.page
        assigned: SectionNode | None = root
        for page_start, node in ordered:
            if page_start <= bp:
                assigned = node
            else:
                break
        if assigned is not None:
            assigned.block_ids.append(block.block_id)


def _cluster_font_sizes(sizes: list[float]) -> dict[float, str]:
    """Map font sizes to tier labels using top-3 frequency anchors."""
    if not sizes:
        return {}
    from collections import Counter
    counts = Counter(round(s, 1) for s in sizes)
    # Sorted descending by size
    by_size = sorted(counts.keys(), reverse=True)
    tiers: dict[float, str] = {}
    tier_labels = [_TIER_TITLE, _TIER_HEADING, _TIER_SUBHEADING]
    assigned = 0
    for size in by_size:
        if assigned < len(tier_labels):
            tiers[size] = tier_labels[assigned]
            assigned += 1
        else:
            tiers[size] = _TIER_BODY
    # Everything not in the top 3 largest sizes is body
    max_heading_size = by_size[min(2, len(by_size) - 1)] if by_size else 0.0
    for size in list(tiers.keys()):
        if size < max_heading_size and tiers[size] not in tier_labels[:2]:
            tiers[size] = _TIER_BODY
    return tiers


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0
