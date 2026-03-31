"""Text parsing."""

from __future__ import annotations

import hashlib
import re

from evidence_enrichment.core.fetch.fetcher import html_to_text
from evidence_enrichment.core.models.contracts import ContentBlock, ParsedDocument, RetrievedDocument

# Patterns for structural HTML extraction
_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_HEADING_RE = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

# Heuristics for plain-text tables (pipe-delimited, tab-delimited, markdown)
_PIPE_ROW_RE = re.compile(r"^\|.+\|", re.MULTILINE)
_TAB_ROW_RE = re.compile(r"^.+\t.+", re.MULTILINE)


class TextParser:
    def parse(self, document: RetrievedDocument) -> ParsedDocument:
        text = html_to_text(document.body)
        excerpt = text[:500]
        return ParsedDocument(
            url=document.final_url,
            title=document.title,
            content_type=document.content_type,
            text=text,
            excerpt=excerpt,
        )

    def parse_with_structure(self, document: RetrievedDocument) -> ParsedDocument:
        """Parse and extract block-level structure (headings, tables, text).

        The existing ``text`` field is populated identically to ``parse()`` for
        backward compatibility.  The new ``blocks`` field carries structured
        content for downstream chunking and retrieval.
        """
        text = html_to_text(document.body)
        excerpt = text[:500]
        blocks = _extract_blocks(document.body)
        full_text = "\n\n".join(b.content for b in blocks) if blocks else text
        content_hash = hashlib.sha256(full_text.encode()).hexdigest()
        mime_type = document.content_type.split(";")[0].strip()
        return ParsedDocument(
            url=document.final_url,
            title=document.title,
            content_type=document.content_type,
            text=text,
            excerpt=excerpt,
            full_text=full_text,
            content_hash=content_hash,
            blocks=blocks,
            mime_type=mime_type,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_tags(html_fragment: str) -> str:
    """Remove all HTML tags and return plain text."""
    return " ".join(_TAG_RE.sub(" ", html_fragment).split()).strip()


def _extract_blocks(body: str) -> list[ContentBlock]:
    """Extract content blocks from HTML, preserving table boundaries.

    Strategy:
    1. Pull out all <table> regions and record their byte offsets.
    2. Pull out all <h1-6> headings.
    3. The remaining text (between structural elements) is split into
       paragraph-sized text blocks.
    4. Detect plain-text tables (pipe/tab-delimited) in leftover text blocks.
    """
    blocks: list[ContentBlock] = []

    # --- Pass 1: locate tables and headings by offset ---
    table_spans: list[tuple[int, int, str]] = []  # (start, end, plain_text)
    for m in _TABLE_RE.finditer(body):
        plain = _strip_tags(m.group())
        if plain:
            table_spans.append((m.start(), m.end(), plain))

    heading_spans: list[tuple[int, int, str]] = []
    for m in _HEADING_RE.finditer(body):
        plain = _strip_tags(m.group())
        if plain:
            heading_spans.append((m.start(), m.end(), plain))

    # --- Pass 2: everything else is text ---
    # Build a sorted list of all known structural regions
    structural: list[tuple[int, int, str, str]] = []  # (start, end, block_type, plain)
    for s, e, p in table_spans:
        structural.append((s, e, "table", p))
    for s, e, p in heading_spans:
        structural.append((s, e, "heading", p))
    structural.sort(key=lambda x: x[0])

    # Collect text regions between structural elements
    cursor = 0
    ordered: list[tuple[int, str, str]] = []  # (start, block_type, plain)
    for s, e, btype, plain in structural:
        if s > cursor:
            gap_html = body[cursor:s]
            gap_text = html_to_text(gap_html)
            if gap_text:
                ordered.extend(_split_text_into_blocks(gap_text, cursor))
        ordered.append((s, btype, plain))
        cursor = e

    # Trailing text
    if cursor < len(body):
        tail_text = html_to_text(body[cursor:])
        if tail_text:
            ordered.extend(_split_text_into_blocks(tail_text, cursor))

    # If the body had no structure at all, fall back to plain text blocks
    if not ordered:
        plain_all = html_to_text(body)
        if plain_all:
            ordered.extend(_split_text_into_blocks(plain_all, 0))

    # Sort by original position and build ContentBlock list
    ordered.sort(key=lambda x: x[0])
    for _, btype, plain in ordered:
        if plain:
            blocks.append(ContentBlock(block_type=btype, content=plain))  # type: ignore[arg-type]

    return blocks


def _split_text_into_blocks(
    text: str, base_offset: int, min_chars: int = 80
) -> list[tuple[int, str, str]]:
    """Split a plain-text region into paragraph chunks.

    Also detects pipe/tab-delimited plain-text tables within the region.
    Returns list of (offset, block_type, content) tuples.
    """
    results: list[tuple[int, str, str]] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    if not paragraphs:
        # Single paragraph — check for plain-text table heuristic
        if len(text) >= min_chars:
            btype = _detect_text_table(text)
            results.append((base_offset, btype, text))
        return results

    for para in paragraphs:
        if len(para) < min_chars:
            continue
        btype = _detect_text_table(para)
        results.append((base_offset, btype, para))

    return results


def _detect_text_table(text: str) -> str:
    """Return 'table' if the text looks like a plain-text table, else 'text'."""
    pipe_rows = _PIPE_ROW_RE.findall(text)
    tab_rows = _TAB_ROW_RE.findall(text)
    lines = text.splitlines()
    # Need at least 2 consistent rows to count as a table
    if len(pipe_rows) >= 2 or len(tab_rows) >= 2:
        return "table"
    # Markdown separator pattern: row of dashes/pipes
    separator_count = sum(1 for l in lines if re.match(r"^\|?[-| :]+\|?$", l.strip()))
    if separator_count >= 1 and len(pipe_rows) >= 1:
        return "table"
    return "text"

