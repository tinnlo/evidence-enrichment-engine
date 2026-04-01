"""Tests for parser.py — block extraction and plain-text table detection."""

from __future__ import annotations

import pytest

from evidence_enrichment.core.parse.parser import _extract_blocks, _html_gap_to_text
from evidence_enrichment.core.providers.agents import ProviderParseError, _extract_json


# ---------------------------------------------------------------------------
# _html_gap_to_text — line structure preservation
# ---------------------------------------------------------------------------


def test_html_gap_to_text_preserves_newlines_from_br() -> None:
    html = "<p>row one</p><br/>row two<br/>row three"
    text = _html_gap_to_text(html)
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) >= 2, f"Expected multiple lines, got: {lines}"


def test_html_gap_to_text_strips_inline_tags() -> None:
    html = "<span><b>hello</b> <i>world</i></span>"
    text = _html_gap_to_text(html)
    assert "<" not in text
    assert "hello" in text
    assert "world" in text


# ---------------------------------------------------------------------------
# _extract_blocks — pipe-delimited plain-text table in non-<table> region
# ---------------------------------------------------------------------------

_PIPE_TABLE_HTML = """\
<div>
<p>Summary paragraph with enough characters to clear the minimum block size threshold here.</p>
<p>| Company | HQ | Revenue |
| --- | --- | --- |
| Acme Corp | USA | $1B |
| Beta Ltd | UK | $500M |</p>
</div>
"""


def test_extract_blocks_detects_pipe_table_in_gap() -> None:
    blocks = _extract_blocks(_PIPE_TABLE_HTML)
    block_types = [b.block_type for b in blocks]
    assert "table" in block_types, f"Expected a table block; got types: {block_types}"


def test_extract_blocks_pipe_table_content_preserved() -> None:
    blocks = _extract_blocks(_PIPE_TABLE_HTML)
    table_blocks = [b for b in blocks if b.block_type == "table"]
    assert table_blocks, "No table block found"
    combined = " ".join(b.content for b in table_blocks)
    assert "Acme Corp" in combined


# ---------------------------------------------------------------------------
# _extract_blocks — tab-delimited table detection
# ---------------------------------------------------------------------------

_TAB_TABLE_HTML = """\
<div>
<p>Header paragraph providing sufficient length for the minimum character check.</p>
<p>Company\tHQ\tRevenue
Acme Corp\tUSA\t$1B
Beta Ltd\tUK\t$500M</p>
</div>
"""


def test_extract_blocks_detects_tab_table_in_gap() -> None:
    blocks = _extract_blocks(_TAB_TABLE_HTML)
    block_types = [b.block_type for b in blocks]
    assert "table" in block_types, f"Expected a table block; got types: {block_types}"


# ---------------------------------------------------------------------------
# Short paragraph preservation
# ---------------------------------------------------------------------------


def test_extract_blocks_preserves_short_paragraphs() -> None:
    """Short but high-signal paragraphs should not be silently dropped."""
    html = (
        "<div>"
        "<p>Headquarters: USA</p>"
        "<p>This is a longer paragraph that provides sufficient context and detail "
        "about the company's operations and global presence across multiple regions.</p>"
        "</div>"
    )
    blocks = _extract_blocks(html)
    combined = " ".join(b.content for b in blocks)
    assert "Headquarters: USA" in combined, f"Short paragraph was dropped; blocks: {[b.content for b in blocks]}"


# ---------------------------------------------------------------------------
# _extract_json — ProviderParseError on malformed output
# ---------------------------------------------------------------------------


def test_extract_json_raises_on_no_json() -> None:
    with pytest.raises(ProviderParseError, match="No JSON object found"):
        _extract_json("This is just plain text with no json at all")


def test_extract_json_raises_on_malformed_json() -> None:
    with pytest.raises(ProviderParseError):
        _extract_json('{"claims": [broken }')


def test_extract_json_succeeds_on_valid_json() -> None:
    result = _extract_json('Here is the result: {"value": "USA", "confidence": 0.95}')
    assert result["value"] == "USA"
