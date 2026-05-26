"""GenericTextParser — wraps TextParser.parse_with_structure for non-HTML text types.

Handles: text/*, application/json, application/xml, application/xhtml.
Preserves current retrieval-mode behaviour (block extraction, full_text,
content_hash, mime_type, plain-text table detection).
"""

from __future__ import annotations

from evidence_enrichment.core.models.contracts import ParsedDocument, RetrievedDocument
from evidence_enrichment.core.parse.parser import TextParser

_HANDLED_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
)


class GenericTextParser:
    """Wraps TextParser.parse_with_structure for all non-HTML text content types."""

    def can_parse(self, doc: RetrievedDocument) -> bool:
        ct = doc.content_type.split(";")[0].strip().lower()
        return ct.startswith(_HANDLED_PREFIXES)

    def parse(self, doc: RetrievedDocument) -> ParsedDocument:
        return TextParser().parse_with_structure(doc)
