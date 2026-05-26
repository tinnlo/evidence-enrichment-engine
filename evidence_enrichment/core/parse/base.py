"""Base interfaces for the pluggable parser stack."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from evidence_enrichment.core.models.contracts import ParsedDocument, RetrievedDocument

# Text MIME prefixes that the GenericTextParser fallback handles.
# Must stay in sync with fetcher._ALLOWED_CONTENT_PREFIXES.
_TEXT_FALLBACK_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
)


class UnsupportedContentTypeError(ValueError):
    """Raised by ParserRegistry when no parser is registered for a binary content type."""


class DocumentParser(Protocol):
    def can_parse(self, doc: "RetrievedDocument") -> bool: ...
    def parse(self, doc: "RetrievedDocument") -> "ParsedDocument": ...


class ParserRegistry:
    """Dispatch document parsing by MIME type.

    Resolution order:
    1. Try each explicitly registered parser in registration order (first wins).
    2. If none match and the content type is a text type, fall back to
       ``GenericTextParser`` (wraps ``TextParser.parse_with_structure``).
    3. If none match and the content type is binary, raise
       ``UnsupportedContentTypeError``.

    This means:
    - ``text/html`` is handled by ``HTMLStructuredParser`` (registered at
      coordinator init) — richer section-tree extraction.
    - ``text/*``, ``application/json``, ``application/xml``, ``application/xhtml``
      fall back to ``GenericTextParser`` — identical to current TextParser behaviour.
    - ``application/pdf`` (and any future binary type) requires an explicit
      registered parser; absence raises a clear error rather than a silent
      empty parse.
    """

    def __init__(self) -> None:
        self._parsers: list[DocumentParser] = []

    def register(self, parser: DocumentParser) -> None:
        self._parsers.append(parser)

    def parse(self, doc: "RetrievedDocument") -> "ParsedDocument":
        for parser in self._parsers:
            if parser.can_parse(doc):
                return parser.parse(doc)

        ct = doc.content_type.split(";")[0].strip().lower()
        if ct.startswith(_TEXT_FALLBACK_PREFIXES):
            from evidence_enrichment.core.parse.generic_text import GenericTextParser
            return GenericTextParser().parse(doc)

        raise UnsupportedContentTypeError(
            f"No parser registered for content_type={doc.content_type!r}. "
            "Install the .[retrieval] extra to enable PDF parsing."
        )
