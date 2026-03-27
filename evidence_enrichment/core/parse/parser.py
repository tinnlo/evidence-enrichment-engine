"""Text parsing."""

from __future__ import annotations

from evidence_enrichment.core.fetch.fetcher import html_to_text
from evidence_enrichment.core.models.contracts import ParsedDocument, RetrievedDocument


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

