"""Text parsing and document assessment."""

from __future__ import annotations

from evidence_enrichment.core.fetch.fetcher import html_to_text, registrable_domain
from evidence_enrichment.core.models.contracts import ParsedDocument, RetrievedDocument
from evidence_enrichment.core.models.enums import DocumentType
from evidence_enrichment.core.evidence.assessment import compute_freshness_score, compute_source_authority
from evidence_enrichment.core.evidence.matching import compute_entity_match_score


class TextParser:
    def parse(self, document: RetrievedDocument, company_name: str) -> ParsedDocument:
        text = html_to_text(document.body)
        excerpt = text[:500]
        domain = registrable_domain(document.final_url)
        doc_type = DocumentType.COMPANY_WEBSITE if any(
            marker in document.final_url.lower() for marker in ["/about", "/company", "/corporate", "/contact"]
        ) else DocumentType.UNKNOWN
        entity_match = compute_entity_match_score(company_name, text)
        authority = compute_source_authority(domain)
        freshness = compute_freshness_score()
        accepted = entity_match >= 0.55
        return ParsedDocument(
            url=document.final_url,
            title=document.title,
            content_type=document.content_type,
            text=text,
            excerpt=excerpt,
            document_type=doc_type,
            entity_match_score=entity_match,
            source_authority_score=authority,
            freshness_score=freshness,
            accepted_for_analysis=accepted,
            rejection_reason=None if accepted else "low_entity_match",
        )

