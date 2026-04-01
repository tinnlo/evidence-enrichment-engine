"""Evidence assessment stage."""

from __future__ import annotations

from evidence_enrichment.core.evidence.assessment import compute_freshness_score, compute_source_authority
from evidence_enrichment.core.evidence.matching import compute_entity_match_score
from evidence_enrichment.core.fetch.fetcher import normalized_hostname
from evidence_enrichment.core.models.contracts import ParsedDocument
from evidence_enrichment.core.models.enums import DocumentType


class EvidenceAssessor:
    def assess(self, document: ParsedDocument, company_name: str) -> ParsedDocument:
        domain = normalized_hostname(document.url)
        doc_type = DocumentType.COMPANY_WEBSITE if any(
            marker in document.url.lower() for marker in ["/about", "/company", "/corporate", "/contact"]
        ) else DocumentType.UNKNOWN
        entity_match = compute_entity_match_score(company_name, document.text)
        authority = compute_source_authority(domain)
        freshness = compute_freshness_score()
        document.document_type = doc_type
        document.entity_match_score = entity_match
        document.source_authority_score = authority
        document.freshness_score = freshness
        document.accepted_for_analysis = entity_match >= 0.55
        document.rejection_reason = None if document.accepted_for_analysis else "low_entity_match"
        return document

