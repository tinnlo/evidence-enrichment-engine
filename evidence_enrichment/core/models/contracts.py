"""Stage contracts and result models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from evidence_enrichment.core.models.enums import DocumentType, ProviderType, ReviewDecision, SourceType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SearchQueryPlan(BaseModel):
    field_name: str
    entity_id: str
    primary_query: str
    query_variants: list[str] = Field(default_factory=list)
    domain_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    provider: ProviderType
    rank: int
    domain: str
    source_tier: str = "other"
    retrieved_at: datetime = Field(default_factory=_utc_now)


class RetrievedDocument(BaseModel):
    url: str
    final_url: str
    title: str
    content_type: str
    body: str
    provider: str
    fetch_success: bool = True
    error: str | None = None


class ParsedDocument(BaseModel):
    url: str
    title: str
    content_type: str
    text: str
    excerpt: str
    document_type: DocumentType = DocumentType.UNKNOWN
    entity_match_score: float = 0.0
    source_authority_score: float = 0.0
    freshness_score: float = 0.7
    accepted_for_analysis: bool = False
    rejection_reason: str | None = None


class FactClaim(BaseModel):
    field_name: str
    candidate_value: str
    supporting_excerpt: str
    source_url: str
    source_title: str
    analysis_confidence: float
    source_authority_score: float
    freshness_score: float
    entity_match_score: float


class AnalysisReport(BaseModel):
    source_url: str
    provider: ProviderType
    claims: list[FactClaim] = Field(default_factory=list)
    reasoning: str | None = None


class ConflictManifest(BaseModel):
    field_name: str
    candidate_values: list[str]
    source_urls: list[str]
    reason: str


class SynthesisResult(BaseModel):
    field_name: str
    value: str | None = None
    normalized_value: str | None = None
    reasoning: str
    synthesis_confidence: float
    supporting_urls: list[str] = Field(default_factory=list)
    conflicts: list[ConflictManifest] = Field(default_factory=list)


class EnrichmentSource(BaseModel):
    source_type: SourceType
    provider: str
    source_url: str | None = None
    title: str | None = None
    snippet: str | None = None
    confidence: float = 0.0


class PipelineRunResult(BaseModel):
    entity_id: str
    field_name: str
    mode: str
    search_plan: SearchQueryPlan
    search_results: list[SearchResult] = Field(default_factory=list)
    parsed_documents: list[ParsedDocument] = Field(default_factory=list)
    analysis_reports: list[AnalysisReport] = Field(default_factory=list)
    synthesis: SynthesisResult
    sources: list[EnrichmentSource] = Field(default_factory=list)
    overall_confidence: float
    decision: ReviewDecision
    gate_reason: str
    output_value: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)

