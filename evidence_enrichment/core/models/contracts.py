"""Stage contracts and result models."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from evidence_enrichment.core.models.enums import (
    DocumentType,
    ProviderType,
    ReviewDecision,
    SourceType,
)
from evidence_enrichment.finops.models import LLMUsage
from evidence_enrichment.guardrails.models import GuardrailsReport


def _clamp_01(v: float) -> float:
    """Clamp a float to [0.0, 1.0].

    Raises ValueError for non-finite values (NaN, inf) so that malformed
    provider output fails loudly rather than silently becoming max confidence.
    """
    if not math.isfinite(v):
        raise ValueError(f"Score must be a finite number, got {v!r}")
    return max(0.0, min(1.0, v))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContentBlock(BaseModel):
    """A structural block extracted from a parsed document."""

    block_type: Literal["text", "table", "heading"]
    content: str
    char_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.content)


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
    # Structured extraction fields (populated by parse_with_structure)
    full_text: str = ""
    content_hash: str = ""
    blocks: list[ContentBlock] = Field(default_factory=list)
    mime_type: str = ""

    @field_validator(
        "entity_match_score", "source_authority_score", "freshness_score", mode="after"
    )
    @classmethod
    def _clamp_scores(cls, v: float) -> float:
        return _clamp_01(v)


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
    supporting_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator(
        "analysis_confidence",
        "source_authority_score",
        "freshness_score",
        "entity_match_score",
        mode="after",
    )
    @classmethod
    def _clamp_scores(cls, v: float) -> float:
        return _clamp_01(v)


class AnalysisReport(BaseModel):
    source_url: str
    provider: ProviderType
    claims: list[FactClaim] = Field(default_factory=list)
    reasoning: str | None = None
    llm_usage: LLMUsage | None = None


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
    llm_usage: LLMUsage | None = None

    @field_validator("synthesis_confidence", mode="after")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return _clamp_01(v)


class EnrichmentSource(BaseModel):
    source_type: SourceType
    provider: str
    source_url: str | None = None
    title: str | None = None
    snippet: str | None = None
    confidence: float = 0.0

    @field_validator("confidence", mode="after")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return _clamp_01(v)


class ContextEntry(BaseModel):
    entry_id: str
    path: str
    priority: str
    required: bool = True
    original_chars: int
    included_chars: int
    approx_tokens: int
    truncated: bool = False
    included: bool = True
    content: str = ""


class StageContextBundle(BaseModel):
    stage: str
    budget_chars: int
    used_chars: int
    approx_tokens: int
    entry_ids: list[str] = Field(default_factory=list)
    entries: list[ContextEntry] = Field(default_factory=list)
    excluded_entry_ids: list[str] = Field(default_factory=list)


class ResolvedContextBundle(BaseModel):
    task_name: str
    manifest_path: str
    entity_id: str
    field_name: str
    load_order: list[str] = Field(default_factory=list)
    stages: dict[str, StageContextBundle] = Field(default_factory=dict)


class PipelineRunResult(BaseModel):
    entity_id: str
    field_name: str
    mode: str
    search_plan: SearchQueryPlan
    resolved_context: ResolvedContextBundle | None = None
    search_results: list[SearchResult] = Field(default_factory=list)
    parsed_documents: list[ParsedDocument] = Field(default_factory=list)
    analysis_reports: list[AnalysisReport] = Field(default_factory=list)
    synthesis: SynthesisResult
    sources: list[EnrichmentSource] = Field(default_factory=list)
    overall_confidence: float
    decision: ReviewDecision
    gate_reason: str
    output_value: str | None = None
    trace_id: str | None = None
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    retrieval_chunk_count: int = 0
    retrieval_top_scores: list[float] = Field(default_factory=list)
    fallback_from_live: bool = False
    retrieval_degraded: bool = False
    guardrails_report: GuardrailsReport | None = None
    finops_summary: dict[str, Any] | None = None
