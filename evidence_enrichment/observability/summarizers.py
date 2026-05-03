"""Vendor-agnostic trace summarizers for all pipeline stages.

These helpers produce compact, serialisable dicts suitable for attaching to
spans in any backend (LangSmith, Langfuse, or local).  They depend only on
domain models and the shared redaction module — no vendor SDK is imported.
"""

from __future__ import annotations

from typing import Any, Mapping

from evidence_enrichment.core.models.contracts import (
    AnalysisReport,
    FactClaim,
    ParsedDocument,
    RetrievedDocument,
    SearchQueryPlan,
    SearchResult,
    SynthesisResult,
)
from evidence_enrichment.observability.redaction import maybe_redact


def trace_payload_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    payload = inputs.get("trace_payload", {})
    raw = dict(payload) if isinstance(payload, Mapping) else {}
    return maybe_redact(raw)


def summarize_query_plan(plan: SearchQueryPlan) -> dict[str, Any]:
    return maybe_redact({
        "field_name": plan.field_name,
        "entity_id": plan.entity_id,
        "primary_query": plan.primary_query,
        "query_variants": len(plan.query_variants),
        "domain_hints": plan.domain_hints,
    })


def summarize_search_results(results: list[SearchResult]) -> dict[str, Any]:
    return maybe_redact({
        "result_count": len(results),
        "top_urls": [row.url for row in results[:5]],
        "providers": sorted({row.provider.value for row in results}),
    })


def summarize_fetched_documents(documents: list[RetrievedDocument]) -> dict[str, Any]:
    return maybe_redact({
        "document_count": len(documents),
        "urls": [document.url for document in documents],
        "success_count": sum(1 for document in documents if document.fetch_success),
    })


def summarize_parsed_documents(documents: list[ParsedDocument]) -> dict[str, Any]:
    return maybe_redact({
        "document_count": len(documents),
        "documents": [
            {
                "url": document.url,
                "title": document.title,
                "text_chars": len(document.text),
            }
            for document in documents
        ],
    })


def summarize_assessed_documents(documents: list[ParsedDocument]) -> dict[str, Any]:
    return maybe_redact({
        "accepted_count": sum(1 for document in documents if document.accepted_for_analysis),
        "rejected_count": sum(1 for document in documents if not document.accepted_for_analysis),
        "documents": [
            {
                "url": document.url,
                "accepted_for_analysis": document.accepted_for_analysis,
                "entity_match_score": document.entity_match_score,
                "source_authority_score": document.source_authority_score,
                "freshness_score": document.freshness_score,
                "rejection_reason": document.rejection_reason,
            }
            for document in documents
        ],
    })


def summarize_analysis_reports(reports: list[AnalysisReport]) -> dict[str, Any]:
    claims = [claim for report in reports for claim in report.claims]
    return maybe_redact({
        "report_count": len(reports),
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
        "source_urls": [report.source_url for report in reports],
    })


def summarize_analysis_stage(output: tuple[list[AnalysisReport], list[FactClaim], str]) -> dict[str, Any]:
    reports, claims, provider = output
    return maybe_redact({
        **summarize_analysis_reports(reports),
        "provider": provider,
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
    })


def summarize_synthesis(synthesis: SynthesisResult) -> dict[str, Any]:
    return maybe_redact({
        "value": synthesis.value,
        "normalized_value": synthesis.normalized_value,
        "synthesis_confidence": synthesis.synthesis_confidence,
        "supporting_urls": synthesis.supporting_urls,
        "conflict_count": len(synthesis.conflicts),
    })


def summarize_synthesis_stage(output: tuple[SynthesisResult, str]) -> dict[str, Any]:
    synthesis, provider = output
    payload = summarize_synthesis(synthesis)
    payload["provider"] = provider
    return payload


def summarize_review_gate(output: Mapping[str, Any]) -> dict[str, Any]:
    # Intentionally no redaction — this summary contains only scores/decisions,
    # not entity-identifying values.
    return {
        "overall_confidence": float(output.get("overall_confidence", 0.0)),
        "decision": output.get("decision"),
        "gate_reason": output.get("gate_reason"),
    }


def summarize_claims(claims: list[FactClaim]) -> dict[str, Any]:
    return maybe_redact({
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
        "source_urls": [claim.source_url for claim in claims],
    })
