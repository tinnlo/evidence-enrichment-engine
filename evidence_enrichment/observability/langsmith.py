"""LangSmith helpers for opt-in tracing."""

from __future__ import annotations

import asyncio
import inspect
import os
from functools import lru_cache
from typing import Any, Mapping

from langsmith import Client

from evidence_enrichment.core.models.contracts import AnalysisReport, FactClaim, ParsedDocument, RetrievedDocument, SearchQueryPlan, SearchResult, SynthesisResult

TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUTHY


def _pick_value(*values: str | None) -> str | None:
    for value in values:
        if value is not None:
            return value
    return None


def apply_langsmith_env(env_values: Mapping[str, str] | None = None) -> bool:
    env_values = env_values or {}
    api_key = _pick_value(
        os.getenv("LANGSMITH_API_KEY"),
        env_values.get("LANGSMITH_API_KEY"),
        os.getenv("LANGCHAIN_API_KEY"),
        env_values.get("LANGCHAIN_API_KEY"),
    )
    project = _pick_value(
        os.getenv("LANGSMITH_PROJECT"),
        env_values.get("LANGSMITH_PROJECT"),
        os.getenv("LANGCHAIN_PROJECT"),
        env_values.get("LANGCHAIN_PROJECT"),
    )
    tracing_value = _pick_value(
        os.getenv("LANGSMITH_TRACING"),
        env_values.get("LANGSMITH_TRACING"),
        os.getenv("LANGCHAIN_TRACING_V2"),
        env_values.get("LANGCHAIN_TRACING_V2"),
    )

    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
    if tracing_value is not None:
        os.environ["LANGSMITH_TRACING"] = "true" if _is_truthy(tracing_value) else "false"

    return _is_truthy(os.getenv("LANGSMITH_TRACING"))


@lru_cache(maxsize=1)
def get_langsmith_client() -> Client | None:
    if not apply_langsmith_env():
        return None
    try:
        return Client()
    except Exception:
        return None


def flush_langsmith_traces() -> None:
    client = get_langsmith_client()
    flush = getattr(client, "flush", None) if client else None
    if flush is None:
        return

    try:
        result = flush()
    except Exception:
        return

    if not inspect.isawaitable(result):
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(result)
    else:
        loop.create_task(result)


def trace_payload_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    payload = inputs.get("trace_payload", {})
    return dict(payload) if isinstance(payload, Mapping) else {}


def summarize_query_plan(plan: SearchQueryPlan) -> dict[str, Any]:
    return {
        "field_name": plan.field_name,
        "entity_id": plan.entity_id,
        "primary_query": plan.primary_query,
        "query_variants": len(plan.query_variants),
        "domain_hints": plan.domain_hints,
    }


def summarize_search_results(results: list[SearchResult]) -> dict[str, Any]:
    return {
        "result_count": len(results),
        "top_urls": [row.url for row in results[:5]],
        "providers": sorted({row.provider.value for row in results}),
    }


def summarize_fetched_documents(documents: list[RetrievedDocument]) -> dict[str, Any]:
    return {
        "document_count": len(documents),
        "urls": [document.url for document in documents],
        "success_count": sum(1 for document in documents if document.fetch_success),
    }


def summarize_parsed_documents(documents: list[ParsedDocument]) -> dict[str, Any]:
    return {
        "document_count": len(documents),
        "documents": [
            {
                "url": document.url,
                "title": document.title,
                "text_chars": len(document.text),
            }
            for document in documents
        ],
    }


def summarize_assessed_documents(documents: list[ParsedDocument]) -> dict[str, Any]:
    return {
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
    }


def summarize_analysis_reports(reports: list[AnalysisReport]) -> dict[str, Any]:
    claims = [claim for report in reports for claim in report.claims]
    return {
        "report_count": len(reports),
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
        "source_urls": [report.source_url for report in reports],
    }


def summarize_analysis_stage(output: tuple[list[AnalysisReport], list[FactClaim], str]) -> dict[str, Any]:
    reports, claims, provider = output
    return {
        **summarize_analysis_reports(reports),
        "provider": provider,
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
    }


def summarize_synthesis(synthesis: SynthesisResult) -> dict[str, Any]:
    return {
        "value": synthesis.value,
        "normalized_value": synthesis.normalized_value,
        "synthesis_confidence": synthesis.synthesis_confidence,
        "supporting_urls": synthesis.supporting_urls,
        "conflict_count": len(synthesis.conflicts),
    }


def summarize_synthesis_stage(output: tuple[SynthesisResult, str]) -> dict[str, Any]:
    synthesis, provider = output
    payload = summarize_synthesis(synthesis)
    payload["provider"] = provider
    return payload


def summarize_review_gate(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall_confidence": float(output.get("overall_confidence", 0.0)),
        "decision": output.get("decision"),
        "gate_reason": output.get("gate_reason"),
    }


def summarize_claims(claims: list[FactClaim]) -> dict[str, Any]:
    return {
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
        "source_urls": [claim.source_url for claim in claims],
    }
