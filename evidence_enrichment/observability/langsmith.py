"""LangSmith helpers for opt-in tracing."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Mapping

# langsmith is an optional dependency; import lazily so the module can be
# loaded (and settings can be applied) even when the package is absent.
if TYPE_CHECKING:
    from langsmith import Client

from evidence_enrichment.core.models.contracts import AnalysisReport, FactClaim, ParsedDocument, RetrievedDocument, SearchQueryPlan, SearchResult, SynthesisResult
from evidence_enrichment.observability.router import langsmith_wrapping_enabled
from evidence_enrichment.observability.runtime import (
    OBSERVABILITY_STATE_LOCK,
    get_runtime_observability_config,
    is_evicted_observability_value,
    resolve_evicted_observability_value,
)

logger = logging.getLogger(__name__)

TRUTHY = {"1", "true", "yes", "on"}

# Fields in trace summaries and trace_payload inputs that may carry extracted
# PII-adjacent values or entity-identifying information.
# When TRACE_REDACT_VALUES=true these are replaced with the sentinel below.
# "documents" is handled separately — it is a list of dicts whose nested
# "url" key is redacted rather than dropping the whole list.
_REDACTED_FIELDS = frozenset({
    # Extracted field candidates and evidence values
    "candidate_values",
    "value",
    "normalized_value",
    # URL fields (various names used across stages)
    "supporting_urls",
    "source_urls",
    "top_urls",
    "urls",
    "document_urls",           # evidence_assessment trace_payload
    "accepted_document_urls",  # analysis trace_payload
    # Query / search fields
    "primary_query",
    "domain_hints",
    # Entity-identifying fields
    "company_name",            # query_plan trace_payload
    "context_entry_ids",       # query_plan trace_payload (opaque IDs but entity-linked)
})
# Keys inside nested document dicts that carry sensitive values.
_REDACTED_NESTED_DOC_FIELDS = frozenset({"url"})
_REDACT_SENTINEL = "[REDACTED]"


def _should_redact() -> bool:
    """Return True when trace value redaction is enabled.

    Resolution order:
    1. Task-local runtime config derived from the active ``Settings`` instance.
    2. Process-global default config written by the last successful
       ``Settings.load()``.
    3. ``TRACE_REDACT_VALUES`` in ``os.environ`` — the canonical signal
       exported by ``Settings.load()`` after a successful load.
    4. When the env var is **absent** (not set at all), defaults to ``True``
       to match the ``Settings.trace_redact_values`` model default.  This
       ensures pre-``Settings.load()`` code paths (e.g. LangSmith client
       wrapping checks during provider initialization) are privacy-safe.

    Deliberately does **not** fall back to ``get_settings()``.  Calling
    ``get_settings()`` from inside a summarizer would trigger a cold-cache
    ``Settings.load()`` if settings have never been loaded, which mutates
    global observability env as a side effect of an otherwise read-only
    redaction check.
    """
    runtime_config = get_runtime_observability_config()
    if runtime_config is not None and runtime_config.trace_redact_values is not None:
        return runtime_config.trace_redact_values

    with OBSERVABILITY_STATE_LOCK:
        raw = os.getenv("TRACE_REDACT_VALUES")
    if raw is None:
        return True
    return str(raw).strip().lower() in TRUTHY


def _redact_nested_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of a document-list with sensitive nested keys replaced."""
    return [
        {k: (_REDACT_SENTINEL if k in _REDACTED_NESTED_DOC_FIELDS else v) for k, v in doc.items()}
        for doc in docs
    ]


def _maybe_redact(summary: dict[str, Any]) -> dict[str, Any]:
    """Replace sensitive fields in a trace summary dict when redaction is on."""
    if not _should_redact():
        return summary
    result: dict[str, Any] = {}
    for k, v in summary.items():
        if k in _REDACTED_FIELDS:
            result[k] = _REDACT_SENTINEL
        elif k == "documents" and isinstance(v, list):
            result[k] = _redact_nested_docs(v)
        else:
            result[k] = v
    return result


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUTHY


def _pick_value(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _mapping_value(mapping: Mapping[str, str | None], key: str) -> str | None:
    value = mapping.get(key)
    return value if value else None


def apply_langsmith_env(
    env_values: Mapping[str, str] | None = None,
    *,
    ambient_env: Mapping[str, str | None] | None = None,
    clear_missing: bool = False,
) -> bool:
    with OBSERVABILITY_STATE_LOCK:
        env_values = env_values or {}
        ambient_env = os.environ if ambient_env is None else ambient_env
        api_key = _pick_value(
            resolve_evicted_observability_value(
                "LANGSMITH_API_KEY",
                _mapping_value(ambient_env, "LANGSMITH_API_KEY"),
            ),
            resolve_evicted_observability_value(
                "LANGSMITH_API_KEY",
                env_values.get("LANGSMITH_API_KEY"),
            ),
            resolve_evicted_observability_value(
                "LANGCHAIN_API_KEY",
                _mapping_value(ambient_env, "LANGCHAIN_API_KEY"),
            ),
            resolve_evicted_observability_value(
                "LANGCHAIN_API_KEY",
                env_values.get("LANGCHAIN_API_KEY"),
            ),
        )
        project = _pick_value(
            _mapping_value(ambient_env, "LANGSMITH_PROJECT"),
            env_values.get("LANGSMITH_PROJECT"),
            _mapping_value(ambient_env, "LANGCHAIN_PROJECT"),
            env_values.get("LANGCHAIN_PROJECT"),
        )
        tracing_value = _pick_value(
            _mapping_value(ambient_env, "LANGSMITH_TRACING"),
            env_values.get("LANGSMITH_TRACING"),
            _mapping_value(ambient_env, "LANGCHAIN_TRACING_V2"),
            env_values.get("LANGCHAIN_TRACING_V2"),
        )

        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGCHAIN_API_KEY"] = api_key
        elif is_evicted_observability_value(
            "LANGSMITH_API_KEY",
            os.getenv("LANGSMITH_API_KEY"),
        ) or is_evicted_observability_value(
            "LANGCHAIN_API_KEY",
            os.getenv("LANGCHAIN_API_KEY"),
        ):
            os.environ.pop("LANGSMITH_API_KEY", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
        elif clear_missing:
            os.environ.pop("LANGSMITH_API_KEY", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)

        if project:
            os.environ["LANGSMITH_PROJECT"] = project
            os.environ["LANGCHAIN_PROJECT"] = project
        elif clear_missing:
            os.environ.pop("LANGSMITH_PROJECT", None)
            os.environ.pop("LANGCHAIN_PROJECT", None)

        if tracing_value is not None:
            normalized = "true" if _is_truthy(tracing_value) else "false"
            os.environ["LANGSMITH_TRACING"] = normalized
            os.environ["LANGCHAIN_TRACING_V2"] = normalized
        elif clear_missing:
            os.environ.pop("LANGSMITH_TRACING", None)
            os.environ.pop("LANGCHAIN_TRACING_V2", None)

        tracing_on = _is_truthy(os.getenv("LANGSMITH_TRACING"))
        has_key = bool(
            os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        )
        fully_configured = tracing_on and has_key

        if not fully_configured:
            if tracing_on and not has_key:
                logger.warning(
                    "LangSmith partially configured: LANGSMITH_TRACING=true but no API key found. "
                    "LangSmith will remain inactive until LANGSMITH_API_KEY is set."
                )
            elif has_key and not tracing_on:
                logger.warning(
                    "LangSmith partially configured: API key present but LANGSMITH_TRACING is not true. "
                    "LangSmith will remain inactive until LANGSMITH_TRACING=true is set."
                )

        return fully_configured

def get_langsmith_client() -> "Client | None":
    """Return a LangSmith client if tracing is active, or None.

    Not cached — re-evaluates env on every call so that repeated
    Settings.load() calls (backend flips, credential rotation) are
    reflected immediately.

    Also respects ``OBSERVABILITY_BACKEND``: if the backend is set to a
    value that does not include LangSmith (e.g. ``langfuse`` or ``none``),
    returns ``None`` even when ``LANGSMITH_TRACING=true`` is present in
    the environment.
    """
    if not langsmith_wrapping_enabled():
        return None

    with OBSERVABILITY_STATE_LOCK:
        if not apply_langsmith_env():
            return None

        try:
            from langsmith import Client  # lazy — optional dependency
        except ImportError:
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
    raw = dict(payload) if isinstance(payload, Mapping) else {}
    return _maybe_redact(raw)


def summarize_query_plan(plan: SearchQueryPlan) -> dict[str, Any]:
    return _maybe_redact({
        "field_name": plan.field_name,
        "entity_id": plan.entity_id,
        "primary_query": plan.primary_query,
        "query_variants": len(plan.query_variants),
        "domain_hints": plan.domain_hints,
    })


def summarize_search_results(results: list[SearchResult]) -> dict[str, Any]:
    return _maybe_redact({
        "result_count": len(results),
        "top_urls": [row.url for row in results[:5]],
        "providers": sorted({row.provider.value for row in results}),
    })


def summarize_fetched_documents(documents: list[RetrievedDocument]) -> dict[str, Any]:
    return _maybe_redact({
        "document_count": len(documents),
        "urls": [document.url for document in documents],
        "success_count": sum(1 for document in documents if document.fetch_success),
    })


def summarize_parsed_documents(documents: list[ParsedDocument]) -> dict[str, Any]:
    return _maybe_redact({
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
    return _maybe_redact({
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
    return _maybe_redact({
        "report_count": len(reports),
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
        "source_urls": [report.source_url for report in reports],
    })


def summarize_analysis_stage(output: tuple[list[AnalysisReport], list[FactClaim], str]) -> dict[str, Any]:
    reports, claims, provider = output
    return _maybe_redact({
        **summarize_analysis_reports(reports),
        "provider": provider,
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
    })


def summarize_synthesis(synthesis: SynthesisResult) -> dict[str, Any]:
    return _maybe_redact({
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
    return {
        "overall_confidence": float(output.get("overall_confidence", 0.0)),
        "decision": output.get("decision"),
        "gate_reason": output.get("gate_reason"),
    }


def summarize_claims(claims: list[FactClaim]) -> dict[str, Any]:
    return _maybe_redact({
        "claim_count": len(claims),
        "candidate_values": [claim.candidate_value for claim in claims],
        "source_urls": [claim.source_url for claim in claims],
    })
