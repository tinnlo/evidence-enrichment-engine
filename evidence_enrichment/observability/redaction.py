"""Vendor-agnostic trace value redaction for the observability layer.

All redaction constants, predicates, and helpers live here so both the
LangSmith and Langfuse adapters share a single source of truth.  Neither
adapter needs to be imported to use these utilities.
"""

from __future__ import annotations

import os
from typing import Any

from evidence_enrichment.observability.runtime import (
    OBSERVABILITY_STATE_LOCK,
    get_runtime_observability_config,
)

TRUTHY = {"1", "true", "yes", "on"}

# Fields in trace summaries and trace_payload inputs that may carry extracted
# PII-adjacent values or entity-identifying information.
# When TRACE_REDACT_VALUES=true these are replaced with REDACT_SENTINEL.
# "documents" is handled separately — it is a list of dicts whose nested
# "url" key is redacted rather than dropping the whole list.
REDACTED_FIELDS: frozenset[str] = frozenset({
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
REDACTED_NESTED_DOC_FIELDS: frozenset[str] = frozenset({"url"})

REDACT_SENTINEL: str = "[REDACTED]"


def should_redact() -> bool:
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


def redact_nested_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of a document-list with sensitive nested keys replaced."""
    return [
        {k: (REDACT_SENTINEL if k in REDACTED_NESTED_DOC_FIELDS else v) for k, v in doc.items()}
        for doc in docs
    ]


def maybe_redact(summary: dict[str, Any]) -> dict[str, Any]:
    """Replace sensitive fields in a trace summary dict when redaction is on.

    Redaction is **shallow by design**: only top-level keys listed in
    ``REDACTED_FIELDS`` are replaced, plus the ``url`` key inside each item of
    a top-level ``"documents"`` list.  Nested dicts under any other key are
    passed through unchanged.

    This is intentional — all summarizers produce flat-ish dicts and the set of
    sensitive fields is known ahead of time.  If a future summarizer introduces
    a new sensitive nested structure it must either:
      - add the top-level key to ``REDACTED_FIELDS``, or
      - call ``redact_nested_docs`` explicitly before passing to ``maybe_redact``.
    """
    if not should_redact():
        return summary
    result: dict[str, Any] = {}
    for k, v in summary.items():
        if k in REDACTED_FIELDS:
            result[k] = REDACT_SENTINEL
        elif k == "documents" and isinstance(v, list):
            result[k] = redact_nested_docs(v)
        else:
            result[k] = v
    return result
