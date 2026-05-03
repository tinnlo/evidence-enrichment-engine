"""LangSmith helpers for opt-in tracing."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import TYPE_CHECKING, Mapping

# langsmith is an optional dependency; import lazily so the module can be
# loaded (and settings can be applied) even when the package is absent.
if TYPE_CHECKING:
    from langsmith import Client

# ---------------------------------------------------------------------------
# Deprecated aliases — kept for backward compatibility only.
# Import from their canonical locations instead:
#   redaction constants/helpers  →  evidence_enrichment.observability.redaction
#   summarize_* / trace_payload  →  evidence_enrichment.observability.summarizers
# These aliases will be removed in a future version.
# ---------------------------------------------------------------------------
from evidence_enrichment.observability.redaction import (  # noqa: F401
    REDACT_SENTINEL as _REDACT_SENTINEL,
    REDACTED_FIELDS as _REDACTED_FIELDS,
    REDACTED_NESTED_DOC_FIELDS as _REDACTED_NESTED_DOC_FIELDS,
    maybe_redact as _maybe_redact,
    redact_nested_docs as _redact_nested_docs,
    should_redact as _should_redact,
)
from evidence_enrichment.observability.router import langsmith_wrapping_enabled
from evidence_enrichment.observability.runtime import (
    OBSERVABILITY_STATE_LOCK,
    is_evicted_observability_value,
    resolve_evicted_observability_value,
)
from evidence_enrichment.observability.summarizers import (  # noqa: F401
    summarize_analysis_reports,
    summarize_analysis_stage,
    summarize_assessed_documents,
    summarize_claims,
    summarize_fetched_documents,
    summarize_parsed_documents,
    summarize_query_plan,
    summarize_review_gate,
    summarize_search_results,
    summarize_synthesis,
    summarize_synthesis_stage,
    trace_payload_inputs,
)

logger = logging.getLogger(__name__)

TRUTHY = {"1", "true", "yes", "on"}


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



