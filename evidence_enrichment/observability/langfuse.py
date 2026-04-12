"""Langfuse helpers for opt-in tracing (parallel to LangSmith)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Callable, Mapping

# Re-export summarize_* helpers so both backends share them via __init__
from evidence_enrichment.observability.langsmith import (  # noqa: F401
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
        if value is not None:
            return value
    return None


def apply_langfuse_env(env_values: Mapping[str, str] | None = None) -> bool:
    """Set LANGFUSE_* env vars from env_values / os.environ.

    Mirrors apply_langsmith_env.  Returns True when secret key is present.
    Accepts legacy LANGFUSE_HOST for back-compat alongside LANGFUSE_BASE_URL.
    """
    env_values = env_values or {}
    secret_key = _pick_value(
        os.getenv("LANGFUSE_SECRET_KEY"),
        env_values.get("LANGFUSE_SECRET_KEY"),
    )
    public_key = _pick_value(
        os.getenv("LANGFUSE_PUBLIC_KEY"),
        env_values.get("LANGFUSE_PUBLIC_KEY"),
    )
    base_url = _pick_value(
        os.getenv("LANGFUSE_BASE_URL"),
        env_values.get("LANGFUSE_BASE_URL"),
        # Legacy alias — accept LANGFUSE_HOST for back-compat
        os.getenv("LANGFUSE_HOST"),
        env_values.get("LANGFUSE_HOST"),
    )

    if secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    if public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    if base_url:
        # Keep both LANGFUSE_BASE_URL and legacy LANGFUSE_HOST in sync
        os.environ["LANGFUSE_BASE_URL"] = base_url
        os.environ["LANGFUSE_HOST"] = base_url

    return bool(os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Return the v4 Langfuse client, or None when disabled/unavailable."""
    if not apply_langfuse_env():
        return None
    try:
        from langfuse import get_client  # type: ignore[import-untyped]

        return get_client()
    except Exception:
        return None


def flush_langfuse_traces() -> None:
    """Flush pending Langfuse traces; mirrors flush_langsmith_traces."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        return


def record_stage_observation(
    stage_name: str,
    trace_payload: dict[str, Any],
    outputs: Any,
    summarizer: Callable[..., dict[str, Any]],
) -> None:
    """Write a compact observation to the current Langfuse span.

    The trace_payload is already the compact dict that the coordinator
    passes as trace_payload= kwarg — do NOT run it through
    trace_payload_inputs() again.  Wraps the entire body in a bare
    try/except so optional observability never breaks the pipeline.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        output_summary = summarizer(outputs)
        client.update_current_span(  # type: ignore[attr-defined]
            input=trace_payload,
            output=output_summary,
        )
    except Exception as exc:
        logger.debug(
            "Langfuse observation for stage %r failed (non-fatal): %s",
            stage_name,
            exc,
        )
