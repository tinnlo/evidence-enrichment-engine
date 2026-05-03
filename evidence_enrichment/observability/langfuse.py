"""Langfuse helpers for opt-in tracing (parallel to LangSmith)."""

from __future__ import annotations

from functools import wraps
from inspect import iscoroutinefunction
import logging
import os
from typing import Any, Callable, Mapping

# Re-export summarize_* helpers so callers can import them from either adapter
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
from evidence_enrichment.observability.redaction import maybe_redact
from evidence_enrichment.observability.router import current_backend, get_active_backends
from evidence_enrichment.observability.runtime import (
    OBSERVABILITY_STATE_LOCK,
    is_evicted_observability_value,
    resolve_evicted_observability_value,
)

logger = logging.getLogger(__name__)

TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUTHY


def _pick_value(*values: str | None) -> str | None:
    """Return the first non-empty value, skipping None and empty strings.

    This ensures an empty ambient env var (``LANGFUSE_SECRET_KEY=""``) does not
    shadow a valid value supplied via a .env file or explicit argument.
    """
    for value in values:
        if value:  # skips None and ""
            return value
    return None


def _mapping_value(mapping: Mapping[str, str | None], key: str) -> str | None:
    value = mapping.get(key)
    return value if value else None


def observe(*args, **kwargs):  # type: ignore[misc]
    """Gate Langfuse's decorator behind backend + credential readiness.

    The raw SDK decorator can initialize Langfuse even when this process has
    selected ``OBSERVABILITY_BACKEND=none`` or does not have a usable public key.
    That causes noisy auth warnings during replay/eval flows that should remain
    local-only. This wrapper keeps the stage function undecorated until Langfuse
    is both selected and fully configured.
    """

    def _decorator(fn):
        decorated_fn = None

        def _get_decorated_fn():
            nonlocal decorated_fn
            if decorated_fn is not None:
                return decorated_fn
            try:
                from langfuse import observe as sdk_observe  # type: ignore[import-untyped]
            except ImportError:
                decorated_fn = fn
                return decorated_fn
            decorated_fn = sdk_observe(*args, **kwargs)(fn)
            return decorated_fn

        if iscoroutinefunction(fn):

            @wraps(fn)
            async def _wrapped(*f_args, **f_kwargs):
                if get_langfuse_client() is None:
                    return await fn(*f_args, **f_kwargs)
                return await _get_decorated_fn()(*f_args, **f_kwargs)

            return _wrapped

        @wraps(fn)
        def _wrapped(*f_args, **f_kwargs):
            if get_langfuse_client() is None:
                return fn(*f_args, **f_kwargs)
            return _get_decorated_fn()(*f_args, **f_kwargs)

        return _wrapped

    return _decorator


def apply_langfuse_env(
    env_values: Mapping[str, str] | None = None,
    *,
    ambient_env: Mapping[str, str | None] | None = None,
    clear_missing: bool = False,
) -> bool:
    """Set LANGFUSE_* env vars from env_values / ambient_env.

    Args:
        env_values: Repo-local .env values or explicit overrides.
        ambient_env: Source of external process env overrides. Defaults to
            ``os.environ``. ``Settings.load()`` passes a pre-mutation snapshot so
            repeated loads do not accidentally reuse stale managed values.
        clear_missing: When True, remove any LANGFUSE_* vars that are absent from
            both sources instead of leaving old process env values in place.

    Returns True when a secret key is present after synchronisation.
    Accepts legacy LANGFUSE_HOST for back-compat alongside LANGFUSE_BASE_URL.
    """
    with OBSERVABILITY_STATE_LOCK:
        env_values = env_values or {}
        ambient_env = os.environ if ambient_env is None else ambient_env
        secret_key = _pick_value(
            resolve_evicted_observability_value(
                "LANGFUSE_SECRET_KEY",
                _mapping_value(ambient_env, "LANGFUSE_SECRET_KEY"),
            ),
            resolve_evicted_observability_value(
                "LANGFUSE_SECRET_KEY",
                env_values.get("LANGFUSE_SECRET_KEY"),
            ),
        )
        public_key = _pick_value(
            resolve_evicted_observability_value(
                "LANGFUSE_PUBLIC_KEY",
                _mapping_value(ambient_env, "LANGFUSE_PUBLIC_KEY"),
            ),
            resolve_evicted_observability_value(
                "LANGFUSE_PUBLIC_KEY",
                env_values.get("LANGFUSE_PUBLIC_KEY"),
            ),
        )
        base_url = _pick_value(
            _mapping_value(ambient_env, "LANGFUSE_BASE_URL"),
            env_values.get("LANGFUSE_BASE_URL"),
            # Legacy alias — accept LANGFUSE_HOST for back-compat
            _mapping_value(ambient_env, "LANGFUSE_HOST"),
            env_values.get("LANGFUSE_HOST"),
        )

        if secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        elif is_evicted_observability_value(
            "LANGFUSE_SECRET_KEY",
            os.getenv("LANGFUSE_SECRET_KEY"),
        ):
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
        elif clear_missing:
            os.environ.pop("LANGFUSE_SECRET_KEY", None)

        if public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        elif is_evicted_observability_value(
            "LANGFUSE_PUBLIC_KEY",
            os.getenv("LANGFUSE_PUBLIC_KEY"),
        ):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        elif clear_missing:
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)

        if base_url:
            # Keep both LANGFUSE_BASE_URL and legacy LANGFUSE_HOST in sync
            os.environ["LANGFUSE_BASE_URL"] = base_url
            os.environ["LANGFUSE_HOST"] = base_url
        elif clear_missing:
            os.environ.pop("LANGFUSE_BASE_URL", None)
            os.environ.pop("LANGFUSE_HOST", None)

        fully_configured = bool(
            os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")
        )
        if not fully_configured:
            present = [
                k
                for k in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY")
                if os.getenv(k)
            ]
            missing = [
                k
                for k in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY")
                if not os.getenv(k)
            ]
            if present:
                # Partial config — warn once; backend stays inactive to avoid silent
                # activation with incomplete credentials.
                logger.warning(
                    "Langfuse partially configured: %s present but %s missing. "
                    "Langfuse will remain inactive until all required credentials are set.",
                    ", ".join(present),
                    ", ".join(missing),
                )
        return fully_configured


def get_langfuse_client():
    """Return the v4 Langfuse client, or None when disabled/unavailable.

    Not cached — Settings.load() may clear LANGFUSE_* keys after the first
    call when the backend selector changes, so we must re-check on every call.

    Also respects ``OBSERVABILITY_BACKEND``: if the backend is set to a value
    that does not include Langfuse (e.g. ``langsmith`` or ``none``), returns
    ``None`` even when ``LANGFUSE_SECRET_KEY`` is present in the environment.
    """
    langfuse_active, _ = get_active_backends(current_backend())
    if not langfuse_active:
        return None

    with OBSERVABILITY_STATE_LOCK:
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
            input=maybe_redact(trace_payload),
            output=output_summary,
        )
    except Exception as exc:
        logger.debug(
            "Langfuse observation for stage %r failed (non-fatal): %s",
            stage_name,
            exc,
        )
