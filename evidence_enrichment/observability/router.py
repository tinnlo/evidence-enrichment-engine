"""Observability backend router.

Resolves the OBSERVABILITY_BACKEND selector to a pair of booleans that
indicate which remote backends should be activated.  The local tracer
(LocalTracer → spans.jsonl etc.) is always active regardless of this setting.

Allowed values for the backend selector:
    langfuse   — Langfuse only (default)
    langsmith  — LangSmith only
    dual       — both Langfuse and LangSmith
    none       — neither remote backend; local artifacts still written
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

from evidence_enrichment.observability.runtime import (
    OBSERVABILITY_STATE_LOCK,
    get_runtime_observability_config,
)

logger = logging.getLogger(__name__)

VALID_BACKENDS = frozenset({"langfuse", "langsmith", "dual", "none"})
DEFAULT_BACKEND = "langfuse"
_INVALID_FALLBACK = "none"


def resolve_backend(
    raw: str | None,
    env_values: Mapping[str, str] | None = None,
) -> str:
    """Return the normalised backend name from *raw* or *env_values*.

    Priority: explicit *raw* argument → ``OBSERVABILITY_BACKEND`` in
    *env_values* → :data:`DEFAULT_BACKEND`.

    Unknown values are replaced with :data:`_INVALID_FALLBACK` (``"none"``,
    fail-closed) and a warning is logged.
    """
    env_values = env_values or {}
    value = (
        raw
        or env_values.get("OBSERVABILITY_BACKEND")
        or DEFAULT_BACKEND
    )
    value = value.strip().lower()
    if value not in VALID_BACKENDS:
        logger.warning(
            "Unknown OBSERVABILITY_BACKEND %r — falling back to %r (fail-closed). "
            "Allowed values: %s",
            value,
            _INVALID_FALLBACK,
            ", ".join(sorted(VALID_BACKENDS)),
        )
        value = _INVALID_FALLBACK
    return value


def get_active_backends(backend: str) -> tuple[bool, bool]:
    """Return ``(langfuse_active, langsmith_active)`` for *backend*.

    Args:
        backend: Normalised backend name — one of ``langfuse``,
            ``langsmith``, ``dual``, ``none``.

    Returns:
        A two-tuple of booleans ``(langfuse_active, langsmith_active)``.
    """
    backend = backend.strip().lower()
    if backend == "langfuse":
        return True, False
    if backend == "langsmith":
        return False, True
    if backend == "dual":
        return True, True
    # "none" or anything unexpected after resolve_backend normalisation
    return False, False


# Repo root inferred from this file's location: router.py → observability/ → evidence_enrichment/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_backend_from_dotenv(env_file: str = ".env") -> str | None:
    """Return OBSERVABILITY_BACKEND from a .env file when present.

    Note: *env_file* is resolved relative to the current working directory,
    not the repository root.  Callers that need repo-root resolution should
    pass an absolute path (e.g. ``str(Path(__file__).parents[N] / ".env")``).
    """
    path = Path(env_file)
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OBSERVABILITY_BACKEND":
            return value.strip().strip("\"'")
    return None



TRUTHY = frozenset({"1", "true", "yes", "on"})


def current_backend() -> str:
    """Resolve the current backend from runtime config, env, or .env files.

    Resolution order:
    1. Task-local runtime overrides installed for an active coordinator run.
    2. The process-global default config written by the last successful
       ``Settings.load()``.
    3. ``os.environ``.
    4. Repo-local / CWD `.env` fallback.
    """

    runtime_config = get_runtime_observability_config()
    if runtime_config is not None and runtime_config.backend is not None:
        return resolve_backend(runtime_config.backend)

    with OBSERVABILITY_STATE_LOCK:
        raw = os.getenv("OBSERVABILITY_BACKEND")
    if raw is None:
        raw = _read_backend_from_dotenv(str(_REPO_ROOT / ".env"))
    if raw is None:
        raw = _read_backend_from_dotenv()
    return resolve_backend(raw)


def _resolve_current_backend() -> str:
    """Back-compat wrapper for the current backend resolver."""

    return current_backend()


def langsmith_wrapping_enabled() -> bool:
    """Return True only when the active backend permits LangSmith client wrapping.

    Resolution order:
    1. Task-local runtime config derived from the active ``Settings`` instance.
    2. Process-global default config from the last successful ``Settings.load()``.
    3. ``OBSERVABILITY_BACKEND`` from the live process environment.
    4. Repo-root ``.env`` file (resolved relative to this module's location,
        so it is stable regardless of the process working directory)
    5. CWD-relative ``.env`` file (fallback for callers outside the repo)
    6. :data:`DEFAULT_BACKEND`

    Wrapping is permitted for ``langsmith`` and ``dual`` only. For
    ``langfuse`` and ``none`` this returns ``False`` even if
    ``LANGSMITH_TRACING=true`` is present in the environment.
    """
    _, langsmith_active = get_active_backends(current_backend())
    return langsmith_active


def langsmith_tracing_ready() -> bool:
    """Return True only when LangSmith is both selected *and* fully configured.

    Requires all three conditions to be satisfied:
    1. The active backend includes LangSmith (``langsmith`` or ``dual``).
    2. ``LANGSMITH_TRACING`` (or its legacy alias ``LANGCHAIN_TRACING_V2``) is
       set to a truthy value.
    3. An API key is present in ``LANGSMITH_API_KEY`` or ``LANGCHAIN_API_KEY``.

    This is the correct gate for wrapping live OpenAI / Anthropic clients, where
    a partially-configured LangSmith would silently capture raw prompts and
    responses without actually delivering them anywhere.
    """
    if not langsmith_wrapping_enabled():
        return False
    with OBSERVABILITY_STATE_LOCK:
        tracing = (
            os.getenv("LANGSMITH_TRACING")
            or os.getenv("LANGCHAIN_TRACING_V2")
            or ""
        )
        api_key = (
            os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or ""
        )
    if tracing.strip().lower() not in TRUTHY:
        return False
    return bool(api_key.strip())
