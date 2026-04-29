"""Observability runtime config, revocation state, and shared state lock."""

from __future__ import annotations

import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass

OBSERVABILITY_STATE_LOCK = threading.RLock()


@dataclass(frozen=True)
class RuntimeObservabilityConfig:
    """Runtime overrides derived from the active Settings instance."""

    backend: str | None = None
    trace_redact_values: bool | None = None


_ACTIVE_RUNTIME_OBSERVABILITY_CONFIG: ContextVar[
    RuntimeObservabilityConfig | None
] = ContextVar("active_runtime_observability_config", default=None)
_DEFAULT_RUNTIME_OBSERVABILITY_CONFIG: RuntimeObservabilityConfig | None = None
_EVICTED_OBSERVABILITY_VALUES: dict[str, str] = {}


def activate_runtime_observability_config(
    *,
    backend: str | None = None,
    trace_redact_values: bool | None = None,
) -> Token[RuntimeObservabilityConfig | None]:
    """Install task-local observability overrides for the current run."""

    return _ACTIVE_RUNTIME_OBSERVABILITY_CONFIG.set(
        RuntimeObservabilityConfig(
            backend=backend,
            trace_redact_values=trace_redact_values,
        )
    )


def get_runtime_observability_config() -> RuntimeObservabilityConfig | None:
    """Return task-local overrides or the process-global default config."""

    runtime_config = _ACTIVE_RUNTIME_OBSERVABILITY_CONFIG.get()
    if runtime_config is not None:
        return runtime_config
    with OBSERVABILITY_STATE_LOCK:
        return _DEFAULT_RUNTIME_OBSERVABILITY_CONFIG


def reset_runtime_observability_config(
    token: Token[RuntimeObservabilityConfig | None],
) -> None:
    """Remove task-local observability overrides for the current run."""

    _ACTIVE_RUNTIME_OBSERVABILITY_CONFIG.reset(token)


def clear_runtime_observability_config() -> None:
    """Clear any task-local observability overrides in the current context."""

    _ACTIVE_RUNTIME_OBSERVABILITY_CONFIG.set(None)


def set_default_runtime_observability_config(
    *,
    backend: str | None = None,
    trace_redact_values: bool | None = None,
) -> None:
    """Persist the last successfully loaded observability settings process-wide."""

    global _DEFAULT_RUNTIME_OBSERVABILITY_CONFIG
    with OBSERVABILITY_STATE_LOCK:
        _DEFAULT_RUNTIME_OBSERVABILITY_CONFIG = RuntimeObservabilityConfig(
            backend=backend,
            trace_redact_values=trace_redact_values,
        )


def clear_default_runtime_observability_config() -> None:
    """Forget the process-global default observability settings."""

    global _DEFAULT_RUNTIME_OBSERVABILITY_CONFIG
    with OBSERVABILITY_STATE_LOCK:
        _DEFAULT_RUNTIME_OBSERVABILITY_CONFIG = None


def remember_evicted_observability_value(key: str, value: str | None) -> None:
    """Tombstone an explicitly evicted observability value.

    A later load may re-enable the same *key* only when the value actually
    changes. This prevents stale credentials from reappearing across repeated
    ``Settings.load()`` calls when they still exist in `.env` or other ambient
    process state.
    """

    if not value:
        return
    with OBSERVABILITY_STATE_LOCK:
        _EVICTED_OBSERVABILITY_VALUES[key] = value


def resolve_evicted_observability_value(
    key: str,
    value: str | None,
) -> str | None:
    """Filter evicted values while allowing rotated replacements.

    Returns ``None`` when *value* matches the tombstoned value for *key*.
    When *value* differs, the tombstone is cleared so the new value is accepted
    on subsequent loads too.
    """

    if value is None:
        return None
    with OBSERVABILITY_STATE_LOCK:
        evicted = _EVICTED_OBSERVABILITY_VALUES.get(key)
        if evicted is None:
            return value
        if value == evicted:
            return None
        _EVICTED_OBSERVABILITY_VALUES.pop(key, None)
        return value


def is_evicted_observability_value(key: str, value: str | None) -> bool:
    """Return True when *value* matches the current tombstone for *key*."""

    if not value:
        return False
    with OBSERVABILITY_STATE_LOCK:
        return _EVICTED_OBSERVABILITY_VALUES.get(key) == value


def clear_evicted_observability_values() -> None:
    """Forget all in-process observability tombstones."""

    with OBSERVABILITY_STATE_LOCK:
        _EVICTED_OBSERVABILITY_VALUES.clear()


def snapshot_evicted_observability_values() -> dict[str, str]:
    """Return a copy of the current in-process observability tombstones."""

    with OBSERVABILITY_STATE_LOCK:
        return dict(_EVICTED_OBSERVABILITY_VALUES)


def restore_evicted_observability_values(snapshot: dict[str, str]) -> None:
    """Restore in-process observability tombstones from a snapshot."""

    with OBSERVABILITY_STATE_LOCK:
        _EVICTED_OBSERVABILITY_VALUES.clear()
        _EVICTED_OBSERVABILITY_VALUES.update(snapshot)
