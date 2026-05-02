"""Settings and repo-local config loading."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from evidence_enrichment.observability.langfuse import apply_langfuse_env
from evidence_enrichment.observability.langsmith import apply_langsmith_env
from evidence_enrichment.observability.runtime import (
    OBSERVABILITY_STATE_LOCK,
    clear_default_runtime_observability_config,
    remember_evicted_observability_value,
    resolve_evicted_observability_value,
    restore_evicted_observability_values,
    snapshot_evicted_observability_values,
    set_default_runtime_observability_config,
)
from evidence_enrichment.observability.router import get_active_backends, resolve_backend

_settings_logger = logging.getLogger(__name__)

_OBSERVABILITY_ENV_KEYS = (
    "OBSERVABILITY_BACKEND",
    "TRACE_REDACT_VALUES",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
)
_LANGFUSE_SECRET_KEYS = frozenset({
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
})
_LANGFUSE_REVOCABLE_KEYS = frozenset({
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
})
_LANGSMITH_SECRET_KEYS = frozenset({
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
})
_LANGSMITH_REVOCABLE_KEYS = frozenset({
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
})
_TOMBSTONED_OBSERVABILITY_KEYS = frozenset(
    _LANGFUSE_SECRET_KEYS | _LANGSMITH_SECRET_KEYS
)

_UNSET = object()
_AMBIENT_OBSERVABILITY_ENV: dict[str, str | None] = {}
_LAST_MANAGED_OBSERVABILITY_ENV: dict[str, str | None] = {}
# Keys whose value was present in os.environ *before* Settings ever wrote them.
# Used to distinguish "shell set this credential" from "we wrote it" so that
# deletion after deactivation can be detected on reactivation.
_SHELL_SET_OBSERVABILITY_KEYS: set[str] = set()
# Keys that have been explicitly evicted via clear_ambient_credentials().
# _refresh_ambient_observability_env() will not re-capture these from os.environ
# on the next load. The eviction flag is consumed after one successful load; for
# secret values, longer-lived tombstones additionally block reintroduction of the
# same stale credential until the value actually changes.
_EVICTION_PENDING_KEYS: set[str] = set()
# Stash the last known value of each secret key just before
# _redact_inactive_secrets() zeros it.  clear_ambient_credentials() consults
# this dict when os.environ / _AMBIENT / _LAST_MANAGED are already zeroed
# (deactivate-then-evict order) so tombstone creation remains order-independent.
_PRE_REDACTION_SECRET_VALUES: dict[str, str] = {}


def clear_ambient_credentials(
    backend: str | None = None,
) -> None:
    """Evict cached credentials from both in-process caches and ``os.environ``.

    ``Settings.load()`` retains a snapshot of shell-provided credentials so they
    can be restored when the active backend is toggled back on.  This function
    removes those credentials from **four places** atomically:

    1. ``os.environ`` — so SDKs and subsequent code can no longer read them.
    2. The internal ambient snapshot (``_AMBIENT_OBSERVABILITY_ENV``) — so they
       are not restored on backend reactivation.
    3. The managed-state cache (``_LAST_MANAGED_OBSERVABILITY_ENV``) — so they
       are not recoverable via Python introspection.
    4. In-process secret tombstones — the *current* secret value is recorded so
       the same stale credential is rejected on all later ``Settings.load()``
       calls until the value actually changes (rotation accepted; same value
       blocked).

    Keys are also marked as eviction-pending so ``_refresh_ambient_observability_env``
    will not re-capture them from any remaining source on the next load.  The
    one-load eviction flag is consumed after the load commits; tombstones persist
    independently until a new (different) secret value is loaded.

    **Tombstone creation is order-independent.** Even if the backend was already
    deactivated (and ``_redact_inactive_secrets`` zeroed the caches) before this
    function is called, ``_redact_inactive_secrets`` now tombstones secrets before
    zeroing them so the tombstone will already be in place.  Calling
    ``clear_ambient_credentials()`` after deactivation is still correct; the
    eviction-pending flag and ``os.environ`` pop are applied regardless.

    **For production / long-lived processes:** prefer restarting the process after
    credential removal rather than relying on runtime cycling.  Also remove stale
    credentials from the ``.env`` file — tombstones only block the in-process value;
    a *new* value in ``.env`` after eviction will be accepted on the next load.

    Args:
        backend: Which backend's credentials to clear.  One of ``"langfuse"``,
            ``"langsmith"``, or ``None`` (clears all observability credential
            keys).  Non-credential keys such as ``OBSERVABILITY_BACKEND`` are
            never cleared.
    """
    with OBSERVABILITY_STATE_LOCK:
        if backend is None:
            keys_to_clear = _LANGFUSE_REVOCABLE_KEYS | _LANGSMITH_REVOCABLE_KEYS
        elif backend.strip().lower() == "langfuse":
            keys_to_clear = _LANGFUSE_REVOCABLE_KEYS
        elif backend.strip().lower() == "langsmith":
            keys_to_clear = _LANGSMITH_REVOCABLE_KEYS
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. Expected 'langfuse', 'langsmith', or None."
            )

        for key in keys_to_clear:
            if key in _TOMBSTONED_OBSERVABILITY_KEYS:
                # Prefer the live value from env/caches; fall back to the
                # pre-redaction stash in case _redact_inactive_secrets() already
                # zeroed the caches (deactivate-then-evict order).
                secret_value = (
                    os.environ.get(key)
                    or _AMBIENT_OBSERVABILITY_ENV.get(key)
                    or _LAST_MANAGED_OBSERVABILITY_ENV.get(key)
                    or _PRE_REDACTION_SECRET_VALUES.get(key)
                )
                remember_evicted_observability_value(key, secret_value)
            _AMBIENT_OBSERVABILITY_ENV[key] = None
            _LAST_MANAGED_OBSERVABILITY_ENV[key] = None
            _PRE_REDACTION_SECRET_VALUES.pop(key, None)
            _SHELL_SET_OBSERVABILITY_KEYS.discard(key)
            _EVICTION_PENDING_KEYS.add(key)
            os.environ.pop(key, None)


def _redact_inactive_secrets(
    langfuse_active: bool,
    langsmith_active: bool,
    ambient_env: dict[str, str | None] | None = None,
) -> None:
    """Zero out secret string values in _LAST_MANAGED and _AMBIENT for inactive backends.

    ``os.environ`` is already cleared by the caller for inactive backends, but
    the credential strings may still linger in ``_LAST_MANAGED_OBSERVABILITY_ENV``
    and ``_AMBIENT_OBSERVABILITY_ENV`` as non-None values.  This helper replaces
    any remaining non-None secret values with ``None`` so they are not recoverable
    via Python introspection and cannot be resurrected on reactivation.

    Keys are left present in both dicts (set to ``None`` rather than popped) so
    that ``_refresh_ambient_observability_env`` can still distinguish
    "managed and absent" from "never seen" on the next ``Settings.load()``.
    """
    keys_to_redact: set[str] = set()
    if not langfuse_active:
        keys_to_redact |= _LANGFUSE_SECRET_KEYS
    if not langsmith_active:
        keys_to_redact |= _LANGSMITH_SECRET_KEYS
    for key in keys_to_redact:
        # Stash the last known secret value before zeroing caches so that a
        # subsequent clear_ambient_credentials() call can still tombstone it
        # even after the caches are gone (deactivate-then-evict order).
        if key in _TOMBSTONED_OBSERVABILITY_KEYS:
            last_value = (
                os.environ.get(key)
                or _LAST_MANAGED_OBSERVABILITY_ENV.get(key)
                or _AMBIENT_OBSERVABILITY_ENV.get(key)
            )
            if last_value:
                _PRE_REDACTION_SECRET_VALUES[key] = last_value
        if _LAST_MANAGED_OBSERVABILITY_ENV.get(key) is not None:
            _LAST_MANAGED_OBSERVABILITY_ENV[key] = None
        if _AMBIENT_OBSERVABILITY_ENV.get(key) is not None:
            _AMBIENT_OBSERVABILITY_ENV[key] = None
        _SHELL_SET_OBSERVABILITY_KEYS.discard(key)


def _refresh_ambient_observability_env() -> dict[str, str | None]:
    """Capture external observability env without treating prior managed values as ambient.

    On repeated ``Settings.load()`` calls we mutate ``os.environ`` ourselves. This
    helper preserves externally supplied shell values across backend flips while
    ignoring stale values previously written by settings loading.

    On first contact with a key (``managed is _UNSET``) the current env value
    is recorded in ``_SHELL_SET_OBSERVABILITY_KEYS`` if non-None so callers can
    distinguish "shell originally set this" from "we wrote it" in later loads.

    Keys in ``_EVICTION_PENDING_KEYS`` (set by ``clear_ambient_credentials()``)
    are suppressed — their ambient value is set to None and they are excluded
    from ``_SHELL_SET_OBSERVABILITY_KEYS`` so they cannot be restored from
    in-process state or ``.env`` (see the ``env_values`` filtering in
    ``Settings.load()``).

    Remaining caveat: if a *new* credential value is written to the repo's `.env`
    file after ``clear_ambient_credentials()``, the next successful
    ``Settings.load()`` will accept it and clear the tombstone. For hard
    revocation, remove stale values from `.env` too.
    """
    for key in _OBSERVABILITY_ENV_KEYS:
        if key in _EVICTION_PENDING_KEYS:
            _AMBIENT_OBSERVABILITY_ENV[key] = None
            _SHELL_SET_OBSERVABILITY_KEYS.discard(key)
            continue
        current = os.getenv(key)
        managed = _LAST_MANAGED_OBSERVABILITY_ENV.get(key, _UNSET)
        if managed is _UNSET:
            if current is not None:
                _SHELL_SET_OBSERVABILITY_KEYS.add(key)
            _AMBIENT_OBSERVABILITY_ENV[key] = current
        elif current != managed or key not in _AMBIENT_OBSERVABILITY_ENV:
            _AMBIENT_OBSERVABILITY_ENV[key] = current
            if current is not None:
                _SHELL_SET_OBSERVABILITY_KEYS.add(key)
            else:
                _SHELL_SET_OBSERVABILITY_KEYS.discard(key)
    # NOTE: _EVICTION_PENDING_KEYS is NOT cleared here.  It is consumed only
    # after the load commits successfully (in Settings.load()), so that a
    # failed load does not lose the eviction flag.
    return dict(_AMBIENT_OBSERVABILITY_ENV)



def _record_managed_observability_env() -> None:
    for key in _OBSERVABILITY_ENV_KEYS:
        _LAST_MANAGED_OBSERVABILITY_ENV[key] = os.getenv(key)



def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


class StageProviderConfig(BaseModel):
    provider_order: list[str] = Field(default_factory=list)


class FieldThresholds(BaseModel):
    auto_approve_min_confidence: float = 0.85
    review_min_confidence: float = 0.50


class GuardrailsSettings(BaseModel):
    """Settings for post-synthesis guardrail checks."""

    confidence_floor: float = 0.4
    pii_entities: list[str] = Field(
        default_factory=lambda: [
            "EMAIL_ADDRESS",
            "IBAN_CODE",
            "UK_NHS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "CRYPTO",
            "US_SSN",
        ]
    )


class RetrievalConfig(BaseModel):
    """Configuration for the optional RAG retrieval pipeline.

    mode:
        "off"   — retrieval disabled (default); pipeline unchanged
        "local" — embed documents, store in Chroma, retrieve before analysis
    persist_path:
        Directory for Chroma persistent storage.
    chunk_size:
        Target character count per text chunk.
    overlap:
        Character overlap between consecutive text chunks.
    max_table_size:
        Max chars before a table block is split (no overlap).
    top_k:
        Number of retrieved chunks returned to the analysis prompt.
    embedding_model:
        OpenAI embedding model name.
    weights:
        Hybrid scoring weights (vector, keyword, table_boost).
    min_doc_chars:
        Minimum document character count to index; shorter docs are skipped.
    """

    mode: str = "off"  # "off" | "local"
    persist_path: str = "examples/output/chroma"
    chunk_size: int = 1500
    overlap: int = 200
    max_table_size: int = 4000
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    weights: tuple[float, float, float] = (0.7, 0.2, 0.1)
    min_doc_chars: int = 2000


class FinOpsSettings(BaseModel):
    """AI FinOps configuration for cost attribution and budget policy."""

    enabled: bool = True
    budget_mode: str = "off"
    max_cost_usd_per_run: float | None = None
    max_cost_usd_per_success: float | None = None
    openai_cheap_model: str = "gpt-4.1-nano"
    anthropic_cheap_model: str = "claude-3-5-haiku-latest"
    pricing_override: dict[str, dict[str, float]] = Field(default_factory=dict)


class Settings(BaseModel):
    default_mode: str = "auto"
    replay_dir: str = "examples/replay"
    prompt_dir: str = "prompts"
    context_dir: str = "context"
    trace_output_dir: str = "examples/output/traces"
    eval_output_dir: str = "evals/output"
    search: StageProviderConfig = Field(default_factory=StageProviderConfig)
    analysis: StageProviderConfig = Field(default_factory=StageProviderConfig)
    synthesis: StageProviderConfig = Field(default_factory=StageProviderConfig)
    thresholds: dict[str, FieldThresholds] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    guardrails: GuardrailsSettings = Field(default_factory=GuardrailsSettings)
    finops: FinOpsSettings = Field(default_factory=FinOpsSettings)
    observability_backend: str = "langfuse"
    trace_redact_values: bool = True
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"
    serper_api_key: str | None = None
    tavily_api_key: str | None = None

    @classmethod
    def load(
        cls,
        *,
        config_file: str = "evidence_enrichment.yaml",
        env_file: str = ".env",
    ) -> "Settings":
        config = _load_yaml(Path(config_file))
        env_values = _parse_env_file(Path(env_file))

        with OBSERVABILITY_STATE_LOCK:
            # Snapshot process env and module-level observability caches BEFORE any
            # refresh/validation work so a failed load can roll back fully.
            _env_snapshot = {k: os.environ.get(k) for k in _OBSERVABILITY_ENV_KEYS}
            _ambient_snapshot = dict(_AMBIENT_OBSERVABILITY_ENV)
            _managed_snapshot = dict(_LAST_MANAGED_OBSERVABILITY_ENV)
            _shell_set_snapshot = set(_SHELL_SET_OBSERVABILITY_KEYS)
            _eviction_snapshot = set(_EVICTION_PENDING_KEYS)
            _evicted_values_snapshot = snapshot_evicted_observability_values()
            _pre_redaction_snapshot = dict(_PRE_REDACTION_SECRET_VALUES)

            try:
                ambient_observability_env = _refresh_ambient_observability_env()
                for key in _TOMBSTONED_OBSERVABILITY_KEYS:
                    sanitized_ambient = resolve_evicted_observability_value(
                        key,
                        ambient_observability_env.get(key),
                    )
                    if sanitized_ambient != ambient_observability_env.get(key):
                        ambient_observability_env[key] = sanitized_ambient
                        _AMBIENT_OBSERVABILITY_ENV[key] = sanitized_ambient
                        if sanitized_ambient is None:
                            _SHELL_SET_OBSERVABILITY_KEYS.discard(key)
                for key in tuple(env_values):
                    if key not in _TOMBSTONED_OBSERVABILITY_KEYS:
                        continue
                    sanitized_value = resolve_evicted_observability_value(
                        key,
                        env_values.get(key),
                    )
                    if sanitized_value is None:
                        env_values.pop(key, None)
                    else:
                        env_values[key] = sanitized_value
                # Evict .env values for keys that are pending eviction so that
                # stale .env credentials cannot be restored after
                # clear_ambient_credentials().  Only evict from env_values (the
                # parsed .env dict), not from os.environ which was already cleared
                # by clear_ambient_credentials() and by the refresh above.
                if _EVICTION_PENDING_KEYS:
                    for evicted_key in list(_EVICTION_PENDING_KEYS):
                        env_values.pop(evicted_key, None)
                data: dict[str, Any] = dict(config)
                data["openai_api_key"] = os.getenv("OPENAI_API_KEY") or env_values.get(
                    "OPENAI_API_KEY"
                )
                data["openai_model"] = (
                    os.getenv("OPENAI_MODEL")
                    or env_values.get("OPENAI_MODEL")
                    or data.get("openai_model", "gpt-4.1-mini")
                )
                data["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY") or env_values.get(
                    "ANTHROPIC_API_KEY"
                )
                data["anthropic_model"] = (
                    os.getenv("ANTHROPIC_MODEL")
                    or env_values.get("ANTHROPIC_MODEL")
                    or data.get("anthropic_model", "claude-3-5-sonnet-latest")
                )
                data["serper_api_key"] = os.getenv("SERPER_API_KEY") or env_values.get(
                    "SERPER_API_KEY"
                )
                data["tavily_api_key"] = os.getenv("TAVILY_API_KEY") or env_values.get(
                    "TAVILY_API_KEY"
                )
                # Guardrails: merge env override into existing YAML subtree so
                # YAML-supplied pii_entities survive env injection.
                floor_raw = os.getenv("GUARDRAILS_CONFIDENCE_FLOOR") or env_values.get(
                    "GUARDRAILS_CONFIDENCE_FLOOR"
                )
                if floor_raw is not None:
                    guardrails_subtree: dict[str, Any] = dict(data.get("guardrails") or {})
                    try:
                        guardrails_subtree["confidence_floor"] = float(floor_raw)
                    except ValueError:
                        pass
                    data["guardrails"] = guardrails_subtree
                raw_backend = ambient_observability_env.get("OBSERVABILITY_BACKEND") or env_values.get(
                    "OBSERVABILITY_BACKEND"
                )
                backend = resolve_backend(raw_backend, env_values)
                data["observability_backend"] = backend

                # trace_redact_values: resolve from ambient/.env, not raw os.getenv().
                # os.getenv() would pick up stale values written by a previous
                # Settings.load(), shadowing fresh .env changes.
                redact_raw = (
                    ambient_observability_env.get("TRACE_REDACT_VALUES")
                    or env_values.get("TRACE_REDACT_VALUES")
                )
                if redact_raw is not None:
                    data["trace_redact_values"] = str(redact_raw).strip().lower() in ("1", "true", "yes", "on")

                # Validate the Settings object before committing any env/cache changes.
                instance = cls(**data)

                # Export backend into os.environ so downstream code using os.getenv()
                # outside an active coordinator run still sees the resolved value.
                os.environ["OBSERVABILITY_BACKEND"] = backend

                # Export trace_redact_values — always write an explicit value so
                # _should_redact() sees a non-None env var (avoiding the default-True
                # fallback when the operator explicitly set false).
                if instance.trace_redact_values:
                    os.environ["TRACE_REDACT_VALUES"] = "true"
                else:
                    os.environ["TRACE_REDACT_VALUES"] = "false"

                langfuse_active, langsmith_active = get_active_backends(backend)

                if langfuse_active:
                    apply_langfuse_env(
                        env_values,
                        ambient_env=ambient_observability_env,
                        clear_missing=True,
                    )
                else:
                    # Clear all Langfuse keys so @observe decorator stays dormant.
                    # Both LANGFUSE_HOST (legacy) and LANGFUSE_BASE_URL must be cleared
                    # because apply_langfuse_env() sets both names in sync.
                    for key in (
                        "LANGFUSE_SECRET_KEY",
                        "LANGFUSE_PUBLIC_KEY",
                        "LANGFUSE_HOST",
                        "LANGFUSE_BASE_URL",
                    ):
                        os.environ.pop(key, None)

                if langsmith_active:
                    apply_langsmith_env(
                        env_values,
                        ambient_env=ambient_observability_env,
                        clear_missing=True,
                    )
                else:
                    # Clear LangSmith tracing flags AND credentials so @traceable stays
                    # dormant and no residual secrets linger in process env.
                    for key in (
                        "LANGSMITH_TRACING",
                        "LANGCHAIN_TRACING_V2",
                        "LANGSMITH_API_KEY",
                        "LANGCHAIN_API_KEY",
                        "LANGSMITH_PROJECT",
                        "LANGCHAIN_PROJECT",
                    ):
                        os.environ.pop(key, None)

                if backend == "none":
                    _settings_logger.warning(
                        "OBSERVABILITY_BACKEND=none: remote tracing disabled. "
                        "Local trace artifacts (spans.jsonl, trace_summary.json, "
                        "trace_timeline.md, openinference_trace.json) are still written."
                    )

                _record_managed_observability_env()
                # Purge secret values from in-memory caches for inactive backends so
                # credentials are not recoverable via Python introspection after
                # deactivation.  Must run after _record_managed_observability_env() so
                # that function cannot re-populate the keys we just cleared.
                _redact_inactive_secrets(
                    langfuse_active,
                    langsmith_active,
                    ambient_observability_env,
                )
                # Consume eviction flags only after a successful load commit.
                _EVICTION_PENDING_KEYS.clear()
                # Pre-redaction stash is no longer needed once the load committed.
                _PRE_REDACTION_SECRET_VALUES.clear()
                # Persist this instance as the canonical singleton so get_settings()
                # returns it instead of re-loading from default file paths.
                global _LAST_EXPLICIT_SETTINGS
                _LAST_EXPLICIT_SETTINGS = _SettingsCache(
                    instance=instance,
                    config_file=config_file,
                    env_file=env_file,
                )
                set_default_runtime_observability_config(
                    backend=instance.observability_backend,
                    trace_redact_values=instance.trace_redact_values,
                )
                return instance

            except Exception:
                # Roll back any partial env/cache mutations so the process stays in
                # its previous clean state.
                for key, value in _env_snapshot.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                _AMBIENT_OBSERVABILITY_ENV.clear()
                _AMBIENT_OBSERVABILITY_ENV.update(_ambient_snapshot)
                _LAST_MANAGED_OBSERVABILITY_ENV.clear()
                _LAST_MANAGED_OBSERVABILITY_ENV.update(_managed_snapshot)
                _SHELL_SET_OBSERVABILITY_KEYS.clear()
                _SHELL_SET_OBSERVABILITY_KEYS.update(_shell_set_snapshot)
                _EVICTION_PENDING_KEYS.clear()
                _EVICTION_PENDING_KEYS.update(_eviction_snapshot)
                restore_evicted_observability_values(_evicted_values_snapshot)
                _PRE_REDACTION_SECRET_VALUES.clear()
                _PRE_REDACTION_SECRET_VALUES.update(_pre_redaction_snapshot)
                raise

    @property
    def replay_path(self) -> Path:
        return Path(self.replay_dir)

    @property
    def context_path(self) -> Path:
        return Path(self.context_dir)

    @property
    def trace_output_path(self) -> Path:
        return Path(self.trace_output_dir)

    @property
    def eval_output_path(self) -> Path:
        return Path(self.eval_output_dir)


@dataclass
class _SettingsCache:
    """Singleton holder that pairs a Settings instance with its provenance."""

    instance: Settings
    config_file: Path | None
    env_file: Path | None


_LAST_EXPLICIT_SETTINGS: _SettingsCache | None = None


def _reset_settings_cache() -> None:
    """Clear the cached settings singleton (for testing).

    Equivalent to the former ``get_settings.cache_clear()``.  Production code
    should not need this — call ``Settings.load()`` to update settings.
    """
    global _LAST_EXPLICIT_SETTINGS
    with OBSERVABILITY_STATE_LOCK:
        _LAST_EXPLICIT_SETTINGS = None
        _PRE_REDACTION_SECRET_VALUES.clear()
        clear_default_runtime_observability_config()


def get_settings() -> Settings:
    """Return the last explicitly loaded Settings, or load defaults on first call.

    After a successful ``Settings.load()``, this function returns that instance
    immediately without re-reading files or re-running env mutations.  If no
    explicit load has been done yet, it performs a default load (using default
    config/env file paths) and caches the result.

    This replaces the former ``@lru_cache`` approach which was vulnerable to
    stale values after explicit ``Settings.load()`` calls with non-default paths.

    **Architectural note** – ``get_settings()`` is a process-global singleton.
    For non-CLI code (e.g. long-running servers, multi-tenant services), prefer
    explicit dependency injection: pass a ``Settings`` instance directly to the
    code that needs it rather than relying on this module-level accessor.
    """
    global _LAST_EXPLICIT_SETTINGS
    with OBSERVABILITY_STATE_LOCK:
        if _LAST_EXPLICIT_SETTINGS is not None:
            return _LAST_EXPLICIT_SETTINGS.instance
        return Settings.load()
