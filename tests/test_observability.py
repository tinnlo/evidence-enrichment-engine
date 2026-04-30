import asyncio
import os
import sys

import pytest

from evidence_enrichment.observability.langsmith import apply_langsmith_env


def test_apply_langsmith_env_prefers_current_vars(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_PROJECT", "current-project")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    enabled = apply_langsmith_env(
        {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "langsmith-key",
            "LANGSMITH_PROJECT": "env-project",
        }
    )

    assert enabled is False
    assert os.environ["LANGSMITH_PROJECT"] == "current-project"
    assert os.environ["LANGSMITH_API_KEY"] == "langsmith-key"


def test_apply_langsmith_env_supports_legacy_aliases(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    enabled = apply_langsmith_env(
        {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "legacy-key",
            "LANGCHAIN_PROJECT": "legacy-project",
        }
    )

    assert enabled is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "legacy-key"
    assert os.environ["LANGSMITH_PROJECT"] == "legacy-project"


# ---------------------------------------------------------------------------
# Langfuse tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_langfuse_cache():
    """Formerly cleared lru_cache; kept as placeholder since get_langfuse_client
    is no longer cached (removed @lru_cache to allow dynamic key-clearing)."""
    yield


def _clear_langfuse_env(monkeypatch):
    for key in (
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST",
    ):
        monkeypatch.delenv(key, raising=False)


def test_langfuse_disabled_when_no_keys(monkeypatch) -> None:
    _clear_langfuse_env(monkeypatch)
    from evidence_enrichment.observability.langfuse import (
        apply_langfuse_env,
        get_langfuse_client,
    )

    enabled = apply_langfuse_env({})
    assert enabled is False
    assert get_langfuse_client() is None


def test_langfuse_apply_env_sets_vars(monkeypatch) -> None:
    _clear_langfuse_env(monkeypatch)
    from evidence_enrichment.observability.langfuse import apply_langfuse_env

    enabled = apply_langfuse_env(
        {
            "LANGFUSE_SECRET_KEY": "sk-test",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
        }
    )
    assert enabled is True
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
    assert os.environ["LANGFUSE_BASE_URL"] == "https://cloud.langfuse.com"
    # Back-compat: LANGFUSE_HOST should also be set
    assert os.environ["LANGFUSE_HOST"] == "https://cloud.langfuse.com"


def test_langfuse_apply_env_legacy_host_backcompat(monkeypatch) -> None:
    """LANGFUSE_HOST is accepted as a legacy alias for LANGFUSE_BASE_URL."""
    _clear_langfuse_env(monkeypatch)
    from evidence_enrichment.observability.langfuse import apply_langfuse_env

    apply_langfuse_env(
        {
            "LANGFUSE_SECRET_KEY": "sk-legacy",
            "LANGFUSE_PUBLIC_KEY": "pk-legacy",
            "LANGFUSE_HOST": "https://self-hosted.example.com",
        }
    )
    assert os.environ["LANGFUSE_BASE_URL"] == "https://self-hosted.example.com"
    assert os.environ["LANGFUSE_HOST"] == "https://self-hosted.example.com"


def test_langfuse_flush_noop_when_disabled(monkeypatch) -> None:
    _clear_langfuse_env(monkeypatch)
    from evidence_enrichment.observability.langfuse import flush_langfuse_traces

    # Must not raise even when no client is available
    flush_langfuse_traces()


def test_langfuse_client_none_when_import_missing(monkeypatch) -> None:
    _clear_langfuse_env(monkeypatch)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    # Simulate langfuse not installed after env activation succeeds
    monkeypatch.setitem(sys.modules, "langfuse", None)  # type: ignore[call-overload]
    from evidence_enrichment.observability.langfuse import get_langfuse_client

    assert get_langfuse_client() is None


def test_langfuse_observe_bypasses_sdk_when_client_unavailable(monkeypatch) -> None:
    """Local-only runs must not touch the Langfuse SDK decorator."""
    from evidence_enrichment.observability import langfuse as lf_module

    class _ExplodingModule:
        @staticmethod
        def observe(*_args, **_kwargs):
            raise AssertionError("Langfuse SDK decorator should not be used")

    monkeypatch.setitem(sys.modules, "langfuse", _ExplodingModule())
    monkeypatch.setattr(lf_module, "get_langfuse_client", lambda: None)

    @lf_module.observe(name="test", as_type="chain")
    def _wrapped(value: str) -> str:
        return value.upper()

    assert _wrapped("ok") == "OK"


def test_langfuse_observe_async_bypasses_sdk_when_client_unavailable(monkeypatch) -> None:
    """Async stage methods must also bypass Langfuse when local-only."""
    from evidence_enrichment.observability import langfuse as lf_module

    class _ExplodingModule:
        @staticmethod
        def observe(*_args, **_kwargs):
            raise AssertionError("Langfuse SDK decorator should not be used")

    monkeypatch.setitem(sys.modules, "langfuse", _ExplodingModule())
    monkeypatch.setattr(lf_module, "get_langfuse_client", lambda: None)

    @lf_module.observe(name="test_async", as_type="chain")
    async def _wrapped(value: str) -> str:
        return value.upper()

    assert asyncio.run(_wrapped("ok")) == "OK"


def test_record_stage_observation_swallows_errors(monkeypatch) -> None:
    """record_stage_observation must not propagate exceptions from the client."""
    _clear_langfuse_env(monkeypatch)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    from evidence_enrichment.observability import langfuse as lf_module

    class _BrokenClient:
        def update_current_span(self, **_kwargs):
            raise RuntimeError("simulated Langfuse failure")

    monkeypatch.setattr(lf_module, "get_langfuse_client", lambda: _BrokenClient())

    # Should not raise
    lf_module.record_stage_observation(
        "test_stage",
        {"mode": "replay"},
        [],
        lambda x: {},
    )


# ---------------------------------------------------------------------------
# Backend routing tests (OBSERVABILITY_BACKEND selector)
# ---------------------------------------------------------------------------


def _clear_all_obs_env(monkeypatch):
    """Remove all observability-related env vars so each test starts clean."""
    for key in (
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
    ):
        monkeypatch.delenv(key, raising=False)


def test_router_langfuse_returns_correct_flags() -> None:
    from evidence_enrichment.observability.router import get_active_backends

    langfuse_active, langsmith_active = get_active_backends("langfuse")
    assert langfuse_active is True
    assert langsmith_active is False


def test_router_langsmith_returns_correct_flags() -> None:
    from evidence_enrichment.observability.router import get_active_backends

    langfuse_active, langsmith_active = get_active_backends("langsmith")
    assert langfuse_active is False
    assert langsmith_active is True


def test_router_dual_returns_correct_flags() -> None:
    from evidence_enrichment.observability.router import get_active_backends

    langfuse_active, langsmith_active = get_active_backends("dual")
    assert langfuse_active is True
    assert langsmith_active is True


def test_router_none_returns_correct_flags() -> None:
    from evidence_enrichment.observability.router import get_active_backends

    langfuse_active, langsmith_active = get_active_backends("none")
    assert langfuse_active is False
    assert langsmith_active is False


def test_router_default_is_langfuse() -> None:
    from evidence_enrichment.observability.router import DEFAULT_BACKEND, resolve_backend

    assert DEFAULT_BACKEND == "langfuse"
    assert resolve_backend(None) == "langfuse"
    assert resolve_backend("") == "langfuse"


def test_router_unknown_value_falls_back_to_none_fail_closed() -> None:
    """Invalid backend values fall back to 'none' (fail-closed), not 'langfuse'."""
    from evidence_enrichment.observability.router import resolve_backend

    result = resolve_backend("invalid-backend")
    assert result == "none"


def test_settings_load_langfuse_backend_activates_langfuse_only(
    monkeypatch,
) -> None:
    """With OBSERVABILITY_BACKEND=langfuse, only Langfuse apply is called."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")

    langfuse_called = []
    langsmith_called = []

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import _reset_settings_cache

    _reset_settings_cache()
    monkeypatch.setattr(
        settings_mod,
        "apply_langfuse_env",
        lambda *args, **kwargs: langfuse_called.append(True) or False,
    )
    monkeypatch.setattr(
        settings_mod,
        "apply_langsmith_env",
        lambda *args, **kwargs: langsmith_called.append(True) or False,
    )

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    s = Settings.load()
    assert s.observability_backend == "langfuse"
    assert langfuse_called, "apply_langfuse_env should have been called"
    assert not langsmith_called, "apply_langsmith_env should NOT have been called"
    _reset_settings_cache()


def test_settings_load_langsmith_backend_activates_langsmith_only(
    monkeypatch,
) -> None:
    """With OBSERVABILITY_BACKEND=langsmith, only LangSmith apply is called."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")

    langfuse_called = []
    langsmith_called = []

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import _reset_settings_cache

    _reset_settings_cache()
    monkeypatch.setattr(
        settings_mod,
        "apply_langfuse_env",
        lambda *args, **kwargs: langfuse_called.append(True) or False,
    )
    monkeypatch.setattr(
        settings_mod,
        "apply_langsmith_env",
        lambda *args, **kwargs: langsmith_called.append(True) or False,
    )

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    s = Settings.load()
    assert s.observability_backend == "langsmith"
    assert not langfuse_called, "apply_langfuse_env should NOT have been called"
    assert langsmith_called, "apply_langsmith_env should have been called"
    _reset_settings_cache()


def test_settings_load_dual_backend_activates_both(monkeypatch) -> None:
    """With OBSERVABILITY_BACKEND=dual, both apply functions are called."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "dual")

    langfuse_called = []
    langsmith_called = []

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import _reset_settings_cache

    _reset_settings_cache()
    monkeypatch.setattr(
        settings_mod,
        "apply_langfuse_env",
        lambda *args, **kwargs: langfuse_called.append(True) or False,
    )
    monkeypatch.setattr(
        settings_mod,
        "apply_langsmith_env",
        lambda *args, **kwargs: langsmith_called.append(True) or False,
    )

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    s = Settings.load()
    assert s.observability_backend == "dual"
    assert langfuse_called, "apply_langfuse_env should have been called"
    assert langsmith_called, "apply_langsmith_env should have been called"
    _reset_settings_cache()


def test_settings_load_none_backend_activates_neither(monkeypatch) -> None:
    """With OBSERVABILITY_BACKEND=none, neither apply function is called."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "none")

    langfuse_called = []
    langsmith_called = []

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import _reset_settings_cache

    _reset_settings_cache()
    monkeypatch.setattr(
        settings_mod,
        "apply_langfuse_env",
        lambda *args, **kwargs: langfuse_called.append(True) or False,
    )
    monkeypatch.setattr(
        settings_mod,
        "apply_langsmith_env",
        lambda *args, **kwargs: langsmith_called.append(True) or False,
    )

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    s = Settings.load()
    assert s.observability_backend == "none"
    assert not langfuse_called, "apply_langfuse_env should NOT have been called"
    assert not langsmith_called, "apply_langsmith_env should NOT have been called"
    _reset_settings_cache()


def test_settings_load_default_is_langfuse(monkeypatch) -> None:
    """When OBSERVABILITY_BACKEND is absent, langfuse is the default."""
    _clear_all_obs_env(monkeypatch)

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()
    langfuse_called = []
    monkeypatch.setattr(
        settings_mod,
        "apply_langfuse_env",
        lambda *args, **kwargs: langfuse_called.append(True) or False,
    )
    monkeypatch.setattr(settings_mod, "apply_langsmith_env", lambda *args, **kwargs: False)

    s = Settings.load()
    assert s.observability_backend == "langfuse"
    assert langfuse_called, "Langfuse should be active by default"
    _reset_settings_cache()


def test_missing_credentials_with_langfuse_backend_no_crash(
    monkeypatch,
) -> None:
    """Selecting langfuse with no credentials must not crash; client stays None."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    # No LANGFUSE_SECRET_KEY set

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache
    from evidence_enrichment.observability.langfuse import get_langfuse_client

    _reset_settings_cache()

    # Must not raise
    settings = Settings.load()
    assert settings.observability_backend == "langfuse"

    # Client must be None — no credentials available
    client = get_langfuse_client()
    assert client is None


# ---------------------------------------------------------------------------
# Client-wrapping integration tests
# ---------------------------------------------------------------------------
# These tests assert that _wrap_openai_client / _wrap_anthropic_client respect
# OBSERVABILITY_BACKEND and do NOT wrap clients when the backend is langfuse
# or none — even when LANGSMITH_TRACING=true is present in the environment.
# ---------------------------------------------------------------------------


def _sentinel_client():
    """Minimal stand-in for an OpenAI/Anthropic client object."""

    class _Sentinel:
        pass

    return _Sentinel()


@pytest.mark.parametrize("backend", ["langfuse", "none"])
def test_wrap_openai_not_called_when_langsmith_inactive(
    monkeypatch, backend: str
) -> None:
    """With LANGSMITH_* credentials present but backend != langsmith/dual,
    _wrap_openai_client must return the original client unchanged."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", backend)
    # Simulate LangSmith credentials present in the shell
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")

    from evidence_enrichment.core.providers.agents import _wrap_openai_client

    original = _sentinel_client()
    result = _wrap_openai_client(original)
    assert result is original, (
        f"_wrap_openai_client should return original client for OBSERVABILITY_BACKEND={backend!r}, "
        "but it returned a wrapped client"
    )


@pytest.mark.parametrize("backend", ["langfuse", "none"])
def test_wrap_anthropic_not_called_when_langsmith_inactive(
    monkeypatch, backend: str
) -> None:
    """With LANGSMITH_* credentials present but backend != langsmith/dual,
    _wrap_anthropic_client must return the original client unchanged."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", backend)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")

    from evidence_enrichment.core.providers.agents import _wrap_anthropic_client

    original = _sentinel_client()
    result = _wrap_anthropic_client(original)
    assert result is original, (
        f"_wrap_anthropic_client should return original client for OBSERVABILITY_BACKEND={backend!r}, "
        "but it returned a wrapped client"
    )


@pytest.mark.parametrize("backend", ["langsmith", "dual"])
def test_wrap_openai_attempts_wrap_when_langsmith_active(
    monkeypatch, backend: str
) -> None:
    """With backend=langsmith or dual, _wrap_openai_client tries to wrap.
    We stub wrap_openai to return a distinct object and verify it is used.
    TRACE_REDACT_VALUES must be false for wrapping to proceed."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", backend)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")
    monkeypatch.setenv("TRACE_REDACT_VALUES", "false")

    wrapped_sentinel = _sentinel_client()


    # Stub the langsmith.wrappers import path used inside _wrap_openai_client
    class _FakeWrappers:
        @staticmethod
        def wrap_openai(_client):
            return wrapped_sentinel

    import sys

    fake_ls = type(sys)('langsmith')
    fake_ls.wrappers = _FakeWrappers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'langsmith', fake_ls)
    monkeypatch.setitem(sys.modules, 'langsmith.wrappers', _FakeWrappers)  # type: ignore[arg-type]

    from evidence_enrichment.core.providers.agents import _wrap_openai_client

    original = _sentinel_client()
    result = _wrap_openai_client(original)
    assert result is wrapped_sentinel, (
        f"_wrap_openai_client should return wrapped client for OBSERVABILITY_BACKEND={backend!r}"
    )


def test_wrap_openai_runtime_backend_override_disables_langsmith(monkeypatch) -> None:
    """An injected runtime backend must override stale env-backed LangSmith state."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")
    monkeypatch.setenv("TRACE_REDACT_VALUES", "false")

    wrapped_sentinel = _sentinel_client()

    class _FakeWrappers:
        @staticmethod
        def wrap_openai(_client):
            return wrapped_sentinel

    fake_ls = type(sys)("langsmith")
    fake_ls.wrappers = _FakeWrappers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langsmith", fake_ls)
    monkeypatch.setitem(sys.modules, "langsmith.wrappers", _FakeWrappers)  # type: ignore[arg-type]

    from evidence_enrichment.core.providers.agents import _wrap_openai_client
    from evidence_enrichment.observability.runtime import (
        activate_runtime_observability_config,
        reset_runtime_observability_config,
    )

    original = _sentinel_client()
    token = activate_runtime_observability_config(
        backend="none",
        trace_redact_values=False,
    )
    try:
        result = _wrap_openai_client(original)
    finally:
        reset_runtime_observability_config(token)

    assert result is original


@pytest.mark.parametrize("backend", ["langsmith", "dual"])
def test_wrap_openai_skipped_when_trace_redact_enabled(
    monkeypatch, backend: str
) -> None:
    """With TRACE_REDACT_VALUES=true, _wrap_openai_client must NOT wrap even
    when LangSmith is fully configured and active."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", backend)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")
    monkeypatch.setenv("TRACE_REDACT_VALUES", "true")

    from evidence_enrichment.core.providers.agents import _wrap_openai_client

    original = _sentinel_client()
    result = _wrap_openai_client(original)
    assert result is original, (
        f"_wrap_openai_client must NOT wrap when TRACE_REDACT_VALUES=true (backend={backend!r})"
    )


@pytest.mark.parametrize("backend", ["langsmith", "dual"])
def test_wrap_anthropic_skipped_when_trace_redact_enabled(
    monkeypatch, backend: str
) -> None:
    """With TRACE_REDACT_VALUES=true, _wrap_anthropic_client must NOT wrap."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", backend)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")
    monkeypatch.setenv("TRACE_REDACT_VALUES", "true")

    from evidence_enrichment.core.providers.agents import _wrap_anthropic_client

    original = _sentinel_client()
    result = _wrap_anthropic_client(original)
    assert result is original, (
        f"_wrap_anthropic_client must NOT wrap when TRACE_REDACT_VALUES=true (backend={backend!r})"
    )


@pytest.mark.parametrize("backend", ["langsmith", "dual"])
def test_wrap_openai_not_called_when_langsmith_backend_but_no_key(
    monkeypatch, backend: str
) -> None:
    """With backend=langsmith/dual but no API key, wrapping must NOT occur.

    Regression: langsmith_tracing_ready() must return False when LANGSMITH_TRACING
    is true but LANGSMITH_API_KEY is absent.
    """
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", backend)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("TRACE_REDACT_VALUES", "false")
    # Deliberately do NOT set LANGSMITH_API_KEY

    from evidence_enrichment.core.providers.agents import _wrap_openai_client

    original = _sentinel_client()
    result = _wrap_openai_client(original)
    assert result is original, (
        f"_wrap_openai_client must NOT wrap when API key is absent (backend={backend!r})"
    )


@pytest.mark.parametrize("backend", ["langsmith", "dual"])
def test_wrap_openai_not_called_when_langsmith_backend_but_tracing_off(
    monkeypatch, backend: str
) -> None:
    """With backend=langsmith/dual and API key but LANGSMITH_TRACING=false, no wrap."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", backend)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")
    monkeypatch.setenv("TRACE_REDACT_VALUES", "false")

    from evidence_enrichment.core.providers.agents import _wrap_openai_client

    original = _sentinel_client()
    result = _wrap_openai_client(original)
    assert result is original, (
        f"_wrap_openai_client must NOT wrap when LANGSMITH_TRACING=false (backend={backend!r})"
    )


def test_langsmith_tracing_ready_requires_both_tracing_and_key(monkeypatch) -> None:
    """langsmith_tracing_ready() only returns True when all three conditions met."""
    from evidence_enrichment.observability.router import langsmith_tracing_ready

    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")

    # Neither tracing nor key
    assert langsmith_tracing_ready() is False

    # Tracing only
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert langsmith_tracing_ready() is False

    # Both
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")
    assert langsmith_tracing_ready() is True

    # Wrong backend (langfuse) even with full credentials
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    assert langsmith_tracing_ready() is False


def test_current_backend_prefers_last_loaded_settings_over_later_env_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    """Helper reads should follow the last successful Settings.load() globally."""
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache
    from evidence_enrichment.observability.router import current_backend

    _clear_all_obs_env(monkeypatch)
    env_file = tmp_path / "observability.env"
    env_file.write_text(
        "OBSERVABILITY_BACKEND=langsmith\n"
        "LANGSMITH_TRACING=true\n"
        "LANGSMITH_API_KEY=sk-ls-test\n"
    )

    _reset_settings_cache()
    Settings.load(env_file=str(env_file))
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")

    assert current_backend() == "langsmith"


def test_get_langfuse_client_runtime_backend_override_disables_langfuse(
    monkeypatch,
) -> None:
    """An injected runtime backend must override stale env-backed Langfuse state."""
    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

    fake_client = object()
    fake_langfuse = type(sys)("langfuse")
    fake_langfuse.get_client = lambda: fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)

    from evidence_enrichment.observability.langfuse import get_langfuse_client
    from evidence_enrichment.observability.runtime import (
        activate_runtime_observability_config,
        reset_runtime_observability_config,
    )

    assert get_langfuse_client() is fake_client

    token = activate_runtime_observability_config(backend="none")
    try:
        assert get_langfuse_client() is None
    finally:
        reset_runtime_observability_config(token)


@pytest.mark.parametrize("backend", ["none", "langfuse"])
def test_langsmith_tracing_ready_false_when_backend_excludes_langsmith(
    monkeypatch, backend: str
) -> None:
    """Full LangSmith credentials must not yield True when backend excludes it."""
    from evidence_enrichment.observability.router import langsmith_tracing_ready

    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", backend)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-test")
    assert langsmith_tracing_ready() is False


def test_apply_langfuse_env_requires_both_keys(monkeypatch) -> None:
    """apply_langfuse_env returns False when only one of secret/public key present."""
    from evidence_enrichment.observability.langfuse import apply_langfuse_env

    _clear_all_obs_env(monkeypatch)

    # Secret key only — should return False (partial config)
    result = apply_langfuse_env({"LANGFUSE_SECRET_KEY": "sk-lf"})
    assert result is False, "apply_langfuse_env must return False with only secret key"

    _clear_all_obs_env(monkeypatch)

    # Public key only — should return False
    result = apply_langfuse_env({"LANGFUSE_PUBLIC_KEY": "pk-lf"})
    assert result is False, "apply_langfuse_env must return False with only public key"

    _clear_all_obs_env(monkeypatch)

    # Both keys — should return True
    result = apply_langfuse_env(
        {"LANGFUSE_SECRET_KEY": "sk-lf", "LANGFUSE_PUBLIC_KEY": "pk-lf"}
    )
    assert result is True, "apply_langfuse_env must return True with both keys"


def test_apply_langsmith_env_requires_both_tracing_and_key(monkeypatch) -> None:
    """apply_langsmith_env returns False when only one credential is present."""
    from evidence_enrichment.observability.langsmith import apply_langsmith_env

    _clear_all_obs_env(monkeypatch)

    # API key only, no tracing flag
    result = apply_langsmith_env({"LANGSMITH_API_KEY": "sk-ls"})
    assert result is False, "apply_langsmith_env must return False without LANGSMITH_TRACING"

    _clear_all_obs_env(monkeypatch)

    # Tracing flag only, no API key
    result = apply_langsmith_env({"LANGSMITH_TRACING": "true"})
    assert result is False, "apply_langsmith_env must return False without API key"

    _clear_all_obs_env(monkeypatch)

    # Both present
    result = apply_langsmith_env(
        {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "sk-ls"}
    )
    assert result is True, "apply_langsmith_env must return True with tracing + API key"


# ---------------------------------------------------------------------------
# Settings.load() env-export and key-clearing behaviour
# ---------------------------------------------------------------------------


def _clear_obs_keys(monkeypatch) -> None:
    """Remove all observability env vars so each test starts from a clean slate."""
    for key in (
        "OBSERVABILITY_BACKEND",
        "TRACE_REDACT_VALUES",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_settings_load_exports_backend_to_os_environ(monkeypatch) -> None:
    """Settings.load() must write the resolved backend into os.environ."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "none")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()
    Settings.load()

    assert os.environ.get("OBSERVABILITY_BACKEND") == "none"


def test_settings_load_clears_langfuse_keys_for_langsmith_backend(monkeypatch) -> None:
    """When backend=langsmith, LANGFUSE_* keys must be cleared from os.environ."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-ambient")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-ambient")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()
    Settings.load()

    assert os.environ.get("LANGFUSE_SECRET_KEY") is None
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") is None


def test_settings_load_clears_langsmith_tracing_for_langfuse_backend(monkeypatch) -> None:
    """When backend=langfuse, LANGSMITH_TRACING must be cleared from os.environ."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()
    Settings.load()

    assert os.environ.get("LANGSMITH_TRACING") is None
    assert os.environ.get("LANGCHAIN_TRACING_V2") is None


def test_resolve_backend_invalid_value_falls_back_to_none() -> None:
    """An unrecognised backend string must resolve to 'none' (fail-closed)."""
    from evidence_enrichment.observability.router import resolve_backend

    result = resolve_backend("bogus_backend")
    assert result == "none"


def test_resolve_backend_missing_value_defaults_to_langfuse() -> None:
    """A missing backend string must resolve to 'langfuse' (default)."""
    from evidence_enrichment.observability.router import resolve_backend

    result = resolve_backend(None)
    assert result == "langfuse"


# ---------------------------------------------------------------------------
# LANGFUSE_BASE_URL cleared on deactivation
# ---------------------------------------------------------------------------


def test_settings_load_clears_langfuse_base_url_for_langsmith_backend(monkeypatch) -> None:
    """When backend=langsmith, LANGFUSE_BASE_URL must also be cleared (not just HOST/keys)."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-ambient")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-ambient")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.com")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example.com")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()
    Settings.load()

    assert os.environ.get("LANGFUSE_BASE_URL") is None
    assert os.environ.get("LANGFUSE_HOST") is None
    assert os.environ.get("LANGFUSE_SECRET_KEY") is None
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") is None


# ---------------------------------------------------------------------------
# _pick_value empty-string precedence
# ---------------------------------------------------------------------------


def test_pick_value_skips_empty_string_and_falls_through(monkeypatch) -> None:
    """apply_langfuse_env must use .env value when ambient env var is empty string."""
    _clear_langfuse_env(monkeypatch)
    # Simulate shell export with empty value — should be ignored
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")

    from evidence_enrichment.observability.langfuse import apply_langfuse_env

    enabled = apply_langfuse_env(
        {
            "LANGFUSE_SECRET_KEY": "sk-from-dotenv",
            "LANGFUSE_PUBLIC_KEY": "pk-from-dotenv",
        }
    )
    assert enabled is True
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-from-dotenv"
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-from-dotenv"


# ---------------------------------------------------------------------------
# .env-only OBSERVABILITY_BACKEND export
# ---------------------------------------------------------------------------


def test_settings_load_exports_dotenv_only_backend(tmp_path, monkeypatch) -> None:
    """.env-only OBSERVABILITY_BACKEND must be exported into os.environ after load."""
    # Ensure no ambient env var overrides the .env file value
    _clear_obs_keys(monkeypatch)

    # Write a minimal .env file with backend=none
    env_file = tmp_path / ".env"
    env_file.write_text("OBSERVABILITY_BACKEND=none\n")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()
    s = Settings.load(env_file=str(env_file))

    assert s.observability_backend == "none"
    assert os.environ.get("OBSERVABILITY_BACKEND") == "none"


# ---------------------------------------------------------------------------
# Stateful reload regressions
# ---------------------------------------------------------------------------


def test_settings_load_backend_flips_restore_ambient_langfuse_credentials(
    monkeypatch,
) -> None:
    """Shell-provided Langfuse creds must survive langfuse→langsmith→langfuse flips
    when they remain present in os.environ or .env for reactivation.

    After the ambient-secret scrubbing fix, credentials are no longer cached in
    the ambient snapshot across deactivations.  They must be re-supplied (still
    present in env, or provided via .env) at the time of reactivation.
    """
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-shell")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-shell")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://shell.langfuse.example.com")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()

    # 1. Activate Langfuse — creds propagated.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    Settings.load()
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-shell"

    # 2. Flip to Langsmith — Langfuse keys cleared from env and ambient scrubbed.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-langsmith")
    Settings.load()
    assert os.environ.get("LANGFUSE_SECRET_KEY") is None
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") is None
    assert os.environ.get("LANGFUSE_BASE_URL") is None

    # 3. Flip back to Langfuse — credentials must be re-supplied in env.
    #    In a real shell the credential would still be in the process's initial
    #    environment; monkeypatch keeps it in its backing store but
    #    Settings.load() popped it in step 2.  Re-set to simulate the shell.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-shell")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-shell")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://shell.langfuse.example.com")
    Settings.load()
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-shell"
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-shell"
    assert os.environ["LANGFUSE_BASE_URL"] == "https://shell.langfuse.example.com"
    assert os.environ["LANGFUSE_HOST"] == "https://shell.langfuse.example.com"



def test_settings_load_backend_flips_restore_ambient_langsmith_credentials(
    monkeypatch,
) -> None:
    """Shell-provided LangSmith creds must survive langsmith→langfuse→langsmith flips
    when they remain present in os.environ for reactivation.

    After the ambient-secret scrubbing fix, credentials are no longer cached in
    the ambient snapshot across deactivations.  They must be re-supplied (still
    present in env, or provided via .env) at the time of reactivation.
    """
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-shell")
    monkeypatch.setenv("LANGSMITH_PROJECT", "proj-shell")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()

    # 1. Activate LangSmith — creds propagated (incl. LANGCHAIN_* aliases).
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    Settings.load()
    assert os.environ["LANGSMITH_API_KEY"] == "sk-ls-shell"
    assert os.environ["LANGCHAIN_API_KEY"] == "sk-ls-shell"
    assert os.environ["LANGSMITH_PROJECT"] == "proj-shell"

    # 2. Flip to Langfuse — LangSmith keys cleared from env and ambient scrubbed.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://lf.example.com")
    Settings.load()
    assert os.environ.get("LANGSMITH_API_KEY") is None
    assert os.environ.get("LANGCHAIN_API_KEY") is None
    assert os.environ.get("LANGSMITH_PROJECT") is None
    assert os.environ.get("LANGCHAIN_PROJECT") is None
    assert os.environ.get("LANGSMITH_TRACING") is None
    assert os.environ.get("LANGCHAIN_TRACING_V2") is None

    # 3. Flip back to LangSmith — re-supply credentials in env.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-shell")
    monkeypatch.setenv("LANGSMITH_PROJECT", "proj-shell")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    # Remove Langfuse keys so they don't interfere.
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    Settings.load()
    assert os.environ["LANGSMITH_API_KEY"] == "sk-ls-shell"
    assert os.environ["LANGCHAIN_API_KEY"] == "sk-ls-shell"
    assert os.environ["LANGSMITH_PROJECT"] == "proj-shell"
    assert os.environ["LANGCHAIN_PROJECT"] == "proj-shell"


def test_settings_load_active_langfuse_reload_does_not_reuse_stale_dotenv_values(
    tmp_path,
    monkeypatch,
) -> None:
    """A later langfuse load with missing creds must clear old .env-managed values."""
    _clear_obs_keys(monkeypatch)

    first_env = tmp_path / "first.env"
    first_env.write_text(
        "\n".join(
            [
                "OBSERVABILITY_BACKEND=langfuse",
                "LANGFUSE_SECRET_KEY=sk-first",
                "LANGFUSE_PUBLIC_KEY=pk-first",
                "LANGFUSE_BASE_URL=https://first.langfuse.example.com",
            ]
        )
        + "\n"
    )
    second_env = tmp_path / "second.env"
    second_env.write_text("OBSERVABILITY_BACKEND=langfuse\n")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache
    from evidence_enrichment.observability.langfuse import get_langfuse_client

    _reset_settings_cache()
    Settings.load(env_file=str(first_env))
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-first"
    assert os.environ["LANGFUSE_BASE_URL"] == "https://first.langfuse.example.com"

    Settings.load(env_file=str(second_env))
    assert os.environ.get("LANGFUSE_SECRET_KEY") is None
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") is None
    assert os.environ.get("LANGFUSE_BASE_URL") is None
    assert os.environ.get("LANGFUSE_HOST") is None
    assert get_langfuse_client() is None



def test_langsmith_wrapping_enabled_reads_dotenv_before_settings_load(
    tmp_path,
    monkeypatch,
) -> None:
    """Pre-settings client wrapping should honour a repo-local .env backend choice."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OBSERVABILITY_BACKEND=langsmith\n")

    from evidence_enrichment.observability.router import langsmith_wrapping_enabled

    assert langsmith_wrapping_enabled() is True


# ---------------------------------------------------------------------------
# LangSmith client — no-cache regression tests
# ---------------------------------------------------------------------------


def test_get_langsmith_client_none_before_enablement(monkeypatch) -> None:
    """get_langsmith_client() returns None when tracing is not enabled."""
    _clear_obs_keys(monkeypatch)

    from evidence_enrichment.observability.langsmith import get_langsmith_client

    assert get_langsmith_client() is None


def test_get_langsmith_client_activates_after_settings_load(
    tmp_path, monkeypatch
) -> None:
    """get_langsmith_client() returns a client after Settings.load() enables LangSmith.

    Previously the @lru_cache would have kept None after first call and never
    reflected the later enablement.

    langsmith.Client is stubbed so this test does not depend on the optional
    package being constructible with a real API key.
    """
    _clear_obs_keys(monkeypatch)

    env_file = tmp_path / "test.env"
    env_file.write_text(
        "OBSERVABILITY_BACKEND=langsmith\n"
        "LANGSMITH_API_KEY=ls-test-key\n"
        "LANGSMITH_TRACING=true\n"
    )

    class _FakeClient:
        pass

    import evidence_enrichment.observability.langsmith as ls_mod

    monkeypatch.setattr(ls_mod, "_langsmith_client_factory", lambda: _FakeClient(), raising=False)

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache
    from evidence_enrichment.observability.langsmith import get_langsmith_client

    # Patch the lazy import inside get_langsmith_client by stubbing the module.
    fake_langsmith = type(sys)("langsmith")
    fake_langsmith.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langsmith", fake_langsmith)

    # First call — no tracing env set yet; must return None.
    assert get_langsmith_client() is None

    # Enable LangSmith via Settings.load().
    _reset_settings_cache()
    Settings.load(env_file=str(env_file))

    # Second call — env now populated; must NOT return cached None.
    client = get_langsmith_client()
    assert client is not None
    assert isinstance(client, _FakeClient)


def test_get_langsmith_client_returns_none_after_backend_flip(
    tmp_path, monkeypatch
) -> None:
    """After flipping backend away from langsmith, get_langsmith_client() returns None.

    Previously the @lru_cache would have kept the old Client alive.
    """
    _clear_obs_keys(monkeypatch)

    ls_env = tmp_path / "langsmith.env"
    ls_env.write_text(
        "OBSERVABILITY_BACKEND=langsmith\n"
        "LANGSMITH_API_KEY=ls-test-key\n"
        "LANGSMITH_TRACING=true\n"
    )
    none_env = tmp_path / "none.env"
    none_env.write_text("OBSERVABILITY_BACKEND=none\n")

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache
    from evidence_enrichment.observability.langsmith import get_langsmith_client

    _reset_settings_cache()
    Settings.load(env_file=str(ls_env))
    assert get_langsmith_client() is not None  # sanity — enabled

    _reset_settings_cache()
    Settings.load(env_file=str(none_env))
    assert get_langsmith_client() is None  # must reflect flip


def test_apply_langsmith_env_mirrors_langchain_aliases_on_write(monkeypatch) -> None:
    """Writing LANGSMITH_API_KEY/PROJECT must also write the LANGCHAIN_* aliases."""
    _clear_obs_keys(monkeypatch)

    from evidence_enrichment.observability.langsmith import apply_langsmith_env

    apply_langsmith_env(
        {
            "LANGSMITH_API_KEY": "ls-mirror-key",
            "LANGSMITH_PROJECT": "mirror-project",
            "LANGSMITH_TRACING": "true",
        }
    )

    assert os.environ.get("LANGCHAIN_API_KEY") == "ls-mirror-key"
    assert os.environ.get("LANGCHAIN_PROJECT") == "mirror-project"


def test_apply_langsmith_env_clears_langchain_aliases_when_clear_missing(
    monkeypatch,
) -> None:
    """clear_missing=True with no api_key must remove both LANGSMITH and LANGCHAIN aliases."""
    monkeypatch.setenv("LANGCHAIN_API_KEY", "stale-lc-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "stale-lc-project")
    monkeypatch.setenv("LANGSMITH_API_KEY", "stale-ls-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "stale-ls-project")

    from evidence_enrichment.observability.langsmith import apply_langsmith_env

    # Call with no api_key/project in env_values and clear_missing=True.
    apply_langsmith_env({}, ambient_env={}, clear_missing=True)

    assert os.environ.get("LANGCHAIN_API_KEY") is None
    assert os.environ.get("LANGCHAIN_PROJECT") is None
    assert os.environ.get("LANGSMITH_API_KEY") is None
    assert os.environ.get("LANGSMITH_PROJECT") is None


# ---------------------------------------------------------------------------
# langsmith optional-dependency tests
# ---------------------------------------------------------------------------


def test_langsmith_client_none_when_import_missing(monkeypatch) -> None:
    """get_langsmith_client() returns None gracefully when langsmith is not installed."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    # Simulate langsmith not installed
    monkeypatch.setitem(sys.modules, "langsmith", None)  # type: ignore[call-overload]

    from evidence_enrichment.observability.langsmith import get_langsmith_client

    assert get_langsmith_client() is None


def test_settings_importable_when_langsmith_missing(monkeypatch) -> None:
    """evidence_enrichment.config.settings and pipeline.coordinator must be importable
    even when langsmith is absent.

    The langsmith package is optional; a missing installation must not prevent
    settings or the pipeline coordinator from loading.  The coordinator uses a
    try/except fallback for both langsmith.traceable and langfuse.observe.
    """
    import importlib

    monkeypatch.setitem(sys.modules, "langsmith", None)  # type: ignore[call-overload]

    import evidence_enrichment.observability.langsmith as ls_mod

    importlib.reload(ls_mod)

    import evidence_enrichment.config.settings as settings_mod  # noqa: F401

    assert settings_mod.Settings is not None
    assert ls_mod.apply_langsmith_env is not None

    # Verify the pipeline coordinator is also importable (uses try/except for
    # both langsmith.traceable and langfuse.observe).
    import evidence_enrichment.pipeline.coordinator as coord_mod  # noqa: F401

    assert coord_mod.EvidenceCoordinator is not None


def test_get_langsmith_client_respects_backend_before_settings_load(
    monkeypatch,
) -> None:
    """get_langsmith_client() returns None when OBSERVABILITY_BACKEND=langfuse
    even if LANGSMITH_TRACING=true is present in env."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-ambient-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    from evidence_enrichment.observability.langsmith import get_langsmith_client

    assert get_langsmith_client() is None


def test_get_langfuse_client_respects_backend_before_settings_load(
    monkeypatch,
) -> None:
    """get_langfuse_client() returns None when OBSERVABILITY_BACKEND=langsmith
    even if LANGFUSE_SECRET_KEY is present in env."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-ambient")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-ambient")

    from evidence_enrichment.observability.langfuse import get_langfuse_client

    assert get_langfuse_client() is None


def test_settings_load_failed_validation_leaves_env_unchanged(
    tmp_path, monkeypatch
) -> None:
    """A Settings.load() that fails validation must not mutate env or caches.

    Previously _refresh_ambient_observability_env() ran before validation and
    before rollback snapshots were taken, so a bad config could still mutate
    module-global observability state even though Settings.load() never
    succeeded.
    """
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-before")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-before")

    # Write a YAML config that will fail pydantic validation (bad type).
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("default_mode: [not, a, string]\n")  # list instead of str

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    # Seed module-global state so we can verify it rolls back too.
    settings_mod._AMBIENT_OBSERVABILITY_ENV["LANGSMITH_API_KEY"] = "sk-seeded"
    settings_mod._LAST_MANAGED_OBSERVABILITY_ENV["LANGSMITH_API_KEY"] = "sk-seeded"
    settings_mod._SHELL_SET_OBSERVABILITY_KEYS.add("LANGSMITH_API_KEY")
    settings_mod._EVICTION_PENDING_KEYS.add("LANGFUSE_SECRET_KEY")

    ambient_before = dict(settings_mod._AMBIENT_OBSERVABILITY_ENV)
    managed_before = dict(settings_mod._LAST_MANAGED_OBSERVABILITY_ENV)
    shell_before = set(settings_mod._SHELL_SET_OBSERVABILITY_KEYS)
    eviction_before = set(settings_mod._EVICTION_PENDING_KEYS)

    _reset_settings_cache()
    with pytest.raises(Exception):
        Settings.load(config_file=str(bad_yaml), env_file=str(tmp_path / "missing.env"))

    # env must be untouched
    assert os.environ.get("LANGFUSE_SECRET_KEY") == "sk-before"
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "pk-before"
    assert os.environ.get("OBSERVABILITY_BACKEND") is None

    # module-global observability caches must also be untouched
    assert settings_mod._AMBIENT_OBSERVABILITY_ENV == ambient_before
    assert settings_mod._LAST_MANAGED_OBSERVABILITY_ENV == managed_before
    assert settings_mod._SHELL_SET_OBSERVABILITY_KEYS == shell_before
    assert settings_mod._EVICTION_PENDING_KEYS == eviction_before


# ---------------------------------------------------------------------------
# get_settings() stale-cache tests
# ---------------------------------------------------------------------------


def test_get_settings_stale_cache_does_not_reflect_backend_change(
    tmp_path, monkeypatch
) -> None:
    """get_settings() returns the cached instance on repeated calls.

    This documents the known lru_cache behavior: callers using get_settings()
    directly see stale config until the cache is explicitly cleared.  The test
    provides a regression baseline for any future cache-invalidation mechanism.
    """
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")

    from evidence_enrichment.config.settings import _reset_settings_cache, get_settings

    _reset_settings_cache()
    first = get_settings()
    assert first.observability_backend == "langfuse"

    # Change backend in env — without clearing cache get_settings() must
    # return the SAME stale object (lru_cache behaviour).
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "none")
    second = get_settings()
    assert second is first  # stale — cache not cleared
    assert second.observability_backend == "langfuse"  # still old value


def test_get_settings_reflects_change_after_cache_clear(
    tmp_path, monkeypatch
) -> None:
    """After _reset_settings_cache(), get_settings() picks up new env."""
    _clear_obs_keys(monkeypatch)

    from evidence_enrichment.config.settings import Settings, _reset_settings_cache, get_settings  # noqa: F401

    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    _reset_settings_cache()
    first = get_settings()
    assert first.observability_backend == "langfuse"

    # Change env and clear cache — get_settings() must reflect the change.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "none")
    _reset_settings_cache()
    second = get_settings()
    assert second.observability_backend == "none"
    assert second is not first


# ---------------------------------------------------------------------------
# In-memory secret redaction tests
# ---------------------------------------------------------------------------


def test_secrets_redacted_from_globals_when_langfuse_inactive(
    tmp_path, monkeypatch
) -> None:
    """After switching away from langfuse, secret values must be zeroed in _LAST_MANAGED."""
    _clear_obs_keys(monkeypatch)

    langfuse_env = tmp_path / "langfuse.env"
    langfuse_env.write_text(
        "OBSERVABILITY_BACKEND=langfuse\n"
        "LANGFUSE_SECRET_KEY=sk-sensitive\n"
        "LANGFUSE_PUBLIC_KEY=pk-sensitive\n"
    )
    none_env = tmp_path / "none.env"
    none_env.write_text("OBSERVABILITY_BACKEND=none\n")

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()
    Settings.load(env_file=str(langfuse_env))
    # Secrets should be tracked in _LAST_MANAGED while Langfuse is active.
    assert settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.get("LANGFUSE_SECRET_KEY") == "sk-sensitive"

    # Flip to none — secret values must be zeroed (set to None, not stored as strings).
    _reset_settings_cache()
    Settings.load(env_file=str(none_env))
    assert settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.get("LANGFUSE_SECRET_KEY") is None
    assert settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.get("LANGFUSE_PUBLIC_KEY") is None
    # os.environ must also be clear.
    assert os.environ.get("LANGFUSE_SECRET_KEY") is None
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") is None


def test_secrets_redacted_from_globals_when_langsmith_inactive(
    tmp_path, monkeypatch
) -> None:
    """After switching away from langsmith, API key values must be zeroed in _LAST_MANAGED."""
    _clear_obs_keys(monkeypatch)

    ls_env = tmp_path / "langsmith.env"
    ls_env.write_text(
        "OBSERVABILITY_BACKEND=langsmith\n"
        "LANGSMITH_API_KEY=ls-sensitive\n"
        "LANGSMITH_TRACING=true\n"
    )
    langfuse_env = tmp_path / "langfuse.env"
    langfuse_env.write_text("OBSERVABILITY_BACKEND=langfuse\n")

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _reset_settings_cache()
    Settings.load(env_file=str(ls_env))
    assert settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.get("LANGSMITH_API_KEY") == "ls-sensitive"

    # Flip to langfuse — LangSmith secret values must be zeroed (None, not strings).
    _reset_settings_cache()
    Settings.load(env_file=str(langfuse_env))
    assert settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.get("LANGSMITH_API_KEY") is None
    assert settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.get("LANGCHAIN_API_KEY") is None
    assert os.environ.get("LANGSMITH_API_KEY") is None
    assert os.environ.get("LANGCHAIN_API_KEY") is None


def test_settings_load_env_mutation_rollback_on_apply_failure(
    tmp_path, monkeypatch
) -> None:
    """If apply_langfuse_env() raises mid-flight, os.environ must be rolled back."""
    _clear_obs_keys(monkeypatch)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-before")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-before")

    lf_env = tmp_path / "lf.env"
    lf_env.write_text(
        "OBSERVABILITY_BACKEND=langfuse\n"
        "LANGFUSE_SECRET_KEY=sk-new\n"
        "LANGFUSE_PUBLIC_KEY=pk-new\n"
    )

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    def _exploding_apply(*args, **kwargs):
        # Partially mutate then raise to simulate a mid-flight failure.
        os.environ["OBSERVABILITY_BACKEND"] = "langfuse"
        raise RuntimeError("simulated apply failure")

    monkeypatch.setattr(settings_mod, "apply_langfuse_env", _exploding_apply)

    _reset_settings_cache()
    with pytest.raises(RuntimeError, match="simulated apply failure"):
        Settings.load(env_file=str(lf_env))

    # env must have been rolled back to the pre-load state.
    assert os.environ.get("LANGFUSE_SECRET_KEY") == "sk-before"
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "pk-before"
    # OBSERVABILITY_BACKEND must have been rolled back too (it was set in the stub).
    assert os.environ.get("OBSERVABILITY_BACKEND") is None


# ---------------------------------------------------------------------------
# trace_redact_values
# ---------------------------------------------------------------------------

class TestTraceRedactValues:
    """Tests for the opt-in value redaction flag in summarize_* helpers."""

    def setup_method(self):
        # Ensure the env flag is off before each test.
        os.environ.pop("TRACE_REDACT_VALUES", None)

    def teardown_method(self):
        os.environ.pop("TRACE_REDACT_VALUES", None)

    def test_summarize_synthesis_no_redaction_when_explicitly_disabled(self) -> None:
        from evidence_enrichment.core.models.contracts import SynthesisResult
        from evidence_enrichment.observability.langsmith import summarize_synthesis

        os.environ["TRACE_REDACT_VALUES"] = "false"
        synthesis = SynthesisResult(
            field_name="company_name",
            value="Acme Corp",
            normalized_value="acme corp",
            synthesis_confidence=0.9,
            supporting_urls=["https://example.com"],
            conflicts=[],
            reasoning="test",
        )
        result = summarize_synthesis(synthesis)
        assert result["value"] == "Acme Corp"
        assert result["normalized_value"] == "acme corp"
        assert result["supporting_urls"] == ["https://example.com"]

    def test_summarize_synthesis_redacts_when_env_var_set(self, monkeypatch) -> None:
        from evidence_enrichment.core.models.contracts import SynthesisResult
        from evidence_enrichment.observability.langsmith import _REDACT_SENTINEL, summarize_synthesis

        monkeypatch.setenv("TRACE_REDACT_VALUES", "true")
        synthesis = SynthesisResult(
            field_name="company_name",
            value="Acme Corp",
            normalized_value="acme corp",
            synthesis_confidence=0.9,
            supporting_urls=["https://example.com"],
            conflicts=[],
            reasoning="test",
        )
        result = summarize_synthesis(synthesis)
        assert result["value"] == _REDACT_SENTINEL
        assert result["normalized_value"] == _REDACT_SENTINEL
        assert result["supporting_urls"] == _REDACT_SENTINEL
        # Non-sensitive fields must pass through.
        assert result["synthesis_confidence"] == 0.9
        assert result["conflict_count"] == 0

    def test_summarize_claims_redacts_candidate_values(self, monkeypatch) -> None:
        from evidence_enrichment.core.models.contracts import FactClaim
        from evidence_enrichment.observability.langsmith import _REDACT_SENTINEL, summarize_claims

        monkeypatch.setenv("TRACE_REDACT_VALUES", "1")
        claims = [
            FactClaim(
                field_name="company_name",
                candidate_value="secret-val",
                supporting_excerpt="excerpt",
                source_url="https://s.com",
                source_title="Source",
                analysis_confidence=0.8,
                source_authority_score=0.8,
                freshness_score=0.8,
                entity_match_score=0.8,
            ),
        ]
        result = summarize_claims(claims)
        assert result["candidate_values"] == _REDACT_SENTINEL
        assert result["source_urls"] == _REDACT_SENTINEL
        assert result["claim_count"] == 1

    def test_summarize_analysis_reports_redacts_candidate_values(self, monkeypatch) -> None:
        from evidence_enrichment.core.models.contracts import AnalysisReport, FactClaim
        from evidence_enrichment.core.models.contracts import ProviderType
        from evidence_enrichment.observability.langsmith import _REDACT_SENTINEL, summarize_analysis_reports

        monkeypatch.setenv("TRACE_REDACT_VALUES", "true")
        claim = FactClaim(
            field_name="company_name",
            candidate_value="pii-val",
            supporting_excerpt="excerpt",
            source_url="https://r.com",
            source_title="Source",
            analysis_confidence=0.7,
            source_authority_score=0.7,
            freshness_score=0.7,
            entity_match_score=0.7,
        )
        report = AnalysisReport(source_url="https://r.com", provider=ProviderType.OPENAI, claims=[claim])
        result = summarize_analysis_reports([report])
        assert result["candidate_values"] == _REDACT_SENTINEL
        assert result["source_urls"] == _REDACT_SENTINEL
        assert result["report_count"] == 1

    def test_trace_redact_values_setting_controls_redaction(self, tmp_path, monkeypatch) -> None:
        """Settings.load() with trace_redact_values=true in .env enables redaction."""
        from evidence_enrichment.config.settings import Settings, _reset_settings_cache
        from evidence_enrichment.core.models.contracts import SynthesisResult
        from evidence_enrichment.observability.langsmith import _REDACT_SENTINEL, summarize_synthesis

        _clear_obs_keys(monkeypatch)
        env_file = tmp_path / "test.env"
        env_file.write_text("TRACE_REDACT_VALUES=true\n")

        _reset_settings_cache()
        settings = Settings.load(env_file=str(env_file))
        assert settings.trace_redact_values is True

        synthesis = SynthesisResult(
            field_name="company_name",
            value="sensitive",
            normalized_value="sensitive-norm",
            synthesis_confidence=0.8,
            supporting_urls=["https://x.com"],
            conflicts=[],
            reasoning="test",
        )
        result = summarize_synthesis(synthesis)
        assert result["value"] == _REDACT_SENTINEL

        # Clean up env flag set by Settings.load.
        monkeypatch.delenv("TRACE_REDACT_VALUES", raising=False)

    def test_summarize_search_results_redacts_top_urls(self, monkeypatch) -> None:
        from evidence_enrichment.core.models.contracts import ProviderType, SearchResult
        from evidence_enrichment.observability.langsmith import _REDACT_SENTINEL, summarize_search_results

        monkeypatch.setenv("TRACE_REDACT_VALUES", "true")
        results = [
            SearchResult(
                url="https://secret.example.com/page",
                provider=ProviderType.OPENAI,
                title="T",
                snippet="S",
                rank=1,
                domain="secret.example.com",
            ),
        ]
        result = summarize_search_results(results)
        assert result["top_urls"] == _REDACT_SENTINEL
        assert result["result_count"] == 1
        # Non-sensitive fields must pass through.
        assert result["providers"] == ["openai"]

    def test_summarize_fetched_documents_redacts_urls(self, monkeypatch) -> None:
        from evidence_enrichment.core.models.contracts import RetrievedDocument
        from evidence_enrichment.observability.langsmith import _REDACT_SENTINEL, summarize_fetched_documents

        monkeypatch.setenv("TRACE_REDACT_VALUES", "true")
        docs = [
            RetrievedDocument(
                url="https://secret.example.com",
                final_url="https://secret.example.com",
                title="Doc",
                content_type="text/html",
                body="<html/>",
                provider="serper",
                fetch_success=True,
            )
        ]
        result = summarize_fetched_documents(docs)
        assert result["urls"] == _REDACT_SENTINEL
        assert result["document_count"] == 1
        assert result["success_count"] == 1

    def test_summarize_parsed_documents_redacts_nested_urls(self, monkeypatch) -> None:
        from evidence_enrichment.core.models.contracts import ParsedDocument
        from evidence_enrichment.observability.langsmith import _REDACT_SENTINEL, summarize_parsed_documents

        monkeypatch.setenv("TRACE_REDACT_VALUES", "true")
        docs = [
            ParsedDocument(
                url="https://secret.example.com",
                title="Title",
                content_type="text/html",
                text="body text",
                excerpt="body",
            )
        ]
        result = summarize_parsed_documents(docs)
        assert result["document_count"] == 1
        nested = result["documents"]
        assert isinstance(nested, list)
        assert nested[0]["url"] == _REDACT_SENTINEL
        # Non-sensitive nested fields must survive.
        assert nested[0]["title"] == "Title"
        assert nested[0]["text_chars"] == len("body text")

    def test_summarize_query_plan_redacts_primary_query_and_domain_hints(self, monkeypatch) -> None:
        from evidence_enrichment.core.models.contracts import SearchQueryPlan
        from evidence_enrichment.observability.langsmith import _REDACT_SENTINEL, summarize_query_plan

        monkeypatch.setenv("TRACE_REDACT_VALUES", "true")
        plan = SearchQueryPlan(
            field_name="ceo_name",
            entity_id="ent-1",
            primary_query="who is the CEO of Acme?",
            query_variants=["Acme CEO", "Acme chief executive"],
            domain_hints=["linkedin.com", "bloomberg.com"],
        )
        result = summarize_query_plan(plan)
        assert result["primary_query"] == _REDACT_SENTINEL
        assert result["domain_hints"] == _REDACT_SENTINEL
        # Non-sensitive fields must survive.
        assert result["field_name"] == "ceo_name"
        assert result["query_variants"] == 2

    def test_redaction_off_when_explicitly_disabled_for_url_and_query_fields(self, monkeypatch) -> None:
        """Ensure URL/query fields are NOT redacted when the flag is explicitly false."""
        from evidence_enrichment.core.models.contracts import ProviderType, SearchResult
        from evidence_enrichment.observability.langsmith import summarize_search_results

        monkeypatch.setenv("TRACE_REDACT_VALUES", "false")
        results = [
            SearchResult(
                url="https://visible.example.com",
                provider=ProviderType.OPENAI,
                title="T",
                snippet="S",
                rank=1,
                domain="visible.example.com",
            ),
        ]
        result = summarize_search_results(results)
        assert result["top_urls"] == ["https://visible.example.com"]

    def test_trace_redact_values_rollback_on_apply_failure(self, tmp_path, monkeypatch) -> None:
        """TRACE_REDACT_VALUES must be rolled back when Settings.load() fails mid-flight."""
        import evidence_enrichment.config.settings as settings_mod
        from evidence_enrichment.config.settings import Settings, _reset_settings_cache

        _clear_obs_keys(monkeypatch)
        # Start with redaction explicitly OFF in env.
        monkeypatch.delenv("TRACE_REDACT_VALUES", raising=False)

        lf_env = tmp_path / "lf.env"
        lf_env.write_text(
            "OBSERVABILITY_BACKEND=langfuse\n"
            "TRACE_REDACT_VALUES=true\n"
            "LANGFUSE_SECRET_KEY=sk-new\n"
            "LANGFUSE_PUBLIC_KEY=pk-new\n"
        )

        def _exploding_apply(*args, **kwargs):
            raise RuntimeError("simulated apply failure")

        monkeypatch.setattr(settings_mod, "apply_langfuse_env", _exploding_apply)

        _reset_settings_cache()
        with pytest.raises(RuntimeError, match="simulated apply failure"):
            Settings.load(env_file=str(lf_env))

        # TRACE_REDACT_VALUES must have been rolled back to its pre-load state (absent).
        assert os.environ.get("TRACE_REDACT_VALUES") is None

def test_should_redact_stays_aligned_with_loaded_settings_until_reload(monkeypatch) -> None:
    """Direct env edits do not override the last successful Settings.load()."""
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache
    from evidence_enrichment.observability.langsmith import _should_redact

    _clear_obs_keys(monkeypatch)
    _reset_settings_cache()
    settings = Settings.load()
    assert settings.trace_redact_values is True
    assert os.environ.get("TRACE_REDACT_VALUES") == "true"
    assert _should_redact() is True

    # A later direct env edit must not override the loaded settings snapshot.
    monkeypatch.setenv("TRACE_REDACT_VALUES", "false")
    assert _should_redact() is True


def test_should_redact_runtime_override_beats_env(monkeypatch) -> None:
    """An injected runtime setting must override stale env-backed redaction state."""
    from evidence_enrichment.observability.langsmith import _should_redact
    from evidence_enrichment.observability.runtime import (
        activate_runtime_observability_config,
        reset_runtime_observability_config,
    )

    _clear_all_obs_env(monkeypatch)
    monkeypatch.setenv("TRACE_REDACT_VALUES", "true")

    token = activate_runtime_observability_config(trace_redact_values=False)
    try:
        assert _should_redact() is False
    finally:
        reset_runtime_observability_config(token)


def test_should_redact_prefers_last_loaded_settings_over_later_env_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    """Direct helper calls should stay aligned with the last loaded Settings."""
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache
    from evidence_enrichment.observability.langsmith import _should_redact

    _clear_all_obs_env(monkeypatch)
    env_file = tmp_path / "redaction.env"
    env_file.write_text("TRACE_REDACT_VALUES=false\n")

    _reset_settings_cache()
    Settings.load(env_file=str(env_file))
    monkeypatch.setenv("TRACE_REDACT_VALUES", "true")

    assert _should_redact() is False


# ---------------------------------------------------------------------------
# _should_redact() side-effect safety
# ---------------------------------------------------------------------------

def test_should_redact_does_not_trigger_settings_load(monkeypatch) -> None:
    """Calling _should_redact() before Settings.load() must not mutate os.environ.

    Previously the fallback path called get_settings() which, on a cold cache,
    triggered Settings.load() and rewrote observability env vars as a side effect
    of an otherwise read-only redaction check.
    """
    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.observability.langsmith import _should_redact

    _clear_obs_keys(monkeypatch)
    settings_mod._reset_settings_cache()
    monkeypatch.delenv("TRACE_REDACT_VALUES", raising=False)

    before = {k: os.environ.get(k) for k in settings_mod._OBSERVABILITY_ENV_KEYS}

    result = _should_redact()

    after = {k: os.environ.get(k) for k in settings_mod._OBSERVABILITY_ENV_KEYS}
    assert after == before
    # Default is True when env var absent (privacy-safe default).
    assert result is True


# ---------------------------------------------------------------------------
# Secret resurrection regression
# ---------------------------------------------------------------------------

def test_deleted_inactive_credential_not_resurrected_on_reactivation(
    monkeypatch,
) -> None:
    """A credential deleted from env while its backend is inactive must NOT be
    restored when the backend is re-enabled.

    The ambient-secret scrubbing fix ensures _redact_inactive_secrets also
    clears _AMBIENT_OBSERVABILITY_ENV for inactive-backend secret keys, so a
    stale ambient value cannot be restored after deactivation.
    """
    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache

    _clear_obs_keys(monkeypatch)
    _reset_settings_cache()
    settings_mod._AMBIENT_OBSERVABILITY_ENV.clear()
    settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.clear()
    settings_mod._SHELL_SET_OBSERVABILITY_KEYS.clear()
    settings_mod._EVICTION_PENDING_KEYS.clear()

    # Step 1: langsmith active with a shell-set key.
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-original")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    Settings.load()
    assert os.environ.get("LANGSMITH_API_KEY") == "sk-original"

    # Step 2: flip to langfuse — langsmith keys cleared from env AND ambient scrubbed.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://lf.example.com")
    Settings.load()
    assert os.environ.get("LANGSMITH_API_KEY") is None

    # Step 3: operator deletes key from shell env between loads.
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    # Step 4: re-enable langsmith — the deleted key must NOT be resurrected.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    Settings.load()
    assert os.environ.get("LANGSMITH_API_KEY") is None, (
        "Deleted inactive credential must not be resurrected from ambient cache"
    )


# ---------------------------------------------------------------------------
# trace_payload input redaction tests
# ---------------------------------------------------------------------------


class TestTracePayloadInputRedaction:
    """Verify that sensitive trace_payload keys are redacted before leaving the process.

    Both the Langfuse (record_stage_observation / input=) and LangSmith
    (trace_payload_inputs) paths must honour TRACE_REDACT_VALUES.
    """

    def setup_method(self):
        os.environ.pop("TRACE_REDACT_VALUES", None)

    def teardown_method(self):
        os.environ.pop("TRACE_REDACT_VALUES", None)

    def test_trace_payload_inputs_redacts_sensitive_keys(self) -> None:
        """trace_payload_inputs redacts primary_query and urls when flag is on."""
        from evidence_enrichment.observability.langsmith import trace_payload_inputs

        os.environ["TRACE_REDACT_VALUES"] = "true"
        payload = {
            "trace_payload": {
                "mode": "live",
                "primary_query": "Company revenue 2023",
                "urls": ["https://example.com/doc1"],
                "query_variant_count": 3,
            }
        }
        result = trace_payload_inputs(payload)
        assert result["primary_query"] == "[REDACTED]"
        assert result["urls"] == "[REDACTED]"
        # Non-sensitive structural fields must be retained.
        assert result["mode"] == "live"
        assert result["query_variant_count"] == 3

    def test_trace_payload_inputs_passes_through_when_explicitly_disabled(self) -> None:
        """trace_payload_inputs returns values unchanged when TRACE_REDACT_VALUES=false."""
        from evidence_enrichment.observability.langsmith import trace_payload_inputs

        os.environ["TRACE_REDACT_VALUES"] = "false"
        payload = {
            "trace_payload": {
                "mode": "live",
                "primary_query": "Company revenue 2023",
                "urls": ["https://example.com/doc1"],
            }
        }
        result = trace_payload_inputs(payload)
        assert result["primary_query"] == "Company revenue 2023"
        assert result["urls"] == ["https://example.com/doc1"]

    def test_trace_payload_inputs_redacts_document_urls_nested(self) -> None:
        """trace_payload_inputs redacts 'url' keys inside documents lists."""
        from evidence_enrichment.observability.langsmith import trace_payload_inputs

        os.environ["TRACE_REDACT_VALUES"] = "true"
        payload = {
            "trace_payload": {
                "mode": "live",
                "documents": [
                    {"url": "https://example.com/a", "title": "Doc A", "text_chars": 500},
                ],
            }
        }
        result = trace_payload_inputs(payload)
        assert result["documents"][0]["url"] == "[REDACTED]"
        assert result["documents"][0]["title"] == "Doc A"
        assert result["documents"][0]["text_chars"] == 500

    def test_record_stage_observation_input_redacted(self, monkeypatch) -> None:
        """record_stage_observation sends a redacted trace_payload as span input."""
        from evidence_enrichment.observability.langfuse import record_stage_observation
        from evidence_enrichment.observability.langsmith import summarize_review_gate

        os.environ["TRACE_REDACT_VALUES"] = "true"

        captured: dict = {}

        class _FakeSpan:
            def update_current_span(self, *, input, output):
                captured["input"] = input
                captured["output"] = output

        fake_span = _FakeSpan()

        # Patch get_langfuse_client to return our fake span object.
        monkeypatch.setattr(
            "evidence_enrichment.observability.langfuse.get_langfuse_client",
            lambda: fake_span,
        )

        sensitive_payload = {
            "mode": "live",
            "primary_query": "secret query",
            "urls": ["https://secret.example.com"],
        }
        output_data = {
            "overall_confidence": 0.9,
            "decision": "approve",
            "gate_reason": "high confidence",
        }

        record_stage_observation("review_gate", sensitive_payload, output_data, summarize_review_gate)

        assert "input" in captured, "update_current_span was not called"
        assert captured["input"]["primary_query"] == "[REDACTED]"
        assert captured["input"]["urls"] == "[REDACTED]"
        assert captured["input"]["mode"] == "live"

    def test_record_stage_observation_input_not_redacted_when_explicitly_disabled(
        self, monkeypatch
    ) -> None:
        """record_stage_observation passes trace_payload through unchanged when flag is false."""
        from evidence_enrichment.observability.langfuse import record_stage_observation
        from evidence_enrichment.observability.langsmith import summarize_review_gate

        os.environ["TRACE_REDACT_VALUES"] = "false"

        captured: dict = {}

        class _FakeSpan:
            def update_current_span(self, *, input, output):
                captured["input"] = input

        monkeypatch.setattr(
            "evidence_enrichment.observability.langfuse.get_langfuse_client",
            lambda: _FakeSpan(),
        )

        payload = {"mode": "live", "primary_query": "open query"}
        record_stage_observation(
            "review_gate",
            payload,
            {"overall_confidence": 0.8, "decision": "approve", "gate_reason": "ok"},
            summarize_review_gate,
        )

        assert captured["input"]["primary_query"] == "open query"


# ---------------------------------------------------------------------------
# Per-stage trace_payload redaction regression tests
# Covers every stage key that appears in coordinator trace_payload dicts.
# ---------------------------------------------------------------------------


class TestPerStageTracePayloadRedaction:
    """Regression tests ensuring every coordinator stage's trace_payload keys
    are correctly classified as sensitive or structural.

    Tests run _maybe_redact directly on representative per-stage payloads,
    mirroring exactly what trace_payload_inputs / record_stage_observation
    will produce after going through _maybe_redact.
    """

    def setup_method(self):
        os.environ["TRACE_REDACT_VALUES"] = "true"

    def teardown_method(self):
        os.environ.pop("TRACE_REDACT_VALUES", None)

    def _redact(self, payload: dict) -> dict:
        from evidence_enrichment.observability.langsmith import _maybe_redact
        return _maybe_redact(payload)

    def test_query_plan_payload(self) -> None:
        payload = {
            "mode": "live",
            "entity_id": "ent-123",
            "field_name": "revenue",
            "company_name": "Acme Corp",
            "context_entry_ids": ["ctx-1", "ctx-2"],
        }
        result = self._redact(payload)
        # Sensitive
        assert result["company_name"] == "[REDACTED]"
        assert result["context_entry_ids"] == "[REDACTED]"
        # Structural (retained)
        assert result["mode"] == "live"
        assert result["entity_id"] == "ent-123"
        assert result["field_name"] == "revenue"

    def test_search_payload(self) -> None:
        payload = {
            "mode": "live",
            "primary_query": "Acme Corp revenue 2023",
            "query_variant_count": 3,
            "provider_order": ["serper", "tavily"],
        }
        result = self._redact(payload)
        assert result["primary_query"] == "[REDACTED]"
        assert result["query_variant_count"] == 3
        assert result["provider_order"] == ["serper", "tavily"]

    def test_fetch_payload(self) -> None:
        payload = {
            "mode": "live",
            "urls": ["https://example.com/a", "https://example.com/b"],
        }
        result = self._redact(payload)
        assert result["urls"] == "[REDACTED]"
        assert result["mode"] == "live"

    def test_parse_payload(self) -> None:
        payload = {
            "mode": "live",
            "urls": ["https://example.com/a"],
        }
        result = self._redact(payload)
        assert result["urls"] == "[REDACTED]"

    def test_evidence_assessment_payload(self) -> None:
        payload = {
            "mode": "live",
            "document_urls": ["https://example.com/a", "https://example.com/b"],
        }
        result = self._redact(payload)
        assert result["document_urls"] == "[REDACTED]"
        assert result["mode"] == "live"

    def test_analysis_payload(self) -> None:
        payload = {
            "mode": "live",
            "accepted_document_urls": ["https://example.com/a"],
            "accepted_count": 1,
            "rejected_count": 0,
            "documents": [
                {
                    "url": "https://example.com/a",
                    "accepted_for_analysis": True,
                    "entity_match_score": 0.9,
                    "source_authority_score": 0.8,
                    "freshness_score": 0.7,
                    "rejection_reason": None,
                }
            ],
        }
        result = self._redact(payload)
        assert result["accepted_document_urls"] == "[REDACTED]"
        # Nested document URL redacted
        assert result["documents"][0]["url"] == "[REDACTED]"
        # Structural fields retained
        assert result["documents"][0]["accepted_for_analysis"] is True
        assert result["documents"][0]["entity_match_score"] == 0.9
        assert result["accepted_count"] == 1

    def test_synthesis_payload_via_summarize_claims(self) -> None:
        """summarize_claims is used as trace_payload base for synthesis stage."""
        from evidence_enrichment.core.models.contracts import FactClaim
        from evidence_enrichment.observability.langsmith import summarize_claims

        claims = [
            FactClaim(
                candidate_value="$1B",
                source_url="https://example.com/a",
                analysis_confidence=0.9,
                field_name="revenue",
                supporting_excerpt="Revenue was $1B",
                source_title="Annual Report",
                source_authority_score=0.8,
                freshness_score=0.7,
                entity_match_score=0.9,
            )
        ]
        result = summarize_claims(claims)
        assert result["candidate_values"] == "[REDACTED]"
        assert result["source_urls"] == "[REDACTED]"
        assert result["claim_count"] == 1

    def test_review_gate_payload(self) -> None:
        """review_gate trace_payload merges summarize_claims + summarize_synthesis."""
        from evidence_enrichment.core.models.contracts import FactClaim, SynthesisResult
        from evidence_enrichment.observability.langsmith import summarize_claims, summarize_synthesis

        claims = [
            FactClaim(
                candidate_value="$1B",
                source_url="https://example.com/a",
                analysis_confidence=0.9,
                field_name="revenue",
                supporting_excerpt="Revenue was $1B",
                source_title="Annual Report",
                source_authority_score=0.8,
                freshness_score=0.7,
                entity_match_score=0.9,
            )
        ]
        synthesis = SynthesisResult(
            field_name="revenue",
            value="$1B",
            normalized_value="1000000000",
            reasoning="Strong evidence",
            synthesis_confidence=0.9,
            supporting_urls=["https://example.com/a"],
            conflicts=[],
        )
        payload = {"mode": "live", **summarize_claims(claims), **summarize_synthesis(synthesis)}
        # All sensitive fields already went through _maybe_redact inside summarize_*
        assert payload["candidate_values"] == "[REDACTED]"
        assert payload["source_urls"] == "[REDACTED]"
        assert payload["value"] == "[REDACTED]"
        assert payload["normalized_value"] == "[REDACTED]"
        assert payload["supporting_urls"] == "[REDACTED]"
        # Structural fields retained
        assert payload["claim_count"] == 1
        assert payload["synthesis_confidence"] == 0.9
        assert payload["conflict_count"] == 0


# ---------------------------------------------------------------------------
# clear_ambient_credentials() API tests
# ---------------------------------------------------------------------------


def test_clear_ambient_credentials_wipes_all(monkeypatch) -> None:
    """clear_ambient_credentials() with no args clears all observability credential keys."""
    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache, clear_ambient_credentials

    _clear_obs_keys(monkeypatch)
    _reset_settings_cache()
    settings_mod._AMBIENT_OBSERVABILITY_ENV.clear()
    settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.clear()
    settings_mod._SHELL_SET_OBSERVABILITY_KEYS.clear()
    settings_mod._EVICTION_PENDING_KEYS.clear()

    # Load with LangSmith active so credentials enter the ambient cache.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls-to-evict")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    Settings.load()
    assert settings_mod._AMBIENT_OBSERVABILITY_ENV.get("LANGSMITH_API_KEY") == "sk-ls-to-evict"

    # Explicitly evict.
    clear_ambient_credentials()
    assert settings_mod._AMBIENT_OBSERVABILITY_ENV.get("LANGSMITH_API_KEY") is None
    assert settings_mod._AMBIENT_OBSERVABILITY_ENV.get("LANGFUSE_SECRET_KEY") is None
    assert "LANGSMITH_API_KEY" not in settings_mod._SHELL_SET_OBSERVABILITY_KEYS


def test_clear_ambient_credentials_targeted_backend(monkeypatch) -> None:
    """clear_ambient_credentials('langsmith') only clears LangSmith keys."""
    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache, clear_ambient_credentials

    _clear_obs_keys(monkeypatch)
    _reset_settings_cache()
    settings_mod._AMBIENT_OBSERVABILITY_ENV.clear()
    settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.clear()
    settings_mod._SHELL_SET_OBSERVABILITY_KEYS.clear()
    settings_mod._EVICTION_PENDING_KEYS.clear()

    monkeypatch.setenv("OBSERVABILITY_BACKEND", "dual")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-ls")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    Settings.load()

    clear_ambient_credentials("langsmith")

    # LangSmith ambient cleared
    assert settings_mod._AMBIENT_OBSERVABILITY_ENV.get("LANGSMITH_API_KEY") is None
    assert settings_mod._AMBIENT_OBSERVABILITY_ENV.get("LANGCHAIN_API_KEY") is None
    # Langfuse ambient untouched
    assert settings_mod._AMBIENT_OBSERVABILITY_ENV.get("LANGFUSE_SECRET_KEY") == "sk-lf"


def test_clear_ambient_credentials_prevents_resurrection(monkeypatch) -> None:
    """After clear_ambient_credentials(), re-enabling a backend does not restore evicted creds."""
    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache, clear_ambient_credentials

    _clear_obs_keys(monkeypatch)
    _reset_settings_cache()
    settings_mod._AMBIENT_OBSERVABILITY_ENV.clear()
    settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.clear()
    settings_mod._SHELL_SET_OBSERVABILITY_KEYS.clear()
    settings_mod._EVICTION_PENDING_KEYS.clear()

    # Step 1: LangSmith active with shell credential.
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-original")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    Settings.load()

    # Step 2: Flip to Langfuse.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    Settings.load()

    # Step 3: Operator evicts stale credential explicitly, then removes from shell.
    clear_ambient_credentials("langsmith")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    # Step 4: Re-enable LangSmith — evicted credential must NOT be restored.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    Settings.load()
    assert os.environ.get("LANGSMITH_API_KEY") is None, (
        "clear_ambient_credentials() must prevent resurrection of evicted credential"
    )


def test_clear_ambient_credentials_eviction_consumed_after_successful_load(
    monkeypatch,
) -> None:
    """A successful Settings.load() consumes pending evictions so later fresh
    values can be picked up normally.
    """
    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache, clear_ambient_credentials

    _clear_obs_keys(monkeypatch)
    _reset_settings_cache()
    settings_mod._AMBIENT_OBSERVABILITY_ENV.clear()
    settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.clear()
    settings_mod._SHELL_SET_OBSERVABILITY_KEYS.clear()
    settings_mod._EVICTION_PENDING_KEYS.clear()

    # 1. Activate LangSmith with an initial credential.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-old")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    Settings.load()

    # 2. Evict the cached credential; pending flag must be set.
    clear_ambient_credentials("langsmith")
    assert "LANGSMITH_API_KEY" in settings_mod._EVICTION_PENDING_KEYS

    # 3. Successful non-LangSmith load consumes the eviction flag.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://lf.example.com")
    Settings.load()
    assert "LANGSMITH_API_KEY" not in settings_mod._EVICTION_PENDING_KEYS

    # 4. The same stale value remains blocked even after the pending flag is consumed.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-old")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    Settings.load()
    assert os.environ.get("LANGSMITH_API_KEY") is None

    # 5. A rotated value set after the successful load is picked up normally.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-rotated")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    Settings.load()
    assert os.environ.get("LANGSMITH_API_KEY") == "sk-rotated"


def test_get_langfuse_client_rejects_reinjected_evicted_secret_until_rotated(
    monkeypatch,
) -> None:
    """Direct helper use must not revive an explicitly evicted stale secret."""
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache, clear_ambient_credentials
    from evidence_enrichment.observability.langfuse import get_langfuse_client

    _clear_all_obs_env(monkeypatch)
    _reset_settings_cache()
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-old")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-old")
    Settings.load()

    fake_client = object()
    fake_langfuse = type(sys)("langfuse")
    fake_langfuse.get_client = lambda: fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)

    clear_ambient_credentials("langfuse")

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-old")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-old")
    assert get_langfuse_client() is None

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-new")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-new")
    assert get_langfuse_client() is fake_client


def test_clear_ambient_credentials_after_deactivation_still_tombstones(
    monkeypatch,
) -> None:
    """Tombstones are created even when clear_ambient_credentials() is called after
    the backend has already been deactivated (deactivate-then-evict order).

    Regression test for: _redact_inactive_secrets() zeroing caches before
    clear_ambient_credentials() had a chance to tombstone them.
    """
    from evidence_enrichment.config.settings import Settings, _reset_settings_cache, clear_ambient_credentials
    from evidence_enrichment.observability.langfuse import get_langfuse_client

    _clear_all_obs_env(monkeypatch)
    _reset_settings_cache()

    # 1. Load with langfuse active so the secret is cached.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-stale")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-stale")
    Settings.load()

    # 2. Deactivate the backend — _redact_inactive_secrets() zeros caches.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "none")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    Settings.load()

    # 3. Evict AFTER deactivation — must still tombstone the old value.
    clear_ambient_credentials("langfuse")

    # 4. Re-activate and try to reload with the same stale credential.
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-stale")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-stale")
    Settings.load()

    fake_client = object()
    fake_langfuse = type(sys)("langfuse")
    fake_langfuse.get_client = lambda: fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)

    # Tombstone must block the stale secret even though eviction happened after deactivation.
    assert get_langfuse_client() is None, (
        "Stale secret should be blocked by tombstone created from pre-redaction stash"
    )

    # 5. A rotated credential must be accepted.
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-new")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-new")
    Settings.load()
    assert get_langfuse_client() is fake_client, (
        "Rotated credential should clear the tombstone and be accepted"
    )


def test_clear_ambient_credentials_invalid_backend_raises() -> None:
    """clear_ambient_credentials() raises ValueError for unknown backend names."""
    from evidence_enrichment.config.settings import clear_ambient_credentials

    with pytest.raises(ValueError, match="Unknown backend"):
        clear_ambient_credentials("unknown_backend")
