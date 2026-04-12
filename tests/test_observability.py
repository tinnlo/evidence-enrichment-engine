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
    """Clear lru_cache before and after each test to prevent env bleed."""
    from evidence_enrichment.observability.langfuse import get_langfuse_client

    get_langfuse_client.cache_clear()
    yield
    get_langfuse_client.cache_clear()


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
    # Simulate langfuse not installed
    monkeypatch.setitem(sys.modules, "langfuse", None)  # type: ignore[call-overload]
    from evidence_enrichment.observability.langfuse import get_langfuse_client

    get_langfuse_client.cache_clear()
    assert get_langfuse_client() is None


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
