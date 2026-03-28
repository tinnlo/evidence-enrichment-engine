import os

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
