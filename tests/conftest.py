from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def reset_observability_env(monkeypatch):
    """Clear all observability env vars and internal settings state before/after each test.

    Covers LangSmith, LangChain legacy aliases, Langfuse, and the backend
    selector so no test can inherit shell-supplied observability state.
    """
    for key in [
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
    ]:
        monkeypatch.delenv(key, raising=False)

    import evidence_enrichment.config.settings as settings_mod
    from evidence_enrichment.observability.runtime import (
        clear_default_runtime_observability_config,
        clear_evicted_observability_values,
        clear_runtime_observability_config,
    )

    settings_mod._AMBIENT_OBSERVABILITY_ENV.clear()
    settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.clear()
    settings_mod._SHELL_SET_OBSERVABILITY_KEYS.clear()
    settings_mod._EVICTION_PENDING_KEYS.clear()
    settings_mod._reset_settings_cache()
    clear_default_runtime_observability_config()
    clear_evicted_observability_values()
    clear_runtime_observability_config()
    yield
    settings_mod._AMBIENT_OBSERVABILITY_ENV.clear()
    settings_mod._LAST_MANAGED_OBSERVABILITY_ENV.clear()
    settings_mod._SHELL_SET_OBSERVABILITY_KEYS.clear()
    settings_mod._EVICTION_PENDING_KEYS.clear()
    settings_mod._reset_settings_cache()
    clear_default_runtime_observability_config()
    clear_evicted_observability_values()
    clear_runtime_observability_config()
