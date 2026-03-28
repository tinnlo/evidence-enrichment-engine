from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evidence_enrichment.observability.langsmith import get_langsmith_client


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def disable_langsmith_tracing(monkeypatch):
    for key in [
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
    ]:
        monkeypatch.delenv(key, raising=False)
    get_langsmith_client.cache_clear()
    yield
    get_langsmith_client.cache_clear()
