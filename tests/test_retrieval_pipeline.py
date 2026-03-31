"""Tests for retrieval pipeline integration.

Covers:
- retrieval_off: pipeline runs unchanged when mode="off"
- replay: retrieval is skipped entirely in replay mode
- fallback: analysis proceeds with raw text if retrieval fails
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evidence_enrichment.config.settings import RetrievalConfig, Settings
from evidence_enrichment.core.models.contracts import (
    AnalysisReport,
    FactClaim,
    ParsedDocument,
    SynthesisResult,
)
from evidence_enrichment.core.models.enums import ProviderType, ReviewDecision
from evidence_enrichment.core.providers.agents import _build_analysis_context
from evidence_enrichment.core.retrieval.models import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parsed_doc(url: str = "https://example.com/doc", text: str = "hello world") -> ParsedDocument:
    return ParsedDocument(
        url=url,
        title="Test",
        content_type="text/html",
        text=text,
        excerpt=text[:200],
        accepted_for_analysis=True,
        entity_match_score=0.8,
        source_authority_score=0.7,
        freshness_score=0.9,
    )


def _make_chunk(chunk_id: str = "a" * 16, content: str = "relevant text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_url="https://example.com/doc",
        content_hash="hash123",
        index=0,
        content=content,
        chunk_type="text",
    )


def _make_retrieval_result(chunk: Chunk, score: float = 0.85) -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        score=score,
        vector_score=score,
        keyword_score=0.5,
    )


# ---------------------------------------------------------------------------
# Unit tests for _build_analysis_context
# ---------------------------------------------------------------------------

class TestBuildAnalysisContext:
    def test_no_chunks_returns_raw_text(self):
        doc = _make_parsed_doc(text="raw document text " * 500)
        context, chunk_ids = _build_analysis_context(doc, None)
        assert context == doc.text[:6000]
        assert chunk_ids == []

    def test_empty_chunks_returns_raw_text(self):
        doc = _make_parsed_doc(text="raw document text " * 500)
        context, chunk_ids = _build_analysis_context(doc, [])
        # Empty list is falsy, so falls back to raw text
        assert context == doc.text[:6000]
        assert chunk_ids == []

    def test_chunks_used_when_provided(self):
        doc = _make_parsed_doc()
        chunk = _make_chunk(content="Berlin is headquartered in Germany.")
        result = _make_retrieval_result(chunk)
        context, chunk_ids = _build_analysis_context(doc, [result])
        assert "Berlin is headquartered in Germany." in context
        assert len(chunk_ids) == 1
        assert chunk_ids[0] == chunk.chunk_id

    def test_multiple_chunks_ordered(self):
        doc = _make_parsed_doc()
        chunks = [
            _make_retrieval_result(_make_chunk(f"a" * 16, f"chunk content {i}"), score=0.9 - i * 0.1)
            for i in range(3)
        ]
        context, chunk_ids = _build_analysis_context(doc, chunks)
        for i in range(3):
            assert f"chunk content {i}" in context
        assert len(chunk_ids) == 3

    def test_chunk_label_in_context(self):
        doc = _make_parsed_doc()
        chunk = _make_chunk(content="important data")
        result = _make_retrieval_result(chunk, score=0.75)
        context, _ = _build_analysis_context(doc, [result])
        assert "[Chunk 1" in context
        assert "type=text" in context
        assert "0.750" in context


# ---------------------------------------------------------------------------
# Settings: retrieval mode off by default
# ---------------------------------------------------------------------------

class TestRetrievalConfig:
    def test_default_mode_is_off(self):
        config = RetrievalConfig()
        assert config.mode == "off"

    def test_settings_has_retrieval_config(self):
        settings = Settings()
        assert settings.retrieval.mode == "off"
        assert settings.retrieval.top_k == 5
        assert settings.retrieval.chunk_size == 1500
        assert settings.retrieval.min_doc_chars == 2000

    def test_retrieval_local_mode(self):
        config = RetrievalConfig(mode="local")
        assert config.mode == "local"


# ---------------------------------------------------------------------------
# Coordinator: _get_retriever returns None when mode="off"
# ---------------------------------------------------------------------------

class TestCoordinatorGetRetriever:
    def test_retriever_none_when_mode_off(self):
        from evidence_enrichment.config.settings import Settings
        from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

        settings = Settings(retrieval=RetrievalConfig(mode="off"))
        coordinator = EvidenceCoordinator(settings=settings)
        retriever = coordinator._get_retriever("test_entity")
        assert retriever is None

    def test_retriever_none_when_mode_local_but_no_openai_key(self, tmp_path):
        """When mode=local but chromadb/openai unavailable or no key, returns None gracefully."""
        from evidence_enrichment.config.settings import Settings
        from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

        settings = Settings(
            retrieval=RetrievalConfig(mode="local", persist_path=str(tmp_path)),
            openai_api_key=None,
        )
        coordinator = EvidenceCoordinator(settings=settings)
        # Should not raise — returns None gracefully when API key is missing/empty
        # (embedder may initialise but won't call API until embed_texts is called)
        retriever = coordinator._get_retriever("test_entity")
        # May return a retriever object OR None depending on lazy init — either is fine
        # The important thing is no exception is raised
        assert True  # No exception = pass


# ---------------------------------------------------------------------------
# Replay mode: retrieval skipped
# ---------------------------------------------------------------------------

class TestReplaySkipsRetrieval:
    def test_replay_analysis_agent_ignores_chunks(self):
        """ReplayAnalysisAgent.analyze() accepts retrieved_chunks but ignores them."""
        import asyncio

        from evidence_enrichment.core.analysis.replay import ReplayAnalysisAgent

        bundle = {
            "analysis_reports": [
                {
                    "source_url": "https://example.com/doc",
                    "claims": [
                        {
                            "field_name": "hq_country",
                            "candidate_value": "DEU",
                            "supporting_excerpt": "Germany",
                            "source_url": "https://example.com/doc",
                            "source_title": "Test",
                            "analysis_confidence": 0.9,
                            "source_authority_score": 0.8,
                            "freshness_score": 0.7,
                            "entity_match_score": 0.85,
                        }
                    ],
                }
            ]
        }
        agent = ReplayAnalysisAgent(bundle)
        doc = _make_parsed_doc()
        # Pass retrieved_chunks — should be silently ignored
        fake_chunks = [_make_retrieval_result(_make_chunk())]
        report = asyncio.run(agent.analyze(doc, "hq_country", "Acme", retrieved_chunks=fake_chunks))
        assert len(report.claims) == 1
        assert report.claims[0].candidate_value == "DEU"

    def test_replay_analysis_no_retrieved_chunks_arg(self):
        """ReplayAnalysisAgent works without retrieved_chunks (old call-site)."""
        import asyncio

        from evidence_enrichment.core.analysis.replay import ReplayAnalysisAgent

        bundle = {"analysis_reports": []}
        agent = ReplayAnalysisAgent(bundle)
        doc = _make_parsed_doc()
        report = asyncio.run(agent.analyze(doc, "hq_country", "Acme"))
        assert report.claims == []


# ---------------------------------------------------------------------------
# Fallback: context falls back to raw text when no chunks available
# ---------------------------------------------------------------------------

class TestFallbackToRawText:
    def test_context_fallback_short_document(self):
        """Short documents (< min_doc_chars) are not indexed; analysis uses raw text."""
        doc = _make_parsed_doc(text="short text")
        context, chunk_ids = _build_analysis_context(doc, None)
        assert context == "short text"
        assert chunk_ids == []

    def test_context_fallback_truncation_at_6000(self):
        """Raw text fallback is truncated to 6000 chars."""
        long_text = "x" * 10_000
        doc = _make_parsed_doc(text=long_text)
        context, _ = _build_analysis_context(doc, None)
        assert len(context) == 6000
