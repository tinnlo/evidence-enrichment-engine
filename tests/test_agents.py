"""Tests that live agents capture LLM token usage at call time.

These tests mock the provider client so no real API calls are made.
They verify that AnalysisReport.llm_usage and SynthesisResult.llm_usage
are populated with provider-reported (or estimated fallback) token counts.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evidence_enrichment.core.models.contracts import AnalysisReport, SynthesisResult
from evidence_enrichment.core.providers.agents import (
    AnthropicAnalysisAgent,
    AnthropicSynthesisAgent,
    OpenAIAnalysisAgent,
    OpenAISynthesisAgent,
)
from evidence_enrichment.finops.models import UsageSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parsed_document():
    from evidence_enrichment.core.models.contracts import ParsedDocument
    from evidence_enrichment.core.models.enums import DocumentType

    return ParsedDocument(
        url="https://example.com/about",
        title="About Us",
        content_type="text/html",
        text="Acme Corp is headquartered in Berlin, Germany.",
        excerpt="Acme Corp is headquartered in Berlin.",
        document_type=DocumentType.UNKNOWN,
        source_authority_score=0.8,
        freshness_score=0.9,
        entity_match_score=0.95,
    )


def _make_claims():
    from evidence_enrichment.core.models.contracts import FactClaim

    return [
        FactClaim(
            field_name="hq_country",
            candidate_value="DEU",
            supporting_excerpt="headquartered in Berlin",
            source_url="https://example.com/about",
            source_title="About Us",
            analysis_confidence=0.9,
            source_authority_score=0.8,
            freshness_score=0.9,
            entity_match_score=0.95,
            supporting_chunk_ids=[],
        )
    ]


def _openai_response_json():
    return json.dumps({
        "reasoning": "The document mentions Berlin headquarters.",
        "claims": [
            {
                "candidate_value": "DEU",
                "supporting_excerpt": "headquartered in Berlin",
                "analysis_confidence": 0.9,
            }
        ],
    })


def _synthesis_response_json():
    return json.dumps({
        "value": "DEU",
        "normalized_value": "Germany",
        "reasoning": "Consistent across all claims.",
        "synthesis_confidence": 0.92,
    })


# ---------------------------------------------------------------------------
# OpenAI Analysis Agent
# ---------------------------------------------------------------------------

class TestOpenAIAnalysisAgentUsage:
    @pytest.mark.asyncio
    async def test_captures_provider_reported_usage(self):
        """report.llm_usage reflects the token counts returned by the API."""
        agent = OpenAIAnalysisAgent(api_key="test-key", model="gpt-4.1-mini")
        doc = _make_parsed_document()

        mock_usage = MagicMock()
        mock_usage.input_tokens = 120
        mock_usage.output_tokens = 55

        mock_response = MagicMock()
        mock_response.output_text = _openai_response_json()
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            report = await agent.analyze(doc, "hq_country", "Acme Corp")

        assert isinstance(report, AnalysisReport)
        assert report.llm_usage is not None
        assert report.llm_usage.input_tokens == 120
        assert report.llm_usage.output_tokens == 55
        assert report.llm_usage.usage_source == UsageSource.PROVIDER_REPORTED

    @pytest.mark.asyncio
    async def test_falls_back_to_estimated_when_usage_none(self):
        """When response.usage is None, llm_usage uses ESTIMATED source with non-zero tokens."""
        agent = OpenAIAnalysisAgent(api_key="test-key", model="gpt-4.1-mini")
        doc = _make_parsed_document()

        mock_response = MagicMock()
        mock_response.output_text = _openai_response_json()
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            report = await agent.analyze(doc, "hq_country", "Acme Corp")

        assert report.llm_usage is not None
        assert report.llm_usage.usage_source == UsageSource.ESTIMATED
        assert report.llm_usage.input_tokens > 0
        assert report.llm_usage.output_tokens > 0


# ---------------------------------------------------------------------------
# Anthropic Analysis Agent
# ---------------------------------------------------------------------------

class TestAnthropicAnalysisAgentUsage:
    @pytest.mark.asyncio
    async def test_captures_provider_reported_usage(self):
        """Anthropic always returns usage; llm_usage should be PROVIDER_REPORTED."""
        agent = AnthropicAnalysisAgent(api_key="test-key", model="claude-3-5-haiku-20241022")
        doc = _make_parsed_document()

        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 80

        mock_content_block = MagicMock()
        mock_content_block.text = _openai_response_json()

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            report = await agent.analyze(doc, "hq_country", "Acme Corp")

        assert isinstance(report, AnalysisReport)
        assert report.llm_usage is not None
        assert report.llm_usage.input_tokens == 200
        assert report.llm_usage.output_tokens == 80
        assert report.llm_usage.usage_source == UsageSource.PROVIDER_REPORTED


# ---------------------------------------------------------------------------
# OpenAI Synthesis Agent
# ---------------------------------------------------------------------------

class TestOpenAISynthesisAgentUsage:
    @pytest.mark.asyncio
    async def test_captures_provider_reported_usage(self):
        """synthesis.llm_usage reflects the token counts returned by the API."""
        agent = OpenAISynthesisAgent(api_key="test-key", model="gpt-4.1-mini")
        claims = _make_claims()

        mock_usage = MagicMock()
        mock_usage.input_tokens = 300
        mock_usage.output_tokens = 60

        mock_response = MagicMock()
        mock_response.output_text = _synthesis_response_json()
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await agent.synthesize(claims, "hq_country", "Acme Corp")

        assert isinstance(result, SynthesisResult)
        assert result.llm_usage is not None
        assert result.llm_usage.input_tokens == 300
        assert result.llm_usage.output_tokens == 60
        assert result.llm_usage.usage_source == UsageSource.PROVIDER_REPORTED


# ---------------------------------------------------------------------------
# Anthropic Synthesis Agent
# ---------------------------------------------------------------------------

class TestAnthropicSynthesisAgentUsage:
    @pytest.mark.asyncio
    async def test_captures_provider_reported_usage(self):
        """Anthropic synthesis usage is always PROVIDER_REPORTED."""
        agent = AnthropicSynthesisAgent(api_key="test-key", model="claude-3-5-haiku-20241022")
        claims = _make_claims()

        mock_usage = MagicMock()
        mock_usage.input_tokens = 250
        mock_usage.output_tokens = 45

        mock_content_block = MagicMock()
        mock_content_block.text = _synthesis_response_json()

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            result = await agent.synthesize(claims, "hq_country", "Acme Corp")

        assert isinstance(result, SynthesisResult)
        assert result.llm_usage is not None
        assert result.llm_usage.input_tokens == 250
        assert result.llm_usage.output_tokens == 45
        assert result.llm_usage.usage_source == UsageSource.PROVIDER_REPORTED
