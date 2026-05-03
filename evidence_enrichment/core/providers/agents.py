"""Live analysis and synthesis agents."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from evidence_enrichment.core.models.contracts import AnalysisReport, ConflictManifest, FactClaim, ParsedDocument, SynthesisResult
from evidence_enrichment.core.models.enums import ProviderType
from evidence_enrichment.core.providers.base import AnalysisAgent, SynthesisAgent
from evidence_enrichment.finops.models import LLMUsage, UsageSource
from evidence_enrichment.observability.redaction import should_redact as _should_redact
from evidence_enrichment.observability.router import langsmith_tracing_ready

if TYPE_CHECKING:
    from evidence_enrichment.core.retrieval.models import RetrievalResult


class ProviderParseError(RuntimeError):
    """Raised when a provider returns unparseable JSON.

    ``llm_usage`` carries token counts from the API response that preceded the
    parse failure.  The provider was paid for the call even though the response
    could not be decoded, so cost must still be attributed.  ``None`` when the
    error occurred before the API responded (should not happen in practice).
    """

    def __init__(self, message: str, llm_usage: "LLMUsage | None" = None) -> None:
        super().__init__(message)
        self.llm_usage = llm_usage


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ProviderParseError(f"No JSON object found in provider response: {text[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ProviderParseError(f"Malformed JSON in provider response: {exc}") from exc


def _wrap_openai_client(client):
    if _should_redact():
        return client
    if not langsmith_tracing_ready():
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception:
        return client


def _wrap_anthropic_client(client):
    if _should_redact():
        return client
    if not langsmith_tracing_ready():
        return client
    try:
        from langsmith.wrappers import wrap_anthropic

        return wrap_anthropic(client)
    except Exception:
        return client


def _build_analysis_context(
    document: ParsedDocument,
    retrieved_chunks: "list[RetrievalResult] | None",
) -> tuple[str, list[str]]:
    """Build the text context for the analysis prompt.

    Returns
    -------
    context_text:
        The text to include in the prompt.
    chunk_ids:
        IDs of the retrieved chunks used (empty when falling back to raw text).
    """
    if retrieved_chunks:
        parts = []
        chunk_ids = []
        for i, result in enumerate(retrieved_chunks, start=1):
            chunk_label = f"[Chunk {i} | type={result.chunk.chunk_type} | score={result.score:.3f}]"
            parts.append(f"{chunk_label}\n{result.chunk.content}")
            chunk_ids.append(result.chunk.chunk_id)
        context_text = "\n\n".join(parts)
        return context_text, chunk_ids
    # Fallback: first 6000 chars of raw text
    return document.text[:6000], []


class OpenAIAnalysisAgent(AnalysisAgent):
    provider_type = ProviderType.OPENAI

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def analyze(
        self,
        document: ParsedDocument,
        field_name: str,
        company_name: str,
        retrieved_chunks: "list[RetrievalResult] | None" = None,
    ) -> AnalysisReport:
        from openai import AsyncOpenAI

        context_text, chunk_ids = _build_analysis_context(document, retrieved_chunks)
        retrieval_note = (
            "The following are the most relevant retrieved chunks from this document.\n"
            if retrieved_chunks
            else ""
        )
        prompt = (
            "Extract structured evidence for the company's headquarters country.\n"
            "Return JSON with keys: reasoning, claims.\n"
            "Each claim must include candidate_value, supporting_excerpt, analysis_confidence.\n"
            "Use ISO3 country codes when possible.\n\n"
            f"Company: {company_name}\n"
            f"URL: {document.url}\n"
            f"Title: {document.title}\n"
            f"{retrieval_note}"
            f"Text:\n{context_text}"
        )
        client = _wrap_openai_client(AsyncOpenAI(api_key=self.api_key))
        response = await client.responses.create(model=self.model, input=prompt)
        # Capture usage BEFORE parsing so parse failures still carry cost.
        if response.usage is not None:
            llm_usage = LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                usage_source=UsageSource.PROVIDER_REPORTED,
            )
        else:
            from evidence_enrichment.finops.estimation import estimate_tokens
            llm_usage = LLMUsage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(response.output_text),
                usage_source=UsageSource.ESTIMATED,
            )
        try:
            parsed = _extract_json(response.output_text)
            claims = [
                FactClaim(
                    field_name=field_name,
                    candidate_value=str(row.get("candidate_value") or ""),
                    supporting_excerpt=str(row.get("supporting_excerpt") or "")[:400],
                    source_url=document.url,
                    source_title=document.title,
                    analysis_confidence=float(row.get("analysis_confidence") or 0.5),
                    source_authority_score=document.source_authority_score,
                    freshness_score=document.freshness_score,
                    entity_match_score=document.entity_match_score,
                    supporting_chunk_ids=chunk_ids,
                )
                for row in parsed.get("claims", [])
                if row.get("candidate_value")
            ]
            return AnalysisReport(
                source_url=document.url,
                provider=self.provider_type,
                claims=claims,
                reasoning=str(parsed.get("reasoning") or ""),
                llm_usage=llm_usage,
            )
        except Exception as exc:
            raise ProviderParseError(str(exc), llm_usage=llm_usage) from exc


class AnthropicAnalysisAgent(AnalysisAgent):
    provider_type = ProviderType.ANTHROPIC

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def analyze(
        self,
        document: ParsedDocument,
        field_name: str,
        company_name: str,
        retrieved_chunks: "list[RetrievalResult] | None" = None,
    ) -> AnalysisReport:
        from anthropic import AsyncAnthropic

        context_text, chunk_ids = _build_analysis_context(document, retrieved_chunks)
        retrieval_note = (
            "The following are the most relevant retrieved chunks from this document.\n"
            if retrieved_chunks
            else ""
        )
        prompt = (
            "Extract structured evidence for the company's headquarters country.\n"
            "Return JSON with keys: reasoning, claims.\n"
            "Each claim must include candidate_value, supporting_excerpt, analysis_confidence.\n"
            "Use ISO3 country codes when possible.\n\n"
            f"Company: {company_name}\n"
            f"URL: {document.url}\n"
            f"Title: {document.title}\n"
            f"{retrieval_note}"
            f"Text:\n{context_text}"
        )
        client = _wrap_anthropic_client(AsyncAnthropic(api_key=self.api_key))
        response = await client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "text", None))
        # Capture usage BEFORE parsing so parse failures still carry cost.
        llm_usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            usage_source=UsageSource.PROVIDER_REPORTED,
        )
        try:
            parsed = _extract_json(text)
            claims = [
                FactClaim(
                    field_name=field_name,
                    candidate_value=str(row.get("candidate_value") or ""),
                    supporting_excerpt=str(row.get("supporting_excerpt") or "")[:400],
                    source_url=document.url,
                    source_title=document.title,
                    analysis_confidence=float(row.get("analysis_confidence") or 0.5),
                    source_authority_score=document.source_authority_score,
                    freshness_score=document.freshness_score,
                    entity_match_score=document.entity_match_score,
                    supporting_chunk_ids=chunk_ids,
                )
                for row in parsed.get("claims", [])
                if row.get("candidate_value")
            ]
            return AnalysisReport(
                source_url=document.url,
                provider=self.provider_type,
                claims=claims,
                reasoning=str(parsed.get("reasoning") or ""),
                llm_usage=llm_usage,
            )
        except Exception as exc:
            raise ProviderParseError(str(exc), llm_usage=llm_usage) from exc


class OpenAISynthesisAgent(SynthesisAgent):
    provider_type = ProviderType.OPENAI

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def synthesize(self, claims: list[FactClaim], field_name: str, company_name: str) -> SynthesisResult:
        from openai import AsyncOpenAI

        payload = [
            {
                "candidate_value": claim.candidate_value,
                "source_url": claim.source_url,
                "supporting_excerpt": claim.supporting_excerpt,
                "analysis_confidence": claim.analysis_confidence,
            }
            for claim in claims
        ]
        prompt = (
            "Resolve the final headquarters country from the structured claims.\n"
            "Return JSON with keys: value, normalized_value, reasoning, synthesis_confidence.\n\n"
            f"Company: {company_name}\n"
            f"Claims: {json.dumps(payload)}"
        )
        client = _wrap_openai_client(AsyncOpenAI(api_key=self.api_key))
        response = await client.responses.create(model=self.model, input=prompt)
        # Capture usage BEFORE parsing so parse failures still carry cost.
        if response.usage is not None:
            llm_usage = LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                usage_source=UsageSource.PROVIDER_REPORTED,
            )
        else:
            from evidence_enrichment.finops.estimation import estimate_tokens
            llm_usage = LLMUsage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(response.output_text),
                usage_source=UsageSource.ESTIMATED,
            )
        try:
            parsed = _extract_json(response.output_text)
            return SynthesisResult(
                field_name=field_name,
                value=str(parsed.get("value") or "") or None,
                normalized_value=str(parsed.get("normalized_value") or "") or None,
                reasoning=str(parsed.get("reasoning") or ""),
                synthesis_confidence=float(parsed.get("synthesis_confidence") or 0.5),
                supporting_urls=[claim.source_url for claim in claims],
                conflicts=_build_conflicts(field_name, claims),
                llm_usage=llm_usage,
            )
        except Exception as exc:
            raise ProviderParseError(str(exc), llm_usage=llm_usage) from exc


class AnthropicSynthesisAgent(SynthesisAgent):
    provider_type = ProviderType.ANTHROPIC

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def synthesize(self, claims: list[FactClaim], field_name: str, company_name: str) -> SynthesisResult:
        from anthropic import AsyncAnthropic

        payload = [
            {
                "candidate_value": claim.candidate_value,
                "source_url": claim.source_url,
                "supporting_excerpt": claim.supporting_excerpt,
                "analysis_confidence": claim.analysis_confidence,
            }
            for claim in claims
        ]
        prompt = (
            "Resolve the final headquarters country from the structured claims.\n"
            "Return JSON with keys: value, normalized_value, reasoning, synthesis_confidence.\n\n"
            f"Company: {company_name}\n"
            f"Claims: {json.dumps(payload)}"
        )
        client = _wrap_anthropic_client(AsyncAnthropic(api_key=self.api_key))
        response = await client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "text", None))
        # Capture usage BEFORE parsing so parse failures still carry cost.
        llm_usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            usage_source=UsageSource.PROVIDER_REPORTED,
        )
        try:
            parsed = _extract_json(text)
            return SynthesisResult(
                field_name=field_name,
                value=str(parsed.get("value") or "") or None,
                normalized_value=str(parsed.get("normalized_value") or "") or None,
                reasoning=str(parsed.get("reasoning") or ""),
                synthesis_confidence=float(parsed.get("synthesis_confidence") or 0.5),
                supporting_urls=[claim.source_url for claim in claims],
                conflicts=_build_conflicts(field_name, claims),
                llm_usage=llm_usage,
            )
        except Exception as exc:
            raise ProviderParseError(str(exc), llm_usage=llm_usage) from exc


def _build_conflicts(field_name: str, claims: list[FactClaim]) -> list[ConflictManifest]:
    counts = Counter(claim.candidate_value for claim in claims if claim.candidate_value)
    if len(counts) <= 1:
        return []
    return [
        ConflictManifest(
            field_name=field_name,
            candidate_values=list(counts.keys()),
            source_urls=[claim.source_url for claim in claims],
            reason="multiple_candidate_values",
        )
    ]
