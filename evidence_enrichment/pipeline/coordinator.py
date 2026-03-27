"""Stage-based coordinator."""

from __future__ import annotations

from pathlib import Path

from evidence_enrichment.config.settings import Settings, get_settings
from evidence_enrichment.core.analysis.replay import ReplayAnalysisAgent
from evidence_enrichment.core.enrichers.base import BaseEnricher
from evidence_enrichment.core.fetch.fetcher import DocumentFetcher
from evidence_enrichment.core.models.contracts import (
    EnrichmentSource,
    FactClaim,
    ParsedDocument,
    PipelineRunResult,
    RetrievedDocument,
    SearchResult,
)
from evidence_enrichment.core.models.enums import ProviderType, SourceType
from evidence_enrichment.core.parse.parser import TextParser
from evidence_enrichment.core.providers.agents import (
    AnthropicAnalysisAgent,
    AnthropicSynthesisAgent,
    OpenAIAnalysisAgent,
    OpenAISynthesisAgent,
)
from evidence_enrichment.core.providers.search import SerperSearchProvider, TavilySearchProvider
from evidence_enrichment.core.quality.gates import compute_overall_confidence, gate_result
from evidence_enrichment.core.search.query_planner import score_search_result
from evidence_enrichment.core.synthesis.replay import ReplaySynthesisAgent
from evidence_enrichment.pipeline.replay import load_replay_bundle


class EvidenceCoordinator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.fetcher = DocumentFetcher()
        self.parser = TextParser()

    async def run(
        self,
        entity: dict,
        enricher: BaseEnricher,
        *,
        mode: str | None = None,
        replay_bundle: str | None = None,
    ) -> PipelineRunResult:
        effective_mode = mode or self.settings.default_mode
        search_plan = enricher.build_query_plan(entity)
        replay_path = self._resolve_replay_path(replay_bundle, entity, enricher)
        if effective_mode == "replay":
            bundle = load_replay_bundle(replay_path)
            return await self._run_with_replay(entity, enricher, search_plan, bundle, effective_mode)
        if effective_mode == "auto" and replay_path.exists():
            try:
                return await self._run_live(entity, enricher, search_plan, effective_mode)
            except Exception:
                bundle = load_replay_bundle(replay_path)
                return await self._run_with_replay(entity, enricher, search_plan, bundle, effective_mode)
        return await self._run_live(entity, enricher, search_plan, effective_mode)

    def _resolve_replay_path(self, replay_bundle: str | None, entity: dict, enricher: BaseEnricher) -> Path:
        if replay_bundle:
            return Path(replay_bundle)
        return self.settings.replay_path / f"{enricher.replay_slug(entity)}.json"

    async def _run_live(self, entity: dict, enricher: BaseEnricher, search_plan, mode: str) -> PipelineRunResult:
        search_results = await self._search_live(search_plan)
        parsed_documents = await self._fetch_and_parse(entity, search_results)
        analysis_agent = self._analysis_agent()
        synthesis_agent = self._synthesis_agent()
        reports = []
        claims: list[FactClaim] = []
        for document in parsed_documents:
            if not document.accepted_for_analysis:
                continue
            report = await analysis_agent.analyze(document, enricher.field_name, search_plan.metadata.get("company_name", ""))
            reports.append(report)
            claims.extend(report.claims)
        synthesis = await synthesis_agent.synthesize(claims, enricher.field_name, search_plan.metadata.get("company_name", ""))
        synthesis = enricher.validate_synthesis(synthesis)
        return self._build_result(entity, search_plan, mode, search_results, parsed_documents, reports, claims, synthesis)

    async def _run_with_replay(self, entity: dict, enricher: BaseEnricher, search_plan, bundle: dict, mode: str) -> PipelineRunResult:
        search_results = [SearchResult(**row) for row in bundle.get("search_results", [])]
        parsed_documents = [ParsedDocument(**row) for row in bundle.get("parsed_documents", [])]
        analysis_agent = ReplayAnalysisAgent(bundle)
        synthesis_agent = ReplaySynthesisAgent(bundle)
        reports = []
        claims: list[FactClaim] = []
        for document in parsed_documents:
            if not document.accepted_for_analysis:
                continue
            report = await analysis_agent.analyze(document, enricher.field_name, search_plan.metadata.get("company_name", ""))
            reports.append(report)
            claims.extend(report.claims)
        synthesis = await synthesis_agent.synthesize(claims, enricher.field_name, search_plan.metadata.get("company_name", ""))
        synthesis = enricher.validate_synthesis(synthesis)
        return self._build_result(entity, search_plan, mode, search_results, parsed_documents, reports, claims, synthesis)

    async def _search_live(self, search_plan) -> list[SearchResult]:
        providers = []
        if self.settings.serper_api_key:
            providers.append(SerperSearchProvider(self.settings.serper_api_key))
        if self.settings.tavily_api_key:
            providers.append(TavilySearchProvider(self.settings.tavily_api_key))
        if not providers:
            raise RuntimeError("No live search provider configured.")
        last_error: Exception | None = None
        for provider in providers:
            try:
                results = await provider.search(search_plan)
                company_name = str(search_plan.metadata.get("company_name") or "")
                ranked = sorted(
                    results,
                    key=lambda row: score_search_result(company_name, row.url, row.title, row.snippet)[0],
                    reverse=True,
                )
                return ranked[:5]
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"All search providers failed: {last_error}")

    async def _fetch_and_parse(self, entity: dict, search_results: list[SearchResult]) -> list[ParsedDocument]:
        company_name = str(entity.get("name") or entity.get("company_name") or "")
        parsed_documents: list[ParsedDocument] = []
        for result in search_results[:3]:
            document: RetrievedDocument = await self.fetcher.fetch(result)
            parsed_documents.append(self.parser.parse(document, company_name))
        return parsed_documents

    def _analysis_agent(self):
        if self.settings.openai_api_key:
            return OpenAIAnalysisAgent(self.settings.openai_api_key, self.settings.openai_model)
        if self.settings.anthropic_api_key:
            return AnthropicAnalysisAgent(self.settings.anthropic_api_key, self.settings.anthropic_model)
        raise RuntimeError("No live analysis agent configured.")

    def _synthesis_agent(self):
        if self.settings.openai_api_key:
            return OpenAISynthesisAgent(self.settings.openai_api_key, self.settings.openai_model)
        if self.settings.anthropic_api_key:
            return AnthropicSynthesisAgent(self.settings.anthropic_api_key, self.settings.anthropic_model)
        raise RuntimeError("No live synthesis agent configured.")

    def _build_result(self, entity: dict, search_plan, mode: str, search_results, parsed_documents, reports, claims, synthesis):
        overall_confidence = compute_overall_confidence(claims, synthesis)
        decision, gate_reason = gate_result(overall_confidence, claims, synthesis)
        sources = [
            EnrichmentSource(
                source_type=SourceType.SEARCH,
                provider=result.provider.value,
                source_url=result.url,
                title=result.title,
                snippet=result.snippet,
                confidence=1.0,
            )
            for result in search_results
        ]
        for report in reports:
            sources.append(
                EnrichmentSource(
                    source_type=SourceType.ANALYSIS,
                    provider=report.provider.value,
                    source_url=report.source_url,
                    confidence=max((claim.analysis_confidence for claim in report.claims), default=0.0),
                )
            )
        sources.append(
            EnrichmentSource(
                source_type=SourceType.SYNTHESIS,
                provider="replay" if mode == "replay" else "live_agent",
                confidence=synthesis.synthesis_confidence,
            )
        )
        return PipelineRunResult(
            entity_id=str(entity.get("entity_id") or entity.get("id") or search_plan.entity_id),
            field_name=search_plan.field_name,
            mode=mode,
            search_plan=search_plan,
            search_results=search_results,
            parsed_documents=parsed_documents,
            analysis_reports=reports,
            synthesis=synthesis,
            sources=sources,
            overall_confidence=overall_confidence,
            decision=decision,
            gate_reason=gate_reason,
            output_value=synthesis.value,
        )

