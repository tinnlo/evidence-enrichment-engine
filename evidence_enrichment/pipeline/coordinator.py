"""Stage-based coordinator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langsmith import traceable

from evidence_enrichment.config.settings import Settings, get_settings
from evidence_enrichment.core.analysis.replay import ReplayAnalysisAgent
from evidence_enrichment.context.resolver import ContextResolver
from evidence_enrichment.core.enrichers.base import BaseEnricher
from evidence_enrichment.core.evidence.assessor import EvidenceAssessor
from evidence_enrichment.core.fetch.fetcher import DocumentFetcher
from evidence_enrichment.core.models.contracts import (
    AnalysisReport,
    EnrichmentSource,
    FactClaim,
    ParsedDocument,
    PipelineRunResult,
    RetrievedDocument,
    SearchResult,
)
from evidence_enrichment.core.models.enums import SourceType
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
from evidence_enrichment.observability.langsmith import (
    summarize_analysis_stage,
    summarize_assessed_documents,
    summarize_claims,
    summarize_fetched_documents,
    summarize_parsed_documents,
    summarize_query_plan,
    summarize_review_gate,
    summarize_search_results,
    summarize_synthesis,
    summarize_synthesis_stage,
    trace_payload_inputs,
)
from evidence_enrichment.observability.tracer import LocalTracer
from evidence_enrichment.pipeline.replay import load_replay_bundle


class EvidenceCoordinator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.fetcher = DocumentFetcher()
        self.parser = TextParser()
        self.assessor = EvidenceAssessor()
        self.context_resolver = ContextResolver(self.settings.context_path / "context_manifest.yaml")

    async def run(
        self,
        entity: dict,
        enricher: BaseEnricher,
        *,
        mode: str | None = None,
        replay_bundle: str | None = None,
        artifact_label: str | None = None,
    ) -> PipelineRunResult:
        effective_mode = mode or self.settings.default_mode
        entity_id = str(entity.get("entity_id") or entity.get("id") or "unknown")
        tracer = LocalTracer(mode=effective_mode, entity_id=entity_id, field_name=enricher.field_name)
        resolved_context = self.context_resolver.resolve(entity_id=entity_id, field_name=enricher.field_name)
        with tracer.span("query_plan", provider="local_context", input_count=1) as span:
            search_plan = self._stage_query_plan(
                entity,
                enricher,
                trace_payload={
                    "mode": effective_mode,
                    "entity_id": entity_id,
                    "field_name": enricher.field_name,
                    "company_name": str(entity.get("name") or entity.get("company_name") or ""),
                    "context_entry_ids": sorted(
                        {
                            entry_id
                            for stage in resolved_context.stages.values()
                            for entry_id in stage.entry_ids
                        }
                    ),
                },
            )
            span["output_count"] = len([search_plan.primary_query, *search_plan.query_variants])
        replay_path = self._resolve_replay_path(replay_bundle, entity, enricher)
        if effective_mode == "replay":
            bundle = load_replay_bundle(replay_path)
            result = await self._run_with_replay(entity, enricher, search_plan, bundle, effective_mode, tracer)
        elif effective_mode == "auto" and replay_path.exists():
            try:
                result = await self._run_live(entity, enricher, search_plan, effective_mode, tracer)
            except Exception:
                bundle = load_replay_bundle(replay_path)
                result = await self._run_with_replay(entity, enricher, search_plan, bundle, effective_mode, tracer)
        else:
            result = await self._run_live(entity, enricher, search_plan, effective_mode, tracer)
        trace_artifacts = tracer.write(self.settings.trace_output_path)
        resolved_context_path = trace_artifacts.trace_dir / "resolved_context.json"
        resolved_context_path.write_text(resolved_context.model_dump_json(indent=2), encoding="utf-8")
        refs = trace_artifacts.as_refs()
        refs["resolved_context"] = str(resolved_context_path)
        result.resolved_context = resolved_context
        result.trace_id = tracer.trace_id
        result.artifact_refs = refs
        return result

    def _resolve_replay_path(self, replay_bundle: str | None, entity: dict, enricher: BaseEnricher) -> Path:
        if replay_bundle:
            return Path(replay_bundle)
        return self.settings.replay_path / f"{enricher.replay_slug(entity)}.json"

    @traceable(
        name="query_plan",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_query_plan,
    )
    def _stage_query_plan(
        self,
        entity: dict[str, Any],
        enricher: BaseEnricher,
        *,
        trace_payload: dict[str, Any],
    ):
        return enricher.build_query_plan(entity)

    @traceable(
        name="search",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_search_results,
    )
    async def _stage_search(self, search_plan, *, bundle: dict[str, Any] | None, trace_payload: dict[str, Any]) -> list[SearchResult]:
        if bundle is not None:
            return [SearchResult(**row) for row in bundle.get("search_results", [])]
        return await self._search_live(search_plan)

    @traceable(
        name="fetch",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_fetched_documents,
    )
    async def _stage_fetch(
        self,
        search_results: list[SearchResult],
        *,
        bundle: dict[str, Any] | None,
        trace_payload: dict[str, Any],
    ) -> list[RetrievedDocument]:
        if bundle is not None:
            parsed_documents = [ParsedDocument(**row) for row in bundle.get("parsed_documents", [])]
            return [
                RetrievedDocument(
                    url=document.url,
                    final_url=document.url,
                    title=document.title,
                    content_type=document.content_type,
                    body=document.text,
                    provider="replay",
                    fetch_success=True,
                )
                for document in parsed_documents
            ]
        return await self._fetch_documents(search_results)

    @traceable(
        name="parse",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_parsed_documents,
    )
    def _stage_parse(
        self,
        fetched_documents: list[RetrievedDocument],
        *,
        bundle: dict[str, Any] | None,
        trace_payload: dict[str, Any],
    ) -> list[ParsedDocument]:
        if bundle is not None:
            return [ParsedDocument(**row) for row in bundle.get("parsed_documents", [])]
        return [self.parser.parse(document) for document in fetched_documents]

    @traceable(
        name="evidence_assessment",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_assessed_documents,
    )
    def _stage_evidence_assessment(
        self,
        parsed_documents: list[ParsedDocument],
        *,
        company_name: str,
        bundle: dict[str, Any] | None,
        trace_payload: dict[str, Any],
    ) -> list[ParsedDocument]:
        if bundle is not None:
            return parsed_documents
        return [self.assessor.assess(document, company_name) for document in parsed_documents]

    @traceable(
        name="analysis",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_analysis_stage,
    )
    async def _stage_analysis(
        self,
        documents: list[ParsedDocument],
        *,
        field_name: str,
        company_name: str,
        bundle: dict[str, Any] | None,
        trace_payload: dict[str, Any],
    ) -> tuple[list[AnalysisReport], list[FactClaim], str]:
        analysis_agent = ReplayAnalysisAgent(bundle) if bundle is not None else self._analysis_agent()
        reports: list[AnalysisReport] = []
        claims: list[FactClaim] = []
        for document in documents:
            if not document.accepted_for_analysis:
                continue
            report = await analysis_agent.analyze(document, field_name, company_name)
            reports.append(report)
            claims.extend(report.claims)
        return reports, claims, analysis_agent.provider_type.value

    @traceable(
        name="synthesis",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_synthesis_stage,
    )
    async def _stage_synthesis(
        self,
        claims: list[FactClaim],
        *,
        field_name: str,
        company_name: str,
        bundle: dict[str, Any] | None,
        trace_payload: dict[str, Any],
    ) -> tuple[Any, str]:
        synthesis_agent = ReplaySynthesisAgent(bundle) if bundle is not None else self._synthesis_agent()
        synthesis = await synthesis_agent.synthesize(claims, field_name, company_name)
        return synthesis, synthesis_agent.provider_type.value

    @traceable(
        name="review_gate",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_review_gate,
    )
    def _stage_review_gate(self, claims: list[FactClaim], synthesis, *, trace_payload: dict[str, Any]) -> dict[str, Any]:
        overall_confidence = compute_overall_confidence(claims, synthesis)
        decision, gate_reason = gate_result(overall_confidence, claims, synthesis)
        return {
            "overall_confidence": overall_confidence,
            "decision": decision.value,
            "decision_enum": decision,
            "gate_reason": gate_reason,
        }

    async def _run_live(self, entity: dict, enricher: BaseEnricher, search_plan, mode: str, tracer: LocalTracer) -> PipelineRunResult:
        query_count = len([search_plan.primary_query, *search_plan.query_variants])
        with tracer.span("search", provider="provider_chain", input_count=query_count) as span:
            search_results = await self._stage_search(
                search_plan,
                bundle=None,
                trace_payload={
                    "mode": mode,
                    "primary_query": search_plan.primary_query,
                    "query_variant_count": len(search_plan.query_variants),
                    "provider_order": self.settings.search.provider_order,
                },
            )
            span["output_count"] = len(search_results)

        with tracer.span("fetch", provider="httpx", input_count=len(search_results)) as span:
            fetched_documents = await self._stage_fetch(
                search_results,
                bundle=None,
                trace_payload={"mode": mode, "urls": [result.url for result in search_results]},
            )
            span["output_count"] = len(fetched_documents)

        with tracer.span("parse", provider="html_to_text", input_count=len(fetched_documents)) as span:
            parsed_documents = self._stage_parse(
                fetched_documents,
                bundle=None,
                trace_payload={"mode": mode, "urls": [document.url for document in fetched_documents]},
            )
            span["output_count"] = len(parsed_documents)

        company_name = str(entity.get("name") or entity.get("company_name") or "")
        with tracer.span("evidence_assessment", provider="local_rules", input_count=len(parsed_documents)) as span:
            assessed_documents = self._stage_evidence_assessment(
                parsed_documents,
                company_name=company_name,
                bundle=None,
                trace_payload={"mode": mode, "document_urls": [document.url for document in parsed_documents]},
            )
            span["output_count"] = sum(1 for document in assessed_documents if document.accepted_for_analysis)

        accepted_documents = [document for document in assessed_documents if document.accepted_for_analysis]
        analysis_provider = self._analysis_agent().provider_type.value
        with tracer.span("analysis", provider=analysis_provider, input_count=len(accepted_documents)) as span:
            reports, claims, _ = await self._stage_analysis(
                assessed_documents,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=None,
                trace_payload={
                    "mode": mode,
                    "accepted_document_urls": [document.url for document in accepted_documents],
                    **summarize_assessed_documents(assessed_documents),
                },
            )
            span["output_count"] = len(claims)

        synthesis_provider = self._synthesis_agent().provider_type.value
        with tracer.span("synthesis", provider=synthesis_provider, input_count=len(claims)) as span:
            synthesis, _ = await self._stage_synthesis(
                claims,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=None,
                trace_payload={"mode": mode, **summarize_claims(claims)},
            )
            span["output_count"] = 1 if synthesis.value else 0
        synthesis = enricher.validate_synthesis(synthesis)
        return self._build_result(entity, search_plan, mode, search_results, assessed_documents, reports, claims, synthesis, tracer)

    async def _run_with_replay(self, entity: dict, enricher: BaseEnricher, search_plan, bundle: dict, mode: str, tracer: LocalTracer) -> PipelineRunResult:
        query_count = len([search_plan.primary_query, *search_plan.query_variants])
        with tracer.span("search", provider="replay", input_count=query_count) as span:
            search_results = await self._stage_search(
                search_plan,
                bundle=bundle,
                trace_payload={
                    "mode": mode,
                    "primary_query": search_plan.primary_query,
                    "query_variant_count": len(search_plan.query_variants),
                    "provider_order": ["replay"],
                },
            )
            span["output_count"] = len(search_results)
        with tracer.span("fetch", provider="replay", input_count=len(search_results)) as span:
            fetched_documents = await self._stage_fetch(
                search_results,
                bundle=bundle,
                trace_payload={"mode": mode, "urls": [result.url for result in search_results]},
            )
            span["output_count"] = len(fetched_documents)
        with tracer.span("parse", provider="replay", input_count=len(bundle.get("parsed_documents", []))) as span:
            parsed_documents = self._stage_parse(
                fetched_documents,
                bundle=bundle,
                trace_payload={"mode": mode, "document_count": len(bundle.get("parsed_documents", []))},
            )
            span["output_count"] = len(parsed_documents)
        with tracer.span("evidence_assessment", provider="replay", input_count=len(parsed_documents)) as span:
            parsed_documents = self._stage_evidence_assessment(
                parsed_documents,
                company_name=str(entity.get("name") or entity.get("company_name") or ""),
                bundle=bundle,
                trace_payload={"mode": mode, "document_urls": [document.url for document in parsed_documents]},
            )
            span["output_count"] = sum(1 for document in parsed_documents if document.accepted_for_analysis)
        accepted_documents = [document for document in parsed_documents if document.accepted_for_analysis]
        with tracer.span("analysis", provider="replay", input_count=len(accepted_documents)) as span:
            reports, claims, _ = await self._stage_analysis(
                parsed_documents,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=bundle,
                trace_payload={
                    "mode": mode,
                    "accepted_document_urls": [document.url for document in accepted_documents],
                    **summarize_assessed_documents(parsed_documents),
                },
            )
            span["output_count"] = len(claims)
        with tracer.span("synthesis", provider="replay", input_count=len(claims)) as span:
            synthesis, _ = await self._stage_synthesis(
                claims,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=bundle,
                trace_payload={"mode": mode, **summarize_claims(claims)},
            )
            span["output_count"] = 1 if synthesis.value else 0
        synthesis = enricher.validate_synthesis(synthesis)
        return self._build_result(entity, search_plan, mode, search_results, parsed_documents, reports, claims, synthesis, tracer)

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

    async def _fetch_documents(self, search_results: list[SearchResult]) -> list[RetrievedDocument]:
        fetched_documents: list[RetrievedDocument] = []
        for result in search_results[:3]:
            fetched_documents.append(await self.fetcher.fetch(result))
        return fetched_documents

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

    def _build_result(self, entity: dict, search_plan, mode: str, search_results, parsed_documents, reports, claims, synthesis, tracer: LocalTracer):
        with tracer.span("review_gate", provider="local_rules", input_count=len(claims)) as span:
            gate_output = self._stage_review_gate(
                claims,
                synthesis,
                trace_payload={"mode": mode, **summarize_claims(claims), **summarize_synthesis(synthesis)},
            )
            overall_confidence = gate_output["overall_confidence"]
            decision = gate_output["decision_enum"]
            gate_reason = gate_output["gate_reason"]
            span["output_count"] = 1
            span["decision"] = decision.value
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
