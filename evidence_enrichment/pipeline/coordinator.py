"""Stage-based coordinator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langsmith import traceable

from evidence_enrichment.config.settings import FieldThresholds, Settings, get_settings
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
    SynthesisResult,
)
from evidence_enrichment.core.models.enums import SourceType
from evidence_enrichment.core.parse.parser import TextParser
from evidence_enrichment.core.providers.agents import (
    AnthropicAnalysisAgent,
    AnthropicSynthesisAgent,
    OpenAIAnalysisAgent,
    OpenAISynthesisAgent,
    ProviderParseError,
)
from evidence_enrichment.core.providers.search import (
    SerperSearchProvider,
    TavilySearchProvider,
)
from evidence_enrichment.core.quality.gates import (
    compute_overall_confidence,
    gate_result,
)
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

if TYPE_CHECKING:
    from evidence_enrichment.core.retrieval.models import RetrievalResult
    from evidence_enrichment.core.retrieval.retriever import HybridRetriever


class EvidenceCoordinator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.fetcher = DocumentFetcher()
        self.parser = TextParser()
        self.assessor = EvidenceAssessor()
        self.context_resolver = ContextResolver(
            self.settings.context_path / "context_manifest.yaml"
        )
        self._retriever: "HybridRetriever | None" = None
        self._retrieval_degraded: bool = False

    def _get_retriever(self, entity_id: str) -> "HybridRetriever | None":
        """Lazy-initialise the HybridRetriever when retrieval mode is active."""
        rc = self.settings.retrieval
        if rc.mode != "local":
            return None
        if self._retriever is not None and self._retriever.entity_id == entity_id:
            return self._retriever
        try:
            from evidence_enrichment.core.retrieval.chunker import TableAwareChunker
            from evidence_enrichment.core.retrieval.embedder import OpenAIEmbedder
            from evidence_enrichment.core.retrieval.retriever import HybridRetriever
            from evidence_enrichment.core.retrieval.store import ChromaVectorStore

            embedder = OpenAIEmbedder(
                model=rc.embedding_model,
                api_key=self.settings.openai_api_key or "",
            )
            store = ChromaVectorStore(
                persist_path=rc.persist_path,
                embedding_model=rc.embedding_model,
            )
            chunker = TableAwareChunker(
                chunk_size=rc.chunk_size,
                overlap=rc.overlap,
                max_table_size=rc.max_table_size,
            )
            self._retriever = HybridRetriever(
                entity_id=entity_id,
                store=store,
                embedder=embedder,
                chunker=chunker,
                top_k=rc.top_k,
                weights=rc.weights,
            )
        except Exception as exc:
            logging.warning(
                "Failed to initialise retriever for entity %s: %s", entity_id, exc
            )
            self._retriever = None
            if rc.mode == "local":
                self._retrieval_degraded = True
        return self._retriever

    async def run(
        self,
        entity: dict,
        enricher: BaseEnricher,
        *,
        mode: str | None = None,
        replay_bundle: str | None = None,
        artifact_label: str | None = None,
    ) -> PipelineRunResult:
        self._retrieval_degraded = False
        effective_mode = mode or self.settings.default_mode
        entity_id = str(entity.get("entity_id") or entity.get("id") or "unknown")
        tracer = LocalTracer(
            mode=effective_mode, entity_id=entity_id, field_name=enricher.field_name
        )
        resolved_context = self.context_resolver.resolve(
            entity_id=entity_id, field_name=enricher.field_name
        )
        with tracer.span("query_plan", provider="local_context", input_count=1) as span:
            search_plan = self._stage_query_plan(
                entity,
                enricher,
                trace_payload={
                    "mode": effective_mode,
                    "entity_id": entity_id,
                    "field_name": enricher.field_name,
                    "company_name": str(
                        entity.get("name") or entity.get("company_name") or ""
                    ),
                    "context_entry_ids": sorted(
                        {
                            entry_id
                            for stage in resolved_context.stages.values()
                            for entry_id in stage.entry_ids
                        }
                    ),
                },
            )
            span["output_count"] = len(
                [search_plan.primary_query, *search_plan.query_variants]
            )
        replay_path = self._resolve_replay_path(replay_bundle, entity, enricher)
        if effective_mode == "replay":
            bundle = load_replay_bundle(replay_path)
            result = await self._run_with_replay(
                entity, enricher, search_plan, bundle, effective_mode, tracer
            )
        elif effective_mode == "auto" and replay_path.exists():
            try:
                self._preflight_live_readiness()
                result = await self._run_live(
                    entity, enricher, search_plan, effective_mode, tracer
                )
            except (RuntimeError, ImportError) as exc:
                logging.warning(
                    "auto mode: live providers unavailable (%s); falling back to replay bundle %s",
                    exc,
                    replay_path,
                )
                bundle = load_replay_bundle(replay_path)
                result = await self._run_with_replay(
                    entity, enricher, search_plan, bundle, "replay", tracer
                )
                result.fallback_from_live = True
        else:
            result = await self._run_live(
                entity, enricher, search_plan, effective_mode, tracer
            )
        trace_artifacts = tracer.write(self.settings.trace_output_path)
        resolved_context_path = trace_artifacts.trace_dir / "resolved_context.json"
        resolved_context_path.write_text(
            resolved_context.model_dump_json(indent=2), encoding="utf-8"
        )
        refs = trace_artifacts.as_refs()
        refs["resolved_context"] = str(resolved_context_path)
        result.resolved_context = resolved_context
        result.trace_id = tracer.trace_id
        result.artifact_refs = refs
        return result

    def _resolve_replay_path(
        self, replay_bundle: str | None, entity: dict, enricher: BaseEnricher
    ) -> Path:
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
    async def _stage_search(
        self,
        search_plan,
        *,
        bundle: dict[str, Any] | None,
        trace_payload: dict[str, Any],
    ) -> list[SearchResult]:
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
            parsed_documents = [
                ParsedDocument(**row) for row in bundle.get("parsed_documents", [])
            ]
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
        use_structured: bool = False,
        trace_payload: dict[str, Any],
    ) -> list[ParsedDocument]:
        if bundle is not None:
            return [ParsedDocument(**row) for row in bundle.get("parsed_documents", [])]
        if use_structured:
            return [
                self.parser.parse_with_structure(document)
                for document in fetched_documents
            ]
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
        return [
            self.assessor.assess(document, company_name)
            for document in parsed_documents
        ]

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
        retrieved_chunks_map: "dict[str, list[RetrievalResult]] | None" = None,
        trace_payload: dict[str, Any],
    ) -> tuple[list[AnalysisReport], list[FactClaim], str]:
        analysis_agent = (
            ReplayAnalysisAgent(bundle)
            if bundle is not None
            else self._analysis_agent()
        )
        reports: list[AnalysisReport] = []
        claims: list[FactClaim] = []
        accepted_count = 0
        failed_count = 0
        for document in documents:
            if not document.accepted_for_analysis:
                continue
            accepted_count += 1
            retrieved_chunks = (retrieved_chunks_map or {}).get(document.url)
            try:
                report = await analysis_agent.analyze(
                    document, field_name, company_name, retrieved_chunks
                )
            except Exception as exc:
                logging.error(
                    "analysis failed for %s (skipping document): %s",
                    document.url,
                    exc,
                )
                failed_count += 1
                reports.append(
                    AnalysisReport(
                        source_url=document.url,
                        provider=analysis_agent.provider_type,
                        claims=[],
                        reasoning="[analysis error — skipped]",
                    )
                )
                continue
            reports.append(report)
            claims.extend(report.claims)
        if accepted_count > 0 and failed_count == accepted_count:
            raise RuntimeError(
                f"All {accepted_count} document analysis call(s) failed; "
                "cannot produce claims."
            )
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
        synthesis_agent = (
            ReplaySynthesisAgent(bundle)
            if bundle is not None
            else self._synthesis_agent()
        )
        try:
            synthesis = await synthesis_agent.synthesize(
                claims, field_name, company_name
            )
        except ProviderParseError as exc:
            logging.error(
                "Synthesis provider returned unparseable response for field %r: %s",
                field_name,
                exc,
            )
            synthesis = SynthesisResult(
                field_name=field_name,
                value=None,
                reasoning="Synthesis failed: provider returned unparseable response.",
                synthesis_confidence=0.0,
            )
        return synthesis, synthesis_agent.provider_type.value

    @traceable(
        name="review_gate",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_review_gate,
    )
    def _stage_review_gate(
        self,
        claims: list[FactClaim],
        synthesis,
        *,
        field_thresholds: FieldThresholds | None = None,
        trace_payload: dict[str, Any],
    ) -> dict[str, Any]:
        overall_confidence = compute_overall_confidence(claims, synthesis)
        thresholds = field_thresholds or FieldThresholds()
        decision, gate_reason = gate_result(
            overall_confidence,
            claims,
            synthesis,
            auto_approve_min=thresholds.auto_approve_min_confidence,
            review_min=thresholds.review_min_confidence,
        )
        return {
            "overall_confidence": overall_confidence,
            "decision": decision.value,
            "decision_enum": decision,
            "gate_reason": gate_reason,
        }

    async def _run_live(
        self,
        entity: dict,
        enricher: BaseEnricher,
        search_plan,
        mode: str,
        tracer: LocalTracer,
    ) -> PipelineRunResult:
        entity_id = str(entity.get("entity_id") or entity.get("id") or "unknown")
        retriever = self._get_retriever(entity_id)
        use_structured = retriever is not None
        rc = self.settings.retrieval

        query_count = len([search_plan.primary_query, *search_plan.query_variants])
        with tracer.span(
            "search", provider="provider_chain", input_count=query_count
        ) as span:
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

        with tracer.span(
            "fetch", provider="httpx", input_count=len(search_results)
        ) as span:
            fetched_documents = await self._stage_fetch(
                search_results,
                bundle=None,
                trace_payload={
                    "mode": mode,
                    "urls": [result.url for result in search_results],
                },
            )
            span["output_count"] = len(fetched_documents)

        with tracer.span(
            "parse", provider="html_to_text", input_count=len(fetched_documents)
        ) as span:
            parsed_documents = self._stage_parse(
                fetched_documents,
                bundle=None,
                use_structured=use_structured,
                trace_payload={
                    "mode": mode,
                    "urls": [document.url for document in fetched_documents],
                },
            )
            span["output_count"] = len(parsed_documents)

        company_name = str(entity.get("name") or entity.get("company_name") or "")
        with tracer.span(
            "evidence_assessment",
            provider="local_rules",
            input_count=len(parsed_documents),
        ) as span:
            assessed_documents = self._stage_evidence_assessment(
                parsed_documents,
                company_name=company_name,
                bundle=None,
                trace_payload={
                    "mode": mode,
                    "document_urls": [document.url for document in parsed_documents],
                },
            )
            span["output_count"] = sum(
                1 for document in assessed_documents if document.accepted_for_analysis
            )

        # Retrieval indexing stage: index accepted documents above min_doc_chars
        retrieval_chunk_count = 0
        retrieved_chunks_map: dict[str, list[RetrievalResult]] = {}
        if retriever is not None:
            accepted_for_indexing = [
                d
                for d in assessed_documents
                if d.accepted_for_analysis
                and len(d.full_text or d.text) >= rc.min_doc_chars
            ]
            with tracer.span(
                "retrieval_indexing",
                provider="chroma",
                input_count=len(accepted_for_indexing),
            ) as span:
                total_chunks = 0
                for document in accepted_for_indexing:
                    try:
                        chunks = retriever.index_document(document)
                        total_chunks += len(chunks)
                    except Exception as exc:
                        logging.warning(
                            "retrieval_indexing failed for %s: %s", document.url, exc
                        )
                        self._retrieval_degraded = True
                span["output_count"] = total_chunks
                retrieval_chunk_count = total_chunks

            # Retrieve for each accepted document before analysis
            with tracer.span(
                "retrieval_query",
                provider="chroma",
                input_count=len(accepted_for_indexing),
            ) as span:
                for document in accepted_for_indexing:
                    try:
                        hits = retriever.retrieve(
                            query=enricher.retrieval_query(entity),
                            document_url=document.url,
                        )
                        if hits:
                            retrieved_chunks_map[document.url] = hits
                    except Exception as exc:
                        logging.warning(
                            "retrieval_query failed for %s: %s", document.url, exc
                        )
                        self._retrieval_degraded = True
                span["output_count"] = sum(
                    len(v) for v in retrieved_chunks_map.values()
                )

        accepted_documents = [
            document
            for document in assessed_documents
            if document.accepted_for_analysis
        ]
        analysis_provider = self._analysis_agent().provider_type.value
        with tracer.span(
            "analysis", provider=analysis_provider, input_count=len(accepted_documents)
        ) as span:
            reports, claims, _ = await self._stage_analysis(
                assessed_documents,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=None,
                retrieved_chunks_map=retrieved_chunks_map or None,
                trace_payload={
                    "mode": mode,
                    "accepted_document_urls": [
                        document.url for document in accepted_documents
                    ],
                    **summarize_assessed_documents(assessed_documents),
                },
            )
            span["output_count"] = len(claims)

        synthesis_provider = self._synthesis_agent().provider_type.value
        with tracer.span(
            "synthesis", provider=synthesis_provider, input_count=len(claims)
        ) as span:
            synthesis, _ = await self._stage_synthesis(
                claims,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=None,
                trace_payload={"mode": mode, **summarize_claims(claims)},
            )
            span["output_count"] = 1 if synthesis.value else 0
        synthesis = enricher.validate_synthesis(synthesis)

        # Collect top retrieval scores for PipelineRunResult
        all_scores = [r.score for hits in retrieved_chunks_map.values() for r in hits]
        top_scores = sorted(all_scores, reverse=True)[: rc.top_k]

        result = self._build_result(
            entity,
            search_plan,
            mode,
            search_results,
            assessed_documents,
            reports,
            claims,
            synthesis,
            tracer,
        )
        result.retrieval_chunk_count = retrieval_chunk_count
        result.retrieval_top_scores = top_scores
        return result

    async def _run_with_replay(
        self,
        entity: dict,
        enricher: BaseEnricher,
        search_plan,
        bundle: dict,
        mode: str,
        tracer: LocalTracer,
    ) -> PipelineRunResult:
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
        with tracer.span(
            "fetch", provider="replay", input_count=len(search_results)
        ) as span:
            fetched_documents = await self._stage_fetch(
                search_results,
                bundle=bundle,
                trace_payload={
                    "mode": mode,
                    "urls": [result.url for result in search_results],
                },
            )
            span["output_count"] = len(fetched_documents)
        with tracer.span(
            "parse",
            provider="replay",
            input_count=len(bundle.get("parsed_documents", [])),
        ) as span:
            parsed_documents = self._stage_parse(
                fetched_documents,
                bundle=bundle,
                use_structured=False,
                trace_payload={
                    "mode": mode,
                    "document_count": len(bundle.get("parsed_documents", [])),
                },
            )
            span["output_count"] = len(parsed_documents)
        with tracer.span(
            "evidence_assessment", provider="replay", input_count=len(parsed_documents)
        ) as span:
            parsed_documents = self._stage_evidence_assessment(
                parsed_documents,
                company_name=str(
                    entity.get("name") or entity.get("company_name") or ""
                ),
                bundle=bundle,
                trace_payload={
                    "mode": mode,
                    "document_urls": [document.url for document in parsed_documents],
                },
            )
            span["output_count"] = sum(
                1 for document in parsed_documents if document.accepted_for_analysis
            )
        accepted_documents = [
            document for document in parsed_documents if document.accepted_for_analysis
        ]
        with tracer.span(
            "analysis", provider="replay", input_count=len(accepted_documents)
        ) as span:
            reports, claims, _ = await self._stage_analysis(
                parsed_documents,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=bundle,
                retrieved_chunks_map=None,
                trace_payload={
                    "mode": mode,
                    "accepted_document_urls": [
                        document.url for document in accepted_documents
                    ],
                    **summarize_assessed_documents(parsed_documents),
                },
            )
            span["output_count"] = len(claims)
        with tracer.span(
            "synthesis", provider="replay", input_count=len(claims)
        ) as span:
            synthesis, _ = await self._stage_synthesis(
                claims,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=bundle,
                trace_payload={"mode": mode, **summarize_claims(claims)},
            )
            span["output_count"] = 1 if synthesis.value else 0
        synthesis = enricher.validate_synthesis(synthesis)
        return self._build_result(
            entity,
            search_plan,
            mode,
            search_results,
            parsed_documents,
            reports,
            claims,
            synthesis,
            tracer,
        )

    def _preflight_live_readiness(self) -> None:
        """Raise RuntimeError if live providers are not configured.

        Called before any network calls in ``auto`` mode so that missing
        credentials are discovered before search/fetch/parse execute.
        """
        errors: list[str] = []
        search_order = self.settings.search.provider_order or ["serper", "tavily"]
        has_search = any(
            (name == "serper" and self.settings.serper_api_key)
            or (name == "tavily" and self.settings.tavily_api_key)
            for name in search_order
        )
        if not has_search:
            errors.append("search")
        analysis_order = self.settings.analysis.provider_order or [
            "openai",
            "anthropic",
        ]
        has_analysis = any(
            (name == "openai" and self.settings.openai_api_key)
            or (name == "anthropic" and self.settings.anthropic_api_key)
            for name in analysis_order
        )
        if not has_analysis:
            errors.append("analysis")
        synthesis_order = self.settings.synthesis.provider_order or [
            "openai",
            "anthropic",
        ]
        has_synthesis = any(
            (name == "openai" and self.settings.openai_api_key)
            or (name == "anthropic" and self.settings.anthropic_api_key)
            for name in synthesis_order
        )
        if not has_synthesis:
            errors.append("synthesis")
        if errors:
            raise RuntimeError(f"No live providers configured for: {', '.join(errors)}")

    async def _search_live(self, search_plan) -> list[SearchResult]:
        order = self.settings.search.provider_order or ["serper", "tavily"]
        available: dict[str, object] = {
            "serper": SerperSearchProvider(self.settings.serper_api_key)
            if self.settings.serper_api_key
            else None,
            "tavily": TavilySearchProvider(self.settings.tavily_api_key)
            if self.settings.tavily_api_key
            else None,
        }
        providers = [p for name in order if (p := available.get(name)) is not None]
        if not providers:
            raise RuntimeError("No live search provider configured.")
        all_queries = [search_plan.primary_query, *search_plan.query_variants]
        seen_urls: set[str] = set()
        merged: list[SearchResult] = []
        last_error: Exception | None = None
        for provider in providers:
            try:
                for query in all_queries:
                    variant_plan = search_plan.model_copy(
                        update={"primary_query": query, "query_variants": []}
                    )
                    results = await provider.search(variant_plan)
                    for result in results:
                        if result.url not in seen_urls:
                            seen_urls.add(result.url)
                            merged.append(result)
                if merged:
                    break
            except Exception as exc:
                last_error = exc
        if not merged:
            if last_error is not None:
                raise RuntimeError(f"All search providers failed: {last_error}")
            # All providers responded but found nothing — legitimate no-hit scenario;
            # let the pipeline continue to AUTO_REJECT at the review gate.
            return []
        company_name = str(search_plan.metadata.get("company_name") or "")
        ranked = sorted(
            merged,
            key=lambda row: score_search_result(
                company_name, row.url, row.title, row.snippet
            )[0],
            reverse=True,
        )
        return ranked[:5]

    async def _fetch_documents(
        self, search_results: list[SearchResult]
    ) -> list[RetrievedDocument]:
        fetched_documents: list[RetrievedDocument] = []
        for result in search_results[:3]:
            try:
                fetched_documents.append(await self.fetcher.fetch(result))
            except Exception as exc:
                logging.warning("fetch failed for %s: %s", result.url, exc)
        if search_results and not fetched_documents:
            raise RuntimeError("All fetch attempts failed")
        return fetched_documents

    def _analysis_agent(self):
        order = self.settings.analysis.provider_order or ["openai", "anthropic"]
        available = {
            "openai": OpenAIAnalysisAgent(
                self.settings.openai_api_key, self.settings.openai_model
            )
            if self.settings.openai_api_key
            else None,
            "anthropic": AnthropicAnalysisAgent(
                self.settings.anthropic_api_key, self.settings.anthropic_model
            )
            if self.settings.anthropic_api_key
            else None,
        }
        for name in order:
            agent = available.get(name)
            if agent is not None:
                return agent
        raise RuntimeError("No live analysis agent configured.")

    def _synthesis_agent(self):
        order = self.settings.synthesis.provider_order or ["openai", "anthropic"]
        available = {
            "openai": OpenAISynthesisAgent(
                self.settings.openai_api_key, self.settings.openai_model
            )
            if self.settings.openai_api_key
            else None,
            "anthropic": AnthropicSynthesisAgent(
                self.settings.anthropic_api_key, self.settings.anthropic_model
            )
            if self.settings.anthropic_api_key
            else None,
        }
        for name in order:
            agent = available.get(name)
            if agent is not None:
                return agent
        raise RuntimeError("No live synthesis agent configured.")

    def _build_result(
        self,
        entity: dict,
        search_plan,
        mode: str,
        search_results,
        parsed_documents,
        reports,
        claims,
        synthesis,
        tracer: LocalTracer,
    ):
        with tracer.span(
            "review_gate", provider="local_rules", input_count=len(claims)
        ) as span:
            gate_output = self._stage_review_gate(
                claims,
                synthesis,
                field_thresholds=self.settings.thresholds.get(search_plan.field_name),
                trace_payload={
                    "mode": mode,
                    **summarize_claims(claims),
                    **summarize_synthesis(synthesis),
                },
            )
            overall_confidence = gate_output["overall_confidence"]
            decision = gate_output["decision_enum"]
            gate_reason = gate_output["gate_reason"]
            span["output_count"] = 1
            span["decision"] = decision.value
            span["overall_confidence"] = overall_confidence
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
                    confidence=max(
                        (claim.analysis_confidence for claim in report.claims),
                        default=0.0,
                    ),
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
            entity_id=str(
                entity.get("entity_id") or entity.get("id") or search_plan.entity_id
            ),
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
            retrieval_degraded=self._retrieval_degraded,
        )
