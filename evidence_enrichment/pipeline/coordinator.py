"""Stage-based coordinator."""

from __future__ import annotations

import json
import logging
import math
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from langsmith import traceable
except ImportError:

    def traceable(*args, **kwargs):  # type: ignore[misc]
        """No-op fallback when langsmith is not installed."""

        def _decorator(fn):
            return fn

        return _decorator


from evidence_enrichment.config.settings import FieldThresholds, Settings, get_settings
from evidence_enrichment.execution_policy import ExecutionPolicyEngine
from evidence_enrichment.execution_policy.models import ActionType
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
from evidence_enrichment.core.models.enums import ReviewDecision, SourceType
from evidence_enrichment.finops.collector import FinOpsCollector
from evidence_enrichment.finops.estimation import (
    estimate_embedding_cost,
    estimate_stage_cost,
    estimate_tokens,
    stage_cost_from_tokens,
)
from evidence_enrichment.finops.models import (
    BudgetDecision,
    BudgetMode,
    DowngradeAction,
    LLMUsage,
    UsageSource,
)
from evidence_enrichment.finops.policy import BudgetPolicyEngine
from evidence_enrichment.finops.pricing import build_catalog
from evidence_enrichment.guardrails import run_guardrails
from evidence_enrichment.core.parse.base import ParserRegistry
from evidence_enrichment.core.parse.html_structured import HTMLStructuredParser
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
    SchemaValidationGate,
    compute_overall_confidence,
    gate_result,
)
from evidence_enrichment.core.search.query_planner import score_search_result
from evidence_enrichment.core.synthesis.replay import ReplaySynthesisAgent
from evidence_enrichment.observability.langfuse import observe, record_stage_observation
from evidence_enrichment.observability.summarizers import (
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
from evidence_enrichment.observability.runtime import (
    activate_remote_tracing_with_policy_check,
    reset_runtime_observability_config,
)
from evidence_enrichment.observability.tracer import LocalTracer
from evidence_enrichment.pipeline.replay import load_replay_bundle

if TYPE_CHECKING:
    from evidence_enrichment.core.retrieval.models import RetrievalResult
    from evidence_enrichment.core.retrieval.retriever import HybridRetriever
    from evidence_enrichment.core.retrieval.agent import RetrievalAgent
    from evidence_enrichment.core.extraction.models import ExtractionResult


class _NullSpan:
    """No-op context manager used when no tracer is available.

    ``_run_schema_extraction`` wraps its API call in a tracer span when a
    ``LocalTracer`` is passed.  When called without one (e.g. in unit tests or
    future non-pipeline callers), this sentinel span is used instead so the
    span context-manager protocol is always satisfied without an ``if tracer``
    guard inside the hot path.
    """

    def __enter__(self) -> dict:
        return {}

    def __exit__(self, *_: object) -> None:
        return None


@dataclass
class _FinOpsRunContext:
    """Holds all mutable FinOps state for a single pipeline run.

    Instantiated fresh at the start of each ``EvidenceCoordinator.run()`` call so
    that concurrent or sequential reuse of the same coordinator cannot mix state
    across runs.
    """

    collector: FinOpsCollector
    policy: BudgetPolicyEngine
    downgrade_actions: list[DowngradeAction] = field(default_factory=list)
    retrieval_degraded: bool = False

    def record_downgrade(self, action: DowngradeAction) -> None:
        self.collector.record_downgrade(action)
        self.downgrade_actions.append(action)


# Per-task ContextVar so that concurrent asyncio.gather() calls on the same
# EvidenceCoordinator each see their own _FinOpsRunContext without contention.
_frc_var: ContextVar[_FinOpsRunContext] = ContextVar("_frc_var")


@dataclass
class _PolicyRunContext:
    """Holds the ExecutionPolicyEngine for a single pipeline run.

    Created fresh at the start of each ``EvidenceCoordinator.run()`` call so
    that concurrent or sequential reuse of the same coordinator cannot mix
    policy state across runs.
    """

    engine: ExecutionPolicyEngine


_prc_var: ContextVar[_PolicyRunContext] = ContextVar("_prc_var")


class EvidenceCoordinator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.fetcher = DocumentFetcher()
        self.parser = ParserRegistry()
        self.parser.register(HTMLStructuredParser())  # text/html — structured section tree
        try:
            from evidence_enrichment.core.parse.pdf_structured import PDFStructuredParser
            self.parser.register(PDFStructuredParser())  # application/pdf (.[retrieval] only)
        except ImportError:
            pass  # PDF deps absent; PDFs already blocked at fetch time by _ALLOWED_BINARY_PREFIXES
        self.assessor = EvidenceAssessor()
        self.context_resolver = ContextResolver(
            self.settings.context_path / "context_manifest.yaml"
        )
        self._retriever: "HybridRetriever | RetrievalAgent | None" = None
        self._finops_catalog = build_catalog(self.settings.finops.pricing_override or None)
        # Per-instance ephemeral FinOps context used when _frc is accessed
        # outside an active run() (e.g. in tests/helpers).  Stored here rather
        # than in the module-global _frc_var so that two different coordinator
        # instances in the same async task cannot share each other's state.
        self._ephemeral_frc: _FinOpsRunContext | None = None
        # Same pattern for policy run context.
        self._ephemeral_prc: _PolicyRunContext | None = None

        # Initialize cache client if enabled
        self._cache = None
        if self.settings.cache.enabled:
            try:
                from evidence_enrichment.cache.client import RedisCache

                self._cache = RedisCache(
                    host=self.settings.cache.redis_host,
                    port=self.settings.cache.redis_port,
                    db=self.settings.cache.redis_db,
                    password=self.settings.cache.redis_password,
                    max_connections=self.settings.cache.max_connections,
                )
                logging.info("Redis cache initialized")
            except Exception as exc:
                logging.warning(f"Cache initialization failed: {exc}")
                self._cache = None

    @property
    def _frc(self) -> _FinOpsRunContext:
        """Return the FinOps run context for the current async task.

        Set at the top of every ``run()`` call via ``_frc_var.set()``.
        If called outside of an active run (e.g. in tests), a single ephemeral
        context is created and stored in the ContextVar for the current task so
        that all ``self._frc`` accesses within the same helper/test call chain
        share one consistent collector and policy instance.  ``run()`` always
        overwrites this with a fresh context and resets it on exit, so the
        ephemeral context never bleeds into a real run.
        """
        try:
            return _frc_var.get()
        except LookupError:
            # Outside an active run() — create a single ephemeral context on
            # this coordinator instance (not in the module-global ContextVar)
            # so that two different EvidenceCoordinator instances in the same
            # async task cannot share each other's collector/policy state.
            # run() always overwrites _frc_var with a fresh context and resets
            # the token on exit, so the ephemeral context never bleeds into a
            # real run.
            if self._ephemeral_frc is None:
                self._ephemeral_frc = self._new_finops_run_context()
            return self._ephemeral_frc

    def _new_finops_run_context(self) -> _FinOpsRunContext:
        return _FinOpsRunContext(
            collector=FinOpsCollector(self._finops_catalog),
            policy=BudgetPolicyEngine(
                mode=BudgetMode(self.settings.finops.budget_mode),
                max_cost_per_run=self.settings.finops.max_cost_usd_per_run,
                max_cost_per_success=self.settings.finops.max_cost_usd_per_success,
                catalog=self._finops_catalog,
            ),
        )

    @property
    def _prc(self) -> _PolicyRunContext:
        """Return the policy run context for the current async task."""
        try:
            return _prc_var.get()
        except LookupError:
            if self._ephemeral_prc is None:
                self._ephemeral_prc = self._new_policy_run_context()
            return self._ephemeral_prc

    def _new_policy_run_context(self) -> _PolicyRunContext:
        return _PolicyRunContext(
            engine=ExecutionPolicyEngine(self.settings.execution_policy),
        )

    def _get_retriever(
        self, entity_id: str
    ) -> "HybridRetriever | RetrievalAgent | None":
        """Lazy-initialise the retriever when retrieval mode is active."""
        rc = self.settings.retrieval
        if rc.mode not in ("local", "agent"):
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
                schema_version=rc.schema_version,
            )
            # Route chunker based on config
            if rc.chunker == "hierarchical":
                from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker
                chunker: object = HierarchicalChunker(
                    target_chunk_tokens=rc.target_chunk_tokens,
                    max_chunk_tokens=rc.max_chunk_tokens,
                    overlap_tokens=rc.overlap_tokens,
                )
            else:
                chunker = TableAwareChunker(
                    chunk_size=rc.chunk_size,
                    overlap=rc.overlap,
                    max_table_size=rc.max_table_size,
                )
            # Route retriever based on config
            if rc.retriever == "hierarchical":
                from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever
                active_retriever: object = HierarchicalRetriever(
                    entity_id=entity_id,
                    store=store,
                    embedder=embedder,
                    chunker=chunker,
                    top_k=rc.top_k,
                    weights=rc.hierarchical_weights,
                    section_top_k=rc.section_top_k,
                    section_min_score=rc.section_min_score,
                )
            else:
                active_retriever = HybridRetriever(
                    entity_id=entity_id,
                    store=store,
                    embedder=embedder,
                    chunker=chunker,
                    top_k=rc.top_k,
                    weights=rc.hybrid_weights,
                )
            if rc.mode == "agent":
                from evidence_enrichment.core.retrieval.agent import RetrievalAgent

                self._retriever = RetrievalAgent(active_retriever)
            else:
                self._retriever = active_retriever
        except Exception as exc:
            logging.warning(
                "Failed to initialise retriever for entity %s: %s", entity_id, exc
            )
            self._retriever = None
            if rc.mode in ("local", "agent"):
                self._frc.retrieval_degraded = True
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
        runtime_token = activate_remote_tracing_with_policy_check(
            backend=self.settings.observability_backend,
            trace_redact_values=self.settings.trace_redact_values,
        )
        _frc_token = _frc_var.set(self._new_finops_run_context())
        _prc_token = _prc_var.set(self._new_policy_run_context())
        try:
            effective_mode = mode or self.settings.default_mode
            entity_id = str(entity.get("entity_id") or entity.get("id") or "unknown")
            tracer = LocalTracer(
                mode=effective_mode,
                entity_id=entity_id,
                field_name=enricher.field_name,
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
            if self.settings.finops.enabled:
                post_run_budget = self._frc.policy.check_post_run(
                    self._frc.collector,
                    succeeded=result.output_value is not None,
                )
                # If the pipeline returned a hard budget-blocked result (gate_reason
                # starts with "budget_blocked:"), execution was cut short.  Regardless
                # of what check_post_run says (may be NOMINAL when cost is near-zero,
                # or EXCEEDED when some stages ran before the block), force BLOCKED so
                # callers can distinguish "run was terminated early" from "run finished
                # but was over budget".
                if result.gate_reason and result.gate_reason.startswith("budget_blocked:"):
                    from evidence_enrichment.finops.models import BudgetStatus
                    post_run_budget.status = BudgetStatus.BLOCKED
                    post_run_budget.budget_reason = (
                        post_run_budget.budget_reason
                        or result.gate_reason.removeprefix("budget_blocked:").strip()
                    )
                # Do NOT overwrite a nominal post-run status just because a downgrade
                # was applied.  A successful cheap-model or retrieval-off run IS nominal
                # — reporting it as BLOCKED is misleading.  The downgrade detail is
                # already captured in RunFinOpsSummary.downgrade_actions.
                finops_summary = self._frc.collector.build_summary(
                    total_latency_ms=sum(s.latency_ms for s in tracer.spans),
                    budget_decision=post_run_budget,
                )
                result.finops_summary = json.loads(finops_summary.model_dump_json())

            # Attach execution policy report — always, regardless of policy mode.
            # In "off" mode the report will have an empty decisions list; this is
            # intentional so the artifact slot is always present and consumers can
            # distinguish "policy off" from "policy not configured".
            policy_report = self._prc.engine.build_report()
            result.execution_policy_report = policy_report.model_dump()

            trace_artifacts = tracer.write(
                self.settings.trace_output_path,
                finops_data=result.finops_summary,
                execution_policy_data=result.execution_policy_report,
            )
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
        finally:
            _frc_var.reset(_frc_token)
            _prc_var.reset(_prc_token)
            if runtime_token is not None:
                reset_runtime_observability_config(runtime_token)

    def _resolve_replay_path(
        self, replay_bundle: str | None, entity: dict, enricher: BaseEnricher
    ) -> Path:
        if replay_bundle:
            return Path(replay_bundle)
        return self.settings.replay_path / f"{enricher.replay_slug(entity)}.json"

    def _effective_model(self, provider: str, *, cheap: bool = False) -> str:
        if provider == "openai":
            return (
                self.settings.finops.openai_cheap_model
                if cheap
                else self.settings.openai_model
            )
        if provider == "anthropic":
            return (
                self.settings.finops.anthropic_cheap_model
                if cheap
                else self.settings.anthropic_model
            )
        return provider

    def _record_stage_finops(
        self,
        span: dict,
        stage: str,
        provider: str,
        model_name: str,
        input_text: str,
        output_text: str,
        call_count: int = 1,
        downgrade_applied: "DowngradeAction | None" = None,
    ) -> None:
        if not self.settings.finops.enabled:
            return
        rec = estimate_stage_cost(
            stage=stage,
            provider=provider,
            model_name=model_name,
            input_text=input_text,
            output_text=output_text,
            call_count=call_count,
            catalog=self._finops_catalog,
        )
        if downgrade_applied is not None:
            rec.downgrade_applied = downgrade_applied
        self._frc.collector.record(rec)
        span["model_name"] = model_name
        span["estimated_input_tokens"] = rec.estimated_input_tokens
        span["estimated_output_tokens"] = rec.estimated_output_tokens
        span["estimated_total_tokens"] = rec.estimated_total_tokens
        span["estimated_cost_usd"] = rec.estimated_cost_usd
        if downgrade_applied is not None and downgrade_applied.value != "none":
            span["downgrade_applied"] = downgrade_applied.value

    def _record_stage_finops_from_tokens(
        self,
        span: dict,
        stage: str,
        provider: str,
        model_name: str,
        total_input_tokens: int,
        total_output_tokens: int,
        call_count: int,
        downgrade_applied: "DowngradeAction | None" = None,
        usage_source: "UsageSource | None" = None,
    ) -> None:
        """Record FinOps for a stage using pre-summed token counts.

        Avoids the chars→ceil(chars/4) round-trip done by ``_record_stage_finops``
        which can change total tokens when per-call sizes are uneven or not
        divisible by 4.  Use this for multi-call stages (analysis) where per-doc
        token counts are already known.

        ``usage_source`` defaults to ``UsageSource.ESTIMATED`` when ``None``.
        Pass ``UsageSource.PROVIDER_REPORTED`` when all token counts originate
        directly from provider API responses.
        """
        if not self.settings.finops.enabled:
            return
        effective_source = usage_source if usage_source is not None else UsageSource.ESTIMATED
        rec = stage_cost_from_tokens(
            stage=stage,
            provider=provider,
            model_name=model_name,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            call_count=call_count,
            catalog=self._finops_catalog,
            usage_source=effective_source,
        )
        if downgrade_applied is not None:
            rec.downgrade_applied = downgrade_applied
        self._frc.collector.record(rec)
        span["model_name"] = model_name
        span["estimated_input_tokens"] = rec.estimated_input_tokens
        span["estimated_output_tokens"] = rec.estimated_output_tokens
        span["estimated_total_tokens"] = rec.estimated_total_tokens
        span["estimated_cost_usd"] = rec.estimated_cost_usd
        if downgrade_applied is not None and downgrade_applied.value != "none":
            span["downgrade_applied"] = downgrade_applied.value

    def _check_budget_before_stage(
        self,
        stage: str,
        *,
        projected_marginal_cost: float = 0.0,
    ) -> BudgetDecision:
        if not self._frc.policy.is_enforcing:
            return BudgetDecision()
        decision = self._frc.policy.check_before_stage(
            self._frc.collector,
            projected_marginal_cost=projected_marginal_cost,
        )
        return decision

    def _check_policy_before_action(self, action: ActionType):
        """Check execution policy for the given action.

        Returns the PolicyDecision.  In ``enforce`` mode, callers must inspect
        ``decision.allowed`` and return a structured blocked result when False.
        In ``off`` and ``audit`` modes, ``decision.allowed`` is always True.
        """
        return self._prc.engine.check_action(action)

    def _build_policy_blocked_result(
        self,
        entity: dict,
        search_plan,
        mode: str,
        action: ActionType,
    ) -> PipelineRunResult:
        """Return a structured PipelineRunResult for a policy-blocked action.

        Does not raise.  Sets gate_reason to ``policy_blocked:<action>`` so
        callers can distinguish policy blocks from budget blocks.
        """
        synthesis = SynthesisResult(
            field_name=search_plan.field_name,
            value=None,
            reasoning=f"Execution policy blocked action: {action.value}",
            synthesis_confidence=0.0,
        )
        return PipelineRunResult(
            entity_id=str(entity.get("entity_id") or entity.get("id") or "unknown"),
            field_name=search_plan.field_name,
            mode=mode,
            search_plan=search_plan,
            synthesis=synthesis,
            overall_confidence=0.0,
            decision=ReviewDecision.AUTO_REJECT,
            gate_reason=f"policy_blocked:{action.value}",
        )

    def _apply_downgrade_before_analysis(
        self,
        decision: BudgetDecision,
        analysis_provider: str,
    ) -> tuple[str, bool]:
        skip_retrieval = self._frc.policy.should_disable_retrieval(decision)
        use_cheap = self._frc.policy.should_use_cheap_model(decision)
        model = self._effective_model(analysis_provider, cheap=use_cheap)
        if use_cheap:
            self._frc.record_downgrade(DowngradeAction.CHEAP_MODEL)
            self._frc.policy.mark_downgrade_exhausted()
        elif skip_retrieval:
            self._frc.record_downgrade(DowngradeAction.RETRIEVAL_OFF)
            self._frc.policy.mark_downgrade_exhausted()
        return model, skip_retrieval

    def _resolve_stage_model_and_budget(
        self,
        stage: str,
        provider: str,
        docs: list,
        *,
        retrieved_chunks_map: "dict | None" = None,
    ) -> "tuple[str, bool, BudgetDecision, DowngradeAction]":
        """Single-phase budget decision for a stage with cascading downgrades.

        Cascade order (strict mode only):
          1. No breach → run at default model with retrieval.
          2. First breach → try CHEAP_MODEL; re-project without retrieval overhead.
          3. Cheap-model still breaches → try RETRIEVAL_OFF at cheap model.
          4. Still breaches → mark downgrade exhausted; final check returns BLOCKED.

        Returns (model, skip_retrieval, final_decision, applied_downgrade_action).
        """
        if not self.settings.finops.enabled or not self._frc.policy.is_enforcing:
            default_model = self._effective_model(provider, cheap=False)
            return default_model, False, BudgetDecision(), DowngradeAction.NONE

        from evidence_enrichment.finops.estimation import estimate_tokens as _et

        def _item_text(item: Any) -> str:
            """Return the text that represents one item in the prompt payload.

            For FactClaims this mirrors the synthesis agent serialisation:
            candidate_value, source_url, supporting_excerpt, analysis_confidence.
            For documents it returns raw text (used when no chunks are available).
            """
            if hasattr(item, "candidate_value"):
                # Mirror synthesis agent JSON serialisation (all four fields).
                parts = [
                    item.candidate_value or "",
                    getattr(item, "source_url", "") or "",
                    getattr(item, "supporting_excerpt", "") or "",
                    str(getattr(item, "analysis_confidence", "") or ""),
                ]
                return " ".join(p for p in parts if p)[:6000]
            # ParsedDocument / AssessedDocument: text field
            if not getattr(item, "accepted_for_analysis", True):
                return ""
            return (getattr(item, "text", None) or "")[:6000]

        def _doc_input_chars(doc: Any, *, with_retrieval: bool) -> int:
            """Chars that will appear in the analysis prompt for one document.

            Mirrors _build_analysis_context: when retrieval is active and chunks
            exist, use chunk content + labels (no cap); fall back to raw text[:6000]
            only when no chunks exist for that document.
            """
            url = getattr(doc, "url", None) or ""
            if with_retrieval and retrieved_chunks_map:
                chunks = retrieved_chunks_map.get(url, [])
                if chunks:
                    # Replicate the label+content format used in _build_analysis_context.
                    parts = []
                    for i, result in enumerate(chunks, start=1):
                        label = (
                            f"[Chunk {i} | type="
                            f"{getattr(getattr(result, 'chunk', result), 'chunk_type', '')} | "
                            f"score={getattr(result, 'score', 0):.3f}]"
                        )
                        content = getattr(getattr(result, "chunk", result), "content", "") or ""
                        parts.append(f"{label}\n{content}")
                    return len("\n\n".join(parts))
            return len(_item_text(doc))

        def _projected(
            model: str,
            *,
            call_count: int | None = None,
            include_retrieval: bool = True,
        ) -> float:
            if not docs:
                return 0.0
            use_ret = include_retrieval and stage == "analysis"
            if call_count == 1:
                # Synthesis: single call over all docs/claims — tokenize the full
                # concatenated input to avoid averaging artifacts.
                total_input_chars = sum(_doc_input_chars(d, with_retrieval=use_ret) for d in docs)
                input_tokens = _et("x" * total_input_chars)
                # Output estimate: synthesis returns value+normalized+reasoning+confidence.
                # Scale with input complexity; floor at 100 tokens.
                avg_output_tokens = max(100, round(input_tokens * 0.10))
                return round(
                    self._finops_catalog.cost_for_tokens(model, input_tokens, avg_output_tokens),
                    8,
                )
            # Analysis: one call per doc — sum individual costs to avoid ceil() loss
            # from floored-average tokenization (ceil is non-linear over averages).
            total_cost = 0.0
            for d in docs:
                doc_input_chars = _doc_input_chars(d, with_retrieval=use_ret)
                doc_input_tokens = _et("x" * doc_input_chars)
                # Output estimate: reasoning + per-claim fields per doc.
                # Scale with input complexity; floor at 100 tokens.
                doc_output_tokens = max(100, round(doc_input_tokens * 0.15))
                total_cost += self._finops_catalog.cost_for_tokens(
                    model, doc_input_tokens, doc_output_tokens,
                )
            return round(total_cost, 8)

        default_model = self._effective_model(provider, cheap=False)
        cheap_model = self._effective_model(provider, cheap=True)

        # For synthesis (stage=="synthesis") expect exactly one output call.
        _call_count = 1 if stage == "synthesis" else None

        # Helper: is the projected cost within budget?
        def _within_budget(projected_cost: float) -> bool:
            limit = self._frc.policy.max_cost_per_run
            if limit is None:
                return True
            return self._frc.collector.accrued_cost_usd + projected_cost <= limit

        # ── Pass 1: default model, retrieval included ──────────────────────────
        # Reset per-stage downgrade exhaustion so each stage gets an independent
        # downgrade opportunity regardless of what prior stages did.
        self._frc.policy.reset_downgrade_exhausted()
        first_decision = self._frc.policy.check_before_stage(
            self._frc.collector,
            projected_marginal_cost=_projected(default_model, call_count=_call_count),
        )

        use_cheap = self._frc.policy.should_use_cheap_model(first_decision)
        skip_retrieval_first = self._frc.policy.should_disable_retrieval(first_decision)

        if not use_cheap and not skip_retrieval_first:
            # No downgrade needed — return as-is.
            return default_model, False, first_decision, DowngradeAction.NONE

        # ── Downgrade cascade (strict mode only) ──────────────────────────────
        # Try progressively cheaper options before marking downgrade exhausted.
        # Order: cheap-model → cheap-model+retrieval-off → retrieval-off only.
        # We use direct cost comparison to avoid the policy's WARN/BLOCKED flip
        # being gated on _downgrade_exhausted (which we haven't set yet).

        if use_cheap:
            cost_cheap = _projected(cheap_model, call_count=_call_count, include_retrieval=True)
            if _within_budget(cost_cheap):
                # CHEAP_MODEL alone is sufficient — record downgrade and return.
                self._frc.record_downgrade(DowngradeAction.CHEAP_MODEL)
                self._frc.policy.mark_downgrade_exhausted()
                final_decision = self._frc.policy.check_before_stage(
                    self._frc.collector, projected_marginal_cost=cost_cheap
                )
                return cheap_model, False, final_decision, DowngradeAction.CHEAP_MODEL

            # Cheap model alone still over budget — try dropping retrieval too.
            cost_cheap_no_ret = _projected(cheap_model, call_count=_call_count, include_retrieval=False)
            self._frc.policy.mark_downgrade_exhausted()
            final_decision = self._frc.policy.check_before_stage(
                self._frc.collector, projected_marginal_cost=cost_cheap_no_ret
            )
            if final_decision.status.value != "blocked":
                # Both downgrades applied and stage will run — record both.
                self._frc.record_downgrade(DowngradeAction.CHEAP_MODEL)
                self._frc.record_downgrade(DowngradeAction.RETRIEVAL_OFF)
            # Return CHEAP_MODEL as primary applied action; skip_retrieval=True
            # signals the caller to also disable retrieval context.
            return cheap_model, True, final_decision, DowngradeAction.CHEAP_MODEL

        # ── RETRIEVAL_OFF only (use_cheap is False) ────────────────────────────
        cost_no_ret = _projected(default_model, call_count=_call_count, include_retrieval=False)
        self._frc.policy.mark_downgrade_exhausted()
        final_decision = self._frc.policy.check_before_stage(
            self._frc.collector, projected_marginal_cost=cost_no_ret
        )
        if final_decision.status.value != "blocked":
            # Downgrade will actually be applied — record it.
            self._frc.record_downgrade(DowngradeAction.RETRIEVAL_OFF)
        return default_model, True, final_decision, DowngradeAction.RETRIEVAL_OFF

    def _build_budget_blocked_result(
        self,
        entity: dict,
        search_plan,
        mode: str,
        tracer: LocalTracer,
        decision: BudgetDecision,
    ) -> PipelineRunResult:
        self._frc.record_downgrade(DowngradeAction.BLOCK)
        synthesis = SynthesisResult(
            field_name=search_plan.field_name,
            value=None,
            reasoning=f"Budget blocked: {decision.budget_reason or 'exceeded'}",
            synthesis_confidence=0.0,
        )
        return PipelineRunResult(
            entity_id=str(entity.get("entity_id") or entity.get("id") or "unknown"),
            field_name=search_plan.field_name,
            mode=mode,
            search_plan=search_plan,
            synthesis=synthesis,
            overall_confidence=0.0,
            decision=ReviewDecision.AUTO_REJECT,
            gate_reason=f"budget_blocked: {decision.budget_reason or 'cost exceeds limit'}",
        )

    @observe(
        name="query_plan", as_type="chain", capture_input=False, capture_output=False
    )
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
        result = enricher.build_query_plan(entity)
        record_stage_observation(
            "query_plan", trace_payload, result, summarize_query_plan
        )
        return result

    @observe(name="search", as_type="chain", capture_input=False, capture_output=False)
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
            result = [SearchResult(**row) for row in bundle.get("search_results", [])]
            record_stage_observation(
                "search", trace_payload, result, summarize_search_results
            )
            return result
        result = await self._search_live(search_plan)
        record_stage_observation(
            "search", trace_payload, result, summarize_search_results
        )
        return result

    @observe(name="fetch", as_type="chain", capture_input=False, capture_output=False)
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
            result = [
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
            record_stage_observation(
                "fetch", trace_payload, result, summarize_fetched_documents
            )
            return result

        # Cache integration with mode isolation
        mode = trace_payload.get("mode", "auto")
        if self._cache and self._cache.is_available():
            from evidence_enrichment.cache.middleware import fetch_with_cache

            result = []
            cache_hits = 0
            cache_misses = 0
            cache_stale = 0
            fetch_failures = 0

            for search_result in search_results[:3]:  # Match coordinator limit
                try:
                    doc, metadata = await fetch_with_cache(
                        url=search_result.url,
                        mode=mode,
                        cache=self._cache,
                        ttl_seconds=self.settings.cache.fetch_ttl_seconds,
                        fetch_fn=lambda sr=search_result: self.fetcher.fetch(sr),
                    )
                    result.append(doc)

                    if metadata.status == "hit":
                        cache_hits += 1
                    elif metadata.status == "miss":
                        cache_misses += 1
                    elif metadata.status == "stale":
                        cache_stale += 1
                except Exception as exc:
                    fetch_failures += 1
                    logging.warning("fetch failed for %s: %s", search_result.url, exc)

            if search_results and not result:
                raise RuntimeError("All fetch attempts failed")

            # Inject cache metadata into trace payload
            trace_payload["cache_hits"] = cache_hits
            trace_payload["cache_misses"] = cache_misses
            trace_payload["cache_stale"] = cache_stale
            trace_payload["fetch_failures"] = fetch_failures
        else:
            result = await self._fetch_documents(search_results)

        record_stage_observation(
            "fetch", trace_payload, result, summarize_fetched_documents
        )
        return result

    @observe(name="parse", as_type="chain", capture_input=False, capture_output=False)
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
            result = [
                ParsedDocument(**row) for row in bundle.get("parsed_documents", [])
            ]
            record_stage_observation(
                "parse", trace_payload, result, summarize_parsed_documents
            )
            return result
        if use_structured:
            result = [
                self.parser.parse(document)
                for document in fetched_documents
            ]
        else:
            result = [self.parser.parse(document) for document in fetched_documents]
        record_stage_observation(
            "parse", trace_payload, result, summarize_parsed_documents
        )
        return result

    @observe(
        name="evidence_assessment",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    @traceable(
        name="evidence_assessment",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_assessed_documents,
    )
    async def _stage_evidence_assessment(
        self,
        parsed_documents: list[ParsedDocument],
        *,
        company_name: str,
        bundle: dict[str, Any] | None,
        trace_payload: dict[str, Any],
    ) -> list[ParsedDocument]:
        if bundle is not None:
            record_stage_observation(
                "evidence_assessment",
                trace_payload,
                parsed_documents,
                summarize_assessed_documents,
            )
            return parsed_documents

        # Cache integration with mode isolation
        mode = trace_payload.get("mode", "auto")
        if self._cache and self._cache.is_available():
            from evidence_enrichment.cache.middleware import assess_with_cache

            result = []
            cache_hits = 0
            cache_misses = 0
            cache_stale = 0

            for doc in parsed_documents:
                # Assessment cache is async
                assessed_doc, metadata = await assess_with_cache(
                    parsed_doc=doc,
                    company_name=company_name,
                    mode=mode,
                    cache=self._cache,
                    ttl_seconds=self.settings.cache.assessment_ttl_seconds,
                    assess_fn=lambda d=doc: self.assessor.assess(d, company_name),
                )
                result.append(assessed_doc)

                if metadata.status == "hit":
                    cache_hits += 1
                elif metadata.status == "miss":
                    cache_misses += 1
                elif metadata.status == "stale":
                    cache_stale += 1

            # Inject cache metadata into trace payload
            trace_payload["cache_hits"] = cache_hits
            trace_payload["cache_misses"] = cache_misses
            trace_payload["cache_stale"] = cache_stale
        else:
            result = [
                self.assessor.assess(document, company_name)
                for document in parsed_documents
            ]

        record_stage_observation(
            "evidence_assessment", trace_payload, result, summarize_assessed_documents
        )
        return result

    async def _run_analysis_stage(
        self,
        documents: list[ParsedDocument],
        *,
        field_name: str,
        company_name: str,
        analysis_agent: Any,
        retrieved_chunks_map: "dict[str, list[RetrievalResult]] | None" = None,
        trace_payload: dict[str, Any],
    ) -> tuple[list[AnalysisReport], list[FactClaim], str]:
        """Shared implementation for analysis stage execution.

        Both ``_stage_analysis`` and ``_stage_analysis_with_agent`` delegate here
        after resolving the concrete agent instance.  All error-handling and
        FinOps-preservation logic lives in exactly one place.
        """
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
                # ProviderParseError (and any broader post-response exception) carries
                # the usage from the paid call.  Preserve it so FinOps records cost
                # even when the response could not be decoded.
                error_usage = getattr(exc, "llm_usage", None) or LLMUsage(
                    input_tokens=0, output_tokens=0
                )
                reports.append(
                    AnalysisReport(
                        source_url=document.url,
                        provider=analysis_agent.provider_type,
                        claims=[],
                        reasoning="[analysis error — skipped]",
                        llm_usage=error_usage,
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
        result = (reports, claims, analysis_agent.provider_type.value)
        record_stage_observation(
            "analysis", trace_payload, result, summarize_analysis_stage
        )
        return result

    @observe(
        name="analysis", as_type="chain", capture_input=False, capture_output=False
    )
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
        return await self._run_analysis_stage(
            documents,
            field_name=field_name,
            company_name=company_name,
            analysis_agent=analysis_agent,
            retrieved_chunks_map=retrieved_chunks_map,
            trace_payload=trace_payload,
        )

    @observe(
        name="analysis", as_type="chain", capture_input=False, capture_output=False
    )
    @traceable(
        name="analysis",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_analysis_stage,
    )
    async def _stage_analysis_with_agent(
        self,
        documents: list[ParsedDocument],
        *,
        field_name: str,
        company_name: str,
        bundle: dict[str, Any] | None,
        agent: Any | None = None,
        retrieved_chunks_map: "dict[str, list[RetrievalResult]] | None" = None,
        trace_payload: dict[str, Any],
    ) -> tuple[list[AnalysisReport], list[FactClaim], str]:
        analysis_agent = (
            ReplayAnalysisAgent(bundle)
            if bundle is not None
            else (agent or self._analysis_agent())
        )
        return await self._run_analysis_stage(
            documents,
            field_name=field_name,
            company_name=company_name,
            analysis_agent=analysis_agent,
            retrieved_chunks_map=retrieved_chunks_map,
            trace_payload=trace_payload,
        )

    async def _run_synthesis_stage(
        self,
        claims: list[FactClaim],
        *,
        field_name: str,
        company_name: str,
        synthesis_agent: Any,
        trace_payload: dict[str, Any],
    ) -> tuple[Any, str]:
        """Shared implementation for synthesis stage execution.

        Both ``_stage_synthesis`` and ``_stage_synthesis_with_agent`` delegate here
        after resolving the concrete agent instance.  All error-handling and
        FinOps-preservation logic lives in exactly one place.
        """
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
            # ProviderParseError (and any broader post-response exception) carries the
            # usage from the paid API call so FinOps records cost even when the
            # response could not be decoded.
            synthesis = SynthesisResult(
                field_name=field_name,
                value=None,
                reasoning="Synthesis failed: provider returned unparseable response.",
                synthesis_confidence=0.0,
                llm_usage=exc.llm_usage,
            )
        result = (synthesis, synthesis_agent.provider_type.value)
        record_stage_observation(
            "synthesis", trace_payload, result, summarize_synthesis_stage
        )
        return result

    @observe(
        name="synthesis", as_type="chain", capture_input=False, capture_output=False
    )
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
        return await self._run_synthesis_stage(
            claims,
            field_name=field_name,
            company_name=company_name,
            synthesis_agent=synthesis_agent,
            trace_payload=trace_payload,
        )

    @observe(
        name="synthesis", as_type="chain", capture_input=False, capture_output=False
    )
    @traceable(
        name="synthesis",
        run_type="chain",
        process_inputs=trace_payload_inputs,
        process_outputs=summarize_synthesis_stage,
    )
    async def _stage_synthesis_with_agent(
        self,
        claims: list[FactClaim],
        *,
        field_name: str,
        company_name: str,
        bundle: dict[str, Any] | None,
        agent: Any | None = None,
        trace_payload: dict[str, Any],
    ) -> tuple[Any, str]:
        synthesis_agent = (
            ReplaySynthesisAgent(bundle)
            if bundle is not None
            else (agent or self._synthesis_agent())
        )
        return await self._run_synthesis_stage(
            claims,
            field_name=field_name,
            company_name=company_name,
            synthesis_agent=synthesis_agent,
            trace_payload=trace_payload,
        )

    @observe(
        name="review_gate", as_type="chain", capture_input=False, capture_output=False
    )
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
        result = {
            "overall_confidence": overall_confidence,
            "decision": decision.value,
            "decision_enum": decision,
            "gate_reason": gate_reason,
        }
        record_stage_observation(
            "review_gate", trace_payload, result, summarize_review_gate
        )
        return result

    async def _run_live(
        self,
        entity: dict,
        enricher: BaseEnricher,
        search_plan,
        mode: str,
        tracer: LocalTracer,
    ) -> PipelineRunResult:
        entity_id = str(entity.get("entity_id") or entity.get("id") or "unknown")

        # Policy gate: live_provider_calls covers all live LLM and API traffic.
        _lpc_decision = self._check_policy_before_action(ActionType.LIVE_PROVIDER_CALLS)
        if not _lpc_decision.allowed:
            return self._build_policy_blocked_result(entity, search_plan, mode, ActionType.LIVE_PROVIDER_CALLS)

        retriever = self._get_retriever(entity_id)
        use_structured = retriever is not None
        rc = self.settings.retrieval
        # Note: retrieval budget enforcement is deferred until after
        # accepted_for_indexing is known so a real embedding cost can be
        # projected and the retrieval stage gets its own independent
        # downgrade opportunity via reset_downgrade_exhausted().
        # See the deferred retrieval gate just before retrieval_indexing span.

        query_count = len([search_plan.primary_query, *search_plan.query_variants])
        with tracer.span(
            "search", provider="provider_chain", input_count=query_count
        ) as span:
            # Policy gate: search
            _search_decision = self._check_policy_before_action(ActionType.SEARCH)
            if not _search_decision.allowed:
                return self._build_policy_blocked_result(entity, search_plan, mode, ActionType.SEARCH)
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
            # Policy gate: fetch
            _fetch_decision = self._check_policy_before_action(ActionType.FETCH)
            if not _fetch_decision.allowed:
                return self._build_policy_blocked_result(entity, search_plan, mode, ActionType.FETCH)
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
            assessed_documents = await self._stage_evidence_assessment(
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
            # Policy gate: retrieval
            _retrieval_decision = self._check_policy_before_action(ActionType.RETRIEVAL)
            if not _retrieval_decision.allowed:
                # Disable retrieval silently (enforce mode) — pipeline continues
                # without retrieval context rather than blocking entirely.
                retriever = None
                use_structured = False
                self._frc.retrieval_degraded = True
            else:
                accepted_for_indexing = [
                    d
                    for d in assessed_documents
                    if d.accepted_for_analysis
                    and len(d.full_text or d.text) >= rc.min_doc_chars
                ]
                # Deferred retrieval budget gate: now that we know which documents will be
                # indexed we can project a real embedding cost before spending any tokens.
                if accepted_for_indexing and self.settings.finops.enabled and self._frc.policy.is_enforcing:
                    from evidence_enrichment.finops.estimation import estimate_tokens as _et
                    # Reset per-stage exhaustion so the retrieval stage gets its own
                    # independent downgrade opportunity.
                    self._frc.policy.reset_downgrade_exhausted()
                    rc_cfg = self.settings.retrieval
                    _gate_chunk_chars: list[int] = []
                    try:
                        if rc_cfg.chunker == "hierarchical":
                            from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker as _Chunker  # type: ignore[assignment]
                            _gate_chunker = _Chunker(
                                target_chunk_tokens=rc_cfg.target_chunk_tokens,
                                max_chunk_tokens=rc_cfg.max_chunk_tokens,
                                overlap_tokens=rc_cfg.overlap_tokens,
                            )
                        else:
                            from evidence_enrichment.core.retrieval.chunker import TableAwareChunker as _FlatChunker
                            _gate_chunker = _FlatChunker(
                                chunk_size=rc_cfg.chunk_size,
                                overlap=rc_cfg.overlap,
                                max_table_size=rc_cfg.max_table_size,
                            )
                        for d in accepted_for_indexing:
                            for c in _gate_chunker.chunk(d):
                                _gate_chunk_chars.append(len(c.content))
                    except Exception:
                        chunk_size = rc_cfg.chunk_size
                        overlap = rc_cfg.overlap
                        stride = max(chunk_size - overlap, 1)
                        for d in accepted_for_indexing:
                            n = math.ceil(len(d.full_text or d.text) / stride)
                            _gate_chunk_chars.extend([chunk_size] * n)
                    projected_index_cost = sum(
                        self._finops_catalog.cost_for_tokens(
                            self.settings.retrieval.embedding_model, _et("x" * clen), 0
                        )
                        for clen in _gate_chunk_chars
                    ) if _gate_chunk_chars else 0.0
                    query_text = enricher.retrieval_query(entity)
                    max_iters = max(getattr(retriever, "max_iterations", None) or 1, 1)
                    try:
                        from evidence_enrichment.core.retrieval.evaluator import _REFINEMENT_SUFFIXES
                        all_suffix_tokens = " ".join(_REFINEMENT_SUFFIXES)
                        worst_case_query_len = len(query_text) + len(all_suffix_tokens)
                    except Exception:
                        worst_case_query_len = len(query_text) * 2
                    projected_query_cost = sum(
                        self._finops_catalog.cost_for_tokens(
                            self.settings.retrieval.embedding_model,
                            _et("x" * worst_case_query_len),
                            0,
                        )
                        for _ in range(len(accepted_for_indexing) * max_iters)
                    )
                    projected_emb_cost = projected_index_cost + projected_query_cost
                    retrieval_gate = self._check_budget_before_stage(
                        "retrieval_indexing", projected_marginal_cost=projected_emb_cost
                    )
                    if retrieval_gate.status.value == "blocked":
                        return self._build_budget_blocked_result(
                            entity, search_plan, mode, tracer, retrieval_gate,
                        )
                    if self._frc.policy.should_disable_retrieval(retrieval_gate):
                        retriever = None
                        use_structured = False
                        self._frc.retrieval_degraded = True
                        self._frc.record_downgrade(DowngradeAction.RETRIEVAL_OFF)
                        self._frc.policy.mark_downgrade_exhausted()
                if retriever is not None:
                    with tracer.span(
                        "retrieval_indexing",
                        provider="chroma",
                        input_count=len(accepted_for_indexing),
                    ) as span:
                        from evidence_enrichment.core.retrieval.retriever import (  # noqa: PLC0415
                            IndexingPartialError,
                        )
                        indexed_chunk_count = 0
                        billed_embedding_count = 0
                        per_chunk_chars: list[int] = []
                        successfully_indexed_urls: set[str] = set()
                        for document in accepted_for_indexing:
                            try:
                                chunks = retriever.index_document(document)
                                indexed_chunk_count += len(chunks)
                                billed_embedding_count += len(chunks)
                                for c in chunks:
                                    per_chunk_chars.append(len(c.content))
                                if chunks:
                                    successfully_indexed_urls.add(document.url)
                            except IndexingPartialError as exc:
                                logging.warning(
                                    "retrieval_indexing upsert failed for %s: %s",
                                    document.url,
                                    exc,
                                )
                                self._frc.retrieval_degraded = True
                                if hasattr(retriever, "evict_document"):
                                    retriever.evict_document(document.url)
                                billed_embedding_count += len(exc.embedded_chunks)
                                for c in exc.embedded_chunks:
                                    per_chunk_chars.append(len(c.content))
                            except Exception as exc:
                                logging.warning(
                                    "retrieval_indexing failed for %s: %s", document.url, exc
                                )
                                self._frc.retrieval_degraded = True
                                if hasattr(retriever, "evict_document"):
                                    retriever.evict_document(document.url)
                        span["output_count"] = indexed_chunk_count
                        retrieval_chunk_count = indexed_chunk_count
                        if self.settings.finops.enabled and accepted_for_indexing:
                            emb_rec = estimate_embedding_cost(
                                stage="retrieval_indexing",
                                provider="openai",
                                model_name=self.settings.retrieval.embedding_model,
                                text_count=billed_embedding_count,
                                total_chars=sum(per_chunk_chars),
                                per_text_chars=per_chunk_chars or None,
                                catalog=self._finops_catalog,
                            )
                            self._frc.collector.record(emb_rec)
                            span["model_name"] = self.settings.retrieval.embedding_model
                            span["estimated_input_tokens"] = emb_rec.estimated_input_tokens
                            span["estimated_total_tokens"] = emb_rec.estimated_total_tokens
                            span["estimated_cost_usd"] = emb_rec.estimated_cost_usd

                    docs_to_query = [
                        d for d in accepted_for_indexing
                        if d.url in successfully_indexed_urls
                    ]
                    with tracer.span(
                        "retrieval_query",
                        provider="chroma",
                        input_count=len(docs_to_query),
                    ) as span:
                        total_query_iterations = 0
                        total_query_chars = 0
                        per_iteration_query_chars: list[int] = []
                        base_query_text = enricher.retrieval_query(entity)
                        for document in docs_to_query:
                            _doc_metrics = None
                            _query_succeeded = False
                            try:
                                if hasattr(retriever, "retrieve_with_metrics"):
                                    hits, _doc_metrics = retriever.retrieve_with_metrics(
                                        query=base_query_text,
                                        document_url=document.url,
                                    )
                                else:
                                    hits = retriever.retrieve(
                                        query=base_query_text,
                                        document_url=document.url,
                                    )
                                _query_succeeded = True
                                if hits:
                                    retrieved_chunks_map[document.url] = hits
                            except Exception as exc:
                                logging.warning(
                                    "retrieval_query failed for %s: %s", document.url, exc
                                )
                                self._frc.retrieval_degraded = True
                                from evidence_enrichment.core.retrieval.agent import RetrievalPartialError
                                if isinstance(exc, RetrievalPartialError):
                                    _doc_metrics = exc.partial_metrics
                                    if exc.partial_results:
                                        retrieved_chunks_map[document.url] = exc.partial_results
                            if _doc_metrics is not None:
                                if _doc_metrics.iterations == 0 and _doc_metrics.total_query_chars == 0:
                                    continue
                                doc_iters = max(_doc_metrics.iterations, 1)
                                total_query_iterations += doc_iters
                                total_query_chars += _doc_metrics.total_query_chars
                                if _doc_metrics.query_char_history:
                                    per_iteration_query_chars.extend(_doc_metrics.query_char_history)
                                else:
                                    per_iter_chars = _doc_metrics.total_query_chars // doc_iters if doc_iters else _doc_metrics.total_query_chars
                                    per_iteration_query_chars.extend([per_iter_chars] * doc_iters)
                            else:
                                if not _query_succeeded:
                                    continue
                                doc_iters = max(getattr(retriever, "last_iterations", None) or 1, 1)
                                total_query_iterations += doc_iters
                                if getattr(retriever, "last_total_query_chars", 0):
                                    doc_total = retriever.last_total_query_chars
                                    total_query_chars += doc_total
                                    per_iter_chars = doc_total // doc_iters if doc_iters else doc_total
                                    per_iteration_query_chars.extend([per_iter_chars] * doc_iters)
                                else:
                                    doc_query_chars = getattr(retriever, "last_query_chars", None) or len(base_query_text)
                                    total_query_chars += doc_query_chars * doc_iters
                                    per_iteration_query_chars.extend([doc_query_chars] * doc_iters)
                        span["output_count"] = sum(
                            len(v) for v in retrieved_chunks_map.values()
                        )
                        span["agent_iterations"] = total_query_iterations
                        # Build top-3 score breakdown across all retrieved chunks
                        _all_results = [
                            r
                            for hits in retrieved_chunks_map.values()
                            for r in hits
                        ]
                        _all_results.sort(key=lambda r: r.score, reverse=True)
                        _top3 = _all_results[:3]
                        span["chunk_count"] = len(_all_results)
                        span["top_chunk_score"] = round(_top3[0].score, 4) if _top3 else None
                        span["retrieval_score_breakdown"] = [
                            {
                                "rank": i + 1,
                                "chunk_id": r.chunk.chunk_id,
                                "vector": round(r.vector_score, 4),
                                "keyword": round(r.keyword_score, 4),
                                "table_boost": round(r.table_boost_score, 4),
                                "fused": round(r.score, 4),
                            }
                            for i, r in enumerate(_top3)
                        ]
                        if self.settings.finops.enabled and accepted_for_indexing:
                            q_rec = estimate_embedding_cost(
                                stage="retrieval_query",
                                provider="openai",
                                model_name=self.settings.retrieval.embedding_model,
                                text_count=total_query_iterations,
                                total_chars=total_query_chars,
                                per_text_chars=per_iteration_query_chars or None,
                                catalog=self._finops_catalog,
                            )
                            self._frc.collector.record(q_rec)
                            span["model_name"] = self.settings.retrieval.embedding_model
                            span["estimated_input_tokens"] = q_rec.estimated_input_tokens
                            span["estimated_total_tokens"] = q_rec.estimated_total_tokens
                            span["estimated_cost_usd"] = q_rec.estimated_cost_usd

        accepted_documents = [
            document
            for document in assessed_documents
            if document.accepted_for_analysis
        ]
        analysis_provider = self._analysis_agent().provider_type.value
        analysis_model, skip_retrieval_for_budget, analysis_budget, analysis_downgrade = (
            self._resolve_stage_model_and_budget(
                "analysis", analysis_provider, accepted_documents,
                retrieved_chunks_map=retrieved_chunks_map,
            )
        )
        if analysis_budget.status.value == "blocked":
            return self._build_budget_blocked_result(
                entity, search_plan, mode, tracer, analysis_budget,
            )
        # Enforce retrieval-off downgrade: pass an empty map into analysis so it
        # does not receive context it was not supposed to see, but preserve
        # retrieved_chunks_map intact for telemetry/result construction — billed
        # retrieval spend must remain visible even when analysis is degraded.
        if skip_retrieval_for_budget:
            analysis_chunks_map: dict = {}
            self._frc.retrieval_degraded = True
        else:
            analysis_chunks_map = retrieved_chunks_map
        with tracer.span(
            "analysis", provider=analysis_provider, input_count=len(accepted_documents)
        ) as span:
            agent = self._analysis_agent_with_model(analysis_provider, analysis_model)
            reports, claims, _ = await self._stage_analysis_with_agent(
                assessed_documents,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=None,
                agent=agent,
                retrieved_chunks_map=analysis_chunks_map or None,
                trace_payload={
                    "mode": mode,
                    "accepted_document_urls": [
                        document.url for document in accepted_documents
                    ],
                    **summarize_assessed_documents(assessed_documents),
                },
            )
            span["output_count"] = len(claims)
            if self.settings.finops.enabled:
                # Sum token usage captured at call time from each report's llm_usage.
                # Reports with llm_usage=None (replay agent) are skipped entirely.
                # Zero-token placeholders (network errors before the API responded)
                # are excluded from call_count AND from usage_source assessment so
                # they don't flip a fully provider-reported stage to ESTIMATED.
                usages = [r.llm_usage for r in reports if r.llm_usage is not None]
                analysis_call_count = sum(1 for u in usages if u.input_tokens > 0)
                if analysis_call_count > 0:
                    total_input_tokens_analysis = sum(u.input_tokens for u in usages)
                    total_output_tokens_analysis = sum(u.output_tokens for u in usages)
                    paid_usages = [u for u in usages if u.input_tokens > 0 or u.output_tokens > 0]
                    all_provider_reported = all(
                        u.usage_source == UsageSource.PROVIDER_REPORTED for u in paid_usages
                    )
                    self._record_stage_finops_from_tokens(
                        span, "analysis", analysis_provider, analysis_model,
                        total_input_tokens=total_input_tokens_analysis,
                        total_output_tokens=total_output_tokens_analysis,
                        call_count=analysis_call_count,
                        downgrade_applied=analysis_downgrade,
                        usage_source=UsageSource.PROVIDER_REPORTED if all_provider_reported else UsageSource.ESTIMATED,
                    )

        synthesis_provider = self._synthesis_agent().provider_type.value
        synthesis_model, _, synthesis_budget, synthesis_downgrade = self._resolve_stage_model_and_budget(
            "synthesis", synthesis_provider, claims
        )
        if synthesis_budget.status.value == "blocked":
            return self._build_budget_blocked_result(
                entity, search_plan, mode, tracer, synthesis_budget,
            )
        with tracer.span(
            "synthesis", provider=synthesis_provider, input_count=len(claims)
        ) as span:
            synth_agent = self._synthesis_agent_with_model(synthesis_provider, synthesis_model)
            synthesis, _ = await self._stage_synthesis_with_agent(
                claims,
                field_name=enricher.field_name,
                company_name=search_plan.metadata.get("company_name", ""),
                bundle=None,
                agent=synth_agent,
                trace_payload={"mode": mode, **summarize_claims(claims)},
            )
            if self.settings.finops.enabled:
                # Use token usage captured at call time from synthesis.llm_usage.
                # llm_usage=None means replay agent — skip recording entirely.
                if synthesis.llm_usage is not None:
                    self._record_stage_finops_from_tokens(
                        span, "synthesis", synthesis_provider, synthesis_model,
                        total_input_tokens=synthesis.llm_usage.input_tokens,
                        total_output_tokens=synthesis.llm_usage.output_tokens,
                        call_count=1,
                        downgrade_applied=synthesis_downgrade,
                        usage_source=synthesis.llm_usage.usage_source,
                    )
            synthesis = enricher.validate_synthesis(synthesis)
            span["output_count"] = 1 if synthesis.value else 0

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

        # Stage C — typed schema extraction (additive; only when opted in)
        if self.settings.retrieval.schema_validation:
            extraction_results = await self._run_schema_extraction(
                field_name=enricher.field_name,
                retrieved_chunks_map=retrieved_chunks_map,
                assessed_documents=assessed_documents,
                tracer=tracer,
            )
            gate = SchemaValidationGate()
            gated: list = []
            for er in extraction_results:
                gate_result_sc = gate.check(er)
                if gate_result_sc.hard_errors:
                    logging.warning(
                        "Schema extraction hard-fail for field %r: %s",
                        er.field_name,
                        gate_result_sc.hard_errors,
                    )
                # Apply gate's confidence penalty: update both the wrapper field
                # and the inner value.extraction_confidence so the two never diverge.
                new_confidence = gate_result_sc.confidence_after
                new_value = er.value
                if hasattr(er.value, "extraction_confidence"):
                    try:
                        new_value = er.value.model_copy(
                            update={"extraction_confidence": new_confidence}
                        )
                    except Exception:  # noqa: BLE001
                        pass  # leave stale on unexpected copy failure
                gated.append(
                    er.model_copy(
                        update={
                            "extraction_confidence": new_confidence,
                            "value": new_value,
                        }
                    )
                )
            result.extraction_results = gated

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
            parsed_documents = await self._stage_evidence_assessment(
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
        # Derive provider from the configured live agent — replay prices/models must
        # match the live provider selection, not be hard-coded to openai.
        # Fall back to "openai" gracefully when no live provider keys are configured
        # (e.g. pure replay/test environments where agents cannot be constructed).
        try:
            _replay_analysis_provider = self._analysis_agent().provider_type.value
        except RuntimeError:
            _replay_analysis_provider = "openai"
        try:
            _replay_synthesis_provider = self._synthesis_agent().provider_type.value
        except RuntimeError:
            _replay_synthesis_provider = "openai"
        # _resolve_stage_model_and_budget is the single authoritative budget gate:
        # it calls reset_downgrade_exhausted() internally then runs the full
        # cheap-model → retrieval-off cascade before returning BLOCKED.
        # No standalone pre-check is needed — doing one before would leave stale
        # _downgrade_exhausted state that blocks the cascade for the next stage.
        analysis_model, _, analysis_budget, _ = self._resolve_stage_model_and_budget(
            "analysis",
            _replay_analysis_provider,
            accepted_documents,
            retrieved_chunks_map=None,
        )
        if analysis_budget.status.value == "blocked":
            return self._build_budget_blocked_result(
                entity, search_plan, mode, tracer, analysis_budget,
            )
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
            if self.settings.finops.enabled:
                # Replay agents return llm_usage=None — skip recording entirely.
                # No LLM call was made; recording estimated tokens would misrepresent
                # the replay as having a real cost.
                usages = [r.llm_usage for r in reports if r.llm_usage is not None]
                replay_call_count = sum(1 for u in usages if u.input_tokens > 0)
                if replay_call_count > 0:
                    paid_usages = [u for u in usages if u.input_tokens > 0 or u.output_tokens > 0]
                    self._record_stage_finops_from_tokens(
                        span, "analysis", _replay_analysis_provider, analysis_model,
                        total_input_tokens=sum(u.input_tokens for u in usages),
                        total_output_tokens=sum(u.output_tokens for u in usages),
                        call_count=replay_call_count,
                        usage_source=(
                            UsageSource.PROVIDER_REPORTED
                            if all(u.usage_source == UsageSource.PROVIDER_REPORTED for u in paid_usages)
                            else UsageSource.ESTIMATED
                        ),
                    )
        # Same pattern as analysis: route exclusively through _resolve_stage_model_and_budget
        # so the downgrade cascade runs before any BLOCKED decision for synthesis.
        synthesis_model, _, synthesis_budget, _ = self._resolve_stage_model_and_budget(
            "synthesis",
            _replay_synthesis_provider,
            claims,
            retrieved_chunks_map=None,
        )
        if synthesis_budget.status.value == "blocked":
            return self._build_budget_blocked_result(
                entity, search_plan, mode, tracer, synthesis_budget,
            )
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
            if self.settings.finops.enabled:
                # Replay agents return llm_usage=None — skip recording entirely.
                if synthesis.llm_usage is not None:
                    self._record_stage_finops_from_tokens(
                        span, "synthesis", _replay_synthesis_provider, synthesis_model,
                        total_input_tokens=synthesis.llm_usage.input_tokens,
                        total_output_tokens=synthesis.llm_usage.output_tokens,
                        call_count=1,
                        usage_source=synthesis.llm_usage.usage_source,
                    )
            synthesis = enricher.validate_synthesis(synthesis)
            span["output_count"] = 1 if synthesis.value else 0
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

    def _analysis_agent_with_model(self, provider: str, model: str):
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OpenAI API key not configured.")
            return OpenAIAnalysisAgent(self.settings.openai_api_key, model)
        if provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise RuntimeError("Anthropic API key not configured.")
            return AnthropicAnalysisAgent(self.settings.anthropic_api_key, model)
        raise RuntimeError(f"Unknown analysis provider: {provider}")

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

    def _synthesis_agent_with_model(self, provider: str, model: str):
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OpenAI API key not configured.")
            return OpenAISynthesisAgent(self.settings.openai_api_key, model)
        if provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise RuntimeError("Anthropic API key not configured.")
            return AnthropicSynthesisAgent(self.settings.anthropic_api_key, model)
        raise RuntimeError(f"Unknown synthesis provider: {provider}")

    async def _run_schema_extraction(
        self,
        field_name: str,
        retrieved_chunks_map: "dict[str, list[RetrievalResult]]",
        assessed_documents: "list[ParsedDocument]",
        tracer: "LocalTracer | None" = None,
    ) -> "list[ExtractionResult]":
        """Run typed extraction for ``field_name`` when ``schema_validation=True``.

        Builds chunk context from retrieved chunks (falling back to plain
        document text when no retrieval was performed), then delegates to
        ``SchemaExtractor``.

        Model selection goes through ``_resolve_stage_model_and_budget`` so
        cheap-model downgrades and budget blocks apply identically to the main
        analysis stage.  Token spend is recorded via ``_record_stage_finops``
        (char-based estimation) so FinOps accounting remains complete.

        Returns an empty list when:
        - the field has no registered schema, or
        - no analysis provider is available, or
        - the budget is blocked for this stage, or
        - the extractor raises unexpectedly.
        """
        from evidence_enrichment.core.extraction.extractor import SchemaExtractor
        from evidence_enrichment.core.extraction.schemas import SCHEMA_REGISTRY

        if field_name not in SCHEMA_REGISTRY:
            return []

        # ── Model / budget resolution ────────────────────────────────────────
        analysis_agent = self._analysis_agent()
        provider = analysis_agent.provider_type.value
        schema_model, _skip_retrieval, schema_budget, schema_downgrade = (
            self._resolve_stage_model_and_budget(
                "schema_extraction",
                provider,
                list(assessed_documents),
            )
        )
        if schema_budget.status.value == "blocked":
            logging.warning(
                "_run_schema_extraction blocked by budget for field %r", field_name
            )
            return []

        # ── Build (chunk_id, text) context ───────────────────────────────────
        chunk_context: list[tuple[str, str]] = []
        if retrieved_chunks_map:
            for hits in retrieved_chunks_map.values():
                for r in hits:
                    chunk_context.append((r.chunk.chunk_id, r.chunk.content))
        else:
            for doc in assessed_documents:
                if doc.accepted_for_analysis:
                    text = (doc.full_text or doc.text)[:6000]
                    chunk_context.append((doc.url, text))

        # ── Build provider callable using budget-resolved model ──────────────
        # Accumulate per-call token counts so the FinOps finally block can
        # report pre-summed totals via _record_stage_finops_from_tokens.
        # Using pre-summed counts avoids the double-scaling that occurs when
        # concatenated text is passed to _record_stage_finops, which treats
        # its input_text as a single call's text and then multiplies by
        # call_count (see estimate_stage_cost docstring).
        _input_tokens_total: list[int] = []
        _output_tokens_total: list[int] = []

        async def _provider_fn(prompt: str) -> str:
            _input_tokens_total.append(estimate_tokens(prompt))
            if provider == "openai":
                from openai import AsyncOpenAI  # noqa: PLC0415
                client = AsyncOpenAI(api_key=self.settings.openai_api_key)
                response = await client.responses.create(
                    model=schema_model, input=prompt
                )
                text = response.output_text
            elif provider == "anthropic":
                import anthropic  # noqa: PLC0415
                client = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
                message = await client.messages.create(
                    model=schema_model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = message.content[0].text
            else:
                raise RuntimeError(
                    f"Unsupported provider for schema extraction: {provider}"
                )
            _output_tokens_total.append(estimate_tokens(text))
            return text

        rc = self.settings.retrieval
        extractor = SchemaExtractor(
            provider_fn=_provider_fn,
            max_repair_attempts=rc.schema_repair_max_attempts,
        )

        span_ctx = tracer.span("schema_extraction", provider=provider, input_count=len(chunk_context)) if tracer is not None else _NullSpan()
        with span_ctx as span:
            try:
                result = await extractor.extract(field_name, chunk_context)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "_run_schema_extraction failed for field %r: %s", field_name, exc
                )
                return []
            finally:
                # Record FinOps using pre-summed per-call token counts so that
                # repair loops are not double-counted.  _record_stage_finops
                # would multiply token counts by call_count; using
                # _record_stage_finops_from_tokens with already-summed totals
                # avoids that double-scaling.
                if self.settings.finops.enabled and _input_tokens_total:
                    self._record_stage_finops_from_tokens(
                        span,
                        stage="schema_extraction",
                        provider=provider,
                        model_name=schema_model,
                        total_input_tokens=sum(_input_tokens_total),
                        total_output_tokens=sum(_output_tokens_total),
                        call_count=len(_input_tokens_total),
                        downgrade_applied=schema_downgrade,
                    )
        return [result] if result is not None else []

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

            # --- Guardrails: run after review gate, still inside span ---
            guardrails_report = run_guardrails(
                synthesis=synthesis,
                analysis_reports=reports,
                parsed_documents=parsed_documents,
                overall_confidence=overall_confidence,
                floor=self.settings.guardrails.confidence_floor,
                entities=self.settings.guardrails.pii_entities,
            )
            if not guardrails_report.passed:
                decision = ReviewDecision.AUTO_REJECT
                gate_reason = guardrails_report.failure_summary()
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
            retrieval_degraded=self._frc.retrieval_degraded,
            guardrails_report=guardrails_report,
        )
