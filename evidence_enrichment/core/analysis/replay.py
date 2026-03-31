"""Replay analysis agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evidence_enrichment.core.models.contracts import AnalysisReport, FactClaim, ParsedDocument
from evidence_enrichment.core.models.enums import ProviderType
from evidence_enrichment.core.providers.base import AnalysisAgent

if TYPE_CHECKING:
    from evidence_enrichment.core.retrieval.models import RetrievalResult


class ReplayAnalysisAgent(AnalysisAgent):
    provider_type = ProviderType.REPLAY

    def __init__(self, bundle: dict):
        self.bundle = bundle

    async def analyze(
        self,
        document: ParsedDocument,
        field_name: str,
        company_name: str,
        retrieved_chunks: "list[RetrievalResult] | None" = None,
    ) -> AnalysisReport:
        # Replay always uses pre-recorded reports; retrieved_chunks are ignored
        reports = self.bundle.get("analysis_reports", [])
        for report in reports:
            if report.get("source_url") == document.url:
                claims = [FactClaim(**claim) for claim in report.get("claims", [])]
                return AnalysisReport(
                    source_url=document.url,
                    provider=self.provider_type,
                    claims=claims,
                    reasoning=report.get("reasoning"),
                )
        return AnalysisReport(source_url=document.url, provider=self.provider_type, claims=[], reasoning="No replay analysis available.")

