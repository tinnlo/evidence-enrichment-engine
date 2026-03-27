"""Replay analysis agent."""

from __future__ import annotations

from evidence_enrichment.core.models.contracts import AnalysisReport, FactClaim, ParsedDocument
from evidence_enrichment.core.models.enums import ProviderType
from evidence_enrichment.core.providers.base import AnalysisAgent


class ReplayAnalysisAgent(AnalysisAgent):
    provider_type = ProviderType.REPLAY

    def __init__(self, bundle: dict):
        self.bundle = bundle

    async def analyze(self, document: ParsedDocument, field_name: str, company_name: str) -> AnalysisReport:
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

