"""Provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from evidence_enrichment.core.models.contracts import AnalysisReport, ParsedDocument, SearchQueryPlan, SearchResult, SynthesisResult
from evidence_enrichment.core.models.enums import ProviderType


class SearchProvider(ABC):
    provider_type: ProviderType

    @abstractmethod
    async def search(self, plan: SearchQueryPlan) -> list[SearchResult]:
        """Return search results for the plan."""


class AnalysisAgent(ABC):
    provider_type: ProviderType

    @abstractmethod
    async def analyze(self, document: ParsedDocument, field_name: str, company_name: str) -> AnalysisReport:
        """Return fact claims for a parsed document."""


class SynthesisAgent(ABC):
    provider_type: ProviderType

    @abstractmethod
    async def synthesize(self, claims: list, field_name: str, company_name: str) -> SynthesisResult:
        """Resolve a final value from claims."""

