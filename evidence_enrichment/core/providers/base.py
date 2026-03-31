"""Provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from evidence_enrichment.core.models.contracts import AnalysisReport, ParsedDocument, SearchQueryPlan, SearchResult, SynthesisResult
from evidence_enrichment.core.models.enums import ProviderType

if TYPE_CHECKING:
    from evidence_enrichment.core.retrieval.models import RetrievalResult


class SearchProvider(ABC):
    provider_type: ProviderType

    @abstractmethod
    async def search(self, plan: SearchQueryPlan) -> list[SearchResult]:
        """Return search results for the plan."""


class AnalysisAgent(ABC):
    provider_type: ProviderType

    @abstractmethod
    async def analyze(
        self,
        document: ParsedDocument,
        field_name: str,
        company_name: str,
        retrieved_chunks: "list[RetrievalResult] | None" = None,
    ) -> AnalysisReport:
        """Return fact claims for a parsed document.

        Parameters
        ----------
        document:
            The parsed document to analyse.
        field_name:
            The field being enriched (e.g. ``hq_country``).
        company_name:
            The company display name for entity matching.
        retrieved_chunks:
            Optional list of RAG-retrieved chunks to include in the analysis
            prompt.  When provided the agent should prefer these chunks over
            the raw document text.  Pass ``None`` (default) to fall back to
            ``document.text[:6000]``.
        """


class SynthesisAgent(ABC):
    provider_type: ProviderType

    @abstractmethod
    async def synthesize(self, claims: list, field_name: str, company_name: str) -> SynthesisResult:
        """Resolve a final value from claims."""

