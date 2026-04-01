"""Evidence-backed enricher contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evidence_enrichment.core.models.contracts import SearchQueryPlan, SynthesisResult


class BaseEnricher(ABC):
    field_name: str = "generic"

    @abstractmethod
    def build_query_plan(self, entity: dict[str, Any]) -> SearchQueryPlan:
        """Build the search plan for the enricher."""

    def retrieval_query(self, entity: dict[str, Any]) -> str:
        """Natural-language query used for RAG retrieval.

        Override in subclasses to provide a domain-specific retrieval prompt.
        Falls back to the primary query from the search plan.
        """
        plan = self.build_query_plan(entity)
        return plan.primary_query

    @abstractmethod
    def validate_synthesis(self, synthesis: SynthesisResult) -> SynthesisResult:
        """Validate and normalize the synthesized output."""

    def replay_slug(self, entity: dict[str, Any]) -> str:
        entity_id = str(entity.get("entity_id") or entity.get("id") or "unknown").lower()
        return f"{entity_id}_{self.field_name}"

