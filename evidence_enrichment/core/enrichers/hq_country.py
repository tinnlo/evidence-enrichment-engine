"""HQ country enricher driven by evidence collection and agent stages."""

from __future__ import annotations

import re
from typing import Any

from evidence_enrichment.core.enrichers.base import BaseEnricher
from evidence_enrichment.core.models.contracts import SearchQueryPlan, SynthesisResult


class HeadquartersCountryEnricher(BaseEnricher):
    field_name = "hq_country"

    def build_query_plan(self, entity: dict[str, Any]) -> SearchQueryPlan:
        entity_id = str(entity.get("entity_id") or entity.get("id") or "unknown")
        name = str(entity.get("name") or entity.get("company_name") or "").strip()
        website = str(entity.get("website") or "").strip()
        legal_suffix = str(entity.get("legal_suffix") or "").strip()
        domain_hints = [website] if website else []
        primary_query = f"\"{name}\" official website headquarters"
        variants = [f"\"{name}\" company headquarters", f"\"{name}\" about us"]
        if legal_suffix:
            variants.append(f"\"{name}\" \"{legal_suffix}\" headquarters")
        return SearchQueryPlan(
            field_name=self.field_name,
            entity_id=entity_id,
            primary_query=primary_query,
            query_variants=variants,
            domain_hints=domain_hints,
            metadata={"company_name": name, "website": website, "legal_suffix": legal_suffix},
        )

    def retrieval_query(self, entity: dict[str, Any]) -> str:
        name = str(entity.get("name") or entity.get("company_name") or "").strip()
        return f"What country is {name} headquartered in?"

    def validate_synthesis(self, synthesis: SynthesisResult) -> SynthesisResult:
        value = (synthesis.normalized_value or synthesis.value or "").strip().upper()
        if value and re.fullmatch(r"[A-Z]{3}", value):
            synthesis.value = value
            synthesis.normalized_value = value
            return synthesis
        synthesis.value = None
        synthesis.normalized_value = None
        synthesis.reasoning = f"{synthesis.reasoning} Validation failed: expected ISO3 country code.".strip()
        synthesis.synthesis_confidence = min(synthesis.synthesis_confidence, 0.49)
        return synthesis

