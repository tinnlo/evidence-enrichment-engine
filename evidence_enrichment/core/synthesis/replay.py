"""Replay synthesis agent."""

from __future__ import annotations

from evidence_enrichment.core.models.contracts import FactClaim, SynthesisResult
from evidence_enrichment.core.models.enums import ProviderType
from evidence_enrichment.core.providers.base import SynthesisAgent


class ReplaySynthesisAgent(SynthesisAgent):
    provider_type = ProviderType.REPLAY

    def __init__(self, bundle: dict):
        self.bundle = bundle

    async def synthesize(self, claims: list[FactClaim], field_name: str, company_name: str) -> SynthesisResult:
        synthesis = self.bundle.get("synthesis", {})
        return SynthesisResult(field_name=field_name, **synthesis)

