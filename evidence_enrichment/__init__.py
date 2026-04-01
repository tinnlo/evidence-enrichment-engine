"""Public package surface for evidence_enrichment."""

from evidence_enrichment.config.settings import Settings
from evidence_enrichment.core.models.contracts import (
    AnalysisReport,
    EnrichmentSource,
    FactClaim,
    ParsedDocument,
    PipelineRunResult,
    SearchQueryPlan,
    SearchResult,
    SynthesisResult,
)
from evidence_enrichment.core.models.enums import (
    DocumentType,
    ProviderType,
    ReviewDecision,
    SourceType,
)
from evidence_enrichment.core.enrichers.base import BaseEnricher
from evidence_enrichment.core.enrichers.hq_country import HeadquartersCountryEnricher
from evidence_enrichment.pipeline.coordinator import EvidenceCoordinator

__all__ = [
    "AnalysisReport",
    "BaseEnricher",
    "DocumentType",
    "EnrichmentSource",
    "EvidenceCoordinator",
    "FactClaim",
    "HeadquartersCountryEnricher",
    "ParsedDocument",
    "PipelineRunResult",
    "ProviderType",
    "ReviewDecision",
    "SearchQueryPlan",
    "SearchResult",
    "Settings",
    "SourceType",
    "SynthesisResult",
]


# MCP server symbols — only available when the [mcp] extra is installed.
# Import lazily to avoid a hard dependency on the `mcp` package.
def __getattr__(name: str):  # noqa: ANN001, ANN201
    _mcp_symbols = {
        "mcp",
        "ClaimsResult",
        "ScenarioComparison",
        "ScenarioInfo",
        "SynthesisSummary",
    }
    if name in _mcp_symbols:
        from evidence_enrichment import mcp_server as _ms

        return getattr(_ms, name)
    raise AttributeError(f"module 'evidence_enrichment' has no attribute {name!r}")
