"""Live search provider adapters."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from evidence_enrichment.core.models.contracts import SearchQueryPlan, SearchResult
from evidence_enrichment.core.models.enums import ProviderType
from evidence_enrichment.core.providers.base import SearchProvider


class SerperSearchProvider(SearchProvider):
    provider_type = ProviderType.SERPER

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, plan: SearchQueryPlan) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"q": plan.primary_query, "num": 8},
            )
            response.raise_for_status()
            data = response.json()
        organic = data.get("organic", []) or []
        results: list[SearchResult] = []
        for idx, item in enumerate(organic):
            url = str(item.get("link") or "")
            results.append(
                SearchResult(
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("snippet") or ""),
                    provider=self.provider_type,
                    rank=idx + 1,
                    domain=urlparse(url).netloc.lower(),
                )
            )
        return results


class TavilySearchProvider(SearchProvider):
    provider_type = ProviderType.TAVILY

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, plan: SearchQueryPlan) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": plan.primary_query, "max_results": 8},
            )
            response.raise_for_status()
            data = response.json()
        rows = data.get("results", []) or []
        results: list[SearchResult] = []
        for idx, item in enumerate(rows):
            url = str(item.get("url") or "")
            results.append(
                SearchResult(
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("content") or ""),
                    provider=self.provider_type,
                    rank=idx + 1,
                    domain=urlparse(url).netloc.lower(),
                )
            )
        return results

