"""Safe document fetching."""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import httpx

from evidence_enrichment.core.models.contracts import RetrievedDocument, SearchResult


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style.*?</style>", re.IGNORECASE | re.DOTALL)


class DocumentFetcher:
    async def fetch(self, result: SearchResult) -> RetrievedDocument:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(
                result.url,
                headers={"User-Agent": "evidence-enrichment-engine/0.1"},
            )
            response.raise_for_status()
            body = response.text
            return RetrievedDocument(
                url=result.url,
                final_url=str(response.url),
                title=result.title,
                content_type=response.headers.get("content-type", "text/html"),
                body=body,
                provider=result.provider.value,
            )


def html_to_text(body: str) -> str:
    cleaned = _SCRIPT_RE.sub(" ", body)
    cleaned = _STYLE_RE.sub(" ", cleaned)
    cleaned = _TAG_RE.sub(" ", cleaned)
    return " ".join(html.unescape(cleaned).split()).strip()


def registrable_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    return domain.replace("www.", "")

