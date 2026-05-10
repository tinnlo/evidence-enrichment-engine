"""Cache-aside middleware for fetch and assessment stages."""

import logging
from datetime import datetime, timezone
from typing import Any

from .client import CacheClient
from .keys import generate_fetch_key, generate_assessment_key
from .models import (
    CachedFetchResult,
    CachedAssessmentResult,
    CacheMetadata,
    FETCH_STALE_THRESHOLD,
    ASSESSMENT_STALE_THRESHOLD,
    is_stale,
)

logger = logging.getLogger(__name__)


async def fetch_with_cache(
    url: str,
    mode: str,
    cache: CacheClient,
    ttl_seconds: int,
    fetch_fn,
) -> tuple[Any, CacheMetadata]:
    """Fetch with cache-aside pattern.

    Args:
        url: URL to fetch
        mode: Execution mode (live/replay/auto)
        cache: Cache client
        ttl_seconds: Cache TTL
        fetch_fn: Async function to call on cache miss (returns RetrievedDocument)

    Returns:
        (document, cache_metadata) tuple
    """
    # Handle disabled cache
    if cache is None:
        logger.info(f"Cache DISABLED for {url}")
        doc = await fetch_fn()
        metadata = CacheMetadata(
            status="disabled",
            age_seconds=None,
            ttl_seconds=ttl_seconds,
            key=None,
        )
        return doc, metadata

    key = generate_fetch_key(url, mode)

    # Try cache first
    cached_value, cached_at = cache.get(key)

    if cached_value is not None and cached_at is not None:
        # Cache hit - check staleness
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        stale = is_stale(cached_at, FETCH_STALE_THRESHOLD)

        if stale:
            # Stale entry - refetch from network
            logger.info(f"Cache STALE for {url} (age: {age:.1f}s) - refetching")
            try:
                doc = await fetch_fn()

                # Update cache with fresh result
                cached_result = CachedFetchResult(
                    url=doc.url,
                    final_url=doc.final_url,
                    title=doc.title,
                    content_type=doc.content_type,
                    body=doc.body,
                    provider=doc.provider,
                    fetch_success=doc.fetch_success,
                    error=doc.error,
                )
                cache.set(key, cached_result, ttl_seconds)

                metadata = CacheMetadata(
                    status="stale",
                    age_seconds=age,
                    ttl_seconds=ttl_seconds,
                    key=key,
                )
                return doc, metadata

            except Exception as e:
                logger.error(f"Refetch failed for stale entry {url}: {e}")
                metadata = CacheMetadata(status="error", key=key)
                raise

        # Fresh cache hit
        logger.info(f"Cache HIT for {url} (age: {age:.1f}s)")

        metadata = CacheMetadata(
            status="hit",
            age_seconds=age,
            ttl_seconds=ttl_seconds,
            key=key,
        )

        # Convert CachedFetchResult back to RetrievedDocument
        from evidence_enrichment.core.models.contracts import RetrievedDocument

        doc = RetrievedDocument(
            url=cached_value.url,
            final_url=cached_value.final_url,
            title=cached_value.title,
            content_type=cached_value.content_type,
            body=cached_value.body,
            provider=cached_value.provider,
            fetch_success=cached_value.fetch_success,
            error=cached_value.error,
        )
        return doc, metadata

    # Cache miss - fetch from network
    logger.info(f"Cache MISS for {url}")

    try:
        doc = await fetch_fn()

        # Cache the result
        cached_result = CachedFetchResult(
            url=doc.url,
            final_url=doc.final_url,
            title=doc.title,
            content_type=doc.content_type,
            body=doc.body,
            provider=doc.provider,
            fetch_success=doc.fetch_success,
            error=doc.error,
        )
        cache.set(key, cached_result, ttl_seconds)

        metadata = CacheMetadata(
            status="miss",
            age_seconds=None,
            ttl_seconds=ttl_seconds,
            key=key,
        )
        return doc, metadata

    except Exception as e:
        logger.error(f"Fetch failed for {url}: {e}")
        metadata = CacheMetadata(status="error", key=key)
        raise


async def assess_with_cache(
    parsed_doc,
    company_name: str,
    mode: str,
    cache: CacheClient,
    ttl_seconds: int,
    assess_fn,
) -> tuple[Any, CacheMetadata]:
    """Assess with cache-aside pattern.

    Args:
        parsed_doc: ParsedDocument to assess
        company_name: Company name for entity matching
        mode: Execution mode (live/replay/auto)
        cache: Cache client
        ttl_seconds: Cache TTL
        assess_fn: Function to call on cache miss (returns ParsedDocument)

    Returns:
        (assessed_document, cache_metadata) tuple
    """
    # Handle disabled cache
    if cache is None:
        logger.info(f"Assessment cache DISABLED for {parsed_doc.url}")
        doc = assess_fn(parsed_doc)
        metadata = CacheMetadata(
            status="disabled",
            age_seconds=None,
            ttl_seconds=ttl_seconds,
            key=None,
        )
        return doc, metadata

    content_hash = parsed_doc.content_hash or ""
    key = generate_assessment_key(content_hash, company_name, mode, parsed_doc.url)

    # Try cache first
    cached_value, cached_at = cache.get(key)

    if cached_value is not None and cached_at is not None:
        # Cache hit - check staleness
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        stale = is_stale(cached_at, ASSESSMENT_STALE_THRESHOLD)

        status = "stale" if stale else "hit"
        logger.info(f"Assessment cache {status.upper()} for {parsed_doc.url} (age: {age:.1f}s)")

        metadata = CacheMetadata(
            status=status,
            age_seconds=age,
            ttl_seconds=ttl_seconds,
            key=key,
        )

        # Reconstruct ParsedDocument with cached scores
        from evidence_enrichment.core.models.contracts import ParsedDocument
        from evidence_enrichment.core.models.enums import DocumentType

        assessed_doc = ParsedDocument(
            url=cached_value.url,
            title=cached_value.title,
            content_type=cached_value.content_type,
            text=cached_value.text,
            excerpt=cached_value.excerpt,
            document_type=DocumentType(cached_value.document_type),
            entity_match_score=cached_value.entity_match_score,
            source_authority_score=cached_value.source_authority_score,
            freshness_score=cached_value.freshness_score,
            accepted_for_analysis=cached_value.accepted_for_analysis,
            rejection_reason=cached_value.rejection_reason,
            full_text=cached_value.full_text,
            content_hash=cached_value.content_hash,
            mime_type=cached_value.mime_type,
        )
        return assessed_doc, metadata

    # Cache miss - run assessment
    logger.info(f"Assessment cache MISS for {parsed_doc.url}")

    assessed_doc = assess_fn(parsed_doc)

    # Cache the result
    cached_result = CachedAssessmentResult(
        url=assessed_doc.url,
        title=assessed_doc.title,
        content_type=assessed_doc.content_type,
        text=assessed_doc.text,
        excerpt=assessed_doc.excerpt,
        document_type=assessed_doc.document_type.value,
        entity_match_score=assessed_doc.entity_match_score,
        source_authority_score=assessed_doc.source_authority_score,
        freshness_score=assessed_doc.freshness_score,
        accepted_for_analysis=assessed_doc.accepted_for_analysis,
        rejection_reason=assessed_doc.rejection_reason,
        full_text=assessed_doc.full_text,
        content_hash=assessed_doc.content_hash,
        mime_type=assessed_doc.mime_type,
    )
    cache.set(key, cached_result, ttl_seconds)

    metadata = CacheMetadata(
        status="miss",
        age_seconds=None,
        ttl_seconds=ttl_seconds,
        key=key,
    )
    return assessed_doc, metadata
