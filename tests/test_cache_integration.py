"""Integration tests for cache middleware and policy integration."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from evidence_enrichment.cache.middleware import fetch_with_cache, assess_with_cache
from evidence_enrichment.cache.client import RedisCache
from evidence_enrichment.cache.keys import generate_fetch_key, generate_assessment_key
from evidence_enrichment.cache.models import (
    CachedFetchResult,
    CachedAssessmentResult,
    FETCH_STALE_THRESHOLD,
)
from evidence_enrichment.core.models.contracts import RetrievedDocument


@pytest.fixture
def redis_cache():
    """Fixture that skips if Redis unavailable."""
    cache = RedisCache(host="localhost", port=6379)
    if not cache.is_available():
        pytest.skip("Redis not available")

    # Clear test keys before each test
    yield cache
    cache.close()


@pytest.mark.asyncio
class TestFetchWithCache:
    """Test fetch_with_cache middleware."""

    async def test_cache_miss_calls_fetch_fn(self, redis_cache):
        """Verify cache miss calls fetch function."""
        url = "https://example.com/test"
        mode = "live"

        # Mock fetch function
        fetch_fn = AsyncMock(return_value=RetrievedDocument(
            url=url,
            final_url=url,
            title="Test Page",
            content_type="text/html",
            body="<html>test</html>",
            provider="test",
        ))

        # Call middleware
        result, metadata = await fetch_with_cache(
            url=url,
            mode=mode,
            fetch_fn=fetch_fn,
            cache=redis_cache,
            ttl_seconds=60,
        )

        # Verify fetch was called
        fetch_fn.assert_called_once()

        # Verify result
        assert result.url == url
        assert result.body == "<html>test</html>"

        # Verify metadata
        assert metadata.status == "miss"
        assert metadata.key is not None
        assert metadata.age_seconds is None

    async def test_cache_hit_skips_fetch_fn(self, redis_cache):
        """Verify cache hit skips fetch function."""
        url = "https://example.com/test"
        mode = "live"

        # Pre-populate cache
        key = generate_fetch_key(url, mode)
        cached_doc = CachedFetchResult(
            url=url,
            final_url=url,
            title="Cached Page",
            content_type="text/html",
            body="<html>cached</html>",
            provider="cache",
        )
        redis_cache.set(key, cached_doc, ttl_seconds=60)

        # Skip if cache set failed (Redis not actually available)
        if not redis_cache.is_available():
            pytest.skip("Redis became unavailable during test setup")

        # Mock fetch function (should not be called)
        fetch_fn = AsyncMock()

        # Call middleware
        result, metadata = await fetch_with_cache(
            url=url,
            mode=mode,
            fetch_fn=fetch_fn,
            cache=redis_cache,
            ttl_seconds=60,
        )

        # Verify fetch was NOT called
        fetch_fn.assert_not_called()

        # Verify cached result returned
        assert result.body == "<html>cached</html>"

        # Verify metadata
        assert metadata.status == "hit"
        assert metadata.age_seconds is not None
        assert metadata.age_seconds >= 0

    async def test_stale_cache_calls_fetch_fn(self, redis_cache):
        """Verify stale cache triggers fetch."""
        url = "https://example.com/test"
        mode = "live"

        # Pre-populate cache with old entry using proper cache.set() then manually backdate
        key = generate_fetch_key(url, mode)
        old_time = datetime.now(timezone.utc) - timedelta(seconds=FETCH_STALE_THRESHOLD + 100)
        cached_doc = CachedFetchResult(
            url=url,
            final_url=url,
            title="Stale Page",
            content_type="text/html",
            body="<html>stale</html>",
            provider="cache",
        )

        # Use cache.set() to store in correct format, then backdate the timestamp
        redis_cache.set(key, cached_doc, ttl_seconds=3600)

        # Manually backdate the cached_at timestamp to make it stale
        try:
            redis_cache._client.hset(key, "cached_at", old_time.isoformat())
        except Exception:
            pytest.skip("Redis not available for manual cache manipulation")

        # Mock fetch function
        fetch_fn = AsyncMock(return_value=RetrievedDocument(
            url=url,
            final_url=url,
            title="Fresh Page",
            content_type="text/html",
            body="<html>fresh</html>",
            provider="test",
        ))

        # Call middleware
        result, metadata = await fetch_with_cache(
            url=url,
            mode=mode,
            fetch_fn=fetch_fn,
            cache=redis_cache,
            ttl_seconds=60,
        )

        # Verify fetch was called
        fetch_fn.assert_called_once()

        # Verify fresh result returned
        assert result.body == "<html>fresh</html>"

        # Verify metadata
        assert metadata.status == "stale"

    async def test_cache_disabled_always_fetches(self):
        """Verify None cache always calls fetch function."""
        url = "https://example.com/test"
        mode = "live"

        # Mock fetch function
        fetch_fn = AsyncMock(return_value=RetrievedDocument(
            url=url,
            final_url=url,
            title="Test Page",
            content_type="text/html",
            body="<html>test</html>",
            provider="test",
        ))

        # Call middleware with no cache
        result, metadata = await fetch_with_cache(
            url=url,
            mode=mode,
            fetch_fn=fetch_fn,
            cache=None,
            ttl_seconds=60,
        )

        # Verify fetch was called
        fetch_fn.assert_called_once()

        # Verify metadata shows disabled
        assert metadata.status == "disabled"


@pytest.mark.asyncio
class TestAssessWithCache:
    """Test assess_with_cache middleware."""

    async def test_cache_miss_calls_assess_fn(self, redis_cache):
        """Verify cache miss calls assessment function."""
        from evidence_enrichment.core.models.contracts import ParsedDocument
        from evidence_enrichment.core.models.enums import DocumentType

        # Create mock ParsedDocument input
        parsed_doc = MagicMock()
        parsed_doc.url = "https://example.com"
        parsed_doc.title = "Test"
        parsed_doc.text = "content"
        parsed_doc.content_hash = "test123"

        company_name = "TestCorp"
        mode = "live"

        # Mock assess function - returns ParsedDocument
        assessed_result = ParsedDocument(
            url=parsed_doc.url,
            title=parsed_doc.title,
            content_type="text/html",
            text=parsed_doc.text,
            excerpt="content",
            document_type=DocumentType.COMPANY_WEBSITE,
            entity_match_score=0.9,
            source_authority_score=0.8,
            freshness_score=0.7,
            accepted_for_analysis=True,
            rejection_reason=None,
        )
        assess_fn = MagicMock(return_value=assessed_result)

        # Call middleware
        result, metadata = await assess_with_cache(
            parsed_doc=parsed_doc,
            company_name=company_name,
            mode=mode,
            assess_fn=assess_fn,
            cache=redis_cache,
            ttl_seconds=60,
        )

        # Verify assess was called
        assess_fn.assert_called_once()

        # Verify result
        assert result.entity_match_score == 0.9

        # Verify metadata
        assert metadata.status == "miss"

    async def test_cache_hit_skips_assess_fn(self, redis_cache):
        """Verify cache hit skips assessment function."""

        # Create mock ParsedDocument
        parsed_doc = MagicMock()
        parsed_doc.url = "https://example.com"
        parsed_doc.title = "Test"
        parsed_doc.text = "content"
        parsed_doc.content_hash = "test123"

        company_name = "TestCorp"
        mode = "live"

        # Pre-populate cache

        key = generate_assessment_key(
            mode=mode,
            content_hash=parsed_doc.content_hash,
            company_name=company_name,
            url=parsed_doc.url,
        )
        cached_result = CachedAssessmentResult(
            url=parsed_doc.url,
            title=parsed_doc.title,
            content_type="text/html",
            text=parsed_doc.text,
            excerpt="cached excerpt",
            document_type="company_page",
            entity_match_score=0.95,
            source_authority_score=0.85,
            freshness_score=0.75,
            accepted_for_analysis=True,
            rejection_reason=None,
        )
        redis_cache.set(key, cached_result, ttl_seconds=60)

        # Skip if cache set failed (Redis not actually available)
        if not redis_cache.is_available():
            pytest.skip("Redis became unavailable during test setup")

        # Mock assess function (should not be called) - use MagicMock not AsyncMock
        assess_fn = MagicMock()

        # Call middleware
        result, metadata = await assess_with_cache(
            parsed_doc=parsed_doc,
            company_name=company_name,
            mode=mode,
            assess_fn=assess_fn,
            cache=redis_cache,
            ttl_seconds=60,
        )

        # Verify assess was NOT called
        assess_fn.assert_not_called()

        # Verify cached result returned
        assert result.entity_match_score == 0.95
        assert result.excerpt == "cached excerpt"

        # Verify metadata
        assert metadata.status == "hit"
