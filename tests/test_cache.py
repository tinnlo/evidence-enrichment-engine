"""Unit tests for Redis cache implementation."""

import pytest
from datetime import datetime, timezone, timedelta

from evidence_enrichment.cache.client import RedisCache
from evidence_enrichment.cache.keys import generate_fetch_key, generate_assessment_key
from evidence_enrichment.cache.models import (
    CachedFetchResult,
    CachedAssessmentResult,
    is_stale,
    FETCH_STALE_THRESHOLD,
)


@pytest.fixture
def redis_cache():
    """Fixture that skips if Redis unavailable."""
    cache = RedisCache(host="localhost", port=6379)
    if not cache.is_available():
        pytest.skip("Redis not available")
    yield cache
    cache.close()


class TestCacheKeys:
    """Test cache key generation with mode isolation."""

    def test_fetch_key_includes_mode(self):
        """Verify fetch keys include mode for replay/live isolation."""
        url = "https://example.com/page"

        live_key = generate_fetch_key(url, "live")
        replay_key = generate_fetch_key(url, "replay")
        auto_key = generate_fetch_key(url, "auto")

        # Keys should be different for different modes
        assert live_key != replay_key
        assert live_key != auto_key
        assert replay_key != auto_key

        # Keys should contain mode
        assert "live" in live_key
        assert "replay" in replay_key
        assert "auto" in auto_key

    def test_fetch_key_determinism(self):
        """Verify same input produces same key."""
        url = "https://example.com/page"
        mode = "live"

        key1 = generate_fetch_key(url, mode)
        key2 = generate_fetch_key(url, mode)

        assert key1 == key2

    def test_assessment_key_includes_mode(self):
        """Verify assessment keys include mode for replay/live isolation."""
        content_hash = "abc123"
        company = "TestCorp"
        url = "https://example.com"

        live_key = generate_assessment_key(content_hash, company, "live", url)
        replay_key = generate_assessment_key(content_hash, company, "replay", url)

        # Keys should be different for different modes
        assert live_key != replay_key

        # Keys should contain mode
        assert "live" in live_key
        assert "replay" in replay_key

    def test_assessment_key_determinism(self):
        """Verify same input produces same key."""
        content_hash = "abc123"
        company = "TestCorp"
        mode = "live"
        url = "https://example.com"

        key1 = generate_assessment_key(content_hash, company, mode, url)
        key2 = generate_assessment_key(content_hash, company, mode, url)

        assert key1 == key2

    def test_assessment_key_includes_url(self):
        """Same content at different URLs should have different cache keys."""
        content_hash = "abc123"
        company = "TestCorp"
        mode = "live"

        key1 = generate_assessment_key(content_hash, company, mode, "https://sec.gov/about")
        key2 = generate_assessment_key(content_hash, company, mode, "https://spam.com/about")

        assert key1 != key2
        assert "v2" in key1  # Verify version bump


class TestStaleness:
    """Test staleness detection logic."""

    def test_is_stale_fresh(self):
        """Verify fresh entries are not marked stale."""
        now = datetime.now(timezone.utc)
        assert not is_stale(now, FETCH_STALE_THRESHOLD)

    def test_is_stale_old(self):
        """Verify old entries are marked stale."""
        old = datetime.now(timezone.utc) - timedelta(hours=13)
        assert is_stale(old, FETCH_STALE_THRESHOLD)

    def test_is_stale_boundary(self):
        """Verify staleness at threshold boundary."""
        # Just under threshold - not stale
        almost_stale = datetime.now(timezone.utc) - timedelta(seconds=FETCH_STALE_THRESHOLD - 1)
        assert not is_stale(almost_stale, FETCH_STALE_THRESHOLD)

        # Just over threshold - stale
        just_stale = datetime.now(timezone.utc) - timedelta(seconds=FETCH_STALE_THRESHOLD + 1)
        assert is_stale(just_stale, FETCH_STALE_THRESHOLD)


@pytest.mark.asyncio
class TestRedisCache:
    """Test Redis cache client."""

    async def test_cache_miss(self, redis_cache):
        """Test cache miss returns None."""
        key = "test:miss:key"
        value, cached_at = redis_cache.get(key)
        assert value is None
        assert cached_at is None

    async def test_cache_roundtrip(self, redis_cache):
        """Test writing and reading from cache."""
        key = "test:roundtrip:key"
        doc = CachedFetchResult(
            url="https://example.com",
            final_url="https://example.com",
            title="Test",
            content_type="text/html",
            body="<html>test</html>",
            provider="test",
        )

        # Write to cache
        success = redis_cache.set(key, doc, ttl_seconds=60)
        if not success:
            pytest.skip("Redis not available for cache operations")
        assert success

        # Read from cache
        cached_value, cached_at = redis_cache.get(key)
        assert cached_value is not None
        assert cached_at is not None
        assert cached_value.url == doc.url
        assert cached_value.body == doc.body

    async def test_cache_ttl_expiration(self, redis_cache):
        """Test cache entries expire after TTL."""
        import time

        key = "test:ttl:key"
        doc = CachedFetchResult(
            url="https://example.com",
            final_url="https://example.com",
            title="Test",
            content_type="text/html",
            body="<html>test</html>",
            provider="test",
        )

        # Write with 1 second TTL
        success = redis_cache.set(key, doc, ttl_seconds=1)
        if not success:
            pytest.skip("Redis not available for cache operations")

        # Should exist immediately
        value, _ = redis_cache.get(key)
        assert value is not None

        # Wait for expiration
        time.sleep(2)

        # Should be gone
        value, _ = redis_cache.get(key)
        assert value is None

    async def test_cache_graceful_degradation(self):
        """Test cache fails gracefully when Redis unavailable."""
        # Connect to invalid host
        cache = RedisCache(host="invalid-host", port=9999, socket_connect_timeout=0.1)

        # Initially available (circuit breaker not tripped yet)
        assert cache.is_available()

        # Get should return None without raising (and trip circuit breaker)
        value, cached_at = cache.get("test:key")
        assert value is None
        assert cached_at is None

        # Circuit breaker should now be tripped
        assert not cache.is_available()

        # Set should return False without raising
        doc = CachedFetchResult(
            url="https://example.com",
            final_url="https://example.com",
            title="Test",
            content_type="text/html",
            body="test",
            provider="test",
        )
        success = cache.set("test:key", doc, ttl_seconds=60)
        assert not success


@pytest.mark.asyncio
class TestReplayLiveIsolation:
    """Test that replay and live caches don't cross-contaminate."""

    async def test_fetch_replay_live_isolation(self, redis_cache):
        """Verify replay and live fetch caches are isolated."""
        url = "https://example.com/test"

        # Create live cache entry
        live_key = generate_fetch_key(url, "live")
        live_doc = CachedFetchResult(
            url=url,
            final_url=url,
            title="Live Doc",
            content_type="text/html",
            body="<html>live</html>",
            provider="live",
        )
        redis_cache.set(live_key, live_doc, ttl_seconds=60)

        # Skip if cache set failed
        if not redis_cache.is_available():
            pytest.skip("Redis not available for cache operations")

        # Verify replay cache is empty
        replay_key = generate_fetch_key(url, "replay")
        replay_cached, _ = redis_cache.get(replay_key)
        assert replay_cached is None

        # Verify live cache has the entry
        live_cached, _ = redis_cache.get(live_key)
        assert live_cached is not None
        assert live_cached.body == "<html>live</html>"

    async def test_assessment_replay_live_isolation(self, redis_cache):
        """Verify replay and live assessment caches are isolated."""
        content_hash = "test123"
        company = "TestCorp"
        url = "https://example.com"

        # Create live cache entry
        live_key = generate_assessment_key(content_hash, company, "live", url)
        live_result = CachedAssessmentResult(
            url=url,
            title="Test",
            content_type="text/html",
            text="test content",
            excerpt="test",
            document_type="company_page",
            entity_match_score=0.9,
            source_authority_score=0.8,
            freshness_score=0.7,
            accepted_for_analysis=True,
            rejection_reason=None,
        )
        redis_cache.set(live_key, live_result, ttl_seconds=60)

        # Skip if cache set failed
        if not redis_cache.is_available():
            pytest.skip("Redis not available for cache operations")

        # Verify replay cache is empty
        replay_key = generate_assessment_key(content_hash, company, "replay", url)
        replay_cached, _ = redis_cache.get(replay_key)
        assert replay_cached is None

        # Verify live cache has the entry
        live_cached, _ = redis_cache.get(live_key)
        assert live_cached is not None
        assert live_cached.entity_match_score == 0.9
