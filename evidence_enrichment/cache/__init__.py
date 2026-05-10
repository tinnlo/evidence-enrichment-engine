"""Redis cache-aside implementation for evidence enrichment pipeline.

This module provides bounded caching for fetch and assessment stages with:
- Mode-isolated cache keys (live/replay/auto) to preserve replay determinism
- Connection pooling and circuit breaker for graceful degradation
- Explicit staleness tracking visible in traces
- Cache-aside pattern (not write-through or write-back)
"""

from .client import CacheClient, RedisCache
from .keys import generate_fetch_key, generate_assessment_key
from .middleware import fetch_with_cache, assess_with_cache
from .models import (
    CachedFetchResult,
    CachedAssessmentResult,
    CacheMetadata,
    FETCH_STALE_THRESHOLD,
    ASSESSMENT_STALE_THRESHOLD,
    is_stale,
)

__all__ = [
    "CacheClient",
    "RedisCache",
    "generate_fetch_key",
    "generate_assessment_key",
    "fetch_with_cache",
    "assess_with_cache",
    "CachedFetchResult",
    "CachedAssessmentResult",
    "CacheMetadata",
    "FETCH_STALE_THRESHOLD",
    "ASSESSMENT_STALE_THRESHOLD",
    "is_stale",
]
