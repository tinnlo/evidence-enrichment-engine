"""Cache metadata models for Redis cache-aside implementation."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


class CachedFetchResult(BaseModel):
    """Wrapper for cached fetch results with metadata."""

    url: str
    final_url: str
    title: str
    content_type: str
    body: str
    provider: str
    fetch_success: bool = True
    error: str | None = None

    # Cache metadata
    cached_at: datetime = Field(default_factory=_utc_now)
    cache_version: str = "v1"


class CachedAssessmentResult(BaseModel):
    """Wrapper for cached assessment results with metadata."""

    url: str
    title: str
    content_type: str
    text: str
    excerpt: str
    document_type: str
    entity_match_score: float
    source_authority_score: float
    freshness_score: float
    accepted_for_analysis: bool
    rejection_reason: str | None
    full_text: str = ""
    content_hash: str = ""
    mime_type: str = ""

    # Cache metadata
    cached_at: datetime = Field(default_factory=_utc_now)
    cache_version: str = "v1"


class CacheMetadata(BaseModel):
    """Cache status metadata for observability."""

    status: str  # "hit" | "miss" | "stale" | "error"
    age_seconds: float | None = None
    ttl_seconds: int | None = None
    key: str | None = None


# Staleness thresholds (50% of TTL)
FETCH_STALE_THRESHOLD = 12 * 3600  # 12 hours (half of 24h TTL)
ASSESSMENT_STALE_THRESHOLD = 3 * 24 * 3600  # 3 days (half of 7d TTL)


def is_stale(cached_at: datetime, threshold_seconds: int) -> bool:
    """Check if cached value exceeds staleness threshold.

    Staleness is separate from TTL:
    - TTL: Redis evicts after expiration
    - Staleness: We warn if age exceeds threshold

    This allows "stale but usable" cache hits to be flagged in traces.
    """
    age = (datetime.now(timezone.utc) - cached_at).total_seconds()
    return age > threshold_seconds
