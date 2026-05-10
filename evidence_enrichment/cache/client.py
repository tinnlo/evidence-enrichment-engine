"""Redis cache client with connection pooling and graceful degradation."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import redis
from redis.connection import ConnectionPool

from .models import CachedFetchResult, CachedAssessmentResult

logger = logging.getLogger(__name__)


class CacheClient(ABC):
    """Abstract cache client interface."""

    @abstractmethod
    def get(self, key: str) -> tuple[Any | None, datetime | None]:
        """Get value and cached_at timestamp from cache.

        Returns:
            (value, cached_at) tuple, or (None, None) on miss
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Set value in cache with TTL.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if cache is available for use."""
        pass


class RedisCache(CacheClient):
    """Redis cache implementation with connection pooling and circuit breaker.

    Circuit breaker: After first Redis failure, cache is disabled for the
    session to prevent cascading failures. Pipeline continues without cache.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        max_connections: int = 10,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
    ):
        """Initialize Redis cache with connection pool.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional, for authenticated instances)
            max_connections: Max connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
        """
        self._pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        self._available = True

        logger.info(
            f"Redis cache initialized: {host}:{port} db={db} "
            f"pool_size={max_connections} auth={'yes' if password else 'no'}"
        )

    def get(self, key: str) -> tuple[Any | None, datetime | None]:
        """Get value from Redis.

        Returns:
            (parsed_value, cached_at) or (None, None) on miss/error
        """
        if not self._available:
            return None, None

        try:
            raw = self._client.get(key)
            if raw is None:
                return None, None

            data = json.loads(raw)
            cached_at = datetime.fromisoformat(data["cached_at"])
            value = data["value"]

            # Reconstruct Pydantic models if needed
            if data.get("type") == "CachedFetchResult":
                value = CachedFetchResult(**value)
            elif data.get("type") == "CachedAssessmentResult":
                value = CachedAssessmentResult(**value)

            return value, cached_at

        except redis.RedisError as e:
            logger.warning(f"Redis get failed for key={key}: {e}")
            self._available = False  # Circuit breaker
            return None, None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Cache deserialization failed for key={key}: {e}")
            return None, None

    def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Set value in Redis with TTL.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable or Pydantic model)
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self._available:
            return False

        try:
            # Prepare payload with type hint for reconstruction
            payload = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "value": value,
            }

            # Add type hint for Pydantic models
            if isinstance(value, CachedFetchResult):
                payload["type"] = "CachedFetchResult"
                payload["value"] = value.model_dump(mode='json')
            elif isinstance(value, CachedAssessmentResult):
                payload["type"] = "CachedAssessmentResult"
                payload["value"] = value.model_dump(mode='json')

            raw = json.dumps(payload)
            self._client.setex(key, ttl_seconds, raw)
            return True

        except redis.RedisError as e:
            logger.warning(f"Redis set failed for key={key}: {e}")
            self._available = False  # Circuit breaker
            return False
        except (TypeError, ValueError) as e:
            logger.warning(f"Cache serialization failed for key={key}: {e}")
            return False

    def is_available(self) -> bool:
        """Check if cache is available."""
        return self._available

    def close(self):
        """Close connection pool."""
        if self._pool:
            self._pool.disconnect()
            logger.info("Redis connection pool closed")
