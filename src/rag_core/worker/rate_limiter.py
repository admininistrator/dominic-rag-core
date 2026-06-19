"""Rate limiter for embedding provider calls.

Implements a simple Redis-backed rate limiter that tracks RPM (requests per
minute) and TPM (tokens per minute) to prevent embedding provider quota
exhaustion.  Only activates for ``embedding_provider=api``; local/ollama
providers bypass rate limiting.

Strategy:
- Track a sliding window of recent calls in Redis with a 60s TTL.
- Before each embedding call, check current usage against configured limits.
- If approaching the limit, sleep (not retry — just wait/pause) until capacity
  frees up.
- Circuit-breaker: after 3 consecutive 429/rate-limit errors from the provider,
  pause embedding queue for 60 seconds.

Configuration via env/settings:
- ``RAG_EMBEDDING_RATE_LIMIT_RPM`` — max requests per minute (default: 60).
- ``RAG_EMBEDDING_RATE_LIMIT_TPM`` — max tokens per minute (default: 100000).
- ``RAG_EMBEDDING_MAX_CONCURRENT_BATCHES`` — max concurrent embedding batches (default: 4).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import redis

logger = logging.getLogger(__name__)

# Defaults matching proposed-worker-model.md
DEFAULT_RPM = 60
DEFAULT_TPM = 100_000
DEFAULT_MAX_CONCURRENT_BATCHES = 4
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 3
DEFAULT_CIRCUIT_BREAKER_COOLDOWN = 60  # seconds

# Redis key prefixes
_REDIS_KEY_PREFIX = "rag_core:rate_limit"


@dataclass
class RateLimitConfig:
    """Rate-limit configuration."""

    rpm: int = DEFAULT_RPM
    tpm: int = DEFAULT_TPM
    max_concurrent_batches: int = DEFAULT_MAX_CONCURRENT_BATCHES
    circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD
    circuit_breaker_cooldown: int = DEFAULT_CIRCUIT_BREAKER_COOLDOWN


@dataclass
class CircuitBreaker:
    """In-process circuit breaker state (not durable across workers)."""

    failures: int = 0
    opened_at: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def is_open(self, cooldown: int) -> bool:
        if self.failures == 0:
            return False
        if self.failures >= DEFAULT_CIRCUIT_BREAKER_THRESHOLD:
            elapsed = time.time() - self.opened_at
            if elapsed < cooldown:
                return True
            # Cooldown expired — half-open, allow one probe
            with self._lock:
                self.failures = 0
                self.opened_at = 0.0
        return False

    def record_failure(self, threshold: int) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= threshold and self.opened_at == 0.0:
                self.opened_at = time.time()
                logger.warning(
                    "Circuit breaker opened after %d consecutive failures",
                    self.failures,
                )

    def reset(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = 0.0


class EmbeddingRateLimiter:
    """Redis-backed rate limiter for embedding provider calls.

    The rate limiter is optional — it activates only when a Redis client is
    provided.  Without Redis, rate limiting is disabled (no-op).

    Args:
        redis_client: A ``redis.Redis`` client. If ``None``, rate limiting is
            disabled and all calls proceed immediately.
        config: Rate-limit threshold configuration.
    """

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        config: RateLimitConfig | None = None,
    ) -> None:
        self._redis = redis_client
        self._config = config or RateLimitConfig()
        self._circuit_breaker = CircuitBreaker()

    @property
    def enabled(self) -> bool:
        return self._redis is not None

    def acquire(self, token_estimate: int = 0) -> float | None:
        """Try to acquire rate-limit capacity.

        Args:
            token_estimate: Estimated tokens in this batch (for TPM tracking).

        Returns:
            ``None`` if capacity is available immediately, or a float sleep
            seconds to wait before retrying.  If the circuit breaker is open
            the returned delay equals the breaker cooldown seconds.
        """
        if not self._redis:
            return None

        if self._circuit_breaker.is_open(self._config.circuit_breaker_cooldown):
            return float(self._config.circuit_breaker_cooldown)

        now = time.time()
        window_key = f"{_REDIS_KEY_PREFIX}:window:{int(now // 60)}"
        try:
            pipe = self._redis.pipeline()
            pipe.incr(f"{window_key}:rpm")
            pipe.expire(f"{window_key}:rpm", 120)
            if token_estimate > 0:
                pipe.incrby(f"{window_key}:tpm", token_estimate)
                pipe.expire(f"{window_key}:tpm", 120)
            results: list[Any] = pipe.execute()
        except redis.exceptions.ConnectionError:
            logger.warning("Redis unreachable for rate limiting; proceeding without rate limiting.")
            return None

        rpm_count = int(results[0] or 0)
        if rpm_count > self._config.rpm:
            logger.debug("Rate limiter: RPM threshold %d exceeded (%d)", self._config.rpm, rpm_count)
            return 1.0

        if token_estimate > 0:
            tpm_count = int(results[2] or 0)
            if tpm_count > self._config.tpm:
                logger.debug("Rate limiter: TPM threshold %d exceeded (%d)", self._config.tpm, tpm_count)
                return 1.0

        return None

    def record_failure(self) -> None:
        """Record a rate-limit / provider failure for circuit-breaking."""
        self._circuit_breaker.record_failure(self._config.circuit_breaker_threshold)

    def record_success(self) -> None:
        """Reset the circuit breaker on successful calls."""
        self._circuit_breaker.reset()


def build_redis_client(settings: Any) -> redis.Redis | None:
    """Build a Redis client from rag-core settings, if available.

    Tries to parse the broker URL for host/port/password. Returns ``None`` if
    the broker URL is not a Redis URL or Redis is unavailable.
    """
    try:
        broker_url = getattr(settings, "celery_broker_url", "") or ""
        if not broker_url.startswith("redis://") and not broker_url.startswith("rediss://"):
            return None
        import redis as _redis

        return _redis.from_url(broker_url, decode_responses=False)
    except Exception:
        logger.warning("Failed to connect to Redis for rate limiting; rate limiter disabled.")
        return None
