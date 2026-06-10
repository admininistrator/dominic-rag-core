"""Worker health check helpers for rag-core Celery worker.

Provides best-effort health probing without requiring a live Redis/Celery
connection (which would fail when ``RAG_CORE_CELERY_ENABLED`` is ``False``).
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


def check_worker_health() -> dict[str, Any]:
    """Best-effort Celery worker health check.

    When Celery is enabled, attempts a Celery ``inspect.ping()`` to verify
    that at least one worker is alive.  Returns a structured dict suitable
    for inclusion in ``/health`` and ``/ready`` responses.

    This function does NOT connect to chat_db or DominicBE databases.

    Returns:
        Dict with ``enabled``, ``ok``, and optional ``latency_ms`` keys.
    """
    from rag_core.api.config import get_service_settings

    settings = get_service_settings()
    if not settings.celery_enabled:
        return {"enabled": False, "ok": True, "note": "Celery worker is disabled; skipping worker health check."}

    started = perf_counter()
    try:
        from rag_core.worker.celery_app import celery_app

        inspect = celery_app.control.inspect(timeout=5.0)
        ping_result = inspect.ping()
        latency_ms = round((perf_counter() - started) * 1000, 2)

        if ping_result:
            worker_count = len(ping_result)
            return {
                "enabled": True,
                "ok": True,
                "workers": worker_count,
                "latency_ms": latency_ms,
            }
        return {
            "enabled": True,
            "ok": False,
            "workers": 0,
            "detail": "No Celery workers responded to ping.",
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        logger.warning("Celery worker health check failed: %s", exc)
        return {
            "enabled": True,
            "ok": False,
            "workers": 0,
            "detail": _sanitize_detail(exc),
            "latency_ms": latency_ms,
        }


def _sanitize_detail(exc: Exception) -> str:
    """Sanitize exception details — never include connection strings or secrets."""
    msg = str(exc).lower()
    # Strip anything that looks like a redis URL or password
    if "redis://" in msg or "password" in msg or "@" in msg:
        return "Connection error (details redacted)."
    return str(exc)[:200]
