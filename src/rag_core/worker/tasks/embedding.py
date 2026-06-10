"""Embedding task — dedicated embedding for decoupled workflows.

Task ``rag.embed_chunks`` allows embedding chunks independently from the
ingestion pipeline (e.g., for re-embedding with a different model).
"""

from __future__ import annotations

import logging
from typing import Any

from celery import shared_task
from sqlalchemy.orm import Session

from rag_core.api.config import get_service_settings
from rag_core.db.database import SessionLocal
from rag_core.db.models.chunks import RagChunk
from rag_core.indexing.pipeline import compute_checksum, prepare_chunks_for_indexing
from rag_core.worker.rate_limiter import EmbeddingRateLimiter, build_redis_client

logger = logging.getLogger(__name__)


def _get_rate_limiter() -> EmbeddingRateLimiter:
    try:
        settings = get_service_settings()
        client = build_redis_client(settings)
        if client is not None:
            from rag_core.worker.rate_limiter import RateLimitConfig

            return EmbeddingRateLimiter(redis_client=client, config=RateLimitConfig())
    except Exception:
        pass
    return EmbeddingRateLimiter()


@shared_task(
    bind=True,
    name="rag.embed_chunks",
    queue="rag.embedding",
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
def embed_chunks(
    self,
    chunk_ids: list[str],
    *,
    provider_name: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Embed a batch of already-chunked text fragments.

    Args:
        chunk_ids: List of ``rag_chunks.id`` UUIDs to embed.
        provider_name: Override embedding provider.
        model: Override embedding model.

    Returns:
        Dict with ``chunks_embedded`` count.
    """
    settings = get_service_settings()
    db: Session = SessionLocal()
    rate_limiter = _get_rate_limiter()

    try:
        chunks = db.query(RagChunk).filter(RagChunk.id.in_(chunk_ids)).all()
        if not chunks:
            return {"chunks_embedded": 0}

        # We need chunk content to embed — but rag_chunks doesn't store content.
        # Content is in Qdrant payload or re-derived from source.
        # For Phase 5, this task is a placeholder that defers to the full
        # ingest_document task for reindexing the source document.
        logger.warning(
            "embed_chunks: content not available in DB chunks (%d chunks). "
            "Use rag.reindex_document for full re-embedding from source.",
            len(chunks),
        )
        return {"chunks_embedded": 0, "warning": "Content not available in DB; use reindex_document."}
    except Exception as exc:
        logger.exception("embed_chunks failed: %s", exc)
        raise
    finally:
        db.close()
