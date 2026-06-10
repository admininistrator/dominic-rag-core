"""Document ingestion pipeline task.

Task ``rag.ingest_document`` runs the full async pipeline:
parse → normalize → chunk → embed → index.

Idempotency: skips completed/cancelled jobs and already-indexed documents.
Retry: exponential backoff (5s → 10s → 20s), max 3 retries.
Rate limiting: gates embedding provider calls via Redis token bucket.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from celery import shared_task
from celery.exceptions import Retry
from sqlalchemy.orm import Session

from rag_core.api.config import get_service_settings
from rag_core.chunking.custom_pipeline import CustomPipeline
from rag_core.db.database import SessionLocal
from rag_core.db.models.chunks import RagChunk
from rag_core.db.models.documents import RagDocument
from rag_core.db.models.ingestion_jobs import RagIngestionJob
from rag_core.db.models.vector_mappings import RagVectorMapping
from rag_core.indexing.pipeline import compute_checksum, prepare_chunks_for_indexing
from rag_core.parsing.file_extractor import extract_text_from_file
from rag_core.parsing.text_normalizer import normalize_text_for_ingestion
from rag_core.services.collection_registry import get_collection
from rag_core.services.object_storage import ObjectStorageService
from rag_core.vector_store.qdrant_adapter import QdrantAdapter
from rag_core.worker.rate_limiter import EmbeddingRateLimiter, build_redis_client

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_BASE_DELAY = 5
_MAX_DELAY = 20


def _max_retries() -> int:
    import os

    try:
        return int(os.environ.get("RAG_INGESTION_MAX_RETRIES", str(_MAX_RETRIES)))
    except (ValueError, TypeError):
        return _MAX_RETRIES


def _retry_delay(retry_count: int) -> int:
    """Exponential backoff with a cap: 5s → 10s → 20s."""
    import os

    base = _BASE_DELAY
    try:
        base = int(os.environ.get("RAG_INGESTION_RETRY_DELAY_SECONDS", str(_BASE_DELAY)))
    except (ValueError, TypeError):
        pass
    return min(base * (2 ** max(retry_count - 1, 0)), _MAX_DELAY)


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _get_rate_limiter() -> EmbeddingRateLimiter:
    """Build a rate limiter if Redis is available."""
    try:
        settings = get_service_settings()
        client = build_redis_client(settings)
        if client is not None:
            from rag_core.worker.rate_limiter import RateLimitConfig

            config = RateLimitConfig(
                rpm=_env_int("RAG_EMBEDDING_RATE_LIMIT_RPM", 60),
                tpm=_env_int("RAG_EMBEDDING_RATE_LIMIT_TPM", 100000),
            )
            return EmbeddingRateLimiter(redis_client=client, config=config)
    except Exception:
        logger.warning("Failed to build rate limiter; proceeding without rate limiting.")
    return EmbeddingRateLimiter()


def _extract_storage_key(source_uri: str | None) -> str:
    """Parse the object storage key from an s3:// or local:// URI."""
    if not source_uri:
        raise RuntimeError("Document has no source_uri; cannot retrieve source bytes.")
    # s3://bucket/key or local://bucket/key
    if "://" in source_uri:
        parts = source_uri.split("://", 1)[1].split("/", 1)
        if len(parts) >= 2:
            return parts[1]
    return source_uri


# ── Job-state helpers ──────────────────────────────────────────────────────────


def _update_job(db: Session, job_id: str, *, status: str | None = None, step: str | None = None) -> None:
    job = db.query(RagIngestionJob).filter(RagIngestionJob.id == job_id).first()
    if job is None:
        return
    if status is not None:
        job.status = status
    if step is not None:
        job.step_current = step


def _update_job_progress(db: Session, job_id: str, *, step: str, progress: float) -> None:
    job = db.query(RagIngestionJob).filter(RagIngestionJob.id == job_id).first()
    if job is None:
        return
    job.step_current = step
    job.step_progress = min(max(progress, 0.0), 1.0)


def _complete_job(db: Session, job_id: str, document_id: str, chunks_processed: int = 0) -> None:
    job = db.query(RagIngestionJob).filter(RagIngestionJob.id == job_id).first()
    if job is None:
        return
    job.status = "completed"
    job.step_current = "completed"
    job.step_progress = 1.0
    job.chunks_total = chunks_processed
    job.chunks_processed = chunks_processed
    job.completed_at = datetime.now(timezone.utc)


def _fail_job(db: Session, job_id: str, document_id: str, error_code: str, error_message: str) -> None:
    job = db.query(RagIngestionJob).filter(RagIngestionJob.id == job_id).first()
    if job is not None:
        job.status = "failed"
        job.error_code = error_code
        job.error_message = error_message
    doc = db.query(RagDocument).filter(RagDocument.id == document_id).first()
    if doc is not None:
        doc.status = "failed"
    db.commit()


# ── Main task ──────────────────────────────────────────────────────────────────


@shared_task(
    bind=True,
    name="rag.ingest_document",
    queue="rag.ingestion",
    max_retries=_MAX_RETRIES,
    default_retry_delay=_BASE_DELAY,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=_MAX_DELAY,
    retry_jitter=True,
)
def ingest_document(self, document_id: str, job_id: str) -> dict[str, Any]:
    """Run the full ingestion pipeline for a document.

    Pipeline: parse → normalize → chunk → embed → index.

    Args:
        document_id: UUID of the ``rag_documents`` row.
        job_id: UUID of the ``rag_ingestion_jobs`` row.

    Returns:
        Dict with ``document_id``, ``job_id``, ``status``, ``chunks_created``,
        and ``vectors_upserted``.
    """
    settings = get_service_settings()
    db: Session = SessionLocal()
    rate_limiter = _get_rate_limiter()

    try:
        # ── Load document and job ──────────────────────────────────────────
        document = db.query(RagDocument).filter(RagDocument.id == document_id).first()
        if document is None:
            logger.error("Document %s not found; marking job %s as failed.", document_id, job_id)
            _fail_job(db, job_id, document_id, "RAG_DOCUMENT_NOT_FOUND", "Document not found.")
            return {"document_id": document_id, "job_id": job_id, "status": "failed"}

        job = db.query(RagIngestionJob).filter(RagIngestionJob.id == job_id).first()
        if job is None:
            logger.error("Job %s not found.", job_id)
            return {"document_id": document_id, "job_id": job_id, "status": "failed"}

        # ── Idempotency guard ──────────────────────────────────────────────
        if job.status in ("completed", "cancelled"):
            logger.info("Job %s is already %s; skipping.", job_id, job.status)
            return {"document_id": document_id, "job_id": job_id, "status": job.status}

        if document.status == "indexed":
            logger.info("Document %s is already indexed; marking job completed.", document_id)
            _complete_job(db, job_id, document_id, chunks_processed=document.chunk_count or 0)
            return {"document_id": document_id, "job_id": job_id, "status": "completed"}

        # ── Start processing ───────────────────────────────────────────────
        _update_job(db, job_id, status="processing", step="processing")
        document.status = "processing"
        db.commit()

        # ── 1. Parse: retrieve source bytes and extract text ───────────────
        _update_job_progress(db, job_id, step="parsing", progress=0.02)
        db.commit()

        storage_service = ObjectStorageService(settings)
        storage_key = _extract_storage_key(document.source_uri)
        file_bytes = storage_service.read_bytes(storage_key)
        raw_text = extract_text_from_file(
            file_bytes,
            filename=document.title or "document",
            mime_type=document.mime_type,
        )

        # ── 2. Normalize ───────────────────────────────────────────────────
        _update_job_progress(db, job_id, step="normalizing", progress=0.08)
        db.commit()
        clean_text = normalize_text_for_ingestion(raw_text)
        content_checksum = compute_checksum(clean_text)
        logger.info("Document %s: checksum=%s chars=%d", document_id, content_checksum[:12], len(clean_text))

        # ── 3. Chunk ───────────────────────────────────────────────────────
        _update_job_progress(db, job_id, step="chunking", progress=0.12)
        db.commit()
        pipeline = CustomPipeline(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = pipeline.chunk_document(
            clean_text,
            title=document.title,
            source_uri=document.source_uri,
        )
        logger.info("Document %s: produced %d chunks.", document_id, len(chunks))

        if not chunks:
            logger.warning("Document %s produced zero chunks; marking completed.", document_id)
            _complete_job(db, job_id, document_id, chunks_processed=0)
            return {"document_id": document_id, "job_id": job_id, "status": "completed", "chunks_created": 0}

        # Build chunk dicts for the indexing pipeline and persist rag_chunks rows
        chunk_dicts: list[dict[str, Any]] = []
        chunk_rows: list[RagChunk] = []

        for idx, chunk_obj in enumerate(chunks):
            chunk_text = chunk_obj.content
            chunk_hash = compute_checksum(chunk_text)
            chunk_dicts.append(
                {
                    "content": chunk_text,
                    "chunk_index": idx,
                    "metadata_json": {
                        **(getattr(chunk_obj, "metadata_json", None) or {}),
                        "content_hash": chunk_hash,
                    },
                }
            )

            rag_chunk = RagChunk(
                id=uuid.uuid4(),
                document_id=document_id,
                collection_id=document.collection_id,
                chunk_index=idx,
                token_count=None,
                embedding_model=settings.embedding_model,
                content_hash=chunk_hash,
                metadata_json=getattr(chunk_obj, "metadata_json", None),
            )
            db.add(rag_chunk)
            chunk_rows.append(rag_chunk)

        _update_job_progress(db, job_id, step="chunking", progress=0.20)
        db.commit()

        # ── 4. Embed (batched) ± rate limiting ────────────────────────────
        _update_job_progress(db, job_id, step="embedding", progress=0.25)
        db.commit()

        # Rate-limit gate before embedding
        if rate_limiter.enabled:
            delay = rate_limiter.acquire(token_estimate=max(len(clean_text.split()), 1) * len(chunk_dicts))
            if delay is not None:
                logger.info("Rate limiter: waiting %.1fs before embedding.", delay)
                time.sleep(delay)

        embedding_batch_size = settings.embedding_batch_size or 16
        all_prepared: list[dict[str, Any]] = []

        for batch_start in range(0, len(chunk_dicts), embedding_batch_size):
            batch = chunk_dicts[batch_start : batch_start + embedding_batch_size]
            prepared = prepare_chunks_for_indexing(
                document_id=0,  # legacy int placeholder
                checksum=content_checksum,
                chunks=batch,
                provider_name=settings.embedding_provider,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                base_url=settings.embedding_base_url,
                timeout_seconds=settings.embedding_timeout_seconds,
                batch_size=embedding_batch_size,
                api_key=settings.embedding_api_key,
                api_type=settings.embedding_api_type,
                api_version=settings.embedding_api_version,
                store_embeddings_in_metadata=settings.store_embeddings_in_metadata,
            )
            all_prepared.extend(prepared)
            progress = 0.25 + (0.40 * (batch_start + len(batch)) / len(chunk_dicts))
            _update_job_progress(db, job_id, step="embedding", progress=min(progress, 0.65))
            db.commit()

        rate_limiter.record_success()
        logger.info("Document %s: embedded %d chunks.", document_id, len(all_prepared))

        # ── 5. Index: upsert vectors to Qdrant ─────────────────────────────
        _update_job_progress(db, job_id, step="indexing", progress=0.70)
        db.commit()

        collection = get_collection(db, str(document.collection_id))
        qdrant = QdrantAdapter(
            collection=str(collection.name),
            url=settings.vector_store_url,
            api_key=settings.vector_store_api_key,
            timeout_seconds=settings.vector_store_timeout_seconds,
            prefer_grpc=settings.vector_store_prefer_grpc,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
        )

        # Upsert via the adapter's existing method — handles collection
        # auto-creation, dimension validation, and payload formatting.
        qdrant.upsert_document_chunks(
            owner_username=document.owner_username or "",
            document_id=0,
            title=document.title or "",
            source_type=document.source_type or "upload",
            source_uri=document.source_uri or "",
            session_id=None,
            chunk_rows=chunk_rows,
            prepared_chunks=all_prepared,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
        )

        # Persist vector mappings
        for idx, chunk_row in enumerate(chunk_rows):
            prepared = all_prepared[idx] if idx < len(all_prepared) else {}
            vector_id = prepared.get("vector_id", f"local:{document_id}:{idx}:{content_checksum[:12]}")
            mapping = RagVectorMapping(
                id=uuid.uuid4(),
                chunk_id=chunk_row.id,
                collection_name=str(collection.name),
                point_id=str(vector_id),
                embedding_model_id=None,
                status="active",
            )
            db.add(mapping)
            if not chunk_row.vector_id:
                chunk_row.vector_id = str(vector_id)

        _update_job_progress(db, job_id, step="indexing", progress=0.95)
        db.commit()

        # ── 6. Complete ────────────────────────────────────────────────────
        document.chunk_count = len(chunk_rows)
        document.status = "indexed"
        _complete_job(db, job_id, document_id, chunks_processed=len(chunk_rows))
        db.commit()

        logger.info(
            "Document %s: ingestion complete. chunks=%d vectors=%d",
            document_id,
            len(chunk_rows),
            len(all_prepared),
        )
        return {
            "document_id": document_id,
            "job_id": job_id,
            "status": "completed",
            "chunks_created": len(chunk_rows),
            "vectors_upserted": len(all_prepared),
        }

    except Retry:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed for document %s (job %s): %s", document_id, job_id, exc)
        rate_limiter.record_failure()

        try:
            _fail_job(db, job_id, document_id, "RAG_INGESTION_FAILED", str(exc)[:1000])
        except Exception:
            logger.exception("Failed to update job status after ingestion error.")

        retries = self.request.retries
        if retries < _max_retries():
            delay = _retry_delay(retries + 1)
            logger.info("Retrying job %s in %ds (attempt %d/%d)", job_id, delay, retries + 1, _max_retries())
            raise self.retry(exc=exc, countdown=delay, max_retries=_max_retries())
        return {"document_id": document_id, "job_id": job_id, "status": "failed", "error": str(exc)[:500]}
    finally:
        db.close()
