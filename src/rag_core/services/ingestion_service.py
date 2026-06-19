"""Synchronous document ingestion pipeline (runs inline, no Celery dependency).

Extracts the core ingestion logic from the Celery task into a standalone
function so it can be called directly from the upload endpoint when
RAG_CORE_SYNC_INGESTION_ENABLED=true and Celery is disabled.

Backward-compatible: the Celery task delegates to the same function.
No DominicBE files touched. No changes to existing API shapes.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from rag_core.api.config import RagCoreServiceSettings, get_service_settings
from rag_core.chunking.custom_pipeline import CustomPipeline
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


# ── Shared ingestion pipeline ─────────────────────────────────────────────────


def process_document_ingestion(
    document_id: str,
    job_id: str,
    db: Session,
    *,
    settings: RagCoreServiceSettings | None = None,
) -> dict[str, Any]:
    """Run the full ingestion pipeline for a document synchronously.

    Pipeline: parse → normalize → chunk → embed → index.

    Args:
        document_id: UUID of the ``rag_documents`` row.
        job_id: UUID of the ``rag_ingestion_jobs`` row.
        db: SQLAlchemy session (managed by the caller).
        settings: Optional settings override; read from env if omitted.

    Returns:
        Dict with ``document_id``, ``job_id``, ``status``, ``chunks_created``,
        and ``vectors_upserted``.
    """
    resolved_settings = settings or get_service_settings()
    rate_limiter = _get_rate_limiter()

    # ── Load document and job ──────────────────────────────────────────
    try:
        document_raw_id = uuid.UUID(document_id)
    except ValueError:
        return {"document_id": document_id, "job_id": job_id, "status": "failed", "error": "Invalid document_id UUID."}

    try:
        job_raw_id = uuid.UUID(job_id)
    except ValueError:
        return {"document_id": document_id, "job_id": job_id, "status": "failed", "error": "Invalid job_id UUID."}

    document = db.query(RagDocument).filter(RagDocument.id == document_raw_id).first()
    if document is None:
        logger.error("Document %s not found; marking job %s as failed.", document_id, job_id)
        _fail_job(db, job_id, document_id, "RAG_DOCUMENT_NOT_FOUND", "Document not found.")
        return {"document_id": document_id, "job_id": job_id, "status": "failed"}

    job = db.query(RagIngestionJob).filter(RagIngestionJob.id == job_raw_id).first()
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

    try:
        # ── 1. Parse: retrieve source bytes and extract text ───────────────
        _update_job_progress(db, job_id, step="parsing", progress=0.02)
        db.commit()

        storage_service = ObjectStorageService(resolved_settings)
        storage_key = _extract_storage_key(document.source_uri)
        file_bytes = storage_service.read_bytes(storage_key)
        # Use the original source filename from metadata_json (e.g. "report.pdf")
        # so the file extractor can detect the format by extension.
        # Fall back to document.title, then "document".
        source_filename: str = "document"
        raw_meta: dict | None = document.metadata_json
        if raw_meta and isinstance(raw_meta, dict):
            source_filename = str(raw_meta.get("source_filename") or document.title or "document")
        else:
            source_filename = document.title or "document"
        raw_text = extract_text_from_file(
            file_bytes,
            filename=source_filename,
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
            chunk_size=resolved_settings.chunk_size,
            chunk_overlap=resolved_settings.chunk_overlap,
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
                document_id=document_raw_id,
                collection_id=document.collection_id,
                chunk_index=idx,
                token_count=None,
                embedding_model=resolved_settings.embedding_model,
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

        embedding_batch_size = resolved_settings.embedding_batch_size or 16
        all_prepared: list[dict[str, Any]] = []

        for batch_start in range(0, len(chunk_dicts), embedding_batch_size):
            batch = chunk_dicts[batch_start: batch_start + embedding_batch_size]
            prepared = prepare_chunks_for_indexing(
                document_id=0,  # legacy int placeholder
                checksum=content_checksum,
                chunks=batch,
                provider_name=resolved_settings.embedding_provider,
                model=resolved_settings.embedding_model,
                dimensions=resolved_settings.embedding_dimensions,
                base_url=resolved_settings.embedding_base_url,
                timeout_seconds=resolved_settings.embedding_timeout_seconds,
                batch_size=embedding_batch_size,
                api_key=resolved_settings.embedding_api_key,
                api_type=resolved_settings.embedding_api_type,
                api_version=resolved_settings.embedding_api_version,
                store_embeddings_in_metadata=resolved_settings.store_embeddings_in_metadata,
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
            url=resolved_settings.vector_store_url,
            api_key=resolved_settings.vector_store_api_key,
            timeout_seconds=resolved_settings.vector_store_timeout_seconds,
            prefer_grpc=resolved_settings.vector_store_prefer_grpc,
            embedding_provider=resolved_settings.embedding_provider,
            embedding_model=resolved_settings.embedding_model,
        )

        # Upsert via the adapter
        qdrant.upsert_document_chunks(
            owner_username=document.owner_username or "",
            document_id=0,
            title=document.title or "",
            source_type=document.source_type or "upload",
            source_uri=document.source_uri or "",
            session_id=None,
            chunk_rows=chunk_rows,
            prepared_chunks=all_prepared,
            embedding_provider=resolved_settings.embedding_provider,
            embedding_model=resolved_settings.embedding_model,
        )

        # TODO: The document_id=0 is a legacy placeholder. Actual document_id
        # is document.id (UUID) which would need Qdrant adapter signature change.
        # For now, payload stores only document_id=0; the actual UUID is stored
        # in the vector_mappings table and can be looked up.

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

        # Update collection denormalized counters
        from rag_core.db.models.collections import RagCollection
        rag_collection = db.query(RagCollection).filter(RagCollection.id == document.collection_id).first()
        if rag_collection:
            rag_collection.document_count = (rag_collection.document_count or 0) + 1
            rag_collection.vector_count = (rag_collection.vector_count or 0) + len(chunk_rows)

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

    except Exception as exc:
        logger.exception("Ingestion failed for document %s (job %s): %s", document_id, job_id, exc)
        rate_limiter.record_failure()
        try:
            _fail_job(db, job_id, document_id, "RAG_INGESTION_FAILED", str(exc)[:1000])
        except Exception:
            logger.exception("Failed to update job status after ingestion error.")
        return {"document_id": document_id, "job_id": job_id, "status": "failed", "error": str(exc)[:500]}
