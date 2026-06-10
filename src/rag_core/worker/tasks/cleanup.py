"""Cleanup and garbage-collection tasks for rag-core worker.

Task ``rag.cleanup_orphaned_jobs`` finds and marks stuck/failed ingestion jobs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from celery import shared_task
from sqlalchemy import or_
from sqlalchemy.orm import Session

from rag_core.db.database import SessionLocal
from rag_core.db.models.documents import RagDocument
from rag_core.db.models.ingestion_jobs import RagIngestionJob

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_STUCK_THRESHOLD_MINUTES = 60
_DEFAULT_CLEANUP_BATCH_SIZE = 100


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


@shared_task(
    bind=True,
    name="rag.cleanup_orphaned_jobs",
    queue="rag.cleanup",
    max_retries=1,
    default_retry_delay=30,
)
def cleanup_orphaned_jobs(self) -> dict[str, Any]:
    """Find and mark stuck/failed ingestion jobs.

    Jobs that have been in ``processing`` state for longer than the threshold
    (default 60 minutes) are marked as ``failed`` with an orphaned error code.
    Documents with failed jobs where all jobs have failed are marked failed too.

    This task is idempotent and safe to run periodically via Celery beat.

    Returns:
        Dict with ``jobs_cleaned``, ``documents_cleaned`` counts.
    """
    db: Session = SessionLocal()
    threshold_minutes = _env_int("RAG_CLEANUP_STUCK_THRESHOLD_MINUTES", _DEFAULT_STUCK_THRESHOLD_MINUTES)
    batch_size = _env_int("RAG_CLEANUP_BATCH_SIZE", _DEFAULT_CLEANUP_BATCH_SIZE)

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=threshold_minutes)

        # Find stuck processing jobs
        stuck_jobs = (
            db.query(RagIngestionJob)
            .filter(
                RagIngestionJob.status == "processing",
                RagIngestionJob.started_at.isnot(None),
                RagIngestionJob.started_at < cutoff,
            )
            .limit(batch_size)
            .all()
        )

        jobs_cleaned = 0
        documents_touched: set[str] = set()

        for job in stuck_jobs:
            job.status = "failed"
            job.error_code = "RAG_JOB_ORPHANED"
            job.error_message = (
                f"Job stuck in processing state for >{threshold_minutes} minutes; marked failed by cleanup task."
            )
            documents_touched.add(str(job.document_id))
            jobs_cleaned += 1

        # Mark documents with all-failed jobs as failed
        documents_cleaned = 0
        if documents_touched:
            for doc_id in documents_touched:
                pending_jobs_count = (
                    db.query(RagIngestionJob)
                    .filter(
                        RagIngestionJob.document_id == doc_id,
                        RagIngestionJob.status.in_(["processing", "queued"]),
                    )
                    .count()
                )
                if pending_jobs_count == 0:
                    doc = db.query(RagDocument).filter(RagDocument.id == doc_id).first()
                    if doc and doc.status not in ("indexed", "failed", "deleted"):
                        doc.status = "failed"
                        documents_cleaned += 1

        db.commit()

        if jobs_cleaned or documents_cleaned:
            logger.info(
                "cleanup_orphaned_jobs: marked %d jobs and %d documents as failed.",
                jobs_cleaned,
                documents_cleaned,
            )

        # Also mark queued jobs that are very old (24h+) as cancelled
        expired_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        expired_jobs = (
            db.query(RagIngestionJob)
            .filter(
                RagIngestionJob.status == "queued",
                RagIngestionJob.created_at < expired_cutoff,
            )
            .limit(batch_size)
            .all()
        )

        expired_count = 0
        for job in expired_jobs:
            job.status = "cancelled"
            job.error_code = "RAG_JOB_EXPIRED"
            job.error_message = "Job expired (>24h queued); cancelled by cleanup task."
            expired_count += 1

        if expired_count:
            db.commit()
            logger.info("cleanup_orphaned_jobs: cancelled %d expired queued jobs.", expired_count)
            jobs_cleaned += expired_count

        return {
            "jobs_cleaned": jobs_cleaned,
            "documents_cleaned": documents_cleaned,
            "expired_cancelled": expired_count,
        }

    except Exception as exc:
        logger.exception("cleanup_orphaned_jobs failed: %s", exc)
        raise
    finally:
        db.close()
