"""Reindex tasks for rag-core worker.

Task ``rag.reindex_document`` re-ingests an already-registered document from
its stored source file.  Task ``rag.reindex_collection`` re-processes all
documents in a collection.

Both tasks are idempotent: they resubmit ingestion jobs for each document.
"""

from __future__ import annotations

import logging
from typing import Any

from celery import shared_task
from sqlalchemy.orm import Session

from rag_core.db.database import SessionLocal
from rag_core.db.models.documents import RagDocument
from rag_core.worker.tasks.ingestion import ingest_document

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


@shared_task(
    bind=True,
    name="rag.reindex_document",
    queue="rag.reindex",
    max_retries=2,
    default_retry_delay=30,
)
def reindex_document(self, document_id: str) -> dict[str, Any]:
    """Re-embed and re-index a single existing document from its source file.

    This submits a new ingestion job for the document.  The existing chunks
    and vectors will be overwritten idempotently via Qdrant upsert.

    Args:
        document_id: UUID of the ``rag_documents`` row.

    Returns:
        Dict with ``document_id``, ``status``, and the child ``job_id`` if created.
    """
    import uuid as _uuid

    db: Session = SessionLocal()
    try:
        document = db.query(RagDocument).filter(RagDocument.id == document_id).first()
        if document is None:
            return {"document_id": document_id, "status": "not_found"}

        # Reset document state
        document.status = "pending"

        # Create a new ingestion job
        from rag_core.db.models.ingestion_jobs import RagIngestionJob

        job_id = str(_uuid.uuid4())
        job = RagIngestionJob(
            id=job_id,
            document_id=document_id,
            collection_id=document.collection_id,
            status="queued",
            step_current="queued",
            step_progress=0.0,
        )
        db.add(job)
        db.commit()

        # Enqueue the ingestion task
        ingest_document.delay(document_id=document_id, job_id=job_id)

        logger.info(
            "reindex_document: enqueued re-ingestion for document %s (job %s)",
            document_id,
            job_id,
        )
        return {"document_id": document_id, "job_id": job_id, "status": "queued"}
    except Exception as exc:
        logger.exception("reindex_document failed for document %s: %s", document_id, exc)
        raise
    finally:
        db.close()


@shared_task(
    bind=True,
    name="rag.reindex_collection",
    queue="rag.reindex",
    max_retries=1,
    default_retry_delay=60,
)
def reindex_collection(self, collection_id: str) -> dict[str, Any]:
    """Re-embed and re-index all documents in a collection.

    Args:
        collection_id: UUID of the ``rag_collections`` row.

    Returns:
        Dict with ``collection_id``, ``documents_submitted`` count, and
        ``job_ids`` list.
    """
    db: Session = SessionLocal()
    try:
        documents = (
            db.query(RagDocument)
            .filter(
                RagDocument.collection_id == collection_id,
                RagDocument.status != "deleted",
            )
            .all()
        )

        if not documents:
            return {"collection_id": collection_id, "documents_submitted": 0, "job_ids": []}

        job_ids: list[str] = []
        for doc in documents:
            result = reindex_document.delay(document_id=str(doc.id))
            job_ids.append(str(doc.id))

        logger.info(
            "reindex_collection: submitted %d documents for reindexing (collection %s).",
            len(documents),
            collection_id,
        )
        return {
            "collection_id": collection_id,
            "documents_submitted": len(documents),
            "job_ids": job_ids,
        }
    except Exception as exc:
        logger.exception("reindex_collection failed for collection %s: %s", collection_id, exc)
        raise
    finally:
        db.close()
