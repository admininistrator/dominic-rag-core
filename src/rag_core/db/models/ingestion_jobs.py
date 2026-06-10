"""RCSI-P2-T05: Model — rag_ingestion_jobs table.

Tracks the full async ingestion job lifecycle for each document.

status values:
  queued -> processing -> embedding -> indexing -> completed | failed | cancelled

Indexes:
  - INDEX(document_id)
  - INDEX(collection_id)
  - INDEX(status)
  - INDEX(celery_task_id)
  - INDEX(created_at)

Relationships:
  - Belongs to RagDocument
  - Belongs to RagCollection (denormalized FK)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_core.db.database import Base

if TYPE_CHECKING:
    from rag_core.db.models.collections import RagCollection
    from rag_core.db.models.documents import RagDocument


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RagIngestionJob(Base):
    """Async ingestion job lifecycle record.

    Each attempt to ingest a document creates a new job record.
    status: queued | processing | embedding | indexing | completed | failed | cancelled
    """

    __tablename__ = "rag_ingestion_jobs"

    # ---- Primary key -------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # ---- FKs ---------------------------------------------------------------
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="Document being ingested",
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_collections.id", ondelete="CASCADE"),
        nullable=False,
        comment="Denormalized collection FK for fast queue queries",
    )

    # ---- Status & Celery tracking -----------------------------------------
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        comment="queued | processing | embedding | indexing | completed | failed | cancelled",
    )
    celery_task_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Celery async result ID for status polling",
    )

    # ---- Progress tracking -------------------------------------------------
    step_current: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Current pipeline step name",
    )
    step_progress: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Progress within current step: 0.0 - 1.0",
    )
    chunks_total: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total chunks to process",
    )
    chunks_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Chunks completed so far",
    )

    # ---- Error details -----------------------------------------------------
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable error description on failure",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Structured error code for programmatic handling",
    )

    # ---- Retry policy ------------------------------------------------------
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of retries attempted",
    )
    max_retries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        comment="Maximum allowed retries before marking failed",
    )

    # ---- Timestamps --------------------------------------------------------
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When processing actually began",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the job reached a terminal state",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now_utc,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now_utc,
        onupdate=_now_utc,
    )

    # ---- Relationships -----------------------------------------------------
    document: Mapped["RagDocument"] = relationship(
        "RagDocument",
        back_populates="ingestion_jobs",
        lazy="noload",
    )
    collection: Mapped["RagCollection"] = relationship(
        "RagCollection",
        back_populates="ingestion_jobs",
        lazy="noload",
    )

    # ---- Table-level indexes -----------------------------------------------
    __table_args__ = (
        Index("ix_rag_ingestion_jobs_document_id", "document_id"),
        Index("ix_rag_ingestion_jobs_collection_id", "collection_id"),
        Index("ix_rag_ingestion_jobs_status", "status"),
        Index("ix_rag_ingestion_jobs_celery_task_id", "celery_task_id"),
        Index("ix_rag_ingestion_jobs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<RagIngestionJob id={self.id} status={self.status!r} "
            f"doc={self.document_id} retry={self.retry_count}/{self.max_retries}>"
        )
