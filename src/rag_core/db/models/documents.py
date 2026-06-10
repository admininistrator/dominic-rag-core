"""RCSI-P2-T04: Model — rag_documents table.

Source of truth for RAG document lifecycle. DominicBE stores only a reference
(external_id) — it must not write to this table directly.

Data NOT stored here (owned by DominicBE):
  user_id, session_id, workspace_id, conversation context, UI preferences.

Indexes:
  - INDEX(collection_id)
  - INDEX(tenant_id)
  - INDEX(external_id)
  - INDEX(status)
  - INDEX(checksum)
  - INDEX(created_at)

Relationships:
  - Belongs to RagCollection
  - One-to-many with RagIngestionJob
  - One-to-many with RagChunk
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_core.db.database import Base

if TYPE_CHECKING:
    from rag_core.db.models.collections import RagCollection
    from rag_core.db.models.ingestion_jobs import RagIngestionJob
    from rag_core.db.models.chunks import RagChunk


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RagDocument(Base):
    """Tracks a document ingested into rag-core.

    source_type values: upload | text | url
    status values: pending | processing | indexed | failed | deleted
    """

    __tablename__ = "rag_documents"

    # ---- Primary key -------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # ---- Collection FK (denormalized tenant_id for fast filtering) ----------
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_collections.id", ondelete="CASCADE"),
        nullable=False,
        comment="Which collection owns this document",
    )
    tenant_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Denormalized from collection for fast tenant-scoped queries",
    )

    # ---- External reference (DominicBE doc ID, etc.) -----------------------
    external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Consumer document ID (e.g. DominicBE knowledge_items.id)",
    )

    # ---- Document metadata -------------------------------------------------
    title: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Document title",
    )
    source_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="upload | text | url",
    )
    source_uri: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="Object storage URI or remote URL",
    )
    mime_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="MIME type of source content",
    )
    checksum: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="SHA-256 of source content (for deduplication)",
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Original file size in bytes",
    )

    # ---- Lifecycle status --------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        comment="pending | processing | indexed | failed | deleted",
    )

    # ---- Denormalized counter ----------------------------------------------
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Denormalized chunk count; updated by worker on completion",
    )

    # ---- Legacy compat (Qdrant filtering) ----------------------------------
    owner_username: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Legacy owner field for Qdrant payload filtering",
    )

    # ---- Extra consumer metadata -------------------------------------------
    metadata_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Consumer-provided metadata (tags, labels, etc.)",
    )

    # ---- Timestamps --------------------------------------------------------
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
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Soft-delete timestamp; hard-purge after retention window",
    )

    # ---- Relationships -----------------------------------------------------
    collection: Mapped["RagCollection"] = relationship(
        "RagCollection",
        back_populates="documents",
        lazy="noload",
    )
    ingestion_jobs: Mapped[list["RagIngestionJob"]] = relationship(
        "RagIngestionJob",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="noload",
    )
    chunks: Mapped[list["RagChunk"]] = relationship(
        "RagChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    # ---- Table-level indexes -----------------------------------------------
    __table_args__ = (
        Index("ix_rag_documents_collection_id", "collection_id"),
        Index("ix_rag_documents_tenant_id", "tenant_id"),
        Index("ix_rag_documents_external_id", "external_id"),
        Index("ix_rag_documents_status", "status"),
        Index("ix_rag_documents_checksum", "checksum"),
        Index("ix_rag_documents_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<RagDocument id={self.id} status={self.status!r} "
            f"source_type={self.source_type!r} tenant={self.tenant_id!r}>"
        )
