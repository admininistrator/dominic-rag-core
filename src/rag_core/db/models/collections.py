"""RCSI-P2-T03: Model — rag_collections table.

Tracks Qdrant collections owned and managed exclusively by rag-core.
DominicBE must NOT write to this table directly; access is through the rag-core API.

Indexes:
  - UNIQUE(name)
  - INDEX(tenant_id)
  - INDEX(status)

Relationships:
  - One-to-many with rag_documents
  - One-to-many with rag_ingestion_jobs (via denormalized collection_id)
  - One-to-many with rag_chunks (via denormalized collection_id)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_core.db.database import Base

if TYPE_CHECKING:
    from rag_core.db.models.chunks import RagChunk
    from rag_core.db.models.documents import RagDocument
    from rag_core.db.models.ingestion_jobs import RagIngestionJob


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RagCollection(Base):
    """Registry of Qdrant collections owned by rag-core.

    Collection name follows the convention: rag_{tenant}_{provider}_{model}
    Status values: active | archived | deleted
    """

    __tablename__ = "rag_collections"

    # ---- Primary key -------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # ---- Identity ----------------------------------------------------------
    name: Mapped[str] = mapped_column(
        String(63),
        nullable=False,
        unique=True,
        comment="Qdrant collection name (max 63 chars). Convention: rag_{tenant}_{provider}_{model}",
    )
    display_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Human-readable label",
    )
    tenant_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="default",
        comment="Tenant/workspace scope",
    )

    # ---- Embedding spec ----------------------------------------------------
    embedding_provider: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="e.g. ollama, api, local",
    )
    embedding_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="e.g. nomic-embed-text, local-hash-v1",
    )
    embedding_dimensions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="e.g. 768, 1536, 64",
    )

    # ---- Lifecycle ---------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        comment="active | archived | deleted",
    )

    # ---- Denormalized counters (updated by worker tasks) -------------------
    document_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Denormalized document count; updated by worker",
    )
    vector_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Denormalized vector count; updated by worker",
    )

    # ---- Extra metadata ----------------------------------------------------
    metadata_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Provider-specific or consumer-provided extra metadata",
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

    # ---- Relationships (back-populated in child models) --------------------
    documents: Mapped[list["RagDocument"]] = relationship(
        "RagDocument",
        back_populates="collection",
        cascade="all, delete-orphan",
        lazy="noload",
    )
    ingestion_jobs: Mapped[list["RagIngestionJob"]] = relationship(
        "RagIngestionJob",
        back_populates="collection",
        cascade="all, delete-orphan",
        lazy="noload",
    )
    chunks: Mapped[list["RagChunk"]] = relationship(
        "RagChunk",
        back_populates="collection",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    # ---- Table-level indexes -----------------------------------------------
    __table_args__ = (
        Index("ix_rag_collections_tenant_id", "tenant_id"),
        Index("ix_rag_collections_status", "status"),
        # UNIQUE(name) is handled by the unique=True on the column above.
    )

    def __repr__(self) -> str:
        return (
            f"<RagCollection id={self.id} name={self.name!r} "
            f"status={self.status!r} tenant={self.tenant_id!r}>"
        )
