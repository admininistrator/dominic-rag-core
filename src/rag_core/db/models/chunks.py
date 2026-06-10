"""RCSI-P2-T06: Model — rag_chunks table.

Tracks chunk metadata for indexed documents. Chunk content is intentionally not
stored here; content lives in Qdrant payloads or is re-derived from source files.

Ownership boundary: rag-core owns this table in the rag_core database.
DominicBE must access chunk state through rag-core APIs only.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_core.db.database import Base

if TYPE_CHECKING:
    from rag_core.db.models.collections import RagCollection
    from rag_core.db.models.documents import RagDocument
    from rag_core.db.models.vector_mappings import RagVectorMapping


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RagChunk(Base):
    """Metadata for a single chunk belonging to a document.

    The (document_id, chunk_index) pair is unique so re-indexing can target a
    stable ordinal within each document.
    """

    __tablename__ = "rag_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="Parent rag_documents row",
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_collections.id", ondelete="CASCADE"),
        nullable=False,
        comment="Denormalized collection FK for management queries",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Zero-based order of this chunk within the document",
    )
    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Approximate token count for the chunk",
    )
    vector_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Qdrant point ID when indexed",
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Embedding model identifier used for this chunk",
    )
    content_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="SHA-256 hash of chunk content for idempotency/change detection",
    )
    metadata_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Section/page/span metadata; never raw chunk content",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now_utc,
    )

    document: Mapped["RagDocument"] = relationship(
        "RagDocument",
        back_populates="chunks",
        lazy="noload",
    )
    collection: Mapped["RagCollection"] = relationship(
        "RagCollection",
        back_populates="chunks",
        lazy="noload",
    )
    vector_mappings: Mapped[list["RagVectorMapping"]] = relationship(
        "RagVectorMapping",
        back_populates="chunk",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_rag_chunks_document_chunk_index"),
        Index("ix_rag_chunks_document_id", "document_id"),
        Index("ix_rag_chunks_collection_id", "collection_id"),
        Index("ix_rag_chunks_vector_id", "vector_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<RagChunk id={self.id} document_id={self.document_id} "
            f"chunk_index={self.chunk_index}>"
        )
