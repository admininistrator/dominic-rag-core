"""RCSI-P2-T08: Model — rag_vector_mappings table.

Maps rag_chunks rows to Qdrant point IDs so vector lifecycle operations can be
performed without scanning Qdrant collections.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_core.db.database import Base

if TYPE_CHECKING:
    from rag_core.db.models.chunks import RagChunk
    from rag_core.db.models.embedding_models import RagEmbeddingModel


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RagVectorMapping(Base):
    """Mapping between a chunk and a Qdrant point ID.

    (collection_name, point_id) is unique because Qdrant point IDs are scoped to
    a collection.
    """

    __tablename__ = "rag_vector_mappings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_chunks.id", ondelete="CASCADE"),
        nullable=False,
        comment="rag_chunks row mapped to the vector point",
    )
    collection_name: Mapped[str] = mapped_column(
        String(63),
        nullable=False,
        comment="Qdrant collection name",
    )
    point_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Qdrant point UUID/ID",
    )
    embedding_model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_embedding_models.id", ondelete="SET NULL"),
        nullable=True,
        comment="Embedding model registry row used for this vector",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        comment="active | deleted | stale",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now_utc,
    )

    chunk: Mapped["RagChunk"] = relationship(
        "RagChunk",
        back_populates="vector_mappings",
        lazy="noload",
    )
    embedding_model: Mapped["RagEmbeddingModel | None"] = relationship(
        "RagEmbeddingModel",
        back_populates="vector_mappings",
        lazy="noload",
    )

    __table_args__ = (
        UniqueConstraint("collection_name", "point_id", name="uq_rag_vector_mappings_collection_point"),
        Index("ix_rag_vector_mappings_chunk_id", "chunk_id"),
        Index("ix_rag_vector_mappings_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<RagVectorMapping id={self.id} collection={self.collection_name!r} "
            f"point_id={self.point_id!r} status={self.status!r}>"
        )
