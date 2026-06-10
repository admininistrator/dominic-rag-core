"""RCSI-P2-T07: Model — rag_embedding_models table.

Registry of embedding providers/models/dimensions used by rag-core collections.
This registry is used to validate collection dimensionality and re-indexing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_core.db.database import Base

if TYPE_CHECKING:
    from rag_core.db.models.vector_mappings import RagVectorMapping


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RagEmbeddingModel(Base):
    """Embedding model registry row.

    The tuple (provider, model_name, dimensions) is unique, allowing the same
    provider/model name to be registered with different dimensional variants only
    when the dimension differs intentionally.
    """

    __tablename__ = "rag_embedding_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Embedding provider, e.g. local, ollama, api",
    )
    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Provider-specific model identifier",
    )
    dimensions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Vector dimension count",
    )
    version: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Optional model version tag",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether this model is available for new collections",
    )
    metadata_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Provider-specific metadata/configuration",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_now_utc,
    )

    vector_mappings: Mapped[list["RagVectorMapping"]] = relationship(
        "RagVectorMapping",
        back_populates="embedding_model",
        lazy="noload",
    )

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "model_name",
            "dimensions",
            name="uq_rag_embedding_models_provider_model_dimensions",
        ),
        Index("ix_rag_embedding_models_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return (
            f"<RagEmbeddingModel id={self.id} provider={self.provider!r} "
            f"model={self.model_name!r} dimensions={self.dimensions}>"
        )
