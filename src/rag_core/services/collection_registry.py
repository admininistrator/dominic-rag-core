"""Collection registration and lifecycle service for rag-core.

RCSI-P3-T02/T03/T04: rag-core owns Qdrant collection names, lifecycle metadata,
and embedding dimension validation through the rag_collections table.

This module imports only rag_core DB models and SQLAlchemy. DominicBE must access
these operations through rag-core's HTTP API, never by importing this service or
writing the rag_core database directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_core.db.database import SessionLocal
from rag_core.db.models.collections import RagCollection
from rag_core.db.models.embedding_models import RagEmbeddingModel
from rag_core.embeddings.collection_naming import suggest_collection_name


class CollectionRegistryError(ValueError):
    """Base exception for collection registry failures."""

    error_code = "RAG_COLLECTION_ERROR"


class CollectionDimensionMismatchError(CollectionRegistryError):
    """Raised when an existing collection has an incompatible embedding spec."""

    error_code = "RAG_DIMENSION_MISMATCH"


class CollectionNotFoundError(CollectionRegistryError):
    """Raised when a requested collection identifier does not exist."""

    error_code = "RAG_COLLECTION_NOT_FOUND"


@dataclass(frozen=True)
class CollectionRegistrationInput:
    tenant_id: str = "default"
    embedding_provider: str = "local"
    embedding_model: str = "local-hash-v1"
    embedding_dimensions: int = 64
    name: str | None = None
    display_name: str | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)

    def normalized_name(self) -> str:
        return self.name or suggest_collection_name(
            self.embedding_provider,
            self.embedding_model,
            tenant_id=self.tenant_id,
        )


def _positive_dimensions(dimensions: int) -> int:
    try:
        value = int(dimensions)
    except (TypeError, ValueError) as exc:
        raise CollectionRegistryError("embedding_dimensions must be a positive integer") from exc
    if value <= 0:
        raise CollectionRegistryError("embedding_dimensions must be a positive integer")
    return value


def _fetch_collection_by_name(db: Session, name: str) -> RagCollection | None:
    return db.execute(select(RagCollection).where(RagCollection.name == name)).scalars().one_or_none()


def _fetch_collection_by_identifier(db: Session, identifier: str) -> RagCollection | None:
    raw = str(identifier or "").strip()
    if not raw:
        return None
    try:
        collection_id = UUID(raw)
    except ValueError:
        return _fetch_collection_by_name(db, raw)
    return db.execute(select(RagCollection).where(RagCollection.id == collection_id)).scalars().one_or_none()


def _validate_existing_collection(existing: RagCollection, payload: CollectionRegistrationInput) -> None:
    requested_dimensions = _positive_dimensions(payload.embedding_dimensions)
    mismatches: list[str] = []

    if existing.tenant_id != payload.tenant_id:
        mismatches.append(f"tenant {existing.tenant_id!r} != requested {payload.tenant_id!r}")
    if existing.embedding_provider != payload.embedding_provider:
        mismatches.append(
            f"provider {existing.embedding_provider!r} != requested {payload.embedding_provider!r}"
        )
    if existing.embedding_model != payload.embedding_model:
        mismatches.append(f"model {existing.embedding_model!r} != requested {payload.embedding_model!r}")
    if existing.embedding_dimensions != requested_dimensions:
        raise CollectionDimensionMismatchError(
            f"Collection {existing.name!r} is registered with {existing.embedding_dimensions} "
            f"dimensions, requested {requested_dimensions} dimensions. Create or use a "
            "separate rag-core collection for this embedding model/dimension."
        )

    if mismatches:
        raise CollectionRegistryError(
            f"Collection {existing.name!r} is already registered with incompatible metadata: "
            + "; ".join(mismatches)
        )


def _get_or_create_embedding_model(db: Session, payload: CollectionRegistrationInput) -> RagEmbeddingModel:
    dimensions = _positive_dimensions(payload.embedding_dimensions)
    existing = (
        db.execute(
            select(RagEmbeddingModel).where(
                RagEmbeddingModel.provider == payload.embedding_provider,
                RagEmbeddingModel.model_name == payload.embedding_model,
                RagEmbeddingModel.dimensions == dimensions,
            )
        )
        .scalars()
        .one_or_none()
    )
    if existing is not None:
        return existing

    model = RagEmbeddingModel(
        provider=payload.embedding_provider,
        model_name=payload.embedding_model,
        dimensions=dimensions,
        version=payload.embedding_model[:32] if payload.embedding_model else None,
        is_active=True,
        metadata_json={},
    )
    db.add(model)
    return model


def register_collection(db: Session, payload: CollectionRegistrationInput) -> RagCollection:
    """Register or validate a rag-core-owned collection.

    Existing active collections are idempotently reused only when tenant,
    provider, model, and dimensions all match. Dimension mismatches fail fast so
    callers do not upsert vectors into an incompatible Qdrant collection.
    """
    dimensions = _positive_dimensions(payload.embedding_dimensions)
    name = payload.normalized_name()
    existing = _fetch_collection_by_name(db, name)

    if existing is not None:
        _validate_existing_collection(existing, payload)
        if existing.status == "deleted":
            existing.status = "active"
            if payload.display_name is not None:
                existing.display_name = payload.display_name
            if payload.metadata_json:
                existing.metadata_json = dict(payload.metadata_json)
            db.flush()
            db.commit()
        return existing

    _get_or_create_embedding_model(db, payload)
    collection = RagCollection(
        name=name,
        display_name=payload.display_name,
        tenant_id=payload.tenant_id or "default",
        embedding_provider=payload.embedding_provider,
        embedding_model=payload.embedding_model,
        embedding_dimensions=dimensions,
        status="active",
        document_count=0,
        vector_count=0,
        metadata_json=dict(payload.metadata_json or {}),
    )
    db.add(collection)
    db.flush()
    db.commit()
    return collection


def list_collections(
    db: Session,
    *,
    tenant_id: str | None = None,
    status_filter: str | None = None,
) -> list[RagCollection]:
    stmt = select(RagCollection)
    if tenant_id:
        stmt = stmt.where(RagCollection.tenant_id == tenant_id)
    if status_filter:
        stmt = stmt.where(RagCollection.status == status_filter)
    stmt = stmt.order_by(RagCollection.created_at.desc())
    return list(db.execute(stmt).scalars().all())


def get_collection(db: Session, identifier: str) -> RagCollection:
    collection = _fetch_collection_by_identifier(db, identifier)
    if collection is None:
        raise CollectionNotFoundError(f"Collection {identifier!r} was not found")
    return collection


def delete_collection(db: Session, identifier: str) -> RagCollection:
    """Soft-delete a collection registration.

    Phase 3 intentionally does not drop a Qdrant collection from this service.
    Runtime deletion is safe/idempotent at the metadata level; destructive vector
    cleanup can be implemented by explicit worker/API behavior in later phases.
    """
    collection = get_collection(db, identifier)
    if collection.status != "deleted":
        collection.status = "deleted"
        db.flush()
        db.commit()
    return collection


def summarize_collections(db: Session) -> dict[str, Any]:
    """Return health-adjacent collection registry counts."""
    rows = db.execute(
        select(RagCollection.status, func.count(RagCollection.id)).group_by(RagCollection.status)
    ).all()
    counts = {str(status): int(count) for status, count in rows}
    registered_count = sum(counts.values())
    return {
        "ok": True,
        "registered_count": registered_count,
        "active_count": counts.get("active", 0),
        "archived_count": counts.get("archived", 0),
        "deleted_count": counts.get("deleted", 0),
        "counts_by_status": counts,
    }


def collection_health_summary() -> dict[str, Any]:
    """Best-effort health summary that never leaks DB URLs or secrets."""
    try:
        with SessionLocal() as db:
            return summarize_collections(db)
    except Exception as exc:  # pragma: no cover - exercised through API monkeypatch/tests
        return {
            "ok": False,
            "registered_count": 0,
            "active_count": 0,
            "deleted_count": 0,
            "error_type": type(exc).__name__,
        }
