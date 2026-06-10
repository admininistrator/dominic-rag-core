"""Document intake service for rag-core-owned source file storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import uuid

from sqlalchemy.orm import Session

from rag_core.api.config import RagCoreServiceSettings, get_service_settings
from rag_core.db.models.documents import RagDocument
from rag_core.db.models.ingestion_jobs import RagIngestionJob
from rag_core.services.collection_registry import (
    CollectionRegistrationInput,
    CollectionRegistryError,
    get_collection,
    register_collection,
)
from rag_core.services.object_storage import ObjectStorageService, StoredObject, sanitize_filename


_ALLOWED_CONTENT_TYPE_PREFIXES = ("text/", "application/pdf", "application/json", "application/xml")
_ALLOWED_CONTENT_TYPES = {
    "application/octet-stream",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/csv",
}


class DocumentIntakeError(ValueError):
    error_code = "RAG_INVALID_REQUEST"


class DocumentStorageError(RuntimeError):
    error_code = "RAG_STORAGE_UNAVAILABLE"


@dataclass(frozen=True)
class DocumentUploadInput:
    tenant_id: str
    filename: str
    content: bytes
    content_type: str | None = None
    external_id: str | None = None
    title: str | None = None
    collection_id: str | None = None
    collection_name: str | None = None
    owner_username: str | None = None
    metadata_json: dict[str, Any] | None = None


@dataclass(frozen=True)
class DocumentIntakeResult:
    document_id: str
    job_id: str
    status: str
    collection_id: str
    collection_name: str
    source_uri: str
    storage: dict[str, Any]
    metadata_json: dict[str, Any]


def _normalize_tenant(value: str | None) -> str:
    return (value or "default").strip() or "default"


def _is_supported_content_type(content_type: str | None) -> bool:
    if not content_type:
        return True
    normalized = content_type.split(";", 1)[0].strip().lower()
    return normalized in _ALLOWED_CONTENT_TYPES or normalized.startswith(_ALLOWED_CONTENT_TYPE_PREFIXES)


def _validate_upload(payload: DocumentUploadInput, settings: RagCoreServiceSettings) -> str:
    filename = sanitize_filename(payload.filename)
    if not payload.content:
        raise DocumentIntakeError("Uploaded file is empty.")
    max_bytes = int(getattr(settings, "object_storage_max_upload_bytes", 25 * 1024 * 1024) or 0)
    if max_bytes > 0 and len(payload.content) > max_bytes:
        raise DocumentIntakeError(f"Uploaded file exceeds max size of {max_bytes} bytes.")
    if not _is_supported_content_type(payload.content_type):
        raise DocumentIntakeError(f"Unsupported content type: {payload.content_type}")
    return filename


def _resolve_collection(db: Session, payload: DocumentUploadInput, settings: RagCoreServiceSettings):
    identifier = (payload.collection_id or payload.collection_name or "").strip()
    if identifier:
        collection = get_collection(db, identifier)
        if str(collection.tenant_id) != _normalize_tenant(payload.tenant_id):
            raise DocumentIntakeError("Collection tenant does not match document tenant.")
        return collection
    return register_collection(
        db,
        CollectionRegistrationInput(
            tenant_id=_normalize_tenant(payload.tenant_id),
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            metadata_json={"registered_from": "document_intake"},
        ),
    )


def create_document_from_upload(
    db: Session,
    payload: DocumentUploadInput,
    *,
    settings: RagCoreServiceSettings | None = None,
    storage_service: ObjectStorageService | None = None,
) -> DocumentIntakeResult:
    """Persist upload bytes into rag-core storage and create document/job metadata.

    This creates a queued job placeholder only. The real async ingestion pipeline is
    Phase 5; Phase 4 establishes the stable API/storage/metadata boundary.
    """
    resolved_settings = settings or get_service_settings()
    filename = _validate_upload(payload, resolved_settings)
    tenant_id = _normalize_tenant(payload.tenant_id)
    checksum = hashlib.sha256(payload.content).hexdigest()

    try:
        collection = _resolve_collection(db, payload, resolved_settings)
    except CollectionRegistryError:
        raise
    except DocumentIntakeError:
        raise
    except Exception as exc:
        raise DocumentIntakeError("Unable to resolve rag-core collection for document intake.") from exc

    document_uuid = uuid.uuid4()
    job_uuid = uuid.uuid4()
    base_metadata = dict(payload.metadata_json or {})
    base_metadata.setdefault("source_filename", filename)

    document = RagDocument(
        id=document_uuid,
        collection_id=collection.id,
        tenant_id=tenant_id,
        external_id=payload.external_id,
        title=payload.title or filename,
        source_type="upload",
        source_uri=None,
        mime_type=payload.content_type,
        checksum=checksum,
        file_size_bytes=len(payload.content),
        status="pending",
        owner_username=payload.owner_username,
        metadata_json=base_metadata,
    )
    db.add(document)
    db.flush()

    service = storage_service or ObjectStorageService(resolved_settings)
    try:
        stored: StoredObject = service.upload_source_file(
            tenant_id=tenant_id,
            document_id=str(document_uuid),
            filename=filename,
            content=payload.content,
            content_type=payload.content_type,
            checksum=checksum,
        )
    except Exception as exc:
        raise DocumentStorageError("Failed to store source file in rag-core object storage.") from exc

    storage_metadata = stored.to_dict()
    document.source_uri = stored.uri
    document.metadata_json = {
        **base_metadata,
        "storage": storage_metadata,
        "intake": {"mode": "multipart", "uri_reference_enabled": False},
    }

    job = RagIngestionJob(
        id=job_uuid,
        document_id=document_uuid,
        collection_id=collection.id,
        status="queued",
        step_current="queued",
        step_progress=0.0,
    )
    db.add(job)
    db.commit()

    return DocumentIntakeResult(
        document_id=str(document_uuid),
        job_id=str(job_uuid),
        status="queued",
        collection_id=str(collection.id),
        collection_name=str(collection.name),
        source_uri=stored.uri,
        storage=storage_metadata,
        metadata_json=dict(document.metadata_json or {}),
    )
