"""Phase 9 DominicBE -> rag-core offline backfill planner/apply helpers.

Safety model:
- Dry-run is the default. ``--apply`` is required for database writes.
- No delete/drop/truncate operation is implemented here.
- Source-file copying is disabled unless both ``--apply`` and
  ``--enable-file-copy`` are set. Reference mode is the default.
- The tool is an offline migration bridge; production app/runtime code must keep
  the DominicBE <-> rag-core HTTP boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rag_core.embeddings.collection_naming import suggest_collection_name
from rag_core.services.object_storage import build_source_file_key, sanitize_filename

_BACKFILL_NAMESPACE = uuid.UUID("9d33400e-10a0-4932-b581-848fbd6fbb9d")
_DEFAULT_COLLECTION_NAME = "knowledge_chunks"


@dataclass(frozen=True)
class BackfillConfig:
    """Configuration for safe DominicBE backfill planning.

    ``apply`` toggles whether the generated actions may be executed by
    ``apply_plan``. Planning itself is pure and side-effect free.
    """

    apply: bool = False
    tenant_id: str = "default"
    collection_names: Sequence[str] = field(default_factory=lambda: [_DEFAULT_COLLECTION_NAME])
    embedding_provider: str = "local"
    embedding_model: str = "local-hash-v1"
    embedding_dimensions: int = 64
    file_strategy: str = "reference"  # reference | copy
    enable_file_copy: bool = False
    object_storage_key_prefix: str = ""
    limit: int | None = None

    def __post_init__(self) -> None:
        strategy = (self.file_strategy or "reference").strip().lower()
        if strategy not in {"reference", "copy"}:
            raise ValueError("file_strategy must be 'reference' or 'copy'")
        if int(self.embedding_dimensions) <= 0:
            raise ValueError("embedding_dimensions must be positive")
        names = [str(name).strip() for name in self.collection_names if str(name).strip()]
        if not names:
            object.__setattr__(self, "collection_names", [_DEFAULT_COLLECTION_NAME])
        else:
            object.__setattr__(self, "collection_names", names)
        object.__setattr__(self, "file_strategy", strategy)


@dataclass(frozen=True)
class BackfillPlan:
    dry_run: bool
    apply_enabled: bool
    file_strategy: str
    tenant_id: str
    collection_registrations: list[dict[str, Any]]
    document_mappings: list[dict[str, Any]]
    chunk_mappings: list[dict[str, Any]]
    vector_mappings: list[dict[str, Any]]
    chatbot_document_mappings: list[dict[str, Any]]
    file_actions: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    counts: dict[str, int]
    plan_checksum: str
    rollback_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "apply_enabled": self.apply_enabled,
            "file_strategy": self.file_strategy,
            "tenant_id": self.tenant_id,
            "collection_registrations": self.collection_registrations,
            "document_mappings": self.document_mappings,
            "chunk_mappings": self.chunk_mappings,
            "vector_mappings": self.vector_mappings,
            "chatbot_document_mappings": self.chatbot_document_mappings,
            "file_actions": self.file_actions,
            "actions": self.actions,
            "counts": self.counts,
            "plan_checksum": self.plan_checksum,
            "rollback_notes": self.rollback_notes,
        }


def build_rollback_notes() -> str:
    return (
        "Phase 9 rollback is manual and starts with a fresh dry-run to capture "
        "current counts/checksums. No destructive rollback is automatic. To revert "
        "an applied backfill, first stop DominicBE/rag-core ingestion writers, take "
        "fresh backups, then remove only rows tagged with metadata_json.backfill.source "
        "= 'dominicbe_phase9' from rag_vector_mappings, rag_chunks, rag_ingestion_jobs, "
        "rag_documents, and backfill-created rag_collections/rag_embedding_models if no "
        "non-backfill rows reference them. Remove corresponding chatbot_documents and "
        "chatbot_rag_job_mappings rows whose metadata_json marks this backfill. If "
        "--enable-file-copy was used, delete only the exact recorded rag-source-files "
        "keys after comparing the plan checksum/manifests. Reference-mode backfills do "
        "not copy source files, so no rag-source-files object rollback is needed. Never "
        "drop chatbot_db, rag_core, Qdrant collections, or buckets as part of this rollback."
    )


def _deterministic_uuid(kind: str, source_id: Any) -> str:
    return str(uuid.uuid5(_BACKFILL_NAMESPACE, f"dominicbe:{kind}:{source_id}"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _stable_checksum(payload: Any) -> str:
    raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _as_dict(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    return dict(row)


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {"legacy_metadata_raw": value}
        return loaded if isinstance(loaded, dict) else {"legacy_metadata": loaded}
    return {"legacy_metadata": _json_safe(value)}


def _first_collection_name(config: BackfillConfig) -> str:
    return list(config.collection_names)[0]


def _status_for_document(source_status: Any) -> str:
    value = str(source_status or "").strip().lower()
    if value == "indexed":
        return "indexed"
    if value == "failed":
        return "failed"
    if value == "processing":
        return "processing"
    if value == "uploaded":
        return "pending"
    return "pending"


def _source_type(value: Any) -> str:
    normalized = str(value or "text").strip().lower()
    return normalized if normalized in {"upload", "text", "url"} else "text"


def _document_source_uri(row: Mapping[str, Any], rag_document_id: str, config: BackfillConfig) -> str:
    source_uri = str(row.get("source_uri") or "").strip()
    if config.file_strategy == "reference":
        return source_uri or f"backfill://chatbot_db/knowledge_documents/{row['id']}"
    filename = Path(source_uri).name if source_uri else f"knowledge_document_{row['id']}.txt"
    key = build_source_file_key(
        tenant_id=config.tenant_id,
        document_id=rag_document_id,
        filename=filename,
        key_prefix=config.object_storage_key_prefix,
    )
    return f"s3://rag-source-files/{key}"


class BackfillPlanner:
    """Builds deterministic, side-effect-free migration plans."""

    def __init__(self, config: BackfillConfig) -> None:
        self.config = config

    def build_plan(
        self,
        documents: Iterable[Mapping[str, Any] | Any],
        chunks: Iterable[Mapping[str, Any] | Any],
        qdrant_collections: Iterable[Mapping[str, Any] | str] | None = None,
    ) -> BackfillPlan:
        docs = [_as_dict(row) for row in documents]
        chunk_rows = [_as_dict(row) for row in chunks]
        if self.config.limit is not None:
            docs = docs[: self.config.limit]
            allowed_doc_ids = {row.get("id") for row in docs}
            chunk_rows = [row for row in chunk_rows if row.get("document_id") in allowed_doc_ids]

        mode = "apply" if self.config.apply else "dry_run"
        collection_registrations = self._collection_registrations(qdrant_collections)
        primary_collection = collection_registrations[0]
        source_doc_to_rag_doc: dict[Any, str] = {}
        document_mappings: list[dict[str, Any]] = []
        chatbot_document_mappings: list[dict[str, Any]] = []
        file_actions: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []

        for collection in collection_registrations:
            actions.append({"action": "register_collection", "mode": mode, "collection_name": collection["name"]})

        for row in docs:
            local_id = row["id"]
            rag_document_id = _deterministic_uuid("knowledge_document", local_id)
            source_doc_to_rag_doc[local_id] = rag_document_id
            source_uri = _document_source_uri(row, rag_document_id, self.config)
            metadata = _normalize_metadata(row.get("metadata_json"))
            metadata.update(
                {
                    "backfill": {
                        "source": "dominicbe_phase9",
                        "local_document_id": local_id,
                        "source_file_strategy": self.config.file_strategy,
                        "original_source_uri": row.get("source_uri"),
                    }
                }
            )
            document_mappings.append(
                {
                    "local_document_id": local_id,
                    "rag_document_id": rag_document_id,
                    "collection_name": primary_collection["name"],
                    "tenant_id": self.config.tenant_id,
                    "external_id": f"DominicBE:knowledge_documents:{local_id}",
                    "title": row.get("title"),
                    "source_type": _source_type(row.get("source_type")),
                    "source_uri": source_uri,
                    "mime_type": row.get("mime_type"),
                    "status": _status_for_document(row.get("status")),
                    "checksum": row.get("checksum"),
                    "owner_username": row.get("owner_username"),
                    "metadata_json": metadata,
                }
            )
            chatbot_document_mappings.append(
                {
                    "local_document_id": local_id,
                    "rag_document_id": rag_document_id,
                    "rag_collection_name": primary_collection["name"],
                    "tenant_id": self.config.tenant_id,
                    "metadata_json": {
                        "backfill": {
                            "source": "dominicbe_phase9",
                            "plan_document_id": rag_document_id,
                            "plan_collection_name": primary_collection["name"],
                        }
                    },
                }
            )
            actions.append({"action": "upsert_rag_document", "mode": mode, "local_document_id": local_id, "rag_document_id": rag_document_id})
            actions.append({"action": "upsert_chatbot_document_mapping", "mode": mode, "local_document_id": local_id, "rag_document_id": rag_document_id})
            file_actions.extend(self._file_actions_for_document(row, rag_document_id, source_uri, mode))

        chunk_mappings: list[dict[str, Any]] = []
        vector_mappings: list[dict[str, Any]] = []
        for row in chunk_rows:
            local_document_id = row.get("document_id")
            rag_document_id = source_doc_to_rag_doc.get(local_document_id)
            if not rag_document_id:
                continue
            local_chunk_id = row["id"]
            rag_chunk_id = _deterministic_uuid("knowledge_chunk", local_chunk_id)
            content = str(row.get("content") or "")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk_metadata = _normalize_metadata(row.get("metadata_json"))
            chunk_metadata.update(
                {
                    "backfill": {
                        "source": "dominicbe_phase9",
                        "local_chunk_id": local_chunk_id,
                        "local_document_id": local_document_id,
                    }
                }
            )
            chunk_mappings.append(
                {
                    "local_chunk_id": local_chunk_id,
                    "local_document_id": local_document_id,
                    "rag_chunk_id": rag_chunk_id,
                    "rag_document_id": rag_document_id,
                    "collection_name": primary_collection["name"],
                    "chunk_index": int(row.get("chunk_index") or 0),
                    "token_count": row.get("token_count"),
                    "embedding_model": row.get("embedding_model") or self.config.embedding_model,
                    "vector_id": row.get("vector_id"),
                    "content_hash": content_hash,
                    "metadata_json": chunk_metadata,
                }
            )
            actions.append({"action": "upsert_rag_chunk", "mode": mode, "local_chunk_id": local_chunk_id, "rag_chunk_id": rag_chunk_id})
            if row.get("vector_id"):
                mapping = {
                    "local_chunk_id": local_chunk_id,
                    "rag_chunk_id": rag_chunk_id,
                    "collection_name": primary_collection["name"],
                    "point_id": str(row["vector_id"]),
                    "embedding_provider": self.config.embedding_provider,
                    "embedding_model": row.get("embedding_model") or self.config.embedding_model,
                    "embedding_dimensions": int(self.config.embedding_dimensions),
                    "status": "active",
                }
                vector_mappings.append(mapping)
                actions.append({"action": "upsert_vector_mapping", "mode": mode, "point_id": mapping["point_id"], "rag_chunk_id": rag_chunk_id})

        actions.extend(file_actions)
        counts = {
            "source_documents": len(docs),
            "source_chunks": len(chunk_rows),
            "registered_collections": len(collection_registrations),
            "planned_rag_documents": len(document_mappings),
            "planned_rag_chunks": len(chunk_mappings),
            "planned_vector_mappings": len(vector_mappings),
            "planned_chatbot_document_mappings": len(chatbot_document_mappings),
            "planned_file_copies": sum(1 for action in file_actions if action["action"] == "copy_source_file"),
            "blocked_file_copies": sum(1 for action in file_actions if action["action"] == "file_copy_blocked"),
            "planned_file_references": sum(1 for action in file_actions if action["action"] == "reference_source_file"),
        }
        checksum_payload = {
            "config": {
                "tenant_id": self.config.tenant_id,
                "collection_names": list(self.config.collection_names),
                "embedding_provider": self.config.embedding_provider,
                "embedding_model": self.config.embedding_model,
                "embedding_dimensions": self.config.embedding_dimensions,
                "file_strategy": self.config.file_strategy,
            },
            "collections": collection_registrations,
            "documents": document_mappings,
            "chunks": chunk_mappings,
            "vectors": vector_mappings,
            "chatbot_mappings": chatbot_document_mappings,
            "counts": counts,
        }
        return BackfillPlan(
            dry_run=not self.config.apply,
            apply_enabled=bool(self.config.apply),
            file_strategy=self.config.file_strategy,
            tenant_id=self.config.tenant_id,
            collection_registrations=collection_registrations,
            document_mappings=document_mappings,
            chunk_mappings=chunk_mappings,
            vector_mappings=vector_mappings,
            chatbot_document_mappings=chatbot_document_mappings,
            file_actions=file_actions,
            actions=actions,
            counts=counts,
            plan_checksum=_stable_checksum(checksum_payload),
            rollback_notes=build_rollback_notes(),
        )

    def _collection_registrations(
        self,
        qdrant_collections: Iterable[Mapping[str, Any] | str] | None,
    ) -> list[dict[str, Any]]:
        configured = list(self.config.collection_names)
        discovered: list[str] = []
        for item in qdrant_collections or []:
            if isinstance(item, str):
                discovered.append(item)
            else:
                name = item.get("name") or item.get("collection_name")
                if name:
                    discovered.append(str(name))
        names = []
        for name in [*configured, *discovered]:
            if name not in names:
                names.append(name)
        registrations = []
        for name in names:
            registrations.append(
                {
                    "name": name,
                    "tenant_id": self.config.tenant_id,
                    "embedding_provider": self.config.embedding_provider,
                    "embedding_model": self.config.embedding_model,
                    "embedding_dimensions": int(self.config.embedding_dimensions),
                    "status": "active",
                    "metadata_json": {
                        "backfill": {
                            "source": "dominicbe_phase9",
                            "legacy_collection_name": name,
                            "canonical_suggestion": suggest_collection_name(
                                self.config.embedding_provider,
                                self.config.embedding_model,
                                tenant_id=self.config.tenant_id,
                            ),
                        }
                    },
                }
            )
        return registrations

    def _file_actions_for_document(
        self,
        row: Mapping[str, Any],
        rag_document_id: str,
        target_source_uri: str,
        mode: str,
    ) -> list[dict[str, Any]]:
        source_uri = str(row.get("source_uri") or "").strip()
        if self.config.file_strategy == "reference":
            return [
                {
                    "action": "reference_source_file",
                    "mode": mode,
                    "local_document_id": row.get("id"),
                    "rag_document_id": rag_document_id,
                    "source_uri": source_uri or None,
                    "target_source_uri": target_source_uri,
                }
            ]
        filename = sanitize_filename(Path(source_uri).name if source_uri else f"knowledge_document_{row.get('id')}.txt")
        key = build_source_file_key(
            tenant_id=self.config.tenant_id,
            document_id=rag_document_id,
            filename=filename,
            key_prefix=self.config.object_storage_key_prefix,
        )
        base = {
            "local_document_id": row.get("id"),
            "rag_document_id": rag_document_id,
            "source_uri": source_uri or None,
            "target_bucket": "rag-source-files",
            "target_key": key,
            "target_source_uri": target_source_uri,
        }
        if not (self.config.apply and self.config.enable_file_copy):
            return [{"action": "file_copy_blocked", "mode": mode, **base, "reason": "copy requires --apply and --enable-file-copy"}]
        return [{"action": "copy_source_file", "mode": mode, **base}]


def load_source_rows(source_database_url: str, *, limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read DominicBE source rows with SQL only; no DominicBE imports."""
    from sqlalchemy import bindparam, create_engine, text

    engine = create_engine(source_database_url)
    limit_clause = " LIMIT :limit" if limit else ""
    with engine.connect() as conn:
        doc_stmt = text(
            "SELECT id, owner_username, title, source_type, source_uri, mime_type, status, "
            "checksum, raw_text, metadata_json, created_at, updated_at "
            "FROM knowledge_documents WHERE deleted_at IS NULL ORDER BY id" + limit_clause
        )
        params = {"limit": int(limit)} if limit else {}
        docs = [dict(row._mapping) for row in conn.execute(doc_stmt, params)]
        if not docs:
            return [], []
        ids = [row["id"] for row in docs]
        chunk_stmt = text(
            "SELECT id, document_id, chunk_index, content, token_count, embedding_model, "
            "vector_id, metadata_json, created_at FROM knowledge_chunks "
            "WHERE document_id IN :document_ids ORDER BY document_id, chunk_index"
        ).bindparams(bindparam("document_ids", expanding=True))
        chunks = [dict(row._mapping) for row in conn.execute(chunk_stmt, {"document_ids": ids})]
    return docs, chunks


def _jsonb(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True)


def apply_plan(source_database_url: str, rag_core_database_url: str, plan: BackfillPlan) -> dict[str, int]:
    """Apply a previously generated plan.

    This function intentionally has no destructive statements. It upserts only
    deterministic rows tagged as a Phase 9 backfill and inserts DominicBE mapping
    rows. File copies are handled by the CLI wrapper after additional operator
    guards; reference mode performs no file writes.
    """
    if not plan.apply_enabled:
        raise RuntimeError("Refusing to apply a dry-run plan; rerun with --apply")

    from sqlalchemy import create_engine, text

    rag_engine = create_engine(rag_core_database_url)
    source_engine = create_engine(source_database_url)
    result = {
        "collections_upserted": 0,
        "documents_upserted": 0,
        "chunks_upserted": 0,
        "vector_mappings_upserted": 0,
        "chatbot_document_mappings_upserted": 0,
    }
    with rag_engine.begin() as conn:
        for collection in plan.collection_registrations:
            existing_collection = conn.execute(
                text(
                    "SELECT tenant_id, embedding_provider, embedding_model, embedding_dimensions "
                    "FROM rag_collections WHERE name=:name"
                ),
                {"name": collection["name"]},
            ).first()
            if existing_collection is not None:
                row = existing_collection._mapping
                mismatches = []
                for key in ("tenant_id", "embedding_provider", "embedding_model"):
                    if str(row[key]) != str(collection[key]):
                        mismatches.append(f"{key} {row[key]!r} != planned {collection[key]!r}")
                if int(row["embedding_dimensions"]) != int(collection["embedding_dimensions"]):
                    mismatches.append(
                        f"embedding_dimensions {row['embedding_dimensions']!r} != planned {collection['embedding_dimensions']!r}"
                    )
                if mismatches:
                    raise RuntimeError(
                        "Refusing to backfill into incompatible existing rag_collections row "
                        f"{collection['name']!r}: " + "; ".join(mismatches)
                    )
            embedding_model_id = str(
                uuid.uuid5(
                    _BACKFILL_NAMESPACE,
                    f"embedding:{collection['embedding_provider']}:{collection['embedding_model']}:{collection['embedding_dimensions']}",
                )
            )
            conn.execute(
                text(
                    "INSERT INTO rag_embedding_models (id, provider, model_name, dimensions, version, is_active, metadata_json, created_at) "
                    "VALUES (:id, :provider, :model_name, :dimensions, :version, true, CAST(:metadata_json AS jsonb), now()) "
                    "ON CONFLICT (provider, model_name, dimensions) DO NOTHING"
                ),
                {
                    "id": embedding_model_id,
                    "provider": collection["embedding_provider"],
                    "model_name": collection["embedding_model"],
                    "dimensions": collection["embedding_dimensions"],
                    "version": str(collection["embedding_model"])[:32],
                    "metadata_json": _jsonb(collection["metadata_json"]),
                },
            )
            conn.execute(
                text(
                    "INSERT INTO rag_collections (id, name, display_name, tenant_id, embedding_provider, embedding_model, embedding_dimensions, status, document_count, vector_count, metadata_json, created_at, updated_at) "
                    "VALUES (:id, :name, :display_name, :tenant_id, :embedding_provider, :embedding_model, :embedding_dimensions, 'active', 0, 0, CAST(:metadata_json AS jsonb), now(), now()) "
                    "ON CONFLICT (name) DO UPDATE SET status='active', updated_at=now()"
                ),
                {
                    "id": _deterministic_uuid("rag_collection", collection["name"]),
                    "name": collection["name"],
                    "display_name": f"Backfilled {collection['name']}",
                    "tenant_id": collection["tenant_id"],
                    "embedding_provider": collection["embedding_provider"],
                    "embedding_model": collection["embedding_model"],
                    "embedding_dimensions": collection["embedding_dimensions"],
                    "metadata_json": _jsonb(collection["metadata_json"]),
                },
            )
            result["collections_upserted"] += 1

        collection_ids = {
            row._mapping["name"]: str(row._mapping["id"])
            for row in conn.execute(text("SELECT id, name FROM rag_collections"))
        }
        embedding_model_ids = {
            (row._mapping["provider"], row._mapping["model_name"], int(row._mapping["dimensions"])): str(row._mapping["id"])
            for row in conn.execute(text("SELECT id, provider, model_name, dimensions FROM rag_embedding_models"))
        }
        for doc in plan.document_mappings:
            conn.execute(
                text(
                    "INSERT INTO rag_documents (id, collection_id, tenant_id, external_id, title, source_type, source_uri, mime_type, checksum, file_size_bytes, status, chunk_count, owner_username, metadata_json, created_at, updated_at) "
                    "VALUES (:id, :collection_id, :tenant_id, :external_id, :title, :source_type, :source_uri, :mime_type, :checksum, NULL, :status, 0, :owner_username, CAST(:metadata_json AS jsonb), now(), now()) "
                    "ON CONFLICT (id) DO UPDATE SET status=EXCLUDED.status, chunk_count=EXCLUDED.chunk_count, updated_at=now()"
                ),
                {
                    "id": doc["rag_document_id"],
                    "collection_id": collection_ids[doc["collection_name"]],
                    "tenant_id": doc["tenant_id"],
                    "external_id": doc["external_id"],
                    "title": doc["title"],
                    "source_type": doc["source_type"],
                    "source_uri": doc["source_uri"],
                    "mime_type": doc["mime_type"],
                    "checksum": doc["checksum"],
                    "status": doc["status"],
                    "owner_username": doc["owner_username"],
                    "metadata_json": _jsonb(doc["metadata_json"]),
                },
            )
            result["documents_upserted"] += 1
        for chunk in plan.chunk_mappings:
            conn.execute(
                text(
                    "INSERT INTO rag_chunks (id, document_id, collection_id, chunk_index, token_count, vector_id, embedding_model, content_hash, metadata_json, created_at) "
                    "VALUES (:id, :document_id, :collection_id, :chunk_index, :token_count, :vector_id, :embedding_model, :content_hash, CAST(:metadata_json AS jsonb), now()) "
                    "ON CONFLICT (document_id, chunk_index) DO UPDATE SET vector_id=EXCLUDED.vector_id, content_hash=EXCLUDED.content_hash"
                ),
                {
                    "id": chunk["rag_chunk_id"],
                    "document_id": chunk["rag_document_id"],
                    "collection_id": collection_ids[chunk["collection_name"]],
                    "chunk_index": chunk["chunk_index"],
                    "token_count": chunk["token_count"],
                    "vector_id": chunk["vector_id"],
                    "embedding_model": chunk["embedding_model"],
                    "content_hash": chunk["content_hash"],
                    "metadata_json": _jsonb(chunk["metadata_json"]),
                },
            )
            result["chunks_upserted"] += 1
        for vector in plan.vector_mappings:
            embedding_model_id = embedding_model_ids.get(
                (vector["embedding_provider"], vector["embedding_model"], int(vector["embedding_dimensions"]))
            )
            conn.execute(
                text(
                    "INSERT INTO rag_vector_mappings (id, chunk_id, collection_name, point_id, embedding_model_id, status, created_at) "
                    "VALUES (:id, :chunk_id, :collection_name, :point_id, :embedding_model_id, 'active', now()) "
                    "ON CONFLICT (collection_name, point_id) DO UPDATE SET status='active'"
                ),
                {
                    "id": _deterministic_uuid("vector_mapping", f"{vector['collection_name']}:{vector['point_id']}"),
                    "chunk_id": vector["rag_chunk_id"],
                    "collection_name": vector["collection_name"],
                    "point_id": vector["point_id"],
                    "embedding_model_id": embedding_model_id,
                },
            )
            result["vector_mappings_upserted"] += 1
        for doc in plan.document_mappings:
            conn.execute(
                text(
                    "UPDATE rag_documents SET chunk_count = (SELECT count(*) FROM rag_chunks WHERE rag_chunks.document_id = rag_documents.id), updated_at=now() WHERE id=:id"
                ),
                {"id": doc["rag_document_id"]},
            )

    with source_engine.begin() as conn:
        for mapping in plan.chatbot_document_mappings:
            conn.execute(
                text(
                    "INSERT INTO chatbot_documents (local_document_id, rag_document_id, rag_collection_id, rag_collection_name, tenant_id, metadata_json, created_at, updated_at) "
                    "VALUES (:local_document_id, :rag_document_id, NULL, :rag_collection_name, :tenant_id, CAST(:metadata_json AS jsonb), now(), now()) "
                    "ON CONFLICT (local_document_id) DO UPDATE SET rag_document_id=EXCLUDED.rag_document_id, rag_collection_name=EXCLUDED.rag_collection_name, updated_at=now()"
                ),
                {
                    "local_document_id": mapping["local_document_id"],
                    "rag_document_id": mapping["rag_document_id"],
                    "rag_collection_name": mapping["rag_collection_name"],
                    "tenant_id": mapping["tenant_id"],
                    "metadata_json": _jsonb(mapping["metadata_json"]),
                },
            )
            result["chatbot_document_mappings_upserted"] += 1
    return result


def _split_collection_names(raw: Sequence[str] | None) -> list[str]:
    names: list[str] = []
    for item in raw or []:
        for name in str(item).split(","):
            name = name.strip()
            if name:
                names.append(name)
    return names or [os.getenv("VECTOR_STORE_COLLECTION", _DEFAULT_COLLECTION_NAME)]


def load_qdrant_collection_names(
    *,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    timeout_seconds: float = 10.0,
) -> list[str]:
    """Read Qdrant collection names for registration planning only.

    This is intentionally read-only: it calls list/get collection metadata and never
    creates, updates, deletes, or aliases Qdrant collections.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=qdrant_url or os.getenv("VECTOR_STORE_URL") or "http://127.0.0.1:6333",
        api_key=qdrant_api_key or os.getenv("VECTOR_STORE_API_KEY") or None,
        timeout=timeout_seconds,
    )
    response = client.get_collections()
    collections = getattr(response, "collections", response)
    names: list[str] = []
    for collection in collections or []:
        name = getattr(collection, "name", None)
        if name:
            names.append(str(name))
    return names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe dry-run DominicBE -> rag-core Phase 9 backfill planner/apply tool")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Plan only; default and safe mode")
    parser.add_argument("--apply", action="store_true", help="Execute non-destructive upserts. Required for writes")
    parser.add_argument("--source-database-url", default=os.getenv("CHATBOT_DATABASE_URL") or os.getenv("DATABASE_URL"), help="DominicBE chatbot_db SQLAlchemy URL")
    parser.add_argument("--rag-core-database-url", default=os.getenv("RAG_CORE_DATABASE_URL"), help="rag_core SQLAlchemy URL")
    parser.add_argument("--tenant-id", default=os.getenv("RAG_CORE_BACKFILL_TENANT_ID", "default"))
    parser.add_argument("--collection-name", action="append", dest="collection_names", help="Existing Qdrant collection name to register; repeatable or comma-separated")
    parser.add_argument("--scan-qdrant", action="store_true", help="Read-only scan of Qdrant collections to add to registration plan")
    parser.add_argument("--qdrant-url", default=os.getenv("VECTOR_STORE_URL"), help="Qdrant URL for --scan-qdrant")
    parser.add_argument("--qdrant-api-key", default=os.getenv("VECTOR_STORE_API_KEY"), help="Qdrant API key for --scan-qdrant")
    parser.add_argument("--qdrant-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--embedding-provider", default=os.getenv("EMBEDDING_PROVIDER", "local"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "local-hash-v1"))
    parser.add_argument("--embedding-dimensions", type=int, default=int(os.getenv("EMBEDDING_DIMENSIONS", "64")))
    parser.add_argument("--file-strategy", choices=["reference", "copy"], default="reference", help="reference is default; copy plans/copies into rag-source-files only with --enable-file-copy")
    parser.add_argument("--enable-file-copy", action="store_true", help="With --apply and --file-strategy copy, permit guarded file copy actions")
    parser.add_argument("--object-storage-key-prefix", default=os.getenv("RAG_CORE_OBJECT_STORAGE_KEY_PREFIX", ""))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default="", help="Optional path for the full JSON plan/result manifest")
    parser.add_argument("--print-rollback-notes", action="store_true", help="Print rollback notes after the plan")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = BackfillConfig(
        apply=bool(args.apply),
        tenant_id=args.tenant_id,
        collection_names=_split_collection_names(args.collection_names),
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        file_strategy=args.file_strategy,
        enable_file_copy=bool(args.enable_file_copy),
        object_storage_key_prefix=args.object_storage_key_prefix,
        limit=args.limit,
    )
    if args.apply and not args.source_database_url:
        parser.error("--apply requires --source-database-url or CHATBOT_DATABASE_URL/DATABASE_URL")
    if args.apply and not args.rag_core_database_url:
        parser.error("--apply requires --rag-core-database-url or RAG_CORE_DATABASE_URL")
    if not args.source_database_url:
        # Support offline/static CLI validation without touching live DBs.
        docs: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
    else:
        docs, chunks = load_source_rows(args.source_database_url, limit=args.limit)
    qdrant_collection_names = []
    if args.scan_qdrant:
        qdrant_collection_names = load_qdrant_collection_names(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            timeout_seconds=args.qdrant_timeout_seconds,
        )
    plan = BackfillPlanner(config).build_plan(docs, chunks, qdrant_collection_names)
    output: dict[str, Any] = {"plan": plan.to_dict(), "qdrant_scan": {"enabled": bool(args.scan_qdrant), "collection_count": len(qdrant_collection_names)}}
    if args.apply:
        output["apply_result"] = apply_plan(args.source_database_url, args.rag_core_database_url, plan)
    rendered = json.dumps(_json_safe(output), indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.print_rollback_notes:
        print("\nRollback notes:\n" + plan.rollback_notes)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
