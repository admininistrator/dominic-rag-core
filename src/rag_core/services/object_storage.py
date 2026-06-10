"""rag-core-owned object storage helpers for source file intake.

rag-core owns the ``rag-source-files`` bucket/prefix. DominicBE must not read or
write this namespace directly; callers send file bytes to the rag-core API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import re

from rag_core.api.config import RagCoreServiceSettings, get_service_settings


_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_FILENAME_LENGTH = 180


@dataclass(frozen=True)
class StoredObject:
    provider: str
    bucket: str
    key: str
    uri: str
    content_type: str | None
    size_bytes: int
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "bucket": self.bucket,
            "key": self.key,
            "uri": self.uri,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }


def _normalize_provider(settings: RagCoreServiceSettings) -> str:
    return (settings.object_storage_provider or "local").strip().lower()


def _sanitize_component(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[./\\]+", "-", (value or "").strip())
    normalized = _SAFE_COMPONENT_RE.sub("-", normalized)
    normalized = normalized.strip(".-_")
    return normalized.lower() or fallback


def sanitize_filename(filename: str | None) -> str:
    """Return a storage-safe filename with path traversal removed."""
    candidate = Path(filename or "").name.strip()
    if not candidate or candidate in {".", ".."}:
        return "upload.bin"
    cleaned = _SAFE_COMPONENT_RE.sub("-", candidate).strip(".-_")
    if not cleaned:
        return "upload.bin"
    path = Path(cleaned)
    if path.suffix:
        stem = path.stem.strip(".-_") or "upload"
        cleaned = f"{stem}{path.suffix}"
    if len(cleaned) <= _MAX_FILENAME_LENGTH:
        return cleaned
    stem = Path(cleaned).stem[:120].strip(".-_") or "upload"
    suffix = Path(cleaned).suffix[:32]
    return f"{stem}{suffix}"[:_MAX_FILENAME_LENGTH]


def build_source_file_key(
    *,
    tenant_id: str,
    document_id: str,
    filename: str | None,
    key_prefix: str = "",
) -> str:
    """Build the rag-core-owned source-file key.

    Shape: ``{optional_prefix/}{tenant}/{document_id}/{safe_filename}``.
    This keeps source files tenant/document scoped without exposing DominicBE's
    ``dominic-knowledge`` bucket/prefix to rag-core.
    """
    tenant_part = _sanitize_component(tenant_id, fallback="default")
    document_part = _sanitize_component(str(document_id), fallback="document")
    filename_part = sanitize_filename(filename)
    parts: list[str] = []
    prefix = (key_prefix or "").strip().strip("/")
    if prefix:
        safe_prefix = "/".join(
            _sanitize_component(part, fallback="prefix")
            for part in prefix.split("/")
            if part.strip()
        )
        if safe_prefix:
            parts.append(safe_prefix)
    parts.extend([tenant_part, document_part, filename_part])
    return "/".join(parts)


def _normalized_endpoint_url(settings: RagCoreServiceSettings) -> str | None:
    endpoint = (settings.object_storage_endpoint or "").strip()
    if not endpoint:
        return None
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    scheme = "https" if settings.object_storage_secure else "http"
    return f"{scheme}://{endpoint}"


def _client_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code") or "")
    return ""


def _is_missing_bucket_error(exc: Exception) -> bool:
    code = _client_error_code(exc).lower()
    return code in {"404", "nosuchbucket", "notfound"}


class ObjectStorageService:
    """Small S3/MinIO/local storage wrapper using rag-core settings."""

    def __init__(self, settings: RagCoreServiceSettings | None = None, *, s3_client: Any | None = None) -> None:
        self.settings = settings or get_service_settings()
        self.provider = _normalize_provider(self.settings)
        self._s3_client = s3_client

    @property
    def bucket(self) -> str:
        return self.settings.object_storage_bucket

    def _local_root(self) -> Path:
        return Path(self.settings.object_storage_local_path).expanduser().resolve()

    def _client(self):
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep installed
            raise RuntimeError("boto3 is required for s3/minio object storage support.") from exc
        self._s3_client = boto3.client(
            "s3",
            endpoint_url=_normalized_endpoint_url(self.settings),
            aws_access_key_id=self.settings.object_storage_access_key or None,
            aws_secret_access_key=self.settings.object_storage_secret_key or None,
            region_name=self.settings.object_storage_region or None,
            use_ssl=self.settings.object_storage_secure,
        )
        return self._s3_client

    def ensure_bucket_exists(self) -> dict[str, Any]:
        """Create the rag-core bucket if missing; never deletes or mutates other buckets."""
        if self.provider == "local":
            self._local_root().mkdir(parents=True, exist_ok=True)
            return {"provider": self.provider, "bucket": self.bucket, "created": False}
        if self.provider not in {"s3", "minio"}:
            raise ValueError(f"Unsupported object storage provider: {self.settings.object_storage_provider}")

        client = self._client()
        try:
            client.head_bucket(Bucket=self.bucket)
            return {"provider": self.provider, "bucket": self.bucket, "created": False}
        except Exception as exc:
            if not _is_missing_bucket_error(exc):
                raise

        create_kwargs: dict[str, Any] = {"Bucket": self.bucket}
        region = (self.settings.object_storage_region or "").strip()
        if region and region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        client.create_bucket(**create_kwargs)
        return {"provider": self.provider, "bucket": self.bucket, "created": True}

    def upload_source_file(
        self,
        *,
        tenant_id: str,
        document_id: str,
        filename: str | None,
        content: bytes,
        content_type: str | None,
        checksum: str | None = None,
    ) -> StoredObject:
        key = build_source_file_key(
            tenant_id=tenant_id,
            document_id=document_id,
            filename=filename,
            key_prefix=self.settings.object_storage_key_prefix,
        )
        self.ensure_bucket_exists()

        if self.provider == "local":
            destination = self._local_root() / self.bucket / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            return StoredObject(
                provider=self.provider,
                bucket=self.bucket,
                key=key,
                uri=f"local://{self.bucket}/{key}",
                content_type=content_type,
                size_bytes=len(content),
                checksum=checksum,
            )

        if self.provider in {"s3", "minio"}:
            put_kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key, "Body": content}
            if content_type:
                put_kwargs["ContentType"] = content_type
            if checksum:
                put_kwargs["Metadata"] = {"sha256": checksum}
            self._client().put_object(**put_kwargs)
            return StoredObject(
                provider=self.provider,
                bucket=self.bucket,
                key=key,
                uri=f"s3://{self.bucket}/{key}",
                content_type=content_type,
                size_bytes=len(content),
                checksum=checksum,
            )

        raise ValueError(f"Unsupported object storage provider: {self.settings.object_storage_provider}")

    def head_object(self, key: str) -> dict[str, Any]:
        if self.provider == "local":
            path = self._local_root() / self.bucket / key
            stat = path.stat()
            return {"provider": self.provider, "bucket": self.bucket, "key": key, "size_bytes": stat.st_size}
        response = self._client().head_object(Bucket=self.bucket, Key=key)
        return {
            "provider": self.provider,
            "bucket": self.bucket,
            "key": key,
            "size_bytes": int(response.get("ContentLength") or 0),
            "content_type": response.get("ContentType"),
        }

    def read_bytes(self, key: str) -> bytes:
        if self.provider == "local":
            return (self._local_root() / self.bucket / key).read_bytes()
        body = self._client().get_object(Bucket=self.bucket, Key=key)["Body"]
        return body.read()

    def presigned_get_url(self, key: str, *, expires_in: int = 3600) -> str | None:
        if self.provider == "local":
            return None
        client = self._client()
        generator = getattr(client, "generate_presigned_url", None)
        if not callable(generator):
            return None
        return generator(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )


def bootstrap_object_storage(settings: RagCoreServiceSettings | None = None) -> dict[str, Any]:
    return ObjectStorageService(settings).ensure_bucket_exists()


def check_object_storage_health(settings: RagCoreServiceSettings | None = None) -> dict[str, Any]:
    started = perf_counter()
    service = ObjectStorageService(settings)
    base = {"provider": service.provider, "bucket": service.bucket}
    try:
        result = service.ensure_bucket_exists()
        return {**base, "ok": True, "bucket_created": bool(result.get("created")), "latency_ms": round((perf_counter() - started) * 1000, 2)}
    except Exception as exc:
        return {**base, "ok": False, "error_type": type(exc).__name__, "latency_ms": round((perf_counter() - started) * 1000, 2)}
