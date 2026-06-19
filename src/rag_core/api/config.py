"""Environment-backed configuration for the rag-core API service.

Phase 1 (RCSI-P1) additions:
  T01 - Database config:       RAG_CORE_DATABASE_URL / RAG_CORE_DB_* vars.
  T02 - Redis/Celery config:   RAG_CORE_CELERY_BROKER_URL / RAG_CORE_CELERY_RESULT_BACKEND / RAG_CORE_QUEUE_* vars.
  T03 - Object storage config: RAG_CORE_OBJECT_STORAGE_* vars.

Ownership boundaries (MUST NOT be crossed):
  - rag-core owns: rag_core Postgres DB, rag-source-files MinIO bucket, Redis DB 2/3, rag.* Celery queues.
  - DominicBE owns: chatbot_db Postgres DB, dominic-knowledge MinIO bucket, Redis DB 0/1, chatbot.* queues.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from typing import Any


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_str(name: str) -> "str | None":
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


@dataclass(frozen=True)
class RagCoreServiceSettings:
    # ---- Existing API service settings ----------------------------------------
    app_name: str = "rag-core"
    host: str = "0.0.0.0"
    port: int = 8010
    api_key: str = ""

    embedding_provider: str = "local"
    embedding_model: str = "local-hash-v1"
    embedding_dimensions: int = 64
    embedding_base_url: str = "http://localhost:11434"
    embedding_timeout_seconds: float = 60.0
    embedding_batch_size: int = 16
    embedding_api_key: str = ""
    embedding_api_type: str = ""
    embedding_api_version: str = ""
    embedding_api_headers: str = ""

    ingestion_pipeline: str = "custom"
    chunk_size: int = 800
    chunk_overlap: int = 100

    vector_store_provider: str = "database"
    vector_store_url: "str | None" = None
    vector_store_api_key: "str | None" = None
    vector_store_collection: str = "rag_default_local_local_hash_v1"
    vector_store_timeout_seconds: float = 10.0
    vector_store_prefer_grpc: bool = False

    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.15
    retrieval_min_lexical_score: float = 0.1
    retrieval_hybrid_semantic_weight: float = 0.4
    retrieval_hybrid_lexical_weight: float = 0.6
    retrieval_enable_query_expansion: bool = True
    retrieval_max_rerank_candidates: int = 12
    retrieval_rerank_title_weight: float = 0.15
    retrieval_rerank_position_weight: float = 0.1
    retrieval_low_confidence_score: float = 0.2
    retrieval_max_context_tokens: int = 4000
    retrieval_max_context_chunks: int = 6

    # ---- RCSI-P1-T01: PostgreSQL database config (rag_core database) -----------
    # rag-core owns the "rag_core" database. DominicBE owns "chatbot_db".
    # They share the same Postgres container in local/dev but are separate databases.
    # RAG_CORE_DATABASE_URL overrides all RAG_CORE_DB_* component vars when set.
    database_url: "str | None" = None       # RAG_CORE_DATABASE_URL
    db_host: str = "127.0.0.1"             # RAG_CORE_DB_HOST
    db_port: int = 5432                     # RAG_CORE_DB_PORT
    db_user: str = "rag_core"              # RAG_CORE_DB_USER
    db_password: str = ""                   # RAG_CORE_DB_PASSWORD (no real value in .env.example)
    db_name: str = "rag_core"              # RAG_CORE_DB_NAME
    db_pool_size: int = 5                   # RAG_CORE_DB_POOL_SIZE
    db_max_overflow: int = 10              # RAG_CORE_DB_MAX_OVERFLOW
    db_pool_recycle: int = 300             # RAG_CORE_DB_POOL_RECYCLE (seconds)
    db_pool_timeout: int = 10             # RAG_CORE_DB_POOL_TIMEOUT (seconds)
    db_connect_timeout: int = 10          # RAG_CORE_DB_CONNECT_TIMEOUT (seconds)
    db_ssl: bool = False                   # RAG_CORE_DB_SSL

    # ---- RCSI-P1-T02: Redis / Celery config ------------------------------------
    # rag-core uses Redis DB 2 (broker) and DB 3 (results).
    # DominicBE uses Redis DB 0 (broker) and DB 1 (results).
    # NEVER share DB indices between rag-core and DominicBE workers.
    # rag-core Celery app name: "rag_core". Queue namespace: "rag.*".
    # RAG_CORE_CELERY_ENABLED defaults False — API starts without a worker process.
    celery_enabled: bool = False                              # RAG_CORE_CELERY_ENABLED

    # Synchronous ingestion fallback — runs the full ingestion pipeline inline
    # within the upload HTTP request when Celery is disabled.
    # Enable with RAG_CORE_SYNC_INGESTION_ENABLED=true in dl-rag-core.env.
    # When both celery_enabled and sync_ingestion_enabled are true, Celery takes
    # priority (async dispatch used instead of inline processing).
    # Backward-compatible default: False — existing DominicBE behavior unchanged.
    sync_ingestion_enabled: bool = False                      # RAG_CORE_SYNC_INGESTION_ENABLED
    celery_broker_url: str = "redis://127.0.0.1:6379/2"      # RAG_CORE_CELERY_BROKER_URL
    celery_result_backend: str = "redis://127.0.0.1:6379/3"  # RAG_CORE_CELERY_RESULT_BACKEND
    celery_app_name: str = "rag_core"                         # RAG_CORE_CELERY_APP_NAME
    celery_default_queue: str = "rag.ingestion"              # RAG_CORE_CELERY_DEFAULT_QUEUE
    celery_task_serializer: str = "json"                      # RAG_CORE_CELERY_TASK_SERIALIZER
    celery_result_serializer: str = "json"                    # RAG_CORE_CELERY_RESULT_SERIALIZER
    celery_worker_concurrency: int = 2                        # RAG_CORE_CELERY_WORKER_CONCURRENCY
    celery_task_time_limit: int = 3600                        # RAG_CORE_CELERY_TASK_TIME_LIMIT (seconds)
    celery_task_soft_time_limit: int = 3300                  # RAG_CORE_CELERY_TASK_SOFT_TIME_LIMIT (seconds)
    celery_max_tasks_per_child: int = 100                     # RAG_CORE_CELERY_MAX_TASKS_PER_CHILD
    queue_ingestion: str = "rag.ingestion"                   # RAG_CORE_QUEUE_INGESTION
    queue_embedding: str = "rag.embedding"                   # RAG_CORE_QUEUE_EMBEDDING
    queue_reindex: str = "rag.reindex"                       # RAG_CORE_QUEUE_REINDEX
    queue_cleanup: str = "rag.cleanup"                       # RAG_CORE_QUEUE_CLEANUP
    queue_eval: str = "rag.eval"                             # RAG_CORE_QUEUE_EVAL

    # ---- RCSI-P1-T03: Object storage config ------------------------------------
    # rag-core owns the "rag-source-files" MinIO bucket.
    # DominicBE owns the "dominic-knowledge" bucket.
    # Key pattern: {tenant}/{document_id}/{filename}
    object_storage_provider: str = "local"        # RAG_CORE_OBJECT_STORAGE_PROVIDER (local|minio|s3)
    object_storage_bucket: str = "rag-source-files"  # RAG_CORE_OBJECT_STORAGE_BUCKET
    object_storage_endpoint: "str | None" = None  # RAG_CORE_OBJECT_STORAGE_ENDPOINT
    object_storage_access_key: "str | None" = None  # RAG_CORE_OBJECT_STORAGE_ACCESS_KEY
    object_storage_secret_key: "str | None" = None  # RAG_CORE_OBJECT_STORAGE_SECRET_KEY
    object_storage_region: "str | None" = None    # RAG_CORE_OBJECT_STORAGE_REGION
    object_storage_secure: bool = True             # RAG_CORE_OBJECT_STORAGE_SECURE
    object_storage_local_path: str = ".rag-storage"  # RAG_CORE_OBJECT_STORAGE_LOCAL_PATH
    object_storage_key_prefix: str = ""            # RAG_CORE_OBJECT_STORAGE_KEY_PREFIX
    object_storage_max_upload_bytes: int = 25 * 1024 * 1024  # RAG_CORE_OBJECT_STORAGE_MAX_UPLOAD_BYTES

    @property
    def embedding_headers(self) -> dict[str, str]:
        raw = (self.embedding_api_headers or "").strip()
        if not raw:
            return {}
        try:
            parsed: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(key): str(value) for key, value in parsed.items()}

    @property
    def qdrant_enabled(self) -> bool:
        return (self.vector_store_provider or "database").strip().lower() == "qdrant"

    @property
    def store_embeddings_in_metadata(self) -> bool:
        return not self.qdrant_enabled

    @property
    def sqlalchemy_database_url(self) -> str:
        """Build the SQLAlchemy connection URL for the rag_core database.

        RAG_CORE_DATABASE_URL takes precedence when set. Falls back to the
        individual RAG_CORE_DB_* components.

        The rag_core database is SEPARATE from DominicBE's chatbot_db.
        Do NOT point this at chatbot_db or share connections across services.
        """
        if (self.database_url or "").strip():
            return self.database_url.strip()
        from urllib.parse import quote_plus
        encoded_password = quote_plus(self.db_password)
        return (
            f"postgresql+psycopg://{self.db_user}:{encoded_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache(maxsize=1)
def get_service_settings() -> RagCoreServiceSettings:
    return RagCoreServiceSettings(
        app_name=_env_str("RAG_CORE_APP_NAME", "rag-core"),
        host=_env_str("RAG_CORE_HOST", "0.0.0.0"),
        port=_env_int("RAG_CORE_PORT", 8010),
        api_key=_env_str("RAG_CORE_API_KEY", ""),
        embedding_provider=_env_str("EMBEDDING_PROVIDER", "local"),
        embedding_model=_env_str("EMBEDDING_MODEL", "local-hash-v1"),
        embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 64),
        embedding_base_url=_env_str("EMBEDDING_BASE_URL", "http://localhost:11434"),
        embedding_timeout_seconds=_env_float("EMBEDDING_TIMEOUT_SECONDS", 60.0),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 16),
        embedding_api_key=_env_str("EMBEDDING_API_KEY", ""),
        embedding_api_type=_env_str("EMBEDDING_API_TYPE", ""),
        embedding_api_version=_env_str("EMBEDDING_API_VERSION", ""),
        embedding_api_headers=_env_str("EMBEDDING_API_HEADERS", ""),
        ingestion_pipeline=_env_str("INGESTION_PIPELINE", "custom"),
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
        vector_store_provider=_env_str("VECTOR_STORE_PROVIDER", "database"),
        vector_store_url=_env_optional_str("VECTOR_STORE_URL"),
        vector_store_api_key=_env_optional_str("VECTOR_STORE_API_KEY"),
        vector_store_collection=_env_str("VECTOR_STORE_COLLECTION", "rag_default_local_local_hash_v1"),
        vector_store_timeout_seconds=_env_float("VECTOR_STORE_TIMEOUT_SECONDS", 10.0),
        vector_store_prefer_grpc=_env_bool("VECTOR_STORE_PREFER_GRPC", False),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 5),
        retrieval_min_score=_env_float("RETRIEVAL_MIN_SCORE", 0.15),
        retrieval_min_lexical_score=_env_float("RETRIEVAL_MIN_LEXICAL_SCORE", 0.1),
        retrieval_hybrid_semantic_weight=_env_float("RETRIEVAL_HYBRID_SEMANTIC_WEIGHT", 0.4),
        retrieval_hybrid_lexical_weight=_env_float("RETRIEVAL_HYBRID_LEXICAL_WEIGHT", 0.6),
        retrieval_enable_query_expansion=_env_bool("RETRIEVAL_ENABLE_QUERY_EXPANSION", True),
        retrieval_max_rerank_candidates=_env_int("RETRIEVAL_MAX_RERANK_CANDIDATES", 12),
        retrieval_rerank_title_weight=_env_float("RETRIEVAL_RERANK_TITLE_WEIGHT", 0.15),
        retrieval_rerank_position_weight=_env_float("RETRIEVAL_RERANK_POSITION_WEIGHT", 0.1),
        retrieval_low_confidence_score=_env_float("RETRIEVAL_LOW_CONFIDENCE_SCORE", 0.2),
        retrieval_max_context_tokens=_env_int("RETRIEVAL_MAX_CONTEXT_TOKENS", 4000),
        retrieval_max_context_chunks=_env_int("RETRIEVAL_MAX_CONTEXT_CHUNKS", 6),
        # RCSI-P1-T01: Database config
        database_url=_env_optional_str("RAG_CORE_DATABASE_URL"),
        db_host=_env_str("RAG_CORE_DB_HOST", "127.0.0.1"),
        db_port=_env_int("RAG_CORE_DB_PORT", 5432),
        db_user=_env_str("RAG_CORE_DB_USER", "rag_core"),
        db_password=_env_str("RAG_CORE_DB_PASSWORD", ""),
        db_name=_env_str("RAG_CORE_DB_NAME", "rag_core"),
        db_pool_size=_env_int("RAG_CORE_DB_POOL_SIZE", 5),
        db_max_overflow=_env_int("RAG_CORE_DB_MAX_OVERFLOW", 10),
        db_pool_recycle=_env_int("RAG_CORE_DB_POOL_RECYCLE", 300),
        db_pool_timeout=_env_int("RAG_CORE_DB_POOL_TIMEOUT", 10),
        db_connect_timeout=_env_int("RAG_CORE_DB_CONNECT_TIMEOUT", 10),
        db_ssl=_env_bool("RAG_CORE_DB_SSL", False),
        # RCSI-P1-T02: Redis/Celery config
        celery_enabled=_env_bool("RAG_CORE_CELERY_ENABLED", False),
        sync_ingestion_enabled=_env_bool("RAG_CORE_SYNC_INGESTION_ENABLED", False),
        celery_broker_url=_env_str("RAG_CORE_CELERY_BROKER_URL", "redis://127.0.0.1:6379/2"),
        celery_result_backend=_env_str("RAG_CORE_CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/3"),
        celery_app_name=_env_str("RAG_CORE_CELERY_APP_NAME", "rag_core"),
        celery_default_queue=_env_str("RAG_CORE_CELERY_DEFAULT_QUEUE", "rag.ingestion"),
        celery_task_serializer=_env_str("RAG_CORE_CELERY_TASK_SERIALIZER", "json"),
        celery_result_serializer=_env_str("RAG_CORE_CELERY_RESULT_SERIALIZER", "json"),
        celery_worker_concurrency=_env_int("RAG_CORE_CELERY_WORKER_CONCURRENCY", 2),
        celery_task_time_limit=_env_int("RAG_CORE_CELERY_TASK_TIME_LIMIT", 3600),
        celery_task_soft_time_limit=_env_int("RAG_CORE_CELERY_TASK_SOFT_TIME_LIMIT", 3300),
        celery_max_tasks_per_child=_env_int("RAG_CORE_CELERY_MAX_TASKS_PER_CHILD", 100),
        queue_ingestion=_env_str("RAG_CORE_QUEUE_INGESTION", "rag.ingestion"),
        queue_embedding=_env_str("RAG_CORE_QUEUE_EMBEDDING", "rag.embedding"),
        queue_reindex=_env_str("RAG_CORE_QUEUE_REINDEX", "rag.reindex"),
        queue_cleanup=_env_str("RAG_CORE_QUEUE_CLEANUP", "rag.cleanup"),
        queue_eval=_env_str("RAG_CORE_QUEUE_EVAL", "rag.eval"),
        # RCSI-P1-T03: Object storage config
        object_storage_provider=_env_str("RAG_CORE_OBJECT_STORAGE_PROVIDER", "local"),
        object_storage_bucket=_env_str("RAG_CORE_OBJECT_STORAGE_BUCKET", "rag-source-files"),
        object_storage_endpoint=_env_optional_str("RAG_CORE_OBJECT_STORAGE_ENDPOINT"),
        object_storage_access_key=_env_optional_str("RAG_CORE_OBJECT_STORAGE_ACCESS_KEY"),
        object_storage_secret_key=_env_optional_str("RAG_CORE_OBJECT_STORAGE_SECRET_KEY"),
        object_storage_region=_env_optional_str("RAG_CORE_OBJECT_STORAGE_REGION"),
        object_storage_secure=_env_bool("RAG_CORE_OBJECT_STORAGE_SECURE", True),
        object_storage_local_path=_env_str("RAG_CORE_OBJECT_STORAGE_LOCAL_PATH", ".rag-storage"),
        object_storage_key_prefix=_env_str("RAG_CORE_OBJECT_STORAGE_KEY_PREFIX", ""),
        object_storage_max_upload_bytes=_env_int("RAG_CORE_OBJECT_STORAGE_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
    )
