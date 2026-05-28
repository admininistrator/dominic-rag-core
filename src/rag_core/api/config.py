"""Environment-backed configuration for the rag-core API service."""

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


def _env_optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


@dataclass(frozen=True)
class RagCoreServiceSettings:
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
    vector_store_url: str | None = None
    vector_store_api_key: str | None = None
    vector_store_collection: str = "knowledge_chunks"
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
        vector_store_collection=_env_str("VECTOR_STORE_COLLECTION", "knowledge_chunks"),
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
    )

