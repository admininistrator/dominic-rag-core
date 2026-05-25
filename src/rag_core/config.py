"""Core configuration dataclass for rag-core.

``RagCoreConfig`` is a plain dataclass (not Pydantic) that holds all
RAG-related configuration values.  It does **not** read environment
variables — it is populated by the consumer (DominicBE) at the
integration boundary.

All defaults mirror the current DominicBE ``Settings`` RAG defaults,
which are the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=False)
class RagCoreConfig:
    """Configuration for rag-core RAG operations.

    Every field has a default that matches the current DominicBE
    ``app.core.config.Settings`` defaults.  Consumers may override
    individual fields after construction.
    """

    # ── Chunking ────────────────────────────────────────────────────
    chunk_size: int = 800
    """Maximum characters per chunk (default: 800)."""

    chunk_overlap: int = 100
    """Character overlap between consecutive chunks (default: 100)."""

    # ── Embedding ────────────────────────────────────────────────────
    embedding_provider: str = "local"
    """Embedding provider type: 'local', 'ollama', or 'api'."""

    embedding_model: str = "local-hash-v1"
    """Model name used for embedding."""

    embedding_dimensions: int = 64
    """Length of embedding vectors."""

    embedding_base_url: str = "http://localhost:11434"
    """Base URL for remote embedding services."""

    embedding_timeout_seconds: float = 60.0
    """Timeout (seconds) for embedding requests."""

    embedding_batch_size: int = 16
    """Maximum texts per batch embedding call."""

    embedding_api_key: str = ""
    """API key for remote embedding services (never logged)."""

    embedding_api_type: str = ""
    """API format type: 'openai', 'cohere', 'voyage', 'huggingface', or ''."""

    embedding_api_version: str = ""
    """Optional API version string (e.g. '2024-02-01' for Azure)."""

    embedding_api_headers: str = ""
    """JSON string for custom HTTP headers sent with embedding requests."""

    # ── Ingestion ────────────────────────────────────────────────────
    ingestion_pipeline: str = "custom"
    """Ingestion pipeline type: 'custom' or 'llamaindex'."""

    # ── Vector Store ─────────────────────────────────────────────────
    vector_store_provider: str = "database"
    """Vector store provider: 'database', 'qdrant', etc."""

    vector_store_url: str | None = None
    """Vector store connection URL (None = disabled / not configured)."""

    vector_store_api_key: str | None = None
    """API key for vector store authentication."""

    vector_store_collection: str = "knowledge_chunks"
    """Default Qdrant collection name."""

    vector_store_timeout_seconds: float = 10.0
    """Timeout (seconds) for vector store operations."""

    vector_store_prefer_grpc: bool = False
    """Whether to prefer gRPC over HTTP for Qdrant."""

    # ── Retrieval ────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    """Number of top chunks to return per query."""

    retrieval_min_score: float = 0.15
    """Minimum hybrid score threshold for results."""

    retrieval_min_lexical_score: float = 0.1
    """Minimum lexical overlap score threshold."""

    retrieval_hybrid_semantic_weight: float = 0.4
    """Weight for semantic (cosine) score in hybrid scoring."""

    retrieval_hybrid_lexical_weight: float = 0.6
    """Weight for lexical (token overlap) score in hybrid scoring."""

    retrieval_enable_query_expansion: bool = True
    """Whether to apply Vietnamese→English query expansion."""

    retrieval_max_rerank_candidates: int = 12
    """Maximum candidates considered during reranking."""

    retrieval_rerank_title_weight: float = 0.15
    """Weight for title overlap boost during reranking."""

    retrieval_rerank_position_weight: float = 0.1
    """Weight for position-based decay during reranking."""

    retrieval_low_confidence_score: float = 0.2
    """Threshold below which evidence is considered low-confidence."""

    retrieval_strict_grounding_for_scoped_docs: bool = True
    """When ``True`` and a ``knowledge_document_id`` is set, weak evidence
    defaults to ``"insufficient_evidence"`` (default: ``True``)."""

    retrieval_max_context_tokens: int = 4000
    """Maximum total tokens allowed in the evidence context window."""

    retrieval_max_context_chunks: int = 6
    """Maximum chunks included in evidence context."""
