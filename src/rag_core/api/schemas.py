"""Pydantic contracts for the internal rag-core HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class EmbeddingMetaPayload(BaseModel):
    provider: str = ""
    model: str = ""
    dimensions: int = 0
    version: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class CollectionCreateRequest(BaseModel):
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    embedding_provider: str = Field(min_length=1, max_length=64)
    embedding_model: str = Field(min_length=1, max_length=128)
    embedding_dimensions: int = Field(gt=0)
    name: str | None = Field(default=None, max_length=63)
    display_name: str | None = Field(default=None, max_length=255)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class CollectionResponse(BaseModel):
    id: str
    name: str
    display_name: str | None = None
    tenant_id: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    status: str
    document_count: int = 0
    vector_count: int = 0
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class CollectionListResponse(BaseModel):
    collections: list[CollectionResponse]
    count: int


class CollectionDeleteResponse(BaseModel):
    ok: bool
    deleted: bool
    collection: CollectionResponse


class StorageObjectPayload(BaseModel):
    provider: str
    bucket: str
    key: str
    uri: str
    content_type: str | None = None
    size_bytes: int
    checksum: str | None = None


class DocumentIntakeResponse(BaseModel):
    document_id: str
    job_id: str
    status: str
    collection_id: str
    collection_name: str
    source_uri: str
    storage: StorageObjectPayload
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class IndexingPrepareRequest(BaseModel):
    document_id: int = Field(ge=1)
    checksum: str = Field(min_length=1)
    chunks: list[dict[str, Any]]
    store_embeddings_in_metadata: bool | None = None
    index_provider: str | None = None


class IndexingPrepareResponse(BaseModel):
    prepared_chunks: list[dict[str, Any]]
    embedding_meta: EmbeddingMetaPayload | None = None


class VectorDocumentPayload(BaseModel):
    owner_username: str = Field(min_length=1)
    document_id: int = Field(ge=1)
    title: str
    source_type: str
    source_uri: str | None = None
    session_id: int | None = None


class VectorChunkRowPayload(BaseModel):
    id: int = Field(ge=1)
    chunk_index: int = Field(ge=0)


class VectorUpsertRequest(BaseModel):
    document: VectorDocumentPayload
    chunk_rows: list[VectorChunkRowPayload]
    prepared_chunks: list[dict[str, Any]]
    embedding_provider: str | None = None
    embedding_model: str | None = None


class VectorUpsertResponse(BaseModel):
    ok: bool
    upserted: bool


class VectorDeleteRequest(BaseModel):
    owner_username: str = Field(min_length=1)
    document_id: int = Field(ge=1)


class VectorDeleteResponse(BaseModel):
    ok: bool
    deleted: bool


class VectorHitPayload(BaseModel):
    chunk_id: int
    document_id: int
    score: float
    vector_id: str


class VectorSearchRequest(BaseModel):
    owner_username: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=200)
    query: str | None = None
    rewritten_query: str | None = None
    query_vector: list[float] | None = None
    document_id: int | None = Field(default=None, ge=1)
    session_id: int | None = Field(default=None, ge=1)
    session_scope: str = "all"

    @model_validator(mode="after")
    def _require_query_or_vector(self):
        if not self.query_vector and not ((self.rewritten_query or self.query or "").strip()):
            raise ValueError("query, rewritten_query, or query_vector is required.")
        return self


class VectorSearchResponse(BaseModel):
    vector_hits: list[VectorHitPayload]
    rewritten_query: str | None = None
    query_expansions: list[str] = Field(default_factory=list)
    embedding_meta: EmbeddingMetaPayload | None = None
    vector_store_attempted: bool = False
    vector_store_failed: bool = False
    vector_store_error_type: str | None = None


class RetrievalCandidatePayload(BaseModel):
    chunk_id: int = Field(ge=1)
    document_id: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    title: str
    source_type: str
    source_uri: str | None = None
    content: str
    token_count: int | None = None
    vector_id: str | None = None
    embedding_model: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class RetrievalRankRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=200)
    candidates: list[RetrievalCandidatePayload]
    rewritten_query: str | None = None
    query_expansions: list[str] = Field(default_factory=list)
    semantic_scores_by_chunk_id: dict[str, float] = Field(default_factory=dict)
    embedding_meta: EmbeddingMetaPayload | None = None
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class RetrievalRankResponse(BaseModel):
    results: list[dict[str, Any]]
    candidate_count: int
    matched_count: int
    reranked_count: int
    rewritten_query: str
    query_expansions: list[str]
    evidence_strength: str
    embedding_meta: EmbeddingMetaPayload
    mixed_space_skip_count: int = 0
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    retrieval_traces: list[dict[str, Any]] = Field(default_factory=list)


class ContextPackRequest(BaseModel):
    results: list[dict[str, Any]]
    max_context_chunks: int | None = Field(default=None, ge=1, le=200)
    max_context_tokens: int | None = Field(default=None, ge=1)


class ContextPackResponse(BaseModel):
    packed_results: list[dict[str, Any]]
    packed_token_estimate: int


# ── Phase 7: Lifecycle / Query / Job schemas ──────────────────────────────


class DocumentGetResponse(BaseModel):
    document_id: str
    tenant_id: str
    collection_id: str
    collection_name: str
    external_id: str | None = None
    title: str | None = None
    source_type: str
    source_uri: str | None = None
    mime_type: str | None = None
    checksum: str | None = None
    file_size_bytes: int | None = None
    status: str
    chunk_count: int = 0
    owner_username: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    deleted_at: str | None = None


class DocumentDeleteResponse(BaseModel):
    ok: bool
    deleted: bool
    document_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    step_current: str | None = None
    step_progress: float | None = None
    chunks_total: int | None = None
    chunks_processed: int = 0
    error_code: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: str | None = None
    owner_username: str | None = None
    top_k: int = Field(default=5, ge=1, le=200)
    collection_id: str | None = None
    collection_name: str | None = None
    filters: dict[str, Any] | None = None


class QueryResultPayload(BaseModel):
    document_id: str
    chunk_id: str | None = None
    chunk_index: int = 0
    score: float
    content: str
    title: str | None = None
    source_type: str | None = None
    source_uri: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    top_k: int
    returned: int
    results: list[QueryResultPayload]

