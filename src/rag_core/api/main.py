"""FastAPI application for the internal rag-core service."""

from __future__ import annotations

import json
from secrets import compare_digest
from types import SimpleNamespace
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from rag_core.api.config import RagCoreServiceSettings, get_service_settings
from rag_core.api.schemas import (
    CollectionCreateRequest,
    CollectionDeleteResponse,
    CollectionListResponse,
    CollectionResponse,
    ContextPackRequest,
    ContextPackResponse,
    DocumentDeleteResponse,
    DocumentGetResponse,
    DocumentIntakeResponse,
    EmbeddingMetaPayload,
    IndexingPrepareRequest,
    IndexingPrepareResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    QueryResultPayload,
    RetrievalCandidatePayload,
    RetrievalRankRequest,
    RetrievalRankResponse,
    VectorDeleteRequest,
    VectorDeleteResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorUpsertRequest,
    VectorUpsertResponse,
)
from rag_core.context.builder import _pack_retrieval_results
from rag_core.db.database import get_db
from rag_core.embeddings.base import EmbeddingMeta
from rag_core.embeddings.factory import get_embedding_provider
from rag_core.indexing.pipeline import prepare_chunks_for_indexing
from rag_core.retrieval.deduplicator import _dedupe_scored_results
from rag_core.retrieval.evidence import (
    _build_snippet,
    _classify_evidence_strength,
    _estimate_token_count,
    _is_embedding_compatible,
)
from rag_core.retrieval.query_processor import _expand_query
from rag_core.retrieval.reranker import _rerank_results
from rag_core.retrieval.scoring import _cosine_similarity, _hybrid_score, _lexical_overlap_score
from rag_core.retrieval.contracts import RetrievalCandidate, RetrievalFilters, RetrievalPipeline, RetrievalQuery
from rag_core.retrieval.fusion import ReciprocalRankFusionStrategy
from rag_core.retrieval.reranking import ProviderReranker, RerankProviderResult
from rag_core.retrieval.sparse import LexicalCorpusRetriever
from rag_core.services.collection_registry import (
    CollectionDimensionMismatchError,
    CollectionNotFoundError,
    CollectionRegistrationInput,
    CollectionRegistryError,
    collection_health_summary,
    delete_collection,
    get_collection,
    list_collections,
    register_collection,
)
from rag_core.services.document_intake import (
    DocumentIntakeError,
    DocumentStorageError,
    DocumentUploadInput,
    create_document_from_upload,
)
from rag_core.vector_store.qdrant_adapter import QdrantAdapter
from rag_core.worker.health import check_worker_health


app = FastAPI(title="rag-core", version="0.1.0")


def _health_response(payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=200 if payload.get("ok") else 503, content=payload)


def _sanitize_exception_type(exc: Exception) -> str:
    return type(exc).__name__


def _collection_health_summary() -> dict[str, Any]:
    """Best-effort collection registry health summary for /health and /ready."""
    return collection_health_summary()


def _collection_error(exc: CollectionRegistryError, *, status_code: int) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": getattr(exc, "error_code", "RAG_COLLECTION_ERROR"),
                "message": str(exc),
                "details": {},
            }
        },
    )


def _document_error(code: str, message: str, *, status_code: int, details: dict[str, Any] | None = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error": {"code": code, "message": message, "details": details or {}}},
    )


def _parse_metadata_json(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _document_error(
            "RAG_INVALID_REQUEST",
            "metadata_json must be valid JSON object text.",
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    if not isinstance(parsed, dict):
        raise _document_error(
            "RAG_INVALID_REQUEST",
            "metadata_json must decode to an object.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return dict(parsed)


def _dt_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _collection_to_response(collection: Any) -> CollectionResponse:
    return CollectionResponse(
        id=str(collection.id),
        name=str(collection.name),
        display_name=collection.display_name,
        tenant_id=str(collection.tenant_id),
        embedding_provider=str(collection.embedding_provider),
        embedding_model=str(collection.embedding_model),
        embedding_dimensions=int(collection.embedding_dimensions),
        status=str(collection.status),
        document_count=int(collection.document_count or 0),
        vector_count=int(collection.vector_count or 0),
        metadata_json=dict(collection.metadata_json or {}),
        created_at=_dt_to_iso(collection.created_at),
        updated_at=_dt_to_iso(collection.updated_at),
    )


def _registration_input_from_request(request: CollectionCreateRequest) -> CollectionRegistrationInput:
    return CollectionRegistrationInput(
        tenant_id=request.tenant_id,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
        embedding_dimensions=request.embedding_dimensions,
        name=request.name,
        display_name=request.display_name,
        metadata_json=dict(request.metadata_json or {}),
    )


def _embedding_dimensions_from_prepared_chunks(prepared_chunks: list[dict[str, Any]], fallback: int) -> int:
    if not prepared_chunks:
        return fallback
    first = prepared_chunks[0] or {}
    metadata = dict(first.get("metadata_json") or {})
    raw_dimensions = metadata.get("embedding_dimensions")
    if raw_dimensions:
        try:
            return int(raw_dimensions)
        except (TypeError, ValueError):
            pass
    embedding = first.get("embedding")
    if isinstance(embedding, list) and embedding:
        return len(embedding)
    return fallback


def _register_collection_for_vector_upsert(
    db: Session,
    settings: RagCoreServiceSettings,
    request: VectorUpsertRequest,
) -> None:
    first_meta = dict((request.prepared_chunks[0].get("metadata_json") if request.prepared_chunks else {}) or {})
    provider = request.embedding_provider or str(first_meta.get("embedding_provider") or settings.embedding_provider)
    model = request.embedding_model or str(first_meta.get("embedding_model") or settings.embedding_model)
    dimensions = _embedding_dimensions_from_prepared_chunks(request.prepared_chunks, settings.embedding_dimensions)
    register_collection(
        db,
        CollectionRegistrationInput(
            tenant_id=str(first_meta.get("tenant_id") or "default"),
            embedding_provider=provider,
            embedding_model=model,
            embedding_dimensions=dimensions,
            name=settings.vector_store_collection,
            metadata_json={"registered_from": "vector_upsert"},
        ),
    )


def _meta_payload(meta: EmbeddingMeta | EmbeddingMetaPayload | dict[str, Any] | None) -> EmbeddingMetaPayload:
    if isinstance(meta, EmbeddingMetaPayload):
        return meta
    if isinstance(meta, EmbeddingMeta):
        return EmbeddingMetaPayload(
            provider=meta.provider,
            model=meta.model,
            dimensions=meta.dimensions,
            version=meta.version,
            extra=dict(meta.extra or {}),
        )
    if isinstance(meta, dict):
        return EmbeddingMetaPayload(
            provider=str(meta.get("provider") or ""),
            model=str(meta.get("model") or ""),
            dimensions=int(meta.get("dimensions") or 0),
            version=str(meta.get("version") or ""),
            extra=dict(meta.get("extra") or {}),
        )
    settings = get_service_settings()
    return EmbeddingMetaPayload(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        version=settings.embedding_model or "",
        extra={},
    )


def _get_embedding_provider(settings: RagCoreServiceSettings):
    return get_embedding_provider(
        default_provider=settings.embedding_provider,
        default_model=settings.embedding_model,
        default_dimensions=settings.embedding_dimensions,
        default_base_url=settings.embedding_base_url,
        default_timeout_seconds=settings.embedding_timeout_seconds,
        default_batch_size=settings.embedding_batch_size,
        default_api_key=settings.embedding_api_key,
        default_api_type=settings.embedding_api_type,
        default_api_version=settings.embedding_api_version,
        default_custom_headers=settings.embedding_headers,
    )


def _get_vector_adapter(settings: RagCoreServiceSettings) -> QdrantAdapter:
    return QdrantAdapter(
        collection=settings.vector_store_collection,
        url=settings.vector_store_url,
        api_key=settings.vector_store_api_key,
        timeout_seconds=settings.vector_store_timeout_seconds,
        prefer_grpc=settings.vector_store_prefer_grpc,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
    )


def _require_internal_auth(authorization: str | None = Header(default=None)) -> None:
    settings = get_service_settings()
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG_CORE_API_KEY is not configured.",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token.")
    token = authorization.removeprefix("Bearer ").strip()
    if not compare_digest(token, settings.api_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid bearer token.")


def _extract_stored_embedding(metadata_json: dict[str, Any]) -> list[float] | None:
    embedding = metadata_json.get("embedding")
    if isinstance(embedding, list) and embedding:
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError):
            return None
    return None


def _build_result_from_candidate(
    candidate: RetrievalCandidatePayload,
    *,
    score: float,
    semantic_score: float,
    lexical_score: float,
    embedding_provider: str,
) -> dict[str, Any]:
    chunk_meta = dict(candidate.metadata_json or {})
    normalized = RetrievalCandidate.from_mapping(
        {
            "document_id": candidate.document_id,
            "chunk_id": candidate.chunk_id,
            "chunk_index": candidate.chunk_index,
            "title": candidate.title,
            "source_type": candidate.source_type,
            "source_uri": candidate.source_uri,
            "score": score,
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "token_count": candidate.token_count,
            "token_estimate": _estimate_token_count(candidate.content, candidate.token_count),
            "snippet": _build_snippet(candidate.content),
            "content": candidate.content,
            "vector_id": candidate.vector_id,
            "embedding_model": candidate.embedding_model,
            "embedding_provider": chunk_meta.get("embedding_provider", embedding_provider),
            "metadata_json": chunk_meta,
        },
        source_stage=str(chunk_meta.get("source_stage") or "rank"),
    ).to_result_dict()
    normalized["metadata_json"] = chunk_meta
    return normalized


def _normalize_retrieval_mode(value: Any) -> str:
    normalized = str(value or "hybrid_rerank").strip().lower().replace("-", "_")
    if normalized in {"vector", "vector_only", "dense", "dense_only"}:
        return "vector"
    if normalized in {"hybrid", "hybrid_only"}:
        return "hybrid"
    return "hybrid_rerank"


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def _build_retrieval_config(request: RetrievalRankRequest) -> dict[str, Any]:
    raw = dict(request.retrieval_config or {})
    mode = _normalize_retrieval_mode(raw.get("retrieval_mode"))
    enable_reranker = raw.get("enable_reranker") if "enable_reranker" in raw else None
    resolved_enable_reranker = bool(enable_reranker) if enable_reranker is not None else mode == "hybrid_rerank"
    return {
        "retrieval_mode": mode,
        "enable_reranker": resolved_enable_reranker,
        "dense_top_k": _positive_int(raw.get("dense_top_k"), request.top_k),
        "sparse_top_k": _positive_int(raw.get("sparse_top_k"), request.top_k),
        "fusion_top_k": _positive_int(raw.get("fusion_top_k"), request.top_k),
        "rerank_top_k": _positive_int(raw.get("rerank_top_k"), request.top_k),
        "reranker_provider": str(raw.get("reranker_provider") or "heuristic"),
        "reranker_model": str(raw.get("reranker_model") or "rag-core-heuristic"),
        "trace_id": raw.get("trace_id") or request.trace_id,
    }


class _CandidateListRetriever:
    def __init__(self, candidates: list[dict[str, Any]], *, source_stage: str) -> None:
        self._candidates = list(candidates or [])
        self._source_stage = source_stage

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate.from_mapping(candidate, source_stage=self._source_stage)
            for candidate in self._candidates
        ]


class _HeuristicRerankerProvider:
    provider_name = "rag_core_heuristic"

    def __init__(self, settings: RagCoreServiceSettings, *, model_name: str | None = None) -> None:
        self.settings = settings
        self.model_name = model_name or "rag-core-heuristic"

    def rerank(self, query: RetrievalQuery, candidates: list[RetrievalCandidate]) -> list[RerankProviderResult]:
        payloads = [candidate.to_result_dict() for candidate in candidates]
        reranked = _rerank_results(
            query.effective_text,
            payloads,
            max_rerank_candidates=self.settings.retrieval_max_rerank_candidates,
            rerank_title_weight=self.settings.retrieval_rerank_title_weight,
            rerank_position_weight=self.settings.retrieval_rerank_position_weight,
        )
        used: set[int] = set()
        results: list[RerankProviderResult] = []
        for item in reranked:
            index = _find_candidate_index(item, candidates, used)
            if index is None:
                continue
            used.add(index)
            score = float(item.get("rerank_score") or item.get("score") or 0.0)
            results.append(RerankProviderResult(candidate_index=index, score=score))
        return results


def _find_candidate_index(item: dict[str, Any], candidates: list[RetrievalCandidate], used: set[int]) -> int | None:
    target = (item.get("document_id"), item.get("chunk_id"), item.get("vector_id"), item.get("content"))
    for index, candidate in enumerate(candidates):
        if index in used:
            continue
        identity = (candidate.document_id, candidate.chunk_id, candidate.vector_id, candidate.content)
        if identity == target:
            return index
    return None


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "rag-core", "status": "running"}


@app.get("/health")
def health() -> JSONResponse:
    settings = get_service_settings()
    qdrant = _get_vector_adapter(settings).check_health()
    embedding = {
        "ok": True,
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }
    collections = _collection_health_summary()
    worker = check_worker_health()
    payload = {
        "ok": bool(embedding.get("ok")) and (not settings.qdrant_enabled or bool(qdrant.get("ok"))),
        "service": settings.app_name,
        "dependencies": {
            "embedding": embedding,
            "qdrant": qdrant,
            "collections": collections,
            "worker": worker,
        },
        "api_key_configured": bool(settings.api_key),
    }
    return _health_response(payload)


@app.get("/ready")
def ready() -> JSONResponse:
    settings = get_service_settings()
    qdrant = _get_vector_adapter(settings).check_health()
    collections = _collection_health_summary()
    worker = check_worker_health()
    payload = {
        "ok": bool(settings.api_key) and (not settings.qdrant_enabled or bool(qdrant.get("ok"))),
        "service": settings.app_name,
        "api_key_configured": bool(settings.api_key),
        "dependencies": {"qdrant": qdrant, "collections": collections, "worker": worker},
    }
    return _health_response(payload)


@app.post(
    "/rag/v1/collections",
    response_model=CollectionResponse,
    dependencies=[Depends(_require_internal_auth)],
)
def collection_create(
    request: CollectionCreateRequest,
    db: Session = Depends(get_db),
) -> CollectionResponse:
    try:
        collection = register_collection(db, _registration_input_from_request(request))
    except CollectionDimensionMismatchError as exc:
        raise _collection_error(exc, status_code=status.HTTP_409_CONFLICT) from exc
    except CollectionRegistryError as exc:
        raise _collection_error(exc, status_code=status.HTTP_400_BAD_REQUEST) from exc
    return _collection_to_response(collection)


@app.get(
    "/rag/v1/collections",
    response_model=CollectionListResponse,
    dependencies=[Depends(_require_internal_auth)],
)
def collection_list(
    tenant_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
) -> CollectionListResponse:
    collections = list_collections(db, tenant_id=tenant_id, status_filter=status_filter)
    responses = [_collection_to_response(collection) for collection in collections]
    return CollectionListResponse(collections=responses, count=len(responses))


@app.get(
    "/rag/v1/collections/{identifier}",
    response_model=CollectionResponse,
    dependencies=[Depends(_require_internal_auth)],
)
def collection_get(identifier: str, db: Session = Depends(get_db)) -> CollectionResponse:
    try:
        collection = get_collection(db, identifier)
    except CollectionNotFoundError as exc:
        raise _collection_error(exc, status_code=status.HTTP_404_NOT_FOUND) from exc
    return _collection_to_response(collection)


@app.delete(
    "/rag/v1/collections/{identifier}",
    response_model=CollectionDeleteResponse,
    dependencies=[Depends(_require_internal_auth)],
)
def collection_delete(identifier: str, db: Session = Depends(get_db)) -> CollectionDeleteResponse:
    try:
        collection = delete_collection(db, identifier)
    except CollectionNotFoundError as exc:
        raise _collection_error(exc, status_code=status.HTTP_404_NOT_FOUND) from exc
    return CollectionDeleteResponse(ok=True, deleted=True, collection=_collection_to_response(collection))


@app.post(
    "/rag/v1/documents",
    response_model=DocumentIntakeResponse,
    dependencies=[Depends(_require_internal_auth)],
)
async def document_create(
    tenant_id: str = Form(default="default"),
    external_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
    collection_id: str | None = Form(default=None),
    collection_name: str | None = Form(default=None),
    owner_username: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    source_uri: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
) -> DocumentIntakeResponse:
    if (source_uri or "").strip():
        raise _document_error(
            "RAG_URI_INTAKE_DISABLED",
            "URI/reference intake is disabled by default; send multipart file bytes to rag-core.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if file is None:
        raise _document_error(
            "RAG_INVALID_REQUEST",
            "Multipart file field 'file' is required.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    metadata = _parse_metadata_json(metadata_json)
    content = await file.read()
    payload = DocumentUploadInput(
        tenant_id=tenant_id,
        external_id=external_id,
        title=title,
        collection_id=collection_id,
        collection_name=collection_name,
        owner_username=owner_username,
        filename=file.filename or "upload.bin",
        content=content,
        content_type=file.content_type,
        metadata_json=metadata,
    )
    try:
        result = create_document_from_upload(db, payload)
    except CollectionNotFoundError as exc:
        raise _collection_error(exc, status_code=status.HTTP_404_NOT_FOUND) from exc
    except CollectionRegistryError as exc:
        raise _collection_error(exc, status_code=status.HTTP_400_BAD_REQUEST) from exc
    except DocumentStorageError as exc:
        raise _document_error(
            getattr(exc, "error_code", "RAG_STORAGE_UNAVAILABLE"),
            str(exc),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        ) from exc
    except DocumentIntakeError as exc:
        raise _document_error(
            getattr(exc, "error_code", "RAG_INVALID_REQUEST"),
            str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    storage_payload = dict(result.storage or {})
    storage_payload.setdefault("uri", result.source_uri)

    # ── Enqueue async ingestion when Celery is enabled (Phase 5) ─────────
    enqueued = False
    settings_for_enqueue = get_service_settings()
    if settings_for_enqueue.celery_enabled:
        try:
            from rag_core.worker.celery_app import celery_app

            celery_app.send_task(
                "rag.ingest_document",
                args=[result.document_id, result.job_id],
                queue=settings_for_enqueue.queue_ingestion,
            )
            enqueued = True
        except Exception:
            # Graceful fallback — the queued job placeholder is already in
            # the DB; the worker will pick it up when it polls or a later
            # explicit enqueue happens.
            pass

    return DocumentIntakeResponse(
        document_id=result.document_id,
        job_id=result.job_id,
        status=result.status if not enqueued else "queued",
        collection_id=result.collection_id,
        collection_name=result.collection_name,
        source_uri=result.source_uri,
        storage=storage_payload,
        metadata_json=result.metadata_json if not enqueued else {
            **result.metadata_json,
            "_celery_enqueued": enqueued,
        },
    )


@app.post("/v1/indexing/prepare", response_model=IndexingPrepareResponse, dependencies=[Depends(_require_internal_auth)])
def indexing_prepare(request: IndexingPrepareRequest) -> IndexingPrepareResponse:
    settings = get_service_settings()
    try:
        prepared = prepare_chunks_for_indexing(
            request.document_id,
            request.checksum,
            request.chunks,
            default_provider=settings.embedding_provider,
            default_model=settings.embedding_model,
            default_dimensions=settings.embedding_dimensions,
            default_base_url=settings.embedding_base_url,
            default_timeout_seconds=settings.embedding_timeout_seconds,
            default_batch_size=settings.embedding_batch_size,
            default_api_key=settings.embedding_api_key,
            default_api_type=settings.embedding_api_type,
            default_api_version=settings.embedding_api_version,
            default_custom_headers=settings.embedding_headers,
            store_embeddings_in_metadata=(
                settings.store_embeddings_in_metadata
                if request.store_embeddings_in_metadata is None
                else request.store_embeddings_in_metadata
            ),
            index_provider=request.index_provider or settings.vector_store_provider,
        )
        meta = None
        if prepared:
            first_meta = dict(prepared[0].get("metadata_json") or {})
            meta = EmbeddingMetaPayload(
                provider=str(first_meta.get("embedding_provider") or settings.embedding_provider),
                model=str(first_meta.get("embedding_model") or settings.embedding_model),
                dimensions=int(first_meta.get("embedding_dimensions") or settings.embedding_dimensions),
                version=str(first_meta.get("embedding_version") or ""),
            )
        return IndexingPrepareResponse(prepared_chunks=prepared, embedding_meta=meta)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error_type": _sanitize_exception_type(exc)},
        ) from exc


@app.post("/v1/vector/upsert", response_model=VectorUpsertResponse, dependencies=[Depends(_require_internal_auth)])
def vector_upsert(request: VectorUpsertRequest, db: Session = Depends(get_db)) -> VectorUpsertResponse:
    settings = get_service_settings()
    if not settings.qdrant_enabled:
        return VectorUpsertResponse(ok=True, upserted=False)
    try:
        _register_collection_for_vector_upsert(db, settings, request)
    except CollectionDimensionMismatchError as exc:
        raise _collection_error(exc, status_code=status.HTTP_409_CONFLICT) from exc
    except CollectionRegistryError as exc:
        raise _collection_error(exc, status_code=status.HTTP_400_BAD_REQUEST) from exc
    adapter = _get_vector_adapter(settings)
    rows = [SimpleNamespace(id=row.id, chunk_index=row.chunk_index) for row in request.chunk_rows]
    first_meta = dict((request.prepared_chunks[0].get("metadata_json") if request.prepared_chunks else {}) or {})
    adapter.upsert_document_chunks(
        owner_username=request.document.owner_username,
        document_id=request.document.document_id,
        title=request.document.title,
        source_type=request.document.source_type,
        source_uri=request.document.source_uri or "",
        session_id=request.document.session_id,
        chunk_rows=rows,
        prepared_chunks=request.prepared_chunks,
        embedding_provider=request.embedding_provider or str(first_meta.get("embedding_provider") or settings.embedding_provider),
        embedding_model=request.embedding_model or str(first_meta.get("embedding_model") or settings.embedding_model),
    )
    return VectorUpsertResponse(ok=True, upserted=bool(rows))


@app.post("/v1/vector/delete", response_model=VectorDeleteResponse, dependencies=[Depends(_require_internal_auth)])
def vector_delete(request: VectorDeleteRequest) -> VectorDeleteResponse:
    settings = get_service_settings()
    if not settings.qdrant_enabled:
        return VectorDeleteResponse(ok=True, deleted=False)
    _get_vector_adapter(settings).delete_document_chunks(request.owner_username, request.document_id)
    return VectorDeleteResponse(ok=True, deleted=True)


@app.post("/v1/vector/search", response_model=VectorSearchResponse, dependencies=[Depends(_require_internal_auth)])
def vector_search(request: VectorSearchRequest) -> VectorSearchResponse:
    settings = get_service_settings()
    rewritten_query = request.rewritten_query
    query_expansions: list[str] = []
    query_vector = request.query_vector
    embedding_meta: EmbeddingMetaPayload | None = None

    if query_vector is None:
        if rewritten_query:
            rewritten = rewritten_query
        else:
            rewritten, query_expansions = _expand_query(
                request.query or "",
                enable_query_expansion=settings.retrieval_enable_query_expansion,
            )
            rewritten_query = rewritten
        provider = _get_embedding_provider(settings)
        embedded = provider.embed_query(rewritten_query or request.query or "")
        query_vector = embedded.vector
        embedding_meta = _meta_payload(embedded.meta)

    if not settings.qdrant_enabled:
        return VectorSearchResponse(
            vector_hits=[],
            rewritten_query=rewritten_query,
            query_expansions=query_expansions,
            embedding_meta=embedding_meta,
            vector_store_attempted=False,
        )

    try:
        hits = _get_vector_adapter(settings).search_similar_chunks(
            request.owner_username,
            query_vector,
            top_k=request.top_k,
            document_id=request.document_id,
            session_id=request.session_id,
            session_scope=request.session_scope,
        )
        return VectorSearchResponse(
            vector_hits=hits,
            rewritten_query=rewritten_query,
            query_expansions=query_expansions,
            embedding_meta=embedding_meta,
            vector_store_attempted=True,
        )
    except Exception as exc:
        return VectorSearchResponse(
            vector_hits=[],
            rewritten_query=rewritten_query,
            query_expansions=query_expansions,
            embedding_meta=embedding_meta,
            vector_store_attempted=True,
            vector_store_failed=True,
            vector_store_error_type=_sanitize_exception_type(exc),
        )


@app.post("/v1/retrieval/rank", response_model=RetrievalRankResponse, dependencies=[Depends(_require_internal_auth)])
def retrieval_rank(request: RetrievalRankRequest) -> RetrievalRankResponse:
    settings = get_service_settings()
    rewritten_query = request.rewritten_query
    query_expansions = list(request.query_expansions or [])
    if not rewritten_query:
        rewritten_query, query_expansions = _expand_query(
            request.query,
            enable_query_expansion=settings.retrieval_enable_query_expansion,
        )

    semantic_scores_by_chunk_id = {
        int(chunk_id): round(max(0.0, float(score or 0.0)), 6)
        for chunk_id, score in (request.semantic_scores_by_chunk_id or {}).items()
    }
    embed_meta = _meta_payload(request.embedding_meta)
    query_embedding: list[float] = []
    provider = None

    if not semantic_scores_by_chunk_id:
        provider = _get_embedding_provider(settings)
        query_embed_result = provider.embed_query(rewritten_query)
        query_embedding = query_embed_result.vector
        embed_meta = _meta_payload(query_embed_result.meta)

    mixed_space_skip_count = 0
    chunk_embeddings_by_id: dict[int, list[float]] = {}
    missing_embedding_inputs: list[tuple[int, str]] = []
    retrieval_config = _build_retrieval_config(request)
    dense_payloads: list[dict[str, Any]] = []
    corpus_payloads: list[dict[str, Any]] = []

    if not semantic_scores_by_chunk_id:
        for candidate in request.candidates:
            chunk_meta = dict(candidate.metadata_json or {})
            if not _is_embedding_compatible(chunk_meta, embed_meta.provider, embed_meta.model):
                mixed_space_skip_count += 1
                continue
            stored_embedding = _extract_stored_embedding(chunk_meta)
            if stored_embedding is not None:
                chunk_embeddings_by_id[candidate.chunk_id] = stored_embedding
            else:
                missing_embedding_inputs.append((candidate.chunk_id, candidate.content))

        if missing_embedding_inputs:
            if provider is None:
                provider = _get_embedding_provider(settings)
            embed_result = provider.embed_texts([content for _chunk_id, content in missing_embedding_inputs])
            for (chunk_id, _content), vector in zip(missing_embedding_inputs, embed_result.vectors):
                chunk_embeddings_by_id[chunk_id] = vector

    scored_results: list[dict[str, Any]] = []
    for candidate in request.candidates:
        chunk_meta = dict(candidate.metadata_json or {})
        if semantic_scores_by_chunk_id:
            semantic_score = semantic_scores_by_chunk_id.get(candidate.chunk_id, 0.0)
        elif _is_embedding_compatible(chunk_meta, embed_meta.provider, embed_meta.model):
            chunk_embedding = chunk_embeddings_by_id.get(candidate.chunk_id)
            semantic_score = (
                round(max(0.0, _cosine_similarity(query_embedding, chunk_embedding)), 6)
                if chunk_embedding is not None
                else 0.0
            )
        else:
            semantic_score = 0.0

        lexical_score = _lexical_overlap_score(rewritten_query, candidate.content)
        score = _hybrid_score(
            semantic_score,
            lexical_score,
            semantic_weight=settings.retrieval_hybrid_semantic_weight,
            lexical_weight=settings.retrieval_hybrid_lexical_weight,
        )
        base_payload = _build_result_from_candidate(
            candidate,
            score=score,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            embedding_provider=embed_meta.provider,
        )
        corpus_payloads.append(
            {
                **base_payload,
                "score": 0.0,
                "lexical_score": None,
                "trace": {"retriever": "sparse_input"},
            }
        )
        if semantic_score >= settings.retrieval_min_score:
            dense_payloads.append(
                {
                    **base_payload,
                    "score": semantic_score,
                    "lexical_score": None,
                    "trace": {"retriever": "dense"},
                }
            )
        if score < settings.retrieval_min_score and lexical_score < settings.retrieval_min_lexical_score:
            continue
        scored_results.append(base_payload)

    dense_payloads.sort(key=lambda item: (-float(item.get("score") or 0.0), item.get("document_id") or 0, item.get("chunk_index") or 0))
    dense_payloads = dense_payloads[: int(retrieval_config["dense_top_k"])]
    query = RetrievalQuery(
        text=request.query,
        top_k=request.top_k,
        rewritten_text=rewritten_query,
        query_expansions=tuple(query_expansions or []),
        trace_id=retrieval_config.get("trace_id"),
        filters=RetrievalFilters(),
        config=dict(retrieval_config),
    )
    pipeline = RetrievalPipeline(
        stages=[
            ("dense", _CandidateListRetriever(dense_payloads, source_stage="dense")),
            (
                "sparse",
                LexicalCorpusRetriever(
                    corpus_payloads,
                    top_k=int(retrieval_config["sparse_top_k"]),
                    min_score=settings.retrieval_min_lexical_score,
                ),
            ),
        ],
        fusion_strategy=ReciprocalRankFusionStrategy(top_k=int(retrieval_config["fusion_top_k"])),
        reranker=ProviderReranker(
            _HeuristicRerankerProvider(settings, model_name=retrieval_config.get("reranker_model")),
            enabled=bool(retrieval_config.get("enable_reranker")),
            top_k=int(retrieval_config["rerank_top_k"]),
            provider_name=str(retrieval_config.get("reranker_provider") or "heuristic"),
            model_name=str(retrieval_config.get("reranker_model") or "rag-core-heuristic"),
        ),
    )
    package = pipeline.run(query)
    retrieval_traces = [trace.to_dict() for trace in package.traces]
    results = [candidate.to_result_dict() for candidate in package.candidates][: request.top_k]
    reranked_count = next(
        (int(trace.get("candidate_count") or 0) for trace in retrieval_traces if trace.get("stage") == "rerank"),
        0,
    )
    return RetrievalRankResponse(
        results=results,
        candidate_count=len(request.candidates),
        matched_count=len(package.candidates),
        reranked_count=reranked_count,
        rewritten_query=rewritten_query,
        query_expansions=query_expansions,
        evidence_strength=_classify_evidence_strength(
            results,
            fallback_used=False,
            low_confidence_score=settings.retrieval_low_confidence_score,
        ),
        embedding_meta=embed_meta,
        mixed_space_skip_count=mixed_space_skip_count,
        retrieval_config=retrieval_config,
        retrieval_traces=retrieval_traces,
    )


@app.post("/v1/context/pack", response_model=ContextPackResponse, dependencies=[Depends(_require_internal_auth)])
def context_pack(request: ContextPackRequest) -> ContextPackResponse:
    settings = get_service_settings()
    packed, token_estimate = _pack_retrieval_results(
        request.results,
        max_context_chunks=request.max_context_chunks or settings.retrieval_max_context_chunks,
        max_context_tokens=request.max_context_tokens or settings.retrieval_max_context_tokens,
    )
    return ContextPackResponse(packed_results=packed, packed_token_estimate=token_estimate)


# ════════════════════════════════════════════════════════════════════════════
# Phase 7: Document lifecycle / Job status / Query endpoints
# ════════════════════════════════════════════════════════════════════════════


@app.get(
    "/rag/v1/documents/{document_id}",
    response_model=DocumentGetResponse,
    dependencies=[Depends(_require_internal_auth)],
)
def document_get(document_id: str, db: Session = Depends(get_db)) -> DocumentGetResponse:
    """Return rag-core-owned document metadata."""
    from rag_core.db.models.documents import RagDocument
    from rag_core.db.models.collections import RagCollection

    import uuid as _uuid_mod

    try:
        doc_uuid = _uuid_mod.UUID(document_id)
    except ValueError:
        raise _document_error(
            "RAG_DOCUMENT_NOT_FOUND",
            "Document not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    doc = db.query(RagDocument).filter(RagDocument.id == doc_uuid).first()
    if doc is None:
        raise _document_error(
            "RAG_DOCUMENT_NOT_FOUND",
            "Document not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    collection = db.query(RagCollection).filter(RagCollection.id == doc.collection_id).first()
    collection_name = str(collection.name) if collection else "unknown"

    return DocumentGetResponse(
        document_id=str(doc.id),
        tenant_id=str(doc.tenant_id),
        collection_id=str(doc.collection_id),
        collection_name=collection_name,
        external_id=doc.external_id,
        title=doc.title,
        source_type=str(doc.source_type),
        source_uri=doc.source_uri,
        mime_type=doc.mime_type,
        checksum=doc.checksum,
        file_size_bytes=doc.file_size_bytes,
        status=str(doc.status),
        chunk_count=int(doc.chunk_count or 0),
        owner_username=doc.owner_username,
        metadata_json=dict(doc.metadata_json or {}),
        created_at=_dt_to_iso(doc.created_at),
        updated_at=_dt_to_iso(doc.updated_at),
        deleted_at=_dt_to_iso(doc.deleted_at),
    )


@app.delete(
    "/rag/v1/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    dependencies=[Depends(_require_internal_auth)],
)
def document_delete(document_id: str, db: Session = Depends(get_db)) -> DocumentDeleteResponse:
    """Soft-delete a rag-core document and schedule vector/index cleanup."""
    from rag_core.db.models.documents import RagDocument
    from rag_core.db.models.ingestion_jobs import RagIngestionJob
    from rag_core.db.models.chunks import RagChunk
    from datetime import datetime as _dt, timezone as _tz

    import uuid as _uuid_mod

    try:
        doc_uuid = _uuid_mod.UUID(document_id)
    except ValueError:
        raise _document_error(
            "RAG_DOCUMENT_NOT_FOUND",
            "Document not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    doc = db.query(RagDocument).filter(RagDocument.id == doc_uuid).first()
    if doc is None:
        raise _document_error(
            "RAG_DOCUMENT_NOT_FOUND",
            "Document not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    if doc.status == "deleted":
        return DocumentDeleteResponse(ok=True, deleted=False, document_id=str(doc.id), status="deleted")

    doc.status = "deleted"
    doc.deleted_at = _dt.now(_tz.utc)
    doc.metadata_json = {
        **dict(doc.metadata_json or {}),
        "_deleted_from": "rag-core-api-v1-delete-endpoint",
    }

    # Cancel any active ingestion jobs
    active_jobs = (
        db.query(RagIngestionJob)
        .filter(
            RagIngestionJob.document_id == doc_uuid,
            RagIngestionJob.status.in_(["queued", "processing", "embedding", "indexing"]),
        )
        .all()
    )
    for job in active_jobs:
        job.status = "cancelled"
        job.error_message = "Document deleted by API."

    chunks = db.query(RagChunk).filter(RagChunk.document_id == doc_uuid).all()
    for chunk in chunks:
        chunk.metadata_json = {
            **dict(chunk.metadata_json or {}),
            "_status": "deleted",
        }

    # Enqueue vector cleanup when Celery is enabled
    settings_for_delete = get_service_settings()
    if settings_for_delete.celery_enabled:
        try:
            from rag_core.worker.celery_app import celery_app

            celery_app.send_task(
                "rag.delete_document_vectors",
                args=[str(doc_uuid)],
                queue=settings_for_delete.queue_cleanup,
            )
        except Exception:
            pass

    db.commit()
    return DocumentDeleteResponse(ok=True, deleted=True, document_id=str(doc.id), status="deleted")


@app.get(
    "/rag/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(_require_internal_auth)],
)
def job_get(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    """Return async ingestion job status."""
    from rag_core.db.models.ingestion_jobs import RagIngestionJob

    import uuid as _uuid_mod

    try:
        job_uuid = _uuid_mod.UUID(job_id)
    except ValueError:
        raise _document_error(
            "RAG_INVALID_REQUEST",
            "Invalid job ID format.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    job = db.query(RagIngestionJob).filter(RagIngestionJob.id == job_uuid).first()
    if job is None:
        raise _document_error(
            "RAG_JOB_NOT_FOUND",
            "Job not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    return JobStatusResponse(
        job_id=str(job.id),
        document_id=str(job.document_id),
        status=str(job.status),
        step_current=job.step_current,
        step_progress=job.step_progress,
        chunks_total=job.chunks_total,
        chunks_processed=int(job.chunks_processed or 0),
        error_code=job.error_code,
        error_message=job.error_message,
        retry_count=int(job.retry_count or 0),
        max_retries=int(job.max_retries or 3),
        started_at=_dt_to_iso(job.started_at),
        completed_at=_dt_to_iso(job.completed_at),
        created_at=_dt_to_iso(job.created_at),
        updated_at=_dt_to_iso(job.updated_at),
    )


@app.post(
    "/rag/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(_require_internal_auth)],
)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    """Semantic search across rag-core-owned collections."""
    from rag_core.db.models.collections import RagCollection
    from rag_core.db.models.documents import RagDocument

    settings_q = get_service_settings()
    if not settings_q.qdrant_enabled:
        return QueryResponse(query=request.query, top_k=request.top_k, returned=0, results=[])

    # Resolve collection
    collection = None
    identifier = (request.collection_id or request.collection_name or "").strip()
    try:
        from rag_core.services.collection_registry import get_collection

        if identifier:
            collection = get_collection(db, identifier)
        else:
            collection = get_collection(db, settings_q.vector_store_collection)
    except Exception:
        collection = None

    if collection is None:
        return QueryResponse(query=request.query, top_k=request.top_k, returned=0, results=[])

    # Get embedding for query
    provider = _get_embedding_provider(settings_q)
    embedded = provider.embed_query(request.query)
    query_vector = embedded.vector

    # Search Qdrant
    adapter = _get_vector_adapter(settings_q)
    try:
        hits = adapter.search_similar_chunks(
            request.owner_username or "default",
            query_vector,
            top_k=request.top_k,
        )
    except Exception:
        return QueryResponse(query=request.query, top_k=request.top_k, returned=0, results=[])

    # Enrich results with document info from rag_core DB
    results: list[QueryResultPayload] = []
    for hit in hits:
        chunk_id_int = int(hit.get("chunk_id", 0)) if isinstance(hit.get("chunk_id"), (int, float)) else None
        document_id_raw = hit.get("document_id")

        doc_title = None
        doc_source_type = None
        doc_source_uri = None
        doc_id_str = str(document_id_raw or "")

        # Try to look up rag-core document by the external_id (DominicBE doc ID in Phase 6)
        if document_id_raw is not None:
            doc = db.query(RagDocument).filter(RagDocument.external_id == str(document_id_raw)).first()
            if doc:
                doc_id_str = str(doc.id)
                doc_title = doc.title
                doc_source_type = str(doc.source_type)
                doc_source_uri = doc.source_uri

        results.append(
            QueryResultPayload(
                document_id=doc_id_str,
                chunk_id=str(chunk_id_int) if chunk_id_int is not None else None,
                chunk_index=int(hit.get("chunk_index", 0)),
                score=float(hit.get("score", 0.0)),
                content=str(hit.get("content") or ""),
                title=doc_title,
                source_type=doc_source_type,
                source_uri=doc_source_uri,
                metadata_json=dict(hit.get("metadata_json") or {}),
            )
        )

    return QueryResponse(
        query=request.query,
        top_k=request.top_k,
        returned=len(results),
        results=results,
    )

