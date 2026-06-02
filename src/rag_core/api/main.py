"""FastAPI application for the internal rag-core service."""

from __future__ import annotations

from secrets import compare_digest
from types import SimpleNamespace
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from rag_core.api.config import RagCoreServiceSettings, get_service_settings
from rag_core.api.schemas import (
    ContextPackRequest,
    ContextPackResponse,
    EmbeddingMetaPayload,
    IndexingPrepareRequest,
    IndexingPrepareResponse,
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
from rag_core.vector_store.qdrant_adapter import QdrantAdapter


app = FastAPI(title="rag-core", version="0.1.0")


def _health_response(payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=200 if payload.get("ok") else 503, content=payload)


def _sanitize_exception_type(exc: Exception) -> str:
    return type(exc).__name__


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
    payload = {
        "ok": bool(embedding.get("ok")) and (not settings.qdrant_enabled or bool(qdrant.get("ok"))),
        "service": settings.app_name,
        "dependencies": {
            "embedding": embedding,
            "qdrant": qdrant,
        },
        "api_key_configured": bool(settings.api_key),
    }
    return _health_response(payload)


@app.get("/ready")
def ready() -> JSONResponse:
    settings = get_service_settings()
    qdrant = _get_vector_adapter(settings).check_health()
    payload = {
        "ok": bool(settings.api_key) and (not settings.qdrant_enabled or bool(qdrant.get("ok"))),
        "service": settings.app_name,
        "api_key_configured": bool(settings.api_key),
        "dependencies": {"qdrant": qdrant},
    }
    return _health_response(payload)


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
def vector_upsert(request: VectorUpsertRequest) -> VectorUpsertResponse:
    settings = get_service_settings()
    if not settings.qdrant_enabled:
        return VectorUpsertResponse(ok=True, upserted=False)
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

