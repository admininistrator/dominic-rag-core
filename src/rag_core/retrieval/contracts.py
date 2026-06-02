"""Provider-neutral retrieval contracts and metadata normalization.

This module defines lightweight, backend-agnostic data structures for composing
retrieval stages in rag-core. It intentionally does not know about auth,
ownership checks, persistence, or API routing; callers may pass already-approved
scope filters as opaque request constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


_METADATA_FIELDS = {
    "page_number",
    "page_range",
    "section_key",
    "section_title",
    "section_level",
    "section_order",
    "char_start",
    "char_end",
    "confidence",
    "section_confidence",
    "source_type",
    "source_uri",
    "fallback_reason",
    "rag_mode",
    "retrieval_scope",
    "selected_document_id",
    "session_id",
    "table_id",
    "source_stage",
}


@dataclass(frozen=True)
class RetrievalFilters:
    """Opaque retrieval constraints supplied by an integrating service.

    These are filters only; rag-core does not interpret them as permission
    decisions. DominicBE remains responsible for validating user/session/
    document access before invoking rag-core.
    """

    document_id: int | None = None
    session_id: int | None = None
    session_scope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "session_id": self.session_id,
            "session_scope": self.session_scope,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RetrievalQuery:
    """Shared request shape for retrieval stages."""

    text: str
    top_k: int = 5
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    rewritten_text: str | None = None
    query_expansions: tuple[str, ...] = ()
    trace_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_text(self) -> str:
        return self.rewritten_text or self.text


@dataclass(frozen=True)
class RetrievalMetadata:
    """Stable metadata contract shared by retrieval stages and downstream eval."""

    page_number: int | None = None
    page_range: list[int] | list[str] | str | None = None
    section_key: str | None = None
    section_title: str | None = None
    section_level: int | None = None
    section_order: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    confidence: float | None = None
    section_confidence: float | None = None
    source_type: str | None = None
    source_uri: str | None = None
    fallback_reason: str | None = None
    rag_mode: str | None = None
    retrieval_scope: str | None = None
    selected_document_id: int | None = None
    session_id: int | None = None
    table_id: str | None = None
    metadata_extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "page_range": self.page_range,
            "section_key": self.section_key,
            "section_title": self.section_title,
            "section_level": self.section_level,
            "section_order": self.section_order,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "confidence": self.confidence,
            "section_confidence": self.section_confidence,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "fallback_reason": self.fallback_reason,
            "rag_mode": self.rag_mode,
            "retrieval_scope": self.retrieval_scope,
            "selected_document_id": self.selected_document_id,
            "session_id": self.session_id,
            "table_id": self.table_id,
            "metadata_extra": dict(self.metadata_extra or {}),
        }


@dataclass(frozen=True)
class RetrievalCandidate:
    """Normalized candidate emitted by a dense, sparse, table, or other stage."""

    document_id: int | None
    chunk_id: int | None
    chunk_index: int | None
    content: str
    score: float = 0.0
    title: str = ""
    source_stage: str = "unknown"
    metadata: RetrievalMetadata = field(default_factory=RetrievalMetadata)
    semantic_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    token_count: int | None = None
    token_estimate: int | None = None
    snippet: str | None = None
    vector_id: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        source_stage: str = "unknown",
    ) -> "RetrievalCandidate":
        data_dict = dict(data or {})
        metadata = _metadata_from_candidate_mapping(data_dict, source_stage=source_stage)
        return cls(
            document_id=_coerce_int(data_dict.get("document_id")),
            chunk_id=_coerce_int(data_dict.get("chunk_id") or data_dict.get("id")),
            chunk_index=_coerce_int(data_dict.get("chunk_index")),
            content=str(data_dict.get("content") or data_dict.get("text") or ""),
            score=_coerce_float(data_dict.get("score"), default=0.0) or 0.0,
            title=str(data_dict.get("title") or ""),
            source_stage=source_stage,
            metadata=metadata,
            semantic_score=_coerce_float(data_dict.get("semantic_score")),
            lexical_score=_coerce_float(data_dict.get("lexical_score")),
            rerank_score=_coerce_float(data_dict.get("rerank_score")),
            token_count=_coerce_int(data_dict.get("token_count")),
            token_estimate=_coerce_int(data_dict.get("token_estimate")),
            snippet=(str(data_dict.get("snippet")) if data_dict.get("snippet") is not None else None),
            vector_id=(str(data_dict.get("vector_id")) if data_dict.get("vector_id") is not None else None),
            embedding_provider=(
                str(data_dict.get("embedding_provider"))
                if data_dict.get("embedding_provider") is not None
                else None
            ),
            embedding_model=(
                str(data_dict.get("embedding_model"))
                if data_dict.get("embedding_model") is not None
                else None
            ),
            trace=dict(data_dict.get("trace") or {}),
        )

    def to_result_dict(self) -> dict[str, Any]:
        payload = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "source_stage": self.source_stage,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
            "rerank_score": self.rerank_score,
            "token_count": self.token_count,
            "token_estimate": self.token_estimate,
            "snippet": self.snippet,
            "vector_id": self.vector_id,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "trace": dict(self.trace or {}),
        }
        payload.update(self.metadata.to_dict())
        return payload


@dataclass(frozen=True)
class RetrievalStageTrace:
    """Small trace entry for eval/debug visibility without sensitive data."""

    stage: str
    candidate_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "candidate_count": self.candidate_count,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RetrievalCandidatePackage:
    """Unfused candidate package emitted by the Phase-1 pipeline foundation."""

    query: RetrievalQuery
    candidates: list[RetrievalCandidate]
    traces: list[RetrievalStageTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": {
                "text": self.query.text,
                "effective_text": self.query.effective_text,
                "top_k": self.query.top_k,
                "filters": self.query.filters.to_dict(),
                "query_expansions": list(self.query.query_expansions),
                "trace_id": self.query.trace_id,
                "config": dict(self.query.config or {}),
            },
            "candidates": [candidate.to_result_dict() for candidate in self.candidates],
            "traces": [trace.to_dict() for trace in self.traces],
        }


@runtime_checkable
class CandidateRetriever(Protocol):
    """Common contract shared by dense and sparse retrieval adapters."""

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalCandidate] | Sequence[Mapping[str, Any]]:
        ...


@runtime_checkable
class DenseRetriever(CandidateRetriever, Protocol):
    """Marker protocol for vector/dense retrievers."""


@runtime_checkable
class SparseRetriever(CandidateRetriever, Protocol):
    """Marker protocol for sparse/lexical retrievers."""


@runtime_checkable
class FusionStrategy(Protocol):
    """Future fusion strategy contract, e.g. RRF or weighted fusion."""

    def fuse(self, query: RetrievalQuery, candidates: Sequence[RetrievalCandidate]) -> Sequence[RetrievalCandidate]:
        ...


@runtime_checkable
class Reranker(Protocol):
    """Future reranker provider contract."""

    def rerank(self, query: RetrievalQuery, candidates: Sequence[RetrievalCandidate]) -> Sequence[RetrievalCandidate]:
        ...


class RetrievalPipeline:
    """Composable retrieval pipeline for dense/sparse/fusion/rerank stages.

    ``collect_candidates`` preserves the Phase-1 unfused behavior. ``run`` adds
    Phase-3 orchestration for vector-only, hybrid, and hybrid+rerank modes while
    keeping algorithms/provider details behind strategy objects.
    """

    def __init__(
        self,
        stages: Sequence[tuple[str, CandidateRetriever]],
        *,
        fusion_strategy: FusionStrategy | None = None,
        reranker: Reranker | None = None,
    ):
        self._stages = list(stages)
        self._fusion_strategy = fusion_strategy
        self._reranker = reranker

    def collect_candidates(self, query: RetrievalQuery) -> RetrievalCandidatePackage:
        candidates, traces = self._collect_from_stages(query, self._stages)
        return RetrievalCandidatePackage(query=query, candidates=candidates, traces=traces)

    def run(self, query: RetrievalQuery) -> RetrievalCandidatePackage:
        """Execute the configured retrieval mode with explicit stage traces."""

        mode = _normalize_retrieval_mode((query.config or {}).get("retrieval_mode") or (query.config or {}).get("rag_mode"))
        selected_stages = self._select_stages_for_mode(mode)
        candidates, traces = self._collect_from_stages(query, selected_stages)

        if mode == "vector":
            return RetrievalCandidatePackage(query=query, candidates=candidates, traces=traces)

        if self._fusion_strategy is not None:
            candidates = list(self._fusion_strategy.fuse(query, candidates) or [])
            traces.append(
                RetrievalStageTrace(
                    stage="fusion",
                    candidate_count=len(candidates),
                    metadata={"strategy": self._fusion_strategy.__class__.__name__},
                )
            )

        if self._should_rerank(query, mode):
            if self._reranker is None:
                traces.append(
                    RetrievalStageTrace(
                        stage="rerank",
                        candidate_count=len(candidates),
                        metadata={"skipped": "no_reranker_configured"},
                    )
                )
            else:
                candidates = list(self._reranker.rerank(query, candidates) or [])
                traces.append(
                    RetrievalStageTrace(
                        stage="rerank",
                        candidate_count=len(candidates),
                        metadata={"provider": self._reranker.__class__.__name__},
                    )
                )

        return RetrievalCandidatePackage(query=query, candidates=candidates, traces=traces)

    def _collect_from_stages(
        self,
        query: RetrievalQuery,
        stages: Sequence[tuple[str, CandidateRetriever]],
    ) -> tuple[list[RetrievalCandidate], list[RetrievalStageTrace]]:
        candidates: list[RetrievalCandidate] = []
        traces: list[RetrievalStageTrace] = []
        for stage_name, retriever in stages:
            raw_results = list(retriever.retrieve(query) or [])
            stage_candidates = [
                item
                if isinstance(item, RetrievalCandidate)
                else RetrievalCandidate.from_mapping(item, source_stage=stage_name)
                for item in raw_results
            ]
            candidates.extend(stage_candidates)
            traces.append(RetrievalStageTrace(stage=stage_name, candidate_count=len(stage_candidates)))
        return candidates, traces

    def _select_stages_for_mode(self, mode: str) -> list[tuple[str, CandidateRetriever]]:
        if mode != "vector":
            return list(self._stages)
        non_vector_stage_names = {"sparse", "lexical", "bm25", "table", "table_sparse", "table_lexical"}
        return [
            (stage_name, retriever)
            for stage_name, retriever in self._stages
            if stage_name.lower() not in non_vector_stage_names
        ]

    def _should_rerank(self, query: RetrievalQuery, mode: str) -> bool:
        if mode == "vector":
            return False
        config = dict(query.config or {})
        if "enable_reranker" in config:
            return _coerce_bool(config.get("enable_reranker"), default=False)
        if "reranker_enabled" in config:
            return _coerce_bool(config.get("reranker_enabled"), default=False)
        return mode == "hybrid_rerank"


def _normalize_retrieval_mode(value: Any) -> str:
    raw = str(value or "hybrid").strip().lower().replace("-", "_").replace("+", "_")
    if raw in {"vector", "vector_only", "dense", "dense_only"}:
        return "vector"
    if raw in {"hybrid_rerank", "hybrid_with_rerank", "rerank"}:
        return "hybrid_rerank"
    return "hybrid"


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(value)


def normalize_retrieval_metadata(
    value: Mapping[str, Any] | str | None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> RetrievalMetadata:
    """Normalize raw retrieval metadata into the shared metadata contract."""

    raw = _ensure_mapping(value)
    default_values = dict(defaults or {})
    merged = {**default_values, **raw}
    confidence = _coerce_float(merged.get("confidence"))
    section_confidence = _coerce_float(merged.get("section_confidence"))
    extra = {
        key: val
        for key, val in raw.items()
        if key not in _METADATA_FIELDS and key != "metadata_json"
    }
    return RetrievalMetadata(
        page_number=_coerce_int(merged.get("page_number")),
        page_range=merged.get("page_range"),
        section_key=_coerce_str_or_none(merged.get("section_key")),
        section_title=_coerce_str_or_none(merged.get("section_title")),
        section_level=_coerce_int(merged.get("section_level")),
        section_order=_coerce_int(merged.get("section_order")),
        char_start=_coerce_int(merged.get("char_start")),
        char_end=_coerce_int(merged.get("char_end")),
        confidence=confidence,
        section_confidence=section_confidence,
        source_type=_coerce_str_or_none(merged.get("source_type")),
        source_uri=_coerce_str_or_none(merged.get("source_uri")),
        fallback_reason=_coerce_str_or_none(merged.get("fallback_reason")),
        rag_mode=_coerce_str_or_none(merged.get("rag_mode")),
        retrieval_scope=_coerce_str_or_none(merged.get("retrieval_scope")),
        selected_document_id=_coerce_int(merged.get("selected_document_id")),
        session_id=_coerce_int(merged.get("session_id")),
        table_id=_coerce_str_or_none(merged.get("table_id")),
        metadata_extra=extra,
    )


def _metadata_from_candidate_mapping(data: Mapping[str, Any], *, source_stage: str) -> RetrievalMetadata:
    raw_meta = _ensure_mapping(data.get("metadata_json"))
    for key in _METADATA_FIELDS:
        if key in data and data.get(key) is not None:
            raw_meta[key] = data.get(key)
    defaults = {
        "source_type": data.get("source_type"),
        "source_uri": data.get("source_uri"),
        "source_stage": source_stage,
    }
    return normalize_retrieval_metadata(raw_meta, defaults=defaults)


def _ensure_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(decoded, Mapping):
            return dict(decoded)
    return {}


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
