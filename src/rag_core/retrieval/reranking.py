"""Provider-neutral reranking adapters for retrieval candidates."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from rag_core.retrieval.contracts import RetrievalCandidate, RetrievalQuery


@dataclass(frozen=True)
class RerankProviderResult:
    """One provider score for a candidate in the input candidate sequence.

    Providers identify candidates by their zero-based input index so local model
    adapters and API adapters can share the same contract without depending on
    rag-core internals or mutating candidate objects.
    """

    candidate_index: int
    score: float
    trace: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RerankerProvider(Protocol):
    """Swappable provider interface for local or API-based rerankers."""

    provider_name: str
    model_name: str | None

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
    ) -> Sequence[RerankProviderResult] | Sequence[Mapping[str, Any]]:
        ...


class ProviderReranker:
    """Adapter that turns provider scores into normalized retrieval candidates.

    The adapter owns common reranker behavior: enabled/disabled fallback,
    top-k bounding, deterministic sorting, score propagation, and safe trace
    metadata. Provider-returned trace payloads are intentionally not propagated;
    API/local providers may see query text internally, but rag-core traces should
    contain only non-sensitive stage metadata needed for evaluation.
    """

    def __init__(
        self,
        provider: RerankerProvider,
        *,
        enabled: bool = True,
        top_k: int | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.provider = provider
        self.enabled = enabled
        self.top_k = top_k
        self.provider_name = provider_name or getattr(provider, "provider_name", provider.__class__.__name__)
        self.model_name = model_name if model_name is not None else getattr(provider, "model_name", None)

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        candidate_list = list(candidates or [])
        if not candidate_list:
            return []
        if not self._is_enabled(query):
            return candidate_list

        limit = self._resolve_top_k(query, len(candidate_list))
        if limit <= 0:
            return []

        raw_results = list(self.provider.rerank(query, candidate_list) or [])
        scored: list[tuple[RerankProviderResult, RetrievalCandidate]] = []
        for raw in raw_results:
            result = _coerce_provider_result(raw)
            if result is None:
                continue
            if result.candidate_index < 0 or result.candidate_index >= len(candidate_list):
                continue
            scored.append((result, candidate_list[result.candidate_index]))

        scored.sort(key=_rerank_sort_key)
        reranked: list[RetrievalCandidate] = []
        for rank, (result, candidate) in enumerate(scored[:limit], start=1):
            safe_trace = dict(candidate.trace or {})
            safe_trace.update(
                {
                    "reranker_provider": self.provider_name,
                    "reranker_model": self.model_name,
                    "rerank_rank": rank,
                    "rerank_input_index": result.candidate_index,
                    "original_score": candidate.score,
                    "prior_source_stage": candidate.source_stage,
                }
            )
            reranked.append(
                replace(
                    candidate,
                    score=round(float(result.score), 12),
                    source_stage="rerank",
                    rerank_score=round(float(result.score), 12),
                    trace=safe_trace,
                )
            )
        return reranked

    def _is_enabled(self, query: RetrievalQuery) -> bool:
        config = dict(query.config or {})
        if "enable_reranker" in config:
            return _coerce_bool(config.get("enable_reranker"), default=self.enabled)
        if "reranker_enabled" in config:
            return _coerce_bool(config.get("reranker_enabled"), default=self.enabled)
        return self.enabled

    def _resolve_top_k(self, query: RetrievalQuery, candidate_count: int) -> int:
        config = dict(query.config or {})
        raw_limit = config.get("rerank_top_k", self.top_k if self.top_k is not None else query.top_k)
        try:
            return max(0, min(int(raw_limit), candidate_count))
        except (TypeError, ValueError):
            return 0


def _coerce_provider_result(value: RerankProviderResult | Mapping[str, Any]) -> RerankProviderResult | None:
    if isinstance(value, RerankProviderResult):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        candidate_index = int(value.get("candidate_index", value.get("index")))
        score = float(value.get("score"))
    except (TypeError, ValueError):
        return None
    trace = value.get("trace") if isinstance(value.get("trace"), Mapping) else {}
    return RerankProviderResult(candidate_index=candidate_index, score=score, trace=dict(trace))


def _rerank_sort_key(item: tuple[RerankProviderResult, RetrievalCandidate]) -> tuple[Any, ...]:
    result, candidate = item
    return (
        -float(result.score),
        -float(candidate.score or 0.0),
        result.candidate_index,
        candidate.document_id if candidate.document_id is not None else 10**18,
        candidate.chunk_id if candidate.chunk_id is not None else 10**18,
        candidate.chunk_index if candidate.chunk_index is not None else 10**18,
        candidate.title,
        candidate.content,
    )


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
