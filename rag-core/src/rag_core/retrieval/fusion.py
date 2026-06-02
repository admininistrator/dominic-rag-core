"""Deterministic fusion strategies for dense and sparse retrieval candidates."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Sequence

from rag_core.retrieval.contracts import RetrievalCandidate, RetrievalQuery


class ReciprocalRankFusionStrategy:
    """Reciprocal Rank Fusion over provider-neutral retrieval candidates.

    Candidates are grouped by stable chunk identity, scored with RRF per source
    stage, and returned with deterministic tie-breaking and a bounded ``top_k``.
    The strategy safely degrades to vector-only behavior when only dense
    candidates are present, while still annotating fusion trace metadata.
    """

    def __init__(self, *, top_k: int | None = None, rrf_k: int = 60) -> None:
        self.top_k = top_k
        self.rrf_k = max(1, int(rrf_k))

    def fuse(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        limit = self._resolve_top_k(query.top_k)
        if limit <= 0:
            return []

        stage_rankings: dict[str, list[RetrievalCandidate]] = defaultdict(list)
        for candidate in candidates:
            stage_rankings[candidate.source_stage or "unknown"].append(candidate)

        ranks_by_key: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
        groups: dict[tuple[Any, ...], list[RetrievalCandidate]] = defaultdict(list)
        first_position: dict[tuple[Any, ...], int] = {}

        for absolute_position, candidate in enumerate(candidates):
            key = _candidate_identity(candidate)
            groups[key].append(candidate)
            first_position.setdefault(key, absolute_position)

        for stage, stage_candidates in stage_rankings.items():
            for rank, candidate in enumerate(stage_candidates, start=1):
                ranks_by_key[_candidate_identity(candidate)][stage] = rank

        fused = [
            self._merge_group(group, ranks_by_key[key], first_position[key])
            for key, group in groups.items()
        ]
        fused.sort(key=_fusion_sort_key)
        return fused[:limit]

    def _merge_group(
        self,
        group: list[RetrievalCandidate],
        rank_by_stage: dict[str, int],
        first_position: int,
    ) -> RetrievalCandidate:
        best = max(group, key=lambda item: float(item.score or 0.0))
        rrf_score = sum(1.0 / (self.rrf_k + rank) for rank in rank_by_stage.values())
        semantic_score = _max_optional(item.semantic_score for item in group)
        lexical_score = _max_optional(item.lexical_score for item in group)
        rerank_score = _max_optional(item.rerank_score for item in group)
        source_stages = sorted(rank_by_stage, key=lambda stage: rank_by_stage[stage])
        trace = dict(best.trace or {})
        trace.update(
            {
                "fusion_strategy": "rrf",
                "rrf_k": self.rrf_k,
                "rrf_score": round(rrf_score, 12),
                "source_stages": source_stages,
                "rank_by_stage": dict(rank_by_stage),
                "original_scores": {
                    candidate.source_stage or "unknown": candidate.score for candidate in group
                },
                "first_position": first_position,
            }
        )
        return replace(
            best,
            score=round(rrf_score, 12),
            source_stage="fusion",
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            rerank_score=rerank_score,
            trace=trace,
        )

    def _resolve_top_k(self, query_top_k: int) -> int:
        raw_limit = self.top_k if self.top_k is not None else query_top_k
        try:
            return max(0, int(raw_limit))
        except (TypeError, ValueError):
            return 0


def _candidate_identity(candidate: RetrievalCandidate) -> tuple[Any, ...]:
    if candidate.document_id is not None and candidate.chunk_id is not None:
        return ("chunk", candidate.document_id, candidate.chunk_id)
    if candidate.vector_id:
        return ("vector", candidate.vector_id)
    return (
        "content",
        candidate.document_id,
        candidate.chunk_index,
        " ".join((candidate.content or "").split()).lower(),
    )


def _max_optional(values: Sequence[float | None] | Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _fusion_sort_key(candidate: RetrievalCandidate) -> tuple[Any, ...]:
    trace = candidate.trace or {}
    rank_by_stage = trace.get("rank_by_stage") or {}
    best_stage_rank = min(rank_by_stage.values()) if rank_by_stage else 10**18
    return (
        -float(candidate.score or 0.0),
        -float(candidate.semantic_score or 0.0),
        -float(candidate.lexical_score or 0.0),
        best_stage_rank,
        trace.get("first_position", 10**18),
        candidate.document_id if candidate.document_id is not None else 10**18,
        candidate.chunk_id if candidate.chunk_id is not None else 10**18,
        candidate.chunk_index if candidate.chunk_index is not None else 10**18,
        candidate.title,
        candidate.content,
    )
