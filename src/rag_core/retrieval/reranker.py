"""Reranking — post-scoring reranking with title overlap and position decay.

Provides:
- ``_rerank_results()``: Apply title overlap boost and position-based decay
  to scored results.

Extracted from DominicBE's ``app/services/retrieval_service.py``.
- ``settings.retrieval_max_rerank_candidates``,
  ``settings.retrieval_rerank_title_weight``, and
  ``settings.retrieval_rerank_position_weight`` replaced with explicit
  parameters.
"""
from __future__ import annotations

from rag_core.retrieval.scoring import _lexical_overlap_score
from rag_core.retrieval.evidence import _estimate_token_count


def _rerank_results(
    query_text: str,
    results: list[dict],
    *,
    max_rerank_candidates: int = 12,
    rerank_title_weight: float = 0.15,
    rerank_position_weight: float = 0.1,
) -> list[dict]:
    """Rerank scored results using title overlap and position decay.

    Each candidate receives a ``rerank_score`` that combines the original
    ``score`` with a title overlap boost and a position-based decay
    (earlier chunks get a small boost). Results are then sorted by
    ``rerank_score`` descending (with ties broken by ``score``, then
    ``document_id``, then ``chunk_index``).

    Args:
        query_text: The original query text (used for title overlap).
        results: List of scored result dicts.
        max_rerank_candidates: Maximum items to rerank (default: ``12``).
        rerank_title_weight: Weight for title overlap boost
            (default: ``0.15``).
        rerank_position_weight: Weight for position-based decay
            (default: ``0.1``).

    Returns:
        Reranked list of result dicts, each augmented with ``rerank_score``
        and ``token_estimate``.
    """
    reranked: list[dict] = []
    for item in results[:max_rerank_candidates]:
        title_score = _lexical_overlap_score(query_text, item.get("title") or "")
        chunk_index = int(item.get("chunk_index") or 0)
        position_score = max(0.0, 1.0 - (chunk_index * 0.05))
        rerank_score = round(
            min(
                1.0,
                float(item.get("score") or 0.0)
                + (title_score * rerank_title_weight)
                + (position_score * rerank_position_weight),
            ),
            6,
        )
        reranked.append(
            {
                **item,
                "rerank_score": rerank_score,
                "token_estimate": _estimate_token_count(
                    item.get("content") or "", item.get("token_count")
                ),
            }
        )

    reranked.sort(
        key=lambda item: (
            -float(item.get("rerank_score") or 0.0),
            -float(item.get("score") or 0.0),
            item.get("document_id") or 0,
            item.get("chunk_index") or 0,
        )
    )
    return reranked
