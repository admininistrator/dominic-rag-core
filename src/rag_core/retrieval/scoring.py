"""Scoring functions — cosine similarity, lexical overlap, hybrid scoring.

Provides:
- ``_cosine_similarity()``: Standard cosine similarity between two vectors.
- ``_lexical_overlap_score()``: Token-overlap-based lexical relevance score.
- ``_hybrid_score()``: Weighted combination of semantic and lexical scores.
- ``_normalize_for_dedupe()``: Normalize text for deduplication comparison.

Extracted from DominicBE's ``app/services/retrieval_service.py``.
- ``settings.retrieval_hybrid_semantic_weight`` and
  ``settings.retrieval_hybrid_lexical_weight`` replaced with explicit
  ``semantic_weight`` and ``lexical_weight`` parameters.
"""
from __future__ import annotations

import re
from math import sqrt

from rag_core.retrieval.query_processor import _tokenize


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        Cosine similarity in ``[0.0, 1.0]``, or ``0.0`` if either vector is
        empty or dimensions don't match.
    """
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalize_for_dedupe(text: str) -> str:
    """Normalize text for deduplication: collapse whitespace, lowercase.

    Args:
        text: Input text.

    Returns:
        Normalized string.
    """
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _lexical_overlap_score(query_text: str, content: str) -> float:
    """Compute a lexical overlap score between query and content tokens.

    The score combines token coverage (fraction of query tokens found) and
    token density (relative token-set size), weighted 75%/25%.

    Args:
        query_text: Query text.
        content: Content text to score against.

    Returns:
        Float in ``[0.0, 1.0]``, rounded to 6 decimal places.
    """
    query_tokens = _tokenize(query_text)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    overlap = query_tokens & content_tokens
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(query_tokens)
    density = len(overlap) / sqrt(len(query_tokens) * len(content_tokens))
    return round(min(1.0, (coverage * 0.75) + (density * 0.25)), 6)


def _hybrid_score(
    semantic_score: float,
    lexical_score: float,
    *,
    semantic_weight: float = 0.4,
    lexical_weight: float = 0.6,
) -> float:
    """Compute a weighted hybrid score from semantic and lexical components.

    Weights are normalised so they sum to 1.0 before combining.

    Args:
        semantic_score: Semantic similarity score (cosine).
        lexical_score: Lexical overlap score.
        semantic_weight: Weight for semantic component (default: ``0.4``).
        lexical_weight: Weight for lexical component (default: ``0.6``).

    Returns:
        Float in ``[0.0, 1.0]``, rounded to 6 decimal places.
    """
    total_weight = semantic_weight + lexical_weight
    normalised_semantic = semantic_weight / total_weight
    normalised_lexical = lexical_weight / total_weight
    return round(
        min(1.0, (semantic_score * normalised_semantic) + (lexical_score * normalised_lexical)),
        6,
    )
