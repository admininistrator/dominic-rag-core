"""Evidence — snippet building, token estimation, embedding compatibility check,
and evidence strength classification.

Provides:
- ``_build_snippet()``: Truncate content to a readable snippet.
- ``_estimate_token_count()``: Estimate token count from text.
- ``_is_embedding_compatible()``: Check if two embeddings can be compared.
- ``_classify_evidence_strength()``: Classify retrieval results as
  ``none``, ``fallback``, ``weak``, or ``grounded``.

Extracted from DominicBE's ``app/services/retrieval_service.py``.
- ``settings.retrieval_low_confidence_score`` replaced with explicit
  ``low_confidence_score`` parameter.
"""
from __future__ import annotations

from typing import Any


def _build_snippet(content: str, *, max_chars: int = 220) -> str:
    """Build a readable snippet from content, truncating at ``max_chars``.

    Args:
        content: Source text.
        max_chars: Maximum snippet length (default: ``220``).

    Returns:
        Truncated snippet string (with ``"..."`` suffix if truncated).
    """
    normalized = " ".join((content or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _estimate_token_count(text: str, explicit_count: int | None = None) -> int:
    """Estimate the token count for a text string.

    If ``explicit_count`` is provided and positive, it is returned directly.
    Otherwise, the estimate is ``max(1, len(text) // 4)``.

    Args:
        text: Input text.
        explicit_count: Optional pre-computed token count.

    Returns:
        Integer token estimate.
    """
    if explicit_count and explicit_count > 0:
        return int(explicit_count)
    normalized = " ".join((text or "").split())
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _ensure_json_mapping(value: Any) -> dict:
    """Safely coerce a value to a dict (no ``app.core.json_utils`` dep)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {}


def _is_embedding_compatible(
    chunk_meta: Any,
    query_provider: str,
    query_model: str,
) -> bool:
    """Check whether a chunk's embedding is compatible with the query provider.

    When ``True``, semantic cosine comparison is safe. When ``False``, the
    caller should skip semantic scoring and rely on lexical fallback only to
    avoid comparing vectors from incompatible embedding spaces.

    Args:
        chunk_meta: Chunk metadata (dict or JSON string).
        query_provider: Query embedding provider name.
        query_model: Query embedding model name.

    Returns:
        ``True`` if embeddings are compatible, ``False`` otherwise.
    """
    meta = _ensure_json_mapping(chunk_meta)
    stored_provider = (meta.get("embedding_provider") or "").strip().lower()
    stored_model = (meta.get("embedding_model") or "").strip().lower()
    if not stored_provider:
        # Legacy chunk with no provider metadata — assume compatible.
        return True
    if stored_provider == query_provider.lower() and stored_model == query_model.lower():
        return True
    return False


def _classify_evidence_strength(
    results: list[dict],
    *,
    fallback_used: bool,
    low_confidence_score: float = 0.2,
) -> str:
    """Classify the overall evidence strength of retrieval results.

    Args:
        results: List of result dicts, each with a ``score`` key.
        fallback_used: Whether fallback results were used.
        low_confidence_score: Score threshold below which results are
            considered low confidence (default: ``0.2``).

    Returns:
        One of ``"none"``, ``"fallback"``, ``"weak"``, or ``"grounded"``.
    """
    if not results:
        return "none"
    if fallback_used:
        return "fallback"
    top_score = float(results[0].get("score") or 0.0)
    if top_score >= low_confidence_score:
        return "grounded"
    return "weak"
