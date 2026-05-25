"""Deduplication — remove duplicate results by document and content.

Provides:
- ``_dedupe_scored_results()``: Deduplicate scored results based on
  ``(document_id, normalized_content)`` key.

Extracted from DominicBE's ``app/services/retrieval_service.py``.
- No ``settings`` access — pure function.
"""
from __future__ import annotations

from rag_core.retrieval.scoring import _normalize_for_dedupe


def _dedupe_scored_results(results: list[dict]) -> list[dict]:
    """Deduplicate scored results by ``(document_id, normalized_content)``.

    The first occurrence of each unique key is kept. Subsequent occurrences
    are discarded. Order is preserved for non-duplicate items.

    Args:
        results: List of scored result dicts, each with ``document_id``
            and either ``content`` or ``snippet``.

    Returns:
        Deduplicated list in original order.
    """
    seen: set[tuple[int, str]] = set()
    deduped: list[dict] = []

    for item in results:
        key = (
            item["document_id"],
            _normalize_for_dedupe(item.get("content") or item.get("snippet") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped
