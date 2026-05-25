"""Source formatting — build knowledge and web source dicts.

Provides:
- ``_build_sources()``: Build knowledge source entries from retrieval results.
- ``_build_web_sources()``: Build web search source entries.

Extracted from DominicBE's ``app/services/chat_service.py``.
- No ``settings`` access — pure functions.
"""
from __future__ import annotations


def _build_sources(results: list[dict]) -> list[dict]:
    """Build knowledge source entries from retrieval results.

    Each source dict contains: ``document_id``, ``chunk_id``, ``title``,
    ``source_type``, ``score``, ``rerank_score``, ``snippet``,
    ``source_uri``, ``rank``, ``url``, ``domain``.

    Args:
        results: List of retrieval result dicts (from ``search_knowledge``).

    Returns:
        List of source dicts with rank starting at 1.
    """
    return [
        {
            "document_id": row["document_id"],
            "chunk_id": row["chunk_id"],
            "title": row["title"],
            "source_type": "knowledge",
            "score": row.get("score"),
            "rerank_score": row.get("rerank_score"),
            "snippet": row.get("snippet") or "",
            "source_uri": row.get("source_uri"),
            "rank": index,
            "url": None,
            "domain": None,
        }
        for index, row in enumerate(results, start=1)
    ]


def _build_web_sources(web_results: list[dict], *, start_rank: int = 1) -> list[dict]:
    """Build web search source entries from web search results.

    Each source dict contains: ``document_id``, ``chunk_id``, ``title``,
    ``source_type``, ``score``, ``rerank_score``, ``snippet``,
    ``source_uri``, ``rank``, ``url``, ``domain``.

    Args:
        web_results: List of web search result dicts.
        start_rank: Starting rank number (default: ``1``).

    Returns:
        List of web source dicts.
    """
    return [
        {
            "document_id": None,
            "chunk_id": None,
            "title": row.get("title") or row.get("url") or f"Web source {index}",
            "source_type": "web",
            "score": row.get("score"),
            "rerank_score": None,
            "snippet": row.get("snippet") or "",
            "source_uri": row.get("url"),
            "rank": start_rank + index - 1,
            "url": row.get("url"),
            "domain": row.get("domain"),
        }
        for index, row in enumerate(web_results, start=1)
    ]
