"""Context building — pack retrieval results and format evidence context.

Provides:
- ``_build_evidence_context()``: Format knowledge and web results as
  ``[Source N]`` citation blocks.
- ``_pack_retrieval_results()``: Pack retrieval results within token and
  chunk budgets.

Extracted from DominicBE's ``app/services/chat_service.py``.
- ``settings.retrieval_max_context_chunks`` and
  ``settings.retrieval_max_context_tokens`` replaced with explicit
  parameters.
"""
from __future__ import annotations


def _build_evidence_context(
    knowledge_results: list[dict],
    web_results: list[dict] | None = None,
) -> str:
    """Format knowledge and web results as ``[Source N]`` citation blocks.

    Each knowledge source is formatted as::

        [Source N] type=knowledge title={title} document_id={id} chunk_id={id} score={score}
        {content or snippet}

    Each web source is formatted as::

        [Source N] type=web title={title} url={url} domain={domain} score={score}
        {snippet}

    Blocks are separated by double newlines.

    Args:
        knowledge_results: List of knowledge retrieval result dicts.
        web_results: Optional list of web search result dicts.

    Returns:
        Formatted evidence context string.
    """
    blocks: list[str] = []
    source_index = 1

    for row in knowledge_results:
        if _is_table_result(row):
            blocks.append(_build_table_evidence_block(row, source_index))
        else:
            blocks.append(
                "\n".join(
                    [
                        f"[Source {source_index}] type=knowledge title={row['title']} document_id={row['document_id']} chunk_id={row['chunk_id']} score={float(row.get('score') or 0):.3f}",
                        row.get("content") or row.get("snippet") or "",
                    ]
                )
            )
        source_index += 1

    for row in web_results or []:
        blocks.append(
            "\n".join(
                [
                    f"[Source {source_index}] type=web title={row.get('title') or row.get('url') or 'Web result'} url={row.get('url') or ''} domain={row.get('domain') or ''} score={float(row.get('score') or 0):.3f}",
                    row.get("snippet") or "",
                ]
            )
        )
        source_index += 1

    return "\n\n".join(blocks)


def _pack_retrieval_results(
    results: list[dict],
    *,
    max_context_chunks: int = 6,
    max_context_tokens: int = 4000,
) -> tuple[list[dict], int]:
    """Pack retrieval results within token and chunk budgets.

    Results are iterated in order and included until either the chunk count
    or token budget is exceeded. The first result is always included even if
    it exceeds the token budget (single-chunk edge case).

    Args:
        results: List of retrieval result dicts (preferably pre-sorted and
            reranked).
        max_context_chunks: Maximum number of chunks to include
            (default: ``6``).
        max_context_tokens: Maximum total token estimate to include
            (default: ``4000``).

    Returns:
        Tuple of ``(packed_results, packed_token_estimate)`` where
        ``packed_results`` is the filtered list and ``packed_token_estimate``
        is the sum of token estimates.
    """
    packed_results: list[dict] = []
    packed_token_estimate = 0

    for row in results:
        token_source = _context_token_source(row)
        token_estimate = int(
            row.get("token_estimate")
            or max(1, len(token_source) // 4)
        )
        if len(packed_results) >= max_context_chunks:
            break
        if packed_results and packed_token_estimate + token_estimate > max_context_tokens:
            continue
        if not packed_results and token_estimate > max_context_tokens:
            packed_results.append({**row, "token_estimate": token_estimate})
            packed_token_estimate += token_estimate
            break

        packed_results.append({**row, "token_estimate": token_estimate})
        packed_token_estimate += token_estimate

    return packed_results, packed_token_estimate


def _is_table_result(row: dict) -> bool:
    return str(row.get("source_type") or "").lower() == "table" or bool(row.get("table_id"))


def _build_table_evidence_block(row: dict, source_index: int) -> str:
    metadata = dict(row.get("metadata_json") or row.get("metadata_extra") or {})
    table_id = row.get("table_id") or metadata.get("table_id") or ""
    page_number = row.get("page_number") or metadata.get("page_number")
    section_key = row.get("section_key") or metadata.get("section_key") or ""
    summary = row.get("text_summary") or metadata.get("text_summary") or row.get("snippet") or ""
    markdown_table = row.get("markdown_table") or metadata.get("markdown_table") or ""
    content = row.get("content") or ""
    evidence_parts: list[str] = []
    if summary:
        evidence_parts.append(f"Table summary: {summary}")
    if markdown_table:
        evidence_parts.append(str(markdown_table))
    elif content:
        evidence_parts.append(str(content))
    header_parts = [
        f"[Source {source_index}] type=table",
        f"title={row.get('title') or table_id or 'Table'}",
        f"document_id={row.get('document_id')}",
        f"chunk_id={row.get('chunk_id')}",
        f"table_id={table_id}",
    ]
    if page_number is not None:
        header_parts.append(f"page_number={page_number}")
    if section_key:
        header_parts.append(f"section_key={section_key}")
    header_parts.append(f"score={float(row.get('score') or 0):.3f}")
    return "\n".join([" ".join(header_parts), *evidence_parts])


def _context_token_source(row: dict) -> str:
    metadata = dict(row.get("metadata_json") or row.get("metadata_extra") or {})
    return str(
        row.get("content")
        or row.get("markdown_table")
        or metadata.get("markdown_table")
        or row.get("text_summary")
        or metadata.get("text_summary")
        or row.get("snippet")
        or ""
    )
