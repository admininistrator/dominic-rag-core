"""Sentence-boundary-aware text chunking.

Extracted from DominicBE's ``app/services/knowledge_service.py``
(``chunk_text()``, ``_split_sentences()``, ``_split_large_sentence()``,
lines 158-249).

Behavior preservation: ``chunk_text(text, 800, 100)`` must produce
identical output to the DominicBE original with default settings.
"""

from __future__ import annotations

import re

from rag_core.parsing.text_normalizer import normalize_text_for_ingestion


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Split text into overlapping chunks.

    Uses sentence-boundary-aware splitting when possible.

    Args:
        text: Raw input text.
        chunk_size: Maximum characters per chunk (default: 800).
        chunk_overlap: Character overlap between consecutive chunks (default: 100).

    Returns:
        List of ``{chunk_index, content, token_count (estimated), metadata_json}`` dicts.
        Empty list for empty/whitespace-only input.
    """
    size = chunk_size or 800
    overlap = chunk_overlap or 100

    normalized_text = normalize_text_for_ingestion(text)
    if not normalized_text:
        return []

    # Split into sentences first for cleaner boundaries
    sentences = _split_sentences(normalized_text)
    chunks: list[dict] = []
    current_chunk: list[str] = []
    current_len = 0
    idx = 0

    expanded_sentences: list[str] = []
    for sentence in sentences:
        expanded_sentences.extend(_split_large_sentence(sentence, size=size, overlap=overlap))

    for sentence in expanded_sentences:
        s_len = len(sentence)
        if current_len + s_len > size and current_chunk:
            chunk_text_str = " ".join(current_chunk).strip()
            if chunk_text_str:
                chunks.append({
                    "chunk_index": idx,
                    "content": chunk_text_str,
                    "token_count": max(1, len(chunk_text_str) // 4),
                    "metadata_json": {"char_count": len(chunk_text_str)},
                })
                idx += 1

            # Keep overlap: walk back from end
            overlap_chunks: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_chunks.insert(0, s)
                overlap_len += len(s)
            current_chunk = overlap_chunks
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += s_len

    # Last chunk
    if current_chunk:
        chunk_text_str = " ".join(current_chunk).strip()
        if chunk_text_str:
            chunks.append({
                "chunk_index": idx,
                "content": chunk_text_str,
                "token_count": max(1, len(chunk_text_str) // 4),
                "metadata_json": {"char_count": len(chunk_text_str)},
            })

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter: split on '. ', '! ', '? ', newlines."""
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


def _split_large_sentence(sentence: str, size: int, overlap: int) -> list[str]:
    """Split a single large sentence into overlapping pieces."""
    if len(sentence) <= size:
        return [sentence]

    parts: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(sentence):
        end = min(len(sentence), start + size)
        piece = sentence[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(sentence):
            break
        start += step
    return parts or [sentence]
