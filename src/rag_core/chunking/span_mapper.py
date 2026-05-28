"""Pure monotonic source span mapping helpers for chunking pipelines."""

from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class ChunkSpan:
    char_start: int | None
    char_end: int | None


def map_chunks_monotonic(source_text: str, chunk_texts: list[str]) -> list[ChunkSpan]:
    """Map chunks to source offsets using a monotonic cursor.

    This avoids repeatedly searching from the start of the source and handles
    repeated chunk text by continuing from the prior match when possible.
    """
    source = source_text or ""
    collapsed_source = " ".join(source.split())
    use_collapsed_coordinates = collapsed_source != source
    cursor = 0
    spans: list[ChunkSpan] = []
    for chunk in chunk_texts:
        needle = (chunk or "").strip()
        if not needle:
            spans.append(ChunkSpan(None, None))
            continue
        haystack = collapsed_source if use_collapsed_coordinates else source
        index = haystack.find(needle, cursor)
        if index < 0:
            index = haystack.find(needle)
        if index < 0:
            spans.append(ChunkSpan(None, None))
            continue
        end = index + len(needle)
        spans.append(ChunkSpan(index, end))
        cursor = end
    return spans

