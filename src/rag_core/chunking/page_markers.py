"""Pure helpers for exact PDF page sentinel stripping and page span mapping."""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import re


PAGE_MARKER_RE = re.compile(r"^<<PAGE:(\d+)>>$")


@dataclass(frozen=True)
class PageMarker:
    page_number: int
    char_start: int
    char_end: int


def strip_page_markers(text: str) -> tuple[str, list[PageMarker]]:
    """Strip exact ``<<PAGE:N>>`` lines and return cleaned text + offsets.

    Offsets are in cleaned-text coordinates and point to the location where the
    marker was removed, enabling downstream chunk span to page mapping.
    """
    cleaned_parts: list[str] = []
    markers: list[PageMarker] = []
    cleaned_cursor = 0
    for raw_line in (text or "").splitlines(keepends=True):
        line_without_newline = raw_line.rstrip("\r\n")
        marker_match = PAGE_MARKER_RE.fullmatch(line_without_newline.strip())
        if marker_match:
            markers.append(
                PageMarker(
                    page_number=int(marker_match.group(1)),
                    char_start=cleaned_cursor,
                    char_end=cleaned_cursor,
                )
            )
            continue
        cleaned_parts.append(raw_line)
        cleaned_cursor += len(raw_line)

    cleaned_text = "".join(cleaned_parts).strip()
    trim_delta = len("".join(cleaned_parts)) - len("".join(cleaned_parts).lstrip())
    if trim_delta:
        markers = [
            PageMarker(m.page_number, max(0, m.char_start - trim_delta), max(0, m.char_end - trim_delta))
            for m in markers
        ]
    return cleaned_text, markers


def page_metadata_for_span(
    markers: list[PageMarker],
    char_start: int | None,
    char_end: int | None,
) -> dict:
    """Map a cleaned-text chunk span to ``page_number`` / ``page_range``."""
    if char_start is None or not markers:
        return {}
    starts = [marker.char_start for marker in markers]
    start_index = bisect.bisect_right(starts, char_start) - 1
    if start_index < 0:
        return {}
    start_page = markers[start_index].page_number
    end_page = start_page
    if char_end is not None:
        end_index = bisect.bisect_right(starts, max(char_start, char_end - 1)) - 1
        if end_index >= 0:
            end_page = markers[end_index].page_number
    if end_page != start_page:
        return {"page_number": start_page, "page_range": [start_page, end_page]}
    return {"page_number": start_page}

