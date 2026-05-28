"""Pure heading detection and section metadata helpers.

This module intentionally has no backend, database, or session dependencies.
It provides conservative line-anchored heading detection plus deterministic
Unicode-based section key normalization for ingestion and section retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import re
import unicodedata


@dataclass(frozen=True)
class HeadingInfo:
    """Detected document heading with source coordinates."""

    level: int
    title: str
    section_key: str
    line_index: int
    char_start: int
    char_end: int
    section_order: int


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.{1,120}?)\s*#*\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^((?:\d{1,3}|[A-Z]|[IVXLCDM]{1,8})[\.)](?:\s+|$)(?:\d{1,3}[\.)]\s+)*)?(.{1,120})$", re.IGNORECASE)
_VIETNAMESE_ACADEMIC_RE = re.compile(
    r"^(bài\s+(?:thực\s+hành|tập|học|kiểm\s+tra|thi|lab)\s+(?:số\s+)?\d+[\w\s\-:]{0,80})$",
    re.IGNORECASE,
)
_ALL_CAPS_RE = re.compile(r"^[A-Z0-9À-ỸĐ][A-Z0-9À-ỸĐ\s\-/&]{2,80}$")
_PAGE_MARKER_RE = re.compile(r"^<<PAGE:(\d+)>>$")


def strip_accents(text: str) -> str:
    """Remove combining marks while preserving base characters."""
    normalized = unicodedata.normalize("NFKD", text or "")
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.replace("đ", "d").replace("Đ", "D")


def normalize_section_key(title: str) -> str:
    """Normalize a heading or query fragment into a stable section key."""
    text = strip_accents(unicodedata.normalize("NFKC", title or "")).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _line_offsets(text: str) -> list[tuple[int, int, str]]:
    offsets: list[tuple[int, int, str]] = []
    cursor = 0
    for raw_line in (text or "").splitlines(keepends=True):
        line_end = cursor + len(raw_line)
        stripped_newline = raw_line.rstrip("\r\n")
        offsets.append((cursor, cursor + len(stripped_newline), stripped_newline))
        cursor = line_end
    if text and not text.endswith(("\n", "\r")) and not offsets:
        offsets.append((0, len(text), text))
    return offsets


def _looks_like_heading(line: str) -> tuple[int, str] | None:
    stripped = (line or "").strip()
    if not stripped or len(stripped) > 140:
        return None
    if _PAGE_MARKER_RE.fullmatch(stripped):
        return None

    markdown_match = _MARKDOWN_HEADING_RE.fullmatch(stripped)
    if markdown_match:
        return len(markdown_match.group(1)), markdown_match.group(2).strip()

    vietnamese_match = _VIETNAMESE_ACADEMIC_RE.fullmatch(stripped)
    if vietnamese_match:
        return 2, vietnamese_match.group(1).strip(" .:-")

    numbered_candidate = _NUMBERED_HEADING_RE.fullmatch(stripped)
    if numbered_candidate:
        title = stripped.strip()
        # Conservative numbered headings: require an explicit numeric/roman prefix
        # and avoid long sentence-like body text.
        if re.match(r"^(\d{1,3}(?:\.\d{1,3})*|[IVXLCDM]{1,8}|[A-Z])[\.)]\s+", title, re.IGNORECASE):
            if len(title.split()) <= 14 and not re.search(r"[,;!?]", title) and not title.endswith("."):
                level = min(6, title.split()[0].count(".") + 2)
                return level, title

    if _ALL_CAPS_RE.fullmatch(stripped) and len(stripped.split()) <= 10:
        return 2, stripped.title()

    return None


def detect_headings(text: str) -> list[HeadingInfo]:
    """Detect conservative document headings in source text."""
    headings: list[HeadingInfo] = []
    for line_index, (start, end, raw_line) in enumerate(_line_offsets(text)):
        detected = _looks_like_heading(raw_line)
        if detected is None:
            continue
        level, title = detected
        section_key = normalize_section_key(title)
        if not section_key:
            continue
        headings.append(
            HeadingInfo(
                level=level,
                title=title,
                section_key=section_key,
                line_index=line_index,
                char_start=start,
                char_end=end,
                section_order=len(headings),
            )
        )
    return headings


def heading_for_offset(headings: list[HeadingInfo], char_start: int | None) -> HeadingInfo | None:
    """Return the nearest preceding heading for a source offset."""
    if char_start is None or not headings:
        return None
    starts = [heading.char_start for heading in headings]
    index = bisect.bisect_right(starts, char_start) - 1
    if index < 0:
        return None
    return headings[index]

