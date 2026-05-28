"""Pure section query intent, matching, and context formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from rag_core.chunking.heading_detector import normalize_section_key, strip_accents


SECTION_CONFIDENCE_THRESHOLD = 0.82

_SECTION_FILLER_TOKENS = {"so", "number", "no", "the", "a", "an"}


@dataclass(frozen=True)
class SectionIntent:
    is_section_query: bool
    normalized_query: str
    candidate_section_key: str | None
    wants_count: bool
    wants_summary: bool
    confidence: float


@dataclass(frozen=True)
class SectionMatch:
    section_key: str | None
    section_title: str | None
    confidence: float
    reason: str
    intent: SectionIntent


def detect_section_query_intent(query: str) -> SectionIntent:
    """Detect whether a query asks about a specific document section."""
    normalized = " ".join((query or "").split())
    folded = strip_accents(normalized).lower()
    wants_count = any(term in folded for term in ("co may", "bao nhieu", "how many", "count"))
    wants_summary = any(term in folded for term in ("tom tat", "summary", "summarize", "tung bai", "each item"))
    candidate = None
    confidence = 0.0
    if "bai thuc hanh" in folded:
        import re
        match = re.search(r"bai\s+thuc\s+hanh\s+(?:so\s+)?\d+", folded)
        if match:
            candidate = normalize_section_key(match.group(0))
            confidence = 0.88
    elif any(term in folded for term in ("section", "chapter", "heading", "muc ")):
        candidate = normalize_section_key(normalized)
        confidence = 0.55

    if candidate and (wants_count or wants_summary):
        confidence = min(1.0, confidence + 0.07)

    return SectionIntent(
        is_section_query=bool(candidate) and confidence >= 0.5,
        normalized_query=normalized,
        candidate_section_key=candidate,
        wants_count=wants_count,
        wants_summary=wants_summary,
        confidence=confidence,
    )


def detect_context_expansion_intent(query: str) -> bool:
    """Detect count/list/summarize queries that benefit from broader context."""
    folded = strip_accents(" ".join((query or "").split())).lower()
    if not folded:
        return False
    count_terms = ("co may", "bao nhieu", "how many", "count", "number of")
    list_terms = ("liet ke", "danh sach", "list", "enumerate", "cac muc", "cac bai")
    summary_terms = ("tom tat", "summarize", "summary", "tung bai", "tung muc", "each item")
    return any(term in folded for term in count_terms + list_terms + summary_terms)


def _section_value(section: Any, key: str) -> Any:
    if isinstance(section, dict):
        return section.get(key)
    return getattr(section, key, None)


def _token_set(value: str) -> set[str]:
    key = normalize_section_key(value)
    return {token for token in key.split("-") if token and token not in _SECTION_FILLER_TOKENS}


def _heading_like_query_key(query: str) -> str | None:
    folded = strip_accents(" ".join((query or "").split())).lower()
    if not folded:
        return None
    patterns = (
        r"bai\s+thuc\s+hanh\s+(?:so\s+)?\d+",
        r"chapter\s+\d+",
        r"section\s+\d+(?:\.\d+)*",
        r"muc\s+\d+(?:\.\d+)*",
    )
    for pattern in patterns:
        match = re.search(pattern, folded)
        if match:
            return normalize_section_key(match.group(0))
    return None


def match_section(query: str, available_sections: list[Any]) -> SectionMatch:
    """Match a section query to backend-supplied section metadata."""
    intent = detect_section_query_intent(query)
    candidate_key = intent.candidate_section_key
    effective_intent = intent
    if not candidate_key:
        heading_key = _heading_like_query_key(query)
        if heading_key:
            candidate_key = heading_key
            effective_intent = SectionIntent(
                is_section_query=True,
                normalized_query=intent.normalized_query,
                candidate_section_key=heading_key,
                wants_count=intent.wants_count,
                wants_summary=intent.wants_summary,
                confidence=0.86 if detect_context_expansion_intent(query) else 0.72,
            )
    if not effective_intent.is_section_query or not candidate_key:
        return SectionMatch(None, None, effective_intent.confidence, "no_section_intent", effective_intent)

    best_key: str | None = None
    best_title: str | None = None
    best_score = 0.0
    candidate_tokens = _token_set(candidate_key)
    for section in available_sections:
        section_key = str(_section_value(section, "section_key") or "")
        section_title = str(_section_value(section, "section_title") or "")
        normalized_title_key = normalize_section_key(section_title)
        keys = [section_key, normalized_title_key]
        score = 0.0
        if candidate_key in keys:
            score = 1.0
        elif any(candidate_key and candidate_key in key for key in keys):
            score = 0.9
        else:
            section_tokens = _token_set(section_key or normalized_title_key)
            if candidate_tokens and section_tokens:
                intersection = len(candidate_tokens & section_tokens)
                union_score = intersection / len(candidate_tokens | section_tokens)
                containment_score = intersection / max(1, min(len(candidate_tokens), len(section_tokens)))
                score = max(union_score, containment_score * 0.94)
        if score > best_score:
            best_score = score
            best_key = section_key or normalized_title_key
            best_title = section_title or best_key

    confidence = round(min(1.0, best_score * effective_intent.confidence), 4)
    if best_key and confidence >= SECTION_CONFIDENCE_THRESHOLD:
        return SectionMatch(best_key, best_title, confidence, "high_confidence_section_match", effective_intent)
    return SectionMatch(best_key, best_title, confidence, "low_section_confidence", effective_intent)


def format_section_context(chunks: list[dict]) -> str:
    """Format ordered section chunks for prompts or smoke-test assertions."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("section_title") or chunk.get("title") or "section"
        page = chunk.get("page_number")
        page_label = f" page={page}" if page is not None else ""
        blocks.append(f"[Section chunk {index}] section={title}{page_label}\n{chunk.get('content') or ''}")
    return "\n\n".join(blocks)

