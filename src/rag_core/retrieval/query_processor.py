"""Query processing — expansion, normalization, and tokenization.

Provides:
- ``_expand_query()``: Expands a user query using Vietnamese→English mapping
  rules.
- ``_normalize_for_search()``: Unicode-normalize and strip punctuation.
- ``_strip_accents()``: Remove combining diacritical marks.
- ``_tokenize()``: Extract word tokens from normalized text.

Extracted from DominicBE's ``app/services/retrieval_service.py``.
- ``settings.retrieval_enable_query_expansion`` replaced with explicit
  ``enable_query_expansion`` parameter.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


# Vietnamese → English query expansion rules.
# These are hard-coded mappings that enable English-only retrievers to match
# Vietnamese user queries more effectively.
QUERY_EXPANSION_RULES: dict[str, list[str]] = {
    "hoan tien": ["refund", "refund policy", "money back"],
    "chinh sach": ["policy"],
    "xu ly": ["review", "process", "processing"],
    "bao lau": ["how long", "duration", "timeline", "days"],
    "mat khau": ["password", "credentials"],
    "dang nhap": ["login", "sign in", "authenticate"],
    "tai lieu": ["document", "knowledge base"],
}


def _strip_accents(text: str) -> str:
    """Remove combining diacritical marks via NFKD normalization.

    Args:
        text: Input string (may be ``None`` or empty).

    Returns:
        Accent-stripped string (empty string for ``None``/empty input).
    """
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_for_search(text: str) -> str:
    """Normalize text for search: strip accents, lowercase, remove punctuation.

    Args:
        text: Input string.

    Returns:
        Normalized, stripped string.
    """
    lowered = _strip_accents(text).lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase word tokens.

    Args:
        text: Input string.

    Returns:
        Set of unique word tokens.
    """
    return {
        token
        for token in re.findall(r"\w+", _normalize_for_search(text), flags=re.UNICODE)
        if token
    }


def _expand_query(
    query: str,
    *,
    enable_query_expansion: bool = True,
) -> tuple[str, list[str]]:
    """Expand a user query using Vietnamese→English mapping rules.

    When ``enable_query_expansion`` is ``True``, recognised Vietnamese phrases
    in the normalized query are replaced with their English candidate terms.

    Args:
        query: Raw user query string.
        enable_query_expansion: Whether to apply expansion rules
            (default: ``True``).

    Returns:
        Tuple of ``(rewritten_query, expansion_terms)`` where
        ``rewritten_query`` is the expanded query text and
        ``expansion_terms`` is the list of expansion terms appended.
    """
    normalized = _normalize_for_search(query)
    expansions: list[str] = []
    if enable_query_expansion:
        for phrase, candidates in QUERY_EXPANSION_RULES.items():
            if phrase in normalized:
                for candidate in candidates:
                    if candidate not in expansions:
                        expansions.append(candidate)

    if not expansions:
        return " ".join((query or "").split()), []

    rewritten_query = " ".join(
        part
        for part in [" ".join((query or "").split()), *expansions]
        if part
    ).strip()
    return rewritten_query, expansions
