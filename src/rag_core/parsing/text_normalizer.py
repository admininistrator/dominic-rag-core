"""Text normalization for ingestion.

Extracted verbatim from DominicBE's ``app/services/knowledge_service.py``
(``normalize_text_for_ingestion()``, lines 47-58).

Behavior preservation: output must be byte-identical to the DominicBE
original for any input.
"""

from __future__ import annotations

import re


def normalize_text_for_ingestion(text: str) -> str:
    """Normalize whitespace while preserving readable paragraph boundaries.

    Args:
        text: Raw input text.

    Returns:
        Normalized text with ``\\r\\n`` → ``\\n``, collapsed intra-paragraph
        whitespace, and ``\\n\\n`` paragraph separators.  Empty/whitespace-only
        input returns ``""``.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""

    paragraphs = []
    for block in re.split(r"\n\s*\n+", raw):
        normalized = re.sub(r"[ \t]+", " ", block).strip()
        if normalized:
            paragraphs.append(normalized)
    return "\n\n".join(paragraphs)
