"""Document chunking — sentence-boundary-aware splitting and metadata helpers."""

from rag_core.chunking.heading_detector import HeadingInfo, detect_headings, normalize_section_key
from rag_core.chunking.sentence_chunker import chunk_text

__all__ = [
    "HeadingInfo",
    "chunk_text",
    "detect_headings",
    "normalize_section_key",
]
