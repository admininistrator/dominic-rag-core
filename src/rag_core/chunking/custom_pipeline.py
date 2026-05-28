"""Custom ingestion pipeline — wraps the existing ``chunk_text()`` behavior.

This pipeline preserves the current sentence-boundary-aware chunking logic
exactly as implemented in ``rag_core.chunking.sentence_chunker.chunk_text()``.
It is the default pipeline (``INGESTION_PIPELINE=custom``) and must produce
output that is byte-for-byte compatible with the DominicBE original.

Extracted from DominicBE's ``app/services/ingestion/custom_pipeline.py``.
The import of ``chunk_text`` now points to ``rag_core.chunking.sentence_chunker``
instead of ``app.services.knowledge_service``.

No dependency on CRUD, vector_store, endpoints, chat, or LlamaIndex.
"""

from __future__ import annotations

import logging

from rag_core.chunking.base import IngestionChunk, IngestionPipelineError
from rag_core.chunking.heading_detector import detect_headings, heading_for_offset
from rag_core.chunking.page_markers import page_metadata_for_span, strip_page_markers
from rag_core.chunking.span_mapper import map_chunks_monotonic
from rag_core.parsing.text_normalizer import normalize_text_for_ingestion

logger = logging.getLogger(__name__)

_PARSER_VERSION = "custom-v1"
_CHUNKER_VERSION = "custom-sentence-v1"


class CustomPipeline:
    """Wraps the existing ``chunk_text()`` behind the ``IngestionPipeline`` protocol.

    Delegates to ``rag_core.chunking.sentence_chunker.chunk_text()``.
    """

    def __init__(self, *, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # IngestionPipeline protocol properties
    @property
    def pipeline_name(self) -> str:
        return "custom"

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    @property
    def chunker_version(self) -> str:
        return _CHUNKER_VERSION

    def chunk_document(
        self,
        text: str,
        *,
        document_id: int | None = None,
        source_uri: str | None = None,
        title: str | None = None,
    ) -> list[IngestionChunk]:
        """Delegate to the existing ``chunk_text()`` implementation.

        Returns canonical IngestionChunk objects with the same content and
        chunk_index values that ``chunk_text()`` would produce.
        """
        if not (text or "").strip():
            return []

        try:
            from rag_core.chunking.sentence_chunker import chunk_text
        except ImportError as exc:
            raise IngestionPipelineError(
                f"CustomPipeline: cannot import chunk_text: {exc}",
                pipeline=self.pipeline_name,
                category="import_error",
            ) from exc

        cleaned_text, page_markers = strip_page_markers(text)
        source_text = normalize_text_for_ingestion(cleaned_text or text)
        raw_chunks = chunk_text(
            source_text,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        spans = map_chunks_monotonic(source_text, [raw.get("content", "") for raw in raw_chunks])
        headings = detect_headings(source_text)

        chunks: list[IngestionChunk] = []
        for raw, span in zip(raw_chunks, spans):
            raw_meta = dict(raw.get("metadata_json") or {})
            raw_meta.pop("char_count", None)
            if span.char_start is not None:
                raw_meta["char_start"] = span.char_start
            if span.char_end is not None:
                raw_meta["char_end"] = span.char_end

            page_meta = page_metadata_for_span(page_markers, span.char_start, span.char_end)
            raw_meta.update(page_meta)

            heading = heading_for_offset(headings, span.char_start)
            if heading is not None:
                raw_meta.setdefault("section_key", heading.section_key)
                raw_meta.setdefault("section_title", heading.title)
                raw_meta.setdefault("section_level", heading.level)
                raw_meta.setdefault("section_order", heading.section_order)

            chunks.append(
                IngestionChunk(
                    chunk_index=raw["chunk_index"],
                    content=raw["content"],
                    token_count=raw.get("token_count", max(1, len(raw["content"]) // 4)),
                    page_number=raw_meta.pop("page_number", None),
                    section_title=raw_meta.pop("section_title", None),
                    parser_version=_PARSER_VERSION,
                    chunker_version=_CHUNKER_VERSION,
                    extra_metadata=raw_meta,
                )
            )

        logger.debug(
            "CustomPipeline: document_id=%s produced %d chunks",
            document_id,
            len(chunks),
        )
        return chunks
