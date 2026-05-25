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

logger = logging.getLogger(__name__)

_PARSER_VERSION = "custom-v1"
_CHUNKER_VERSION = "custom-sentence-v1"


class CustomPipeline:
    """Wraps the existing ``chunk_text()`` behind the ``IngestionPipeline`` protocol.

    Delegates to ``rag_core.chunking.sentence_chunker.chunk_text()``.
    """

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

        raw_chunks = chunk_text(text)

        chunks: list[IngestionChunk] = []
        for raw in raw_chunks:
            raw_meta = dict(raw.get("metadata_json") or {})
            raw_meta.pop("char_count", None)

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
