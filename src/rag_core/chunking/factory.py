"""Ingestion pipeline factory — selects pipeline by name.

Usage::

    from rag_core.chunking.factory import get_ingestion_pipeline
    pipeline = get_ingestion_pipeline(pipeline="custom")
    chunks = pipeline.chunk_document(normalized_text)

The factory accepts an explicit pipeline name.  LlamaIndex import is lazy
and optional — if ``llama-index-core`` is not installed and
``pipeline="llamaindex"``, a clear ``IngestionPipelineError`` is raised.

Extracted from DominicBE's ``app/services/ingestion/factory.py``.
The ``settings`` import is replaced with explicit ``pipeline`` parameter.
"""

from __future__ import annotations

import logging

from rag_core.chunking.base import IngestionPipeline, IngestionPipelineError

logger = logging.getLogger(__name__)

_PIPELINE_CUSTOM = "custom"
_PIPELINE_LLAMAINDEX = "llamaindex"


def get_ingestion_pipeline(
    *,
    pipeline: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionPipeline:
    """Return the configured ingestion pipeline.

    Args:
        pipeline:      Pipeline name (``'custom'`` or ``'llamaindex'``).
                       Defaults to ``'custom'``.
        chunk_size:    Optional chunk size override (passed to the pipeline).
        chunk_overlap: Optional chunk overlap override (passed to the pipeline).

    Returns:
        An ``IngestionPipeline`` implementation.

    Raises:
        IngestionPipelineError: If the pipeline name is unknown or if the
            required dependency (e.g. ``llama-index-core``) is not installed.
    """
    selected = (pipeline or _PIPELINE_CUSTOM).strip().lower()

    if selected == _PIPELINE_CUSTOM:
        from rag_core.chunking.custom_pipeline import CustomPipeline

        logger.debug("get_ingestion_pipeline: returning CustomPipeline")
        return CustomPipeline()

    if selected == _PIPELINE_LLAMAINDEX:
        try:
            from rag_core.chunking.llamaindex_pipeline import LlamaIndexPipeline
        except ImportError as exc:
            raise IngestionPipelineError(
                f"pipeline={selected} requires llama-index-core. "
                f"Install with: pip install llama-index-core. "
                f"Original error: {exc}",
                pipeline=selected,
                category="missing_dependency",
            ) from exc
        logger.debug("get_ingestion_pipeline: returning LlamaIndexPipeline")
        return LlamaIndexPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise IngestionPipelineError(
        f"Unknown pipeline value: '{selected}'. "
        f"Supported values: '{_PIPELINE_CUSTOM}', '{_PIPELINE_LLAMAINDEX}'.",
        pipeline=selected,
        category="configuration_error",
    )
