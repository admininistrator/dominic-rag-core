"""Indexing pipeline — chunk preparation and vector ID building.

Provides:
- ``build_vector_id()``: Deterministic vector ID string.
- ``prepare_chunks_for_indexing()``: Embeds chunks and attaches provenance
  metadata.

Replaces ``settings`` access with explicit parameters while preserving the
exact metadata schema and vector ID format from DominicBE's
``app/services/knowledge_service.py``.

Extracted from DominicBE's ``app/services/knowledge_service.py``.
- ``get_embedding_provider`` imported from ``rag_core.embeddings.factory``.
- ``settings`` access replaced with explicit function parameters.
- ``vector_store.should_store_embeddings_in_database()`` replaced with
  ``store_embeddings_in_metadata`` parameter.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from rag_core.embeddings.factory import get_embedding_provider

logger = logging.getLogger(__name__)

# Provider metadata version strings (matching DominicBE defaults)
_CUSTOM_PARSER_VERSION = "custom-v1"
_CUSTOM_CHUNKER_VERSION = "custom-sentence-v1"


def build_vector_id(document_id: int, chunk_index: int, checksum: str) -> str:
    """Build a deterministic vector ID string.

    Format: ``"local:{document_id}:{chunk_index}:{checksum[:12]}"``

    Args:
        document_id: Source document ID.
        chunk_index: Sequential chunk index within the document.
        checksum: Document checksum string (e.g. SHA-256 hex digest).

    Returns:
        Formatted vector ID string.
    """
    return f"local:{document_id}:{chunk_index}:{checksum[:12]}"


def prepare_chunks_for_indexing(
    document_id: int,
    checksum: str,
    chunks: list[dict[str, Any]],
    *,
    # Embedding provider overrides (passed to get_embedding_provider)
    provider_name: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    batch_size: int | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
    api_version: str | None = None,
    custom_headers: dict | None = None,
    # Defaults (used when explicit params are None)
    default_provider: str = "local",
    default_model: str = "local-hash-v1",
    default_dimensions: int = 64,
    default_base_url: str = "http://localhost:11434",
    default_timeout_seconds: float = 60.0,
    default_batch_size: int = 16,
    default_api_key: str = "",
    default_api_type: str = "",
    default_api_version: str = "",
    default_custom_headers: dict | None = None,
    # Indexing config
    store_embeddings_in_metadata: bool = True,
    index_provider: str = "database",
    parser_version: str = _CUSTOM_PARSER_VERSION,
    chunker_version: str = _CUSTOM_CHUNKER_VERSION,
) -> list[dict[str, Any]]:
    """Embed chunks and attach provenance metadata.

    For each input chunk this function:
    1. Batch-embeds all chunk texts using the configured provider.
    2. Attaches provenance metadata (embedding provider, model, dimensions,
       version, parser/chunker versions).
    3. Optionally stores the raw embedding vector in ``metadata_json``.
    4. Generates a ``vector_id`` for each chunk.

    Args:
        document_id: Source document ID.
        checksum: Document checksum (e.g. SHA-256 hex digest).
        chunks: List of chunk dicts, each with ``content`` and
            ``chunk_index`` keys.
        store_embeddings_in_metadata: Whether to store the embedding
            vector inside ``metadata_json["embedding"]`` (default:
            ``True`` for database-backed mode).
        index_provider: Provider name stored in metadata as
            ``index_provider`` (default: ``"database"``).
        parser_version: Parser version string (default:
            ``"custom-v1"``).
        chunker_version: Chunker version string (default:
            ``"custom-sentence-v1"``).

    Returns:
        List of prepared chunk dicts, each augmented with:
        - ``embedding``: The embedding vector.
        - ``embedding_model``: Model name used.
        - ``vector_id``: Deterministic vector ID.
        - ``metadata_json``: Enhanced with provenance fields.

    Raises:
        EmbeddingProviderError: If the embedding provider fails.
    """
    if not chunks:
        return []

    provider = get_embedding_provider(
        provider_name=provider_name,
        model=model,
        dimensions=dimensions,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
        api_key=api_key,
        api_type=api_type,
        api_version=api_version,
        custom_headers=custom_headers,
        default_provider=default_provider,
        default_model=default_model,
        default_dimensions=default_dimensions,
        default_base_url=default_base_url,
        default_timeout_seconds=default_timeout_seconds,
        default_batch_size=default_batch_size,
        default_api_key=default_api_key,
        default_api_type=default_api_type,
        default_api_version=default_api_version,
        default_custom_headers=default_custom_headers,
    )
    provider_meta = provider.meta

    texts = [chunk["content"] for chunk in chunks]
    embed_result = provider.embed_texts(texts)

    prepared: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, embed_result.vectors):
        actual_meta = embed_result.meta

        metadata_json: dict[str, Any] = {
            **(chunk.get("metadata_json") or {}),
            "index_provider": index_provider,
            "embedding_provider": actual_meta.provider,
            "embedding_model": actual_meta.model,
            "embedding_dimensions": actual_meta.dimensions,
            "embedding_version": actual_meta.version,
        }
        # Preserve pipeline-supplied parser/chunker versions, falling back
        # to custom defaults for legacy/backfill paths.
        metadata_json.setdefault("parser_version", parser_version)
        metadata_json.setdefault("chunker_version", chunker_version)
        if store_embeddings_in_metadata:
            metadata_json["embedding"] = vector

        prepared.append(
            {
                **chunk,
                "embedding": vector,
                "embedding_model": actual_meta.model,
                "vector_id": build_vector_id(
                    document_id, chunk["chunk_index"], checksum
                ),
                "metadata_json": metadata_json,
            }
        )

    logger.info(
        "prepare_chunks_for_indexing: document_id=%d provider=%s model=%s dims=%d chunks=%d",
        document_id,
        provider_meta.provider,
        embed_result.meta.model,
        embed_result.meta.dimensions,
        len(prepared),
    )

    return prepared


def compute_checksum(text: str) -> str:
    """Compute a SHA-256 hex digest for the given text.

    This is used for de-duplication and vector ID generation.

    Args:
        text: Input text to hash.

    Returns:
        SHA-256 hex digest string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
