"""Vector store protocol and interface definitions.

Defines the ``VectorStore`` protocol that all vector store adapters
must implement.  The protocol mirrors the current function signatures
from DominicBE's ``app/services/vector_store.py``.

Extracted from DominicBE's ``app/services/vector_store.py``.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from rag_core.vector_store.types import VectorSearchResult, VectorStoreHealth


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store operations.

    Implementations provide concrete storage and retrieval for document
    chunk embeddings (e.g. Qdrant, in-memory, mock).
    """

    def is_enabled(self) -> bool:
        """Return whether the vector store is enabled/configured."""
        ...

    def check_health(self) -> VectorStoreHealth:
        """Return health check information for the vector store."""
        ...

    def upsert_document_chunks(
        self,
        *,
        owner_username: str,
        document_id: int,
        title: str,
        source_type: str,
        source_uri: str,
        session_id: int | None,
        chunk_rows: list[Any],
        prepared_chunks: list[dict[str, Any]],
        embedding_provider: str,
        embedding_model: str,
    ) -> None:
        """Upsert document chunk vectors into the store.

        Args:
            owner_username: Owner/tenant identifier.
            document_id: Source document ID.
            title: Document title.
            source_type: Document source type.
            source_uri: Document source URI.
            session_id: Optional session ID for scoped documents.
            chunk_rows: Database chunk row objects (must have ``id``,
                ``chunk_index`` attributes).
            prepared_chunks: Prepared chunk dicts from the indexing
                pipeline (must have ``chunk_index``, ``embedding``,
                ``metadata_json`` keys).
            embedding_provider: Provider name for provenance metadata.
            embedding_model: Model name for provenance metadata.
        """
        ...

    def delete_document_chunks(self, owner_username: str, document_id: int) -> None:
        """Delete all vector chunks for a given document.

        Args:
            owner_username: Owner/tenant identifier.
            document_id: Source document ID to delete.
        """
        ...

    def search_similar_chunks(
        self,
        owner_username: str,
        query_vector: list[float],
        *,
        top_k: int,
        document_id: int | None = None,
        session_id: int | None = None,
        session_scope: str = "all",
    ) -> list[dict[str, Any]]:
        """Search for similar chunks in the vector store.

        Args:
            owner_username: Owner/tenant identifier filter.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            document_id: Optional document ID filter.
            session_id: Optional session ID filter.
            session_scope: Scope filter (``"all"``, ``"session"``,
                ``"global"``).

        Returns:
            List of result dicts with keys ``chunk_id``, ``document_id``,
            ``score``, ``vector_id``.
        """
        ...


class VectorStoreError(RuntimeError):
    """Base exception for vector store operations.

    Attributes:
        provider: Provider name (e.g. ``"qdrant"``).
        category: Error category (e.g. ``"connection"``, ``"operation"``).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.category = category
