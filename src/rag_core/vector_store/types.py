"""Type definitions for vector store operations.

Defines ``VectorPayload``, ``VectorSearchResult``, and related types
that represent the shape of data flowing through the vector store layer.

The payload schema must exactly match the Qdrant payload used by DominicBE's
``app/services/vector_store.py`` to preserve retrieval filter compatibility.

Extracted from DominicBE's ``app/services/vector_store.py`` — exact payload schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorPayload:
    """Payload attached to every vector store point.

    This schema **must** be preserved exactly — it is used by retrieval
    filters in ``search_similar_chunks()`` and must remain compatible
    with existing indexed points in Qdrant.

    Attributes:
        owner_username: Owner/tenant identifier for multi-tenant isolation.
        document_id: ID of the source document.
        chunk_id: ID of the specific chunk row.
        chunk_index: Sequential index of the chunk within the document.
        title: Document title.
        source_type: Document source type (e.g. ``"upload"``, ``"text"``).
        source_uri: URI/path of the original source.
        session_id: Optional session identifier for scoped documents.
        session_scope: Either ``"session"`` or ``"global"``.
        embedding_provider: Provider name (e.g. ``"local"``, ``"ollama"``).
        embedding_model: Model name used for embedding.
        embedding_dimensions: Embedding vector dimensions.
        embedding_version: Provider implementation version string.
        parser_version: Parser version that extracted the text.
        chunker_version: Chunker version that split the text.
    """

    owner_username: str
    document_id: int
    chunk_id: int
    chunk_index: int
    title: str
    source_type: str
    source_uri: str
    session_id: int | None = None
    session_scope: str = "global"
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_version: str = ""
    parser_version: str = ""
    chunker_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return payload as a plain dict for Qdrant PointStruct."""
        return {
            "owner_username": self.owner_username,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "session_id": self.session_id,
            "session_scope": self.session_scope,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_version": self.embedding_version,
            "parser_version": self.parser_version,
            "chunker_version": self.chunker_version,
        }


@dataclass
class VectorSearchResult:
    """A single result from a vector store search.

    Attributes:
        chunk_id: ID of the matching chunk.
        document_id: ID of the source document.
        score: Similarity score returned by the vector store.
        vector_id: Internal vector store point ID.
        payload: Optional full payload dict (if requested).
    """

    chunk_id: int
    document_id: int
    score: float
    vector_id: str
    payload: dict[str, Any] | None = None


@dataclass
class VectorStoreHealth:
    """Health check result for the vector store.

    Attributes:
        ok: Whether the store is healthy.
        provider: Provider name (e.g. ``"qdrant"``, ``"database"``).
        collection: Collection name (if applicable).
        url: Connection URL (if applicable).
        detail: Human-readable detail message.
        latency_ms: Health check latency in milliseconds.
        collection_exists: Whether the configured collection exists.
        collections_count: Total number of collections (if applicable).
    """

    ok: bool = False
    provider: str = "database"
    collection: str = ""
    url: str | None = None
    detail: str = ""
    latency_ms: float = 0.0
    collection_exists: bool | None = None
    collections_count: int | None = None
