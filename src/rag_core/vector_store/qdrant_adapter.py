"""Qdrant vector store adapter.

Implements the ``VectorStore`` protocol for the Qdrant vector database.
Replaces ``settings`` access with explicit constructor parameters while
preserving the exact payload schema and filter logic from DominicBE's
``app/services/vector_store.py``.

Extracted from DominicBE's ``app/services/vector_store.py``.
- ``settings`` import replaced with explicit constructor parameters.
- ``collection_naming.suggest_collection_name`` imported from rag_core.
- All Qdrant-specific logic preserved exactly (payload schema, filter
  construction, collection auto-creation, dimension validation).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

from rag_core.embeddings.collection_naming import suggest_collection_name

logger = logging.getLogger(__name__)


class QdrantAdapter:
    """Qdrant vector store adapter.

    Wraps QdrantClient operations behind a simple interface that matches
    the current DominicBE ``app/services/vector_store.py`` function signatures.

    Args:
        collection: Qdrant collection name (default: ``"knowledge_chunks"``).
        url: Qdrant server URL (default: ``None`` → disabled).
        api_key: Qdrant API key (default: ``None``).
        timeout_seconds: Timeout for Qdrant requests (default: ``10.0``).
        prefer_grpc: Whether to prefer gRPC over HTTP (default: ``False``).
        embedding_provider: Provider name for dimension guard messaging
            (default: ``"local"``).
        embedding_model: Model name for dimension guard messaging
            (default: ``"local-hash-v1"``).
    """

    def __init__(
        self,
        collection: str = "knowledge_chunks",
        url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        prefer_grpc: bool = False,
        embedding_provider: str = "local",
        embedding_model: str = "local-hash-v1",
    ) -> None:
        self._collection = collection
        self._url = url
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._prefer_grpc = prefer_grpc
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Public interface (matches VectorStore protocol)
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return whether Qdrant is configured and enabled."""
        return bool(self._url and self._url.strip())

    def check_health(self) -> dict:
        """Return health check information for the Qdrant store.

        Returns:
            Dict with keys: ok, provider, collection, url, latency_ms,
            detail, collection_exists, collections_count.
        """
        started_at = perf_counter()
        base: dict = {
            "ok": False,
            "provider": "qdrant",
            "collection": self._collection,
            "url": self._url,
        }

        if not self.is_enabled():
            return {
                **base,
                "latency_ms": round((perf_counter() - started_at) * 1000, 2),
                "detail": "External vector store is disabled.",
            }

        try:
            client = self._get_client()
            collections_response = client.get_collections()
            collection_names = [item.name for item in collections_response.collections]
            return {
                **base,
                "ok": True,
                "collection_exists": self._collection in collection_names,
                "collections_count": len(collection_names),
                "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        except Exception as exc:
            return {
                **base,
                "latency_ms": round((perf_counter() - started_at) * 1000, 2),
                "detail": str(exc),
            }

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
        """Upsert document chunk vectors into Qdrant.

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
            embedding_provider: Provider name for Qdrant payload metadata.
            embedding_model: Model name for Qdrant payload metadata.

        Raises:
            ValueError: If no embeddings are available for upsert.
        """
        if not self.is_enabled() or not chunk_rows:
            return

        # Build lookup maps from prepared chunks
        embeddings_by_index: dict[int, list[float]] = {
            int(chunk["chunk_index"]): chunk.get("embedding") or []
            for chunk in prepared_chunks
        }
        meta_by_index: dict[int, dict] = {
            int(chunk["chunk_index"]): (chunk.get("metadata_json") or {})
            for chunk in prepared_chunks
        }

        first_vector = next(
            (v for v in embeddings_by_index.values() if v), None
        )
        if not first_vector:
            raise ValueError("No embeddings prepared for vector upsert.")

        self._ensure_collection(len(first_vector))
        client = self._get_client()
        from qdrant_client import models  # type: ignore[import-untyped]

        session_scope = "session" if session_id is not None else "global"
        points = []
        for row in chunk_rows:
            vector = embeddings_by_index.get(int(row.chunk_index)) or []
            if not vector:
                continue
            chunk_meta = meta_by_index.get(int(row.chunk_index), {})
            points.append(
                models.PointStruct(
                    id=int(row.id),
                    vector=vector,
                    payload={
                        "owner_username": owner_username,
                        "document_id": int(document_id),
                        "chunk_id": int(row.id),
                        "chunk_index": int(row.chunk_index),
                        "title": title,
                        "source_type": source_type,
                        "source_uri": source_uri,
                        "session_id": int(session_id) if session_id is not None else None,
                        "session_scope": session_scope,
                        "embedding_provider": chunk_meta.get("embedding_provider", embedding_provider),
                        "embedding_model": chunk_meta.get("embedding_model", embedding_model),
                        "embedding_dimensions": chunk_meta.get("embedding_dimensions", len(vector)),
                        "embedding_version": chunk_meta.get("embedding_version", ""),
                        "parser_version": chunk_meta.get("parser_version", ""),
                        "chunker_version": chunk_meta.get("chunker_version", ""),
                    },
                )
            )

        if not points:
            logger.warning("No Qdrant points built for document id=%s", document_id)
            return

        client.upsert(
            collection_name=self._collection,
            wait=True,
            points=points,
        )

    def delete_document_chunks(self, owner_username: str, document_id: int) -> None:
        """Delete all Qdrant points for a given document.

        Args:
            owner_username: Owner/tenant identifier.
            document_id: Source document ID to delete.
        """
        if not self.is_enabled():
            return

        client = self._get_client()
        from qdrant_client import models  # type: ignore[import-untyped]
        from qdrant_client.http.exceptions import UnexpectedResponse  # type: ignore[import-untyped]

        try:
            client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="owner_username",
                                match=models.MatchValue(value=owner_username),
                            ),
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            ),
                        ]
                    )
                ),
                wait=True,
            )
        except (UnexpectedResponse, ValueError) as exc:
            logger.info(
                "Skipping Qdrant delete for missing collection or filter support: %s",
                exc,
            )

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
        """Search for similar chunks in Qdrant.

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
        if not self.is_enabled():
            return []

        client = self._get_client()
        from qdrant_client import models  # type: ignore[import-untyped]

        must_conditions: list[Any] = [
            models.FieldCondition(
                key="owner_username",
                match=models.MatchValue(value=owner_username),
            )
        ]

        if document_id is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            )
        elif session_scope == "session" and session_id is not None:
            must_conditions.extend(
                [
                    models.FieldCondition(
                        key="session_scope",
                        match=models.MatchValue(value="session"),
                    ),
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id),
                    ),
                ]
            )
        elif session_scope == "global":
            must_conditions.append(
                models.FieldCondition(
                    key="session_scope",
                    match=models.MatchValue(value="global"),
                )
            )

        points = client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=models.Filter(must=must_conditions),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "chunk_id": int(point.payload.get("chunk_id") or point.id),
                "document_id": int(point.payload.get("document_id")),
                "score": float(point.score or 0.0),
                "vector_id": str(point.id),
            }
            for point in points
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @lru_cache  # noqa: B019
    def _get_client(self):
        """Return (and cache) the QdrantClient instance."""
        if not self.is_enabled():
            return None

        try:
            from qdrant_client import QdrantClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for Qdrant vector store support."
            ) from exc

        return QdrantClient(
            url=self._url,
            api_key=self._api_key or None,
            timeout=self._timeout,
            prefer_grpc=self._prefer_grpc,
        )

    def _ensure_collection(self, vector_size: int) -> None:
        """Ensure the Qdrant collection exists with the expected vector size.

        If the collection already exists, validates that its vector dimension
        matches *vector_size*.  If it does not match, raises ``ValueError``.

        If the collection does not exist, it is created automatically.
        """
        client = self._get_client()
        if client is None:
            return

        from qdrant_client import models  # type: ignore[import-untyped]
        from qdrant_client.http.exceptions import UnexpectedResponse  # type: ignore[import-untyped]

        try:
            collection_info = client.get_collection(self._collection)
        except (UnexpectedResponse, ValueError):
            # Collection does not exist — create it.
            client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            return

        # Dimension guard — fail fast if collection dimension mismatches.
        existing_size = collection_info.config.params.vectors.size
        if existing_size != vector_size:
            suggested = suggest_collection_name(
                self._embedding_provider, self._embedding_model
            )
            raise ValueError(
                f"Qdrant collection {self._collection!r} has "
                f"vector dimension {existing_size}, but the current embedding "
                f"provider ({self._embedding_provider}/{self._embedding_model}) produces "
                f"{vector_size}-dimensional vectors. "
                f"Use a dedicated collection (e.g. {suggested}) "
                f"for the new provider, or reindex the collection with the correct size."
            )
