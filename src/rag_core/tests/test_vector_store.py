"""Tests for the vector store module.

Covers:
- ``VectorPayload`` schema and ``to_dict()`` output
- ``VectorSearchResult`` dataclass
- ``VectorStoreHealth`` dataclass
- ``QdrantAdapter`` constructor and helper methods
- Payload schema shape (all required keys)
"""
from __future__ import annotations

from rag_core.vector_store.base import VectorStore, VectorStoreError
from rag_core.vector_store.qdrant_adapter import QdrantAdapter
from rag_core.vector_store.types import VectorPayload, VectorSearchResult, VectorStoreHealth


# ---------------------------------------------------------------------------
# VectorPayload schema tests
# ---------------------------------------------------------------------------

class TestVectorPayload:
    """Verify VectorPayload schema matches DominicBE qdrant payload."""

    REQUIRED_KEYS = {
        "owner_username",
        "document_id",
        "chunk_id",
        "chunk_index",
        "title",
        "source_type",
        "source_uri",
        "session_id",
        "session_scope",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "embedding_version",
        "parser_version",
        "chunker_version",
    }

    def test_full_payload_shape(self):
        """to_dict() should contain all required keys."""
        payload = VectorPayload(
            owner_username="user1",
            document_id=42,
            chunk_id=100,
            chunk_index=0,
            title="Test Doc",
            source_type="upload",
            source_uri="/path/to/doc.pdf",
            session_id=5,
            session_scope="session",
            embedding_provider="local",
            embedding_model="local-hash-v1",
            embedding_dimensions=64,
            embedding_version="local-hash-v1",
            parser_version="custom-v1",
            chunker_version="custom-sentence-v1",
        )
        d = payload.to_dict()
        assert set(d.keys()) == self.REQUIRED_KEYS

    def test_minimal_payload_defaults(self):
        """Minimal payload should fill required fields and default optional."""
        payload = VectorPayload(
            owner_username="user1",
            document_id=1,
            chunk_id=10,
            chunk_index=0,
            title="Doc",
            source_type="text",
            source_uri="",
        )
        d = payload.to_dict()
        assert d["session_id"] is None
        assert d["session_scope"] == "global"
        assert d["embedding_provider"] == ""
        assert d["embedding_model"] == ""
        assert d["embedding_dimensions"] == 0

    def test_payload_types(self):
        """All payload values should have correct types."""
        payload = VectorPayload(
            owner_username="test_user",
            document_id=99,
            chunk_id=999,
            chunk_index=3,
            title="My Title",
            source_type="upload",
            source_uri="s3://bucket/key",
            session_id=7,
            session_scope="global",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
            embedding_version="v1",
            parser_version="custom-v1",
            chunker_version="custom-sentence-v1",
        )
        d = payload.to_dict()
        assert isinstance(d["owner_username"], str)
        assert isinstance(d["document_id"], int)
        assert isinstance(d["chunk_id"], int)
        assert isinstance(d["chunk_index"], int)
        assert isinstance(d["title"], str)
        assert isinstance(d["source_type"], str)
        assert isinstance(d["source_uri"], str)
        assert d["session_id"] is None or isinstance(d["session_id"], int)
        assert isinstance(d["session_scope"], str)


# ---------------------------------------------------------------------------
# VectorSearchResult tests
# ---------------------------------------------------------------------------

class TestVectorSearchResult:
    """Verify VectorSearchResult dataclass."""

    def test_minimal_result(self):
        result = VectorSearchResult(
            chunk_id=1,
            document_id=42,
            score=0.85,
            vector_id="some_id",
        )
        assert result.chunk_id == 1
        assert result.document_id == 42
        assert result.score == 0.85
        assert result.vector_id == "some_id"
        assert result.payload is None

    def test_result_with_payload(self):
        result = VectorSearchResult(
            chunk_id=2,
            document_id=99,
            score=0.95,
            vector_id="point_123",
            payload={"title": "Test"},
        )
        assert result.payload == {"title": "Test"}


# ---------------------------------------------------------------------------
# VectorStoreHealth tests
# ---------------------------------------------------------------------------

class TestVectorStoreHealth:
    """Verify VectorStoreHealth dataclass."""

    def test_default_health(self):
        health = VectorStoreHealth()
        assert health.ok is False
        assert health.provider == "database"
        assert health.latency_ms == 0.0

    def test_healthy_qdrant(self):
        health = VectorStoreHealth(
            ok=True,
            provider="qdrant",
            collection="knowledge_chunks",
            url="http://localhost:6333",
            detail="All good",
            latency_ms=5.2,
            collection_exists=True,
            collections_count=3,
        )
        assert health.ok is True


# ---------------------------------------------------------------------------
# VectorStore protocol test
# ---------------------------------------------------------------------------

class TestVectorStoreProtocol:
    """Verify VectorStore protocol is importable and checkable."""

    def test_protocol_importable(self):
        """VectorStore should be importable and runtime-checkable."""
        from rag_core.vector_store.base import VectorStore as VS
        assert VS is not None

    def test_vector_store_error(self):
        error = VectorStoreError("test error", provider="qdrant", category="connection")
        assert str(error) == "test error"
        assert error.provider == "qdrant"
        assert error.category == "connection"


# ---------------------------------------------------------------------------
# QdrantAdapter tests
# ---------------------------------------------------------------------------

class TestQdrantAdapter:
    """Verify QdrantAdapter construction and helper behavior."""

    def test_default_constructor(self):
        adapter = QdrantAdapter()
        assert adapter._collection == "knowledge_chunks"
        assert adapter._url is None
        assert adapter._timeout == 10.0
        assert adapter._prefer_grpc is False

    def test_is_enabled_false_when_no_url(self):
        adapter = QdrantAdapter()
        assert adapter.is_enabled() is False

    def test_is_enabled_true_with_url(self):
        adapter = QdrantAdapter(url="http://localhost:6333")
        assert adapter.is_enabled() is True

    def test_is_enabled_false_with_empty_url(self):
        adapter = QdrantAdapter(url="")
        assert adapter.is_enabled() is False

    def test_is_enabled_false_with_whitespace_url(self):
        adapter = QdrantAdapter(url="   ")
        assert adapter.is_enabled() is False

    def test_health_disabled(self):
        adapter = QdrantAdapter()
        health = adapter.check_health()
        assert health["ok"] is False
        assert health["provider"] == "qdrant"
        assert "disabled" in health["detail"]

    def test_constructor_with_all_params(self):
        adapter = QdrantAdapter(
            collection="custom_collection",
            url="https://qdrant.example.com:6333",
            api_key="secret",
            timeout_seconds=30.0,
            prefer_grpc=True,
            embedding_provider="api",
            embedding_model="text-embedding-3-small",
        )
        assert adapter._collection == "custom_collection"
        assert adapter._url == "https://qdrant.example.com:6333"
        assert adapter._api_key == "secret"
        assert adapter._timeout == 30.0
        assert adapter._prefer_grpc is True
        assert adapter._embedding_provider == "api"
        assert adapter._embedding_model == "text-embedding-3-small"

    def test_search_returns_empty_when_disabled(self):
        adapter = QdrantAdapter()
        results = adapter.search_similar_chunks(
            "user1", [0.1, 0.2, 0.3], top_k=5
        )
        assert results == []

    def test_delete_noop_when_disabled(self):
        adapter = QdrantAdapter()
        # Should not raise
        adapter.delete_document_chunks("user1", 42)
