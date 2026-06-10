from __future__ import annotations

from fastapi.testclient import TestClient

import rag_core.api.main as api_main
from rag_core.api.config import get_service_settings


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("RAG_CORE_API_KEY", "test-secret")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "database")
    get_service_settings.cache_clear()
    return TestClient(api_main.app)


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-secret"}


def test_health_is_public(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "rag-core"


def test_v1_requires_bearer_token(monkeypatch):
    client = _client(monkeypatch)
    response = client.post("/v1/context/pack", json={"results": []})
    assert response.status_code == 401


def test_context_pack_endpoint(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/context/pack",
        headers=_headers(),
        json={
            "results": [
                {"document_id": 1, "chunk_index": 0, "content": "A" * 100, "token_estimate": 25},
                {"document_id": 1, "chunk_index": 1, "content": "B" * 100, "token_estimate": 25},
            ],
            "max_context_chunks": 1,
            "max_context_tokens": 4000,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["packed_results"]) == 1
    assert data["packed_token_estimate"] == 25


def test_indexing_prepare_local_provider(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/indexing/prepare",
        headers=_headers(),
        json={
            "document_id": 7,
            "checksum": "abcdef1234567890",
            "chunks": [
                {"chunk_index": 0, "content": "refund policy details", "token_count": 5, "metadata_json": {}}
            ],
        },
    )
    assert response.status_code == 200
    prepared = response.json()["prepared_chunks"]
    assert prepared[0]["vector_id"] == "local:7:0:abcdef123456"
    assert prepared[0]["metadata_json"]["embedding_provider"] == "local"
    assert "embedding" in prepared[0]["metadata_json"]


def test_retrieval_rank_endpoint(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/retrieval/rank",
        headers=_headers(),
        json={
            "query": "refund policy",
            "top_k": 3,
            "candidates": [
                {
                    "chunk_id": 10,
                    "document_id": 1,
                    "chunk_index": 0,
                    "title": "Policy",
                    "source_type": "text",
                    "source_uri": None,
                    "content": "The refund policy allows refunds within 30 days.",
                    "token_count": 10,
                    "vector_id": "local:1:0:abc",
                    "embedding_model": "local-hash-v1",
                    "metadata_json": {},
                }
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"]
    assert data["results"][0]["chunk_id"] == 10
    assert data["matched_count"] == 1


def test_retrieval_rank_endpoint_emits_pipeline_config_and_traces(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/retrieval/rank",
        headers=_headers(),
        json={
            "query": "refund policy",
            "top_k": 2,
            "semantic_scores_by_chunk_id": {"10": 0.9},
            "retrieval_config": {
                "retrieval_mode": "vector",
                "enable_reranker": True,
                "dense_top_k": 5,
                "sparse_top_k": 5,
                "fusion_top_k": 5,
                "rerank_top_k": 2,
            },
            "trace_id": "trace-api-1",
            "candidates": [
                {
                    "chunk_id": 10,
                    "document_id": 1,
                    "chunk_index": 0,
                    "title": "Policy",
                    "source_type": "text",
                    "source_uri": None,
                    "content": "The refund policy allows refunds within 30 days.",
                    "token_count": 10,
                    "vector_id": "local:1:0:abc",
                    "embedding_model": "local-hash-v1",
                    "metadata_json": {},
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_config"]["retrieval_mode"] == "vector"
    assert data["retrieval_config"]["trace_id"] == "trace-api-1"
    assert [trace["stage"] for trace in data["retrieval_traces"]] == ["dense"]
    assert data["results"][0]["source_stage"] == "dense"


def test_retrieval_rank_hybrid_mode_does_not_enable_reranker_by_default(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/retrieval/rank",
        headers=_headers(),
        json={
            "query": "refund policy",
            "top_k": 2,
            "semantic_scores_by_chunk_id": {"10": 0.9},
            "retrieval_config": {"retrieval_mode": "hybrid"},
            "candidates": [
                {
                    "chunk_id": 10,
                    "document_id": 1,
                    "chunk_index": 0,
                    "title": "Policy",
                    "source_type": "text",
                    "source_uri": None,
                    "content": "The refund policy allows refunds within 30 days.",
                    "token_count": 10,
                    "vector_id": "local:1:0:abc",
                    "embedding_model": "local-hash-v1",
                    "metadata_json": {},
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_config"]["retrieval_mode"] == "hybrid"
    assert data["retrieval_config"]["enable_reranker"] is False
    assert [trace["stage"] for trace in data["retrieval_traces"]] == ["dense", "sparse", "fusion"]


def test_retrieval_rank_endpoint_emits_normalized_metadata(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/retrieval/rank",
        headers=_headers(),
        json={
            "query": "refund policy",
            "top_k": 1,
            "semantic_scores_by_chunk_id": {"11": 0.9},
            "candidates": [
                {
                    "chunk_id": 11,
                    "document_id": 2,
                    "chunk_index": 1,
                    "title": "Policy",
                    "source_type": "pdf",
                    "source_uri": "file://policy.pdf",
                    "content": "Refund policy metadata test.",
                    "metadata_json": {
                        "page_number": 8,
                        "section_key": "returns",
                        "char_start": 20,
                        "char_end": 80,
                        "fallback_reason": "none",
                        "rag_mode": "hybrid_rerank",
                        "retrieval_scope": "session",
                        "selected_document_id": 2,
                        "session_id": 123,
                        "future_field": "preserved",
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["page_number"] == 8
    assert result["section_key"] == "returns"
    assert result["char_start"] == 20
    assert result["char_end"] == 80
    assert result["fallback_reason"] == "none"
    assert result["rag_mode"] == "hybrid_rerank"
    assert result["retrieval_scope"] == "session"
    assert result["selected_document_id"] == 2
    assert result["session_id"] == 123
    assert result["metadata_extra"] == {"future_field": "preserved"}


def test_vector_endpoints_delegate_to_adapter(monkeypatch):
    class FakeAdapter:
        def __init__(self):
            self.deleted = False
            self.upserted = False

        def delete_document_chunks(self, owner_username, document_id):
            self.deleted = (owner_username, document_id)

        def upsert_document_chunks(self, **kwargs):
            self.upserted = kwargs

        def search_similar_chunks(self, owner_username, query_vector, **kwargs):
            return [{"chunk_id": 5, "document_id": 9, "score": 0.7, "vector_id": "5"}]

    fake = FakeAdapter()
    monkeypatch.setenv("RAG_CORE_API_KEY", "test-secret")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "qdrant")
    monkeypatch.setenv("VECTOR_STORE_URL", "http://qdrant:6333")
    get_service_settings.cache_clear()
    monkeypatch.setattr(api_main, "_get_vector_adapter", lambda settings: fake)
    monkeypatch.setattr(api_main, "register_collection", lambda db, payload: None)

    def fake_get_db():
        yield object()

    api_main.app.dependency_overrides[api_main.get_db] = fake_get_db
    client = TestClient(api_main.app)

    try:
        delete_response = client.post(
            "/v1/vector/delete",
            headers=_headers(),
            json={"owner_username": "alice", "document_id": 9},
        )
        assert delete_response.status_code == 200
        assert fake.deleted == ("alice", 9)

        upsert_response = client.post(
            "/v1/vector/upsert",
            headers=_headers(),
            json={
                "document": {
                    "owner_username": "alice",
                    "document_id": 9,
                    "title": "Doc",
                    "source_type": "text",
                    "source_uri": None,
                    "session_id": None,
                },
                "chunk_rows": [{"id": 5, "chunk_index": 0}],
                "prepared_chunks": [
                    {
                        "chunk_index": 0,
                        "embedding": [0.1, 0.2],
                        "metadata_json": {"embedding_provider": "local", "embedding_model": "local-hash-v1"},
                    }
                ],
            },
        )
        assert upsert_response.status_code == 200
        assert fake.upserted["document_id"] == 9

        search_response = client.post(
            "/v1/vector/search",
            headers=_headers(),
            json={"owner_username": "alice", "top_k": 1, "query_vector": [0.1, 0.2]},
        )
        assert search_response.status_code == 200
        assert search_response.json()["vector_hits"][0]["chunk_id"] == 5
    finally:
        api_main.app.dependency_overrides.clear()

