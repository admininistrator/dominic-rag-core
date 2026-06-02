"""Tests for the context module.

Covers:
- ``_build_sources()``: Knowledge source shape and content
- ``_build_web_sources()``: Web source shape and ranking
- ``_build_evidence_context()``: Citation format string
- ``_pack_retrieval_results()``: Token and chunk budgeting
"""
from __future__ import annotations

from rag_core.context.builder import _build_evidence_context, _pack_retrieval_results
from rag_core.context.sources import _build_sources, _build_web_sources


# ---------------------------------------------------------------------------
# Source building tests
# ---------------------------------------------------------------------------

class TestBuildSources:
    def test_build_sources_shape(self):
        results = [
            {
                "document_id": 1,
                "chunk_id": 10,
                "title": "Doc 1",
                "score": 0.85,
                "rerank_score": 0.90,
                "snippet": "Some content here",
                "source_uri": "/docs/doc1.pdf",
            }
        ]
        sources = _build_sources(results)
        assert len(sources) == 1
        source = sources[0]
        assert source["document_id"] == 1
        assert source["chunk_id"] == 10
        assert source["title"] == "Doc 1"
        assert source["source_type"] == "knowledge"
        assert source["score"] == 0.85
        assert source["rerank_score"] == 0.90
        assert source["snippet"] == "Some content here"
        assert source["source_uri"] == "/docs/doc1.pdf"
        assert source["rank"] == 1
        assert source["url"] is None
        assert source["domain"] is None

    def test_multiple_sources_sequential_rank(self):
        results = [
            {"document_id": 1, "chunk_id": 1, "title": "A", "score": 0.9},
            {"document_id": 2, "chunk_id": 2, "title": "B", "score": 0.8},
        ]
        sources = _build_sources(results)
        assert sources[0]["rank"] == 1
        assert sources[1]["rank"] == 2

    def test_empty_results(self):
        assert _build_sources([]) == []

    def test_missing_optional_fields(self):
        results = [{"document_id": 1, "chunk_id": 1, "title": "T"}]
        sources = _build_sources(results)
        assert sources[0]["score"] is None
        assert sources[0]["snippet"] == ""


class TestBuildWebSources:
    def test_basic_web_source(self):
        results = [{"title": "Web Page", "url": "https://example.com", "domain": "example.com"}]
        sources = _build_web_sources(results)
        assert len(sources) == 1
        source = sources[0]
        assert source["document_id"] is None
        assert source["chunk_id"] is None
        assert source["source_type"] == "web"
        assert source["title"] == "Web Page"
        assert source["url"] == "https://example.com"
        assert source["domain"] == "example.com"
        assert source["rank"] == 1

    def test_sequential_rank(self):
        results = [
            {"title": "A", "url": "http://a.com"},
            {"title": "B", "url": "http://b.com"},
        ]
        sources = _build_web_sources(results)
        assert sources[0]["rank"] == 1
        assert sources[1]["rank"] == 2

    def test_start_rank(self):
        results = [{"title": "A", "url": "http://a.com"}]
        sources = _build_web_sources(results, start_rank=5)
        assert sources[0]["rank"] == 5

    def test_empty_results(self):
        assert _build_web_sources([]) == []

    def test_fallback_title(self):
        results = [{"url": "http://example.com"}]
        sources = _build_web_sources(results)
        assert sources[0]["title"] == "http://example.com"


# ---------------------------------------------------------------------------
# Evidence context formatting tests
# ---------------------------------------------------------------------------

class TestBuildEvidenceContext:
    def test_knowledge_only(self):
        results = [
            {
                "document_id": 1,
                "chunk_id": 10,
                "title": "Doc 1",
                "score": 0.85,
                "content": "Hello world.",
            }
        ]
        context = _build_evidence_context(results)
        assert "[Source 1]" in context
        assert "type=knowledge" in context
        assert "title=Doc 1" in context
        assert "document_id=1" in context
        assert "chunk_id=10" in context
        assert "score=0.850" in context
        assert "Hello world." in context

    def test_web_and_knowledge(self):
        knowledge = [
            {"document_id": 1, "chunk_id": 10, "title": "Doc", "score": 0.9, "content": "Doc content."}
        ]
        web = [
            {"title": "Web", "url": "http://web.com", "domain": "web.com", "score": 0.8, "snippet": "Web snippet."}
        ]
        context = _build_evidence_context(knowledge, web)
        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "type=web" in context
        assert "url=http://web.com" in context

    def test_empty_knowledge(self):
        assert _build_evidence_context([]) == ""

    def test_web_only(self):
        web = [{"title": "Web", "url": "http://w.com", "domain": "w.com", "score": 0.7, "snippet": "Snippet."}]
        context = _build_evidence_context([], web)
        assert "[Source 1]" in context
        assert "type=web" in context

    def test_table_evidence_preserves_citation_metadata_and_markdown(self):
        results = [
            {
                "document_id": 10,
                "chunk_id": 700,
                "title": "Revenue table",
                "score": 0.75,
                "source_type": "table",
                "table_id": "tbl-revenue-2026",
                "page_number": 4,
                "section_key": "finance.revenue",
                "text_summary": "Table Revenue with 3 rows and 3 columns.",
                "markdown_table": "| Quarter | Revenue | Margin |\n| --- | --- | --- |\n| Q2 | $1.5M | 44% |",
            }
        ]

        context = _build_evidence_context(results)

        assert "[Source 1] type=table" in context
        assert "document_id=10" in context
        assert "chunk_id=700" in context
        assert "table_id=tbl-revenue-2026" in context
        assert "page_number=4" in context
        assert "section_key=finance.revenue" in context
        assert "Table summary: Table Revenue with 3 rows and 3 columns." in context
        assert "| Q2 | $1.5M | 44% |" in context


# ---------------------------------------------------------------------------
# Retrieval packing tests
# ---------------------------------------------------------------------------

class TestPackRetrievalResults:
    def test_basic_packing(self):
        results = [
            {"document_id": 1, "chunk_index": 0, "score": 0.9, "content": "A" * 100, "token_estimate": 25},
            {"document_id": 2, "chunk_index": 1, "score": 0.8, "content": "B" * 100, "token_estimate": 25},
        ]
        packed, total = _pack_retrieval_results(results)
        assert len(packed) == 2
        assert total > 0

    def test_respects_max_chunks(self):
        results = [
            {"document_id": i, "chunk_index": 0, "score": 0.9, "content": "A", "token_estimate": 10}
            for i in range(20)
        ]
        packed, total = _pack_retrieval_results(results, max_context_chunks=3)
        assert len(packed) == 3

    def test_respects_token_budget(self):
        results = [
            {"document_id": i, "chunk_index": 0, "score": 0.9, "content": "A" * 100, "token_estimate": 2000}
            for i in range(10)
        ]
        packed, total = _pack_retrieval_results(results, max_context_chunks=10, max_context_tokens=2500)
        assert len(packed) <= 2  # 2000 + 2000 > 2500, so at most 2
        assert total <= 2500  # but first result always included

    def test_first_result_always_included(self):
        """Even if the first result exceeds the token budget, it is included."""
        results = [
            {"document_id": 1, "chunk_index": 0, "score": 0.9, "content": "A" * 1000},
        ]
        packed, total = _pack_retrieval_results(results, max_context_tokens=10)
        assert len(packed) == 1

    def test_empty_results(self):
        packed, total = _pack_retrieval_results([])
        assert packed == []
        assert total == 0

    def test_fallback_token_estimate(self):
        results = [
            {"document_id": 1, "chunk_index": 0, "score": 0.9, "content": "hello world"},
        ]
        packed, total = _pack_retrieval_results(results)
        assert len(packed) == 1
        assert packed[0]["token_estimate"] > 0
