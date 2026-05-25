"""Tests for the retrieval module.

Covers:
- Query processing: normalisation, tokenisation, accent stripping, expansion
- Cosine similarity: matching, mismatching, empty, zero-norm vectors
- Lexical overlap: full, partial, empty, single-token edge cases
- Hybrid scoring: weight application, edge weights
- Reranking: title boost, position decay, sort order
- Deduplication: duplicate content, same-document duplicates
- Evidence classification: none, fallback, weak, grounded
- Snippet building: short, long, empty
- Token estimation: explicit, implicit, empty
- Embedding compatibility: matching, mismatching, legacy
"""
from __future__ import annotations

import math

from rag_core.retrieval.query_processor import (
    QUERY_EXPANSION_RULES,
    _expand_query,
    _normalize_for_search,
    _strip_accents,
    _tokenize,
)
from rag_core.retrieval.scoring import (
    _cosine_similarity,
    _hybrid_score,
    _lexical_overlap_score,
    _normalize_for_dedupe,
)
from rag_core.retrieval.reranker import _rerank_results
from rag_core.retrieval.deduplicator import _dedupe_scored_results
from rag_core.retrieval.evidence import (
    _build_snippet,
    _classify_evidence_strength,
    _estimate_token_count,
    _is_embedding_compatible,
)


# ---------------------------------------------------------------------------
# Query processing tests
# ---------------------------------------------------------------------------

class TestStripAccents:
    def test_plain_text(self):
        assert _strip_accents("hello") == "hello"

    def test_vietnamese_accents(self):
        assert "a" in _strip_accents("Xin chào")

    def test_empty(self):
        assert _strip_accents("") == ""

    def test_none(self):
        assert _strip_accents(None) == ""


class TestNormalizeForSearch:
    def test_basic(self):
        result = _normalize_for_search("Hello, World!")
        assert result == "hello world"

    def test_punctuation(self):
        result = _normalize_for_search("What's up?")
        assert "'" not in result

    def test_multiple_spaces(self):
        result = _normalize_for_search("hello    world")
        assert result == "hello world"

    def test_empty(self):
        assert _normalize_for_search("") == ""


class TestTokenize:
    def test_basic_tokens(self):
        tokens = _tokenize("hello world")
        assert tokens == {"hello", "world"}

    def test_punctuation_removed(self):
        tokens = _tokenize("hello, world!")
        assert tokens == {"hello", "world"}

    def test_empty(self):
        assert _tokenize("") == set()


class TestExpandQuery:
    def test_no_expansion_needed(self):
        rewritten, expansions = _expand_query("hello world")
        assert expansions == []
        assert rewritten == "hello world"

    def test_vietnamese_expansion(self):
        rewritten, expansions = _expand_query("chinh sach hoan tien")
        assert "policy" in expansions or "refund" in expansions
        assert " ".join((rewritten or "").split()) != "chinh sach hoan tien"

    def test_expansion_disabled(self):
        rewritten, expansions = _expand_query("hoan tien", enable_query_expansion=False)
        assert expansions == []
        assert rewritten == "hoan tien"

    def test_empty_query(self):
        rewritten, expansions = _expand_query("")
        assert rewritten == ""
        assert expansions == []

    def test_all_expansion_rules_have_candidates(self):
        for phrase, candidates in QUERY_EXPANSION_RULES.items():
            assert len(candidates) >= 1, f"Rule '{phrase}' has no candidates"


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_partial_match(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        result = _cosine_similarity(a, b)
        expected = 1.0 / math.sqrt(2.0)
        assert abs(result - expected) < 1e-10


class TestLexicalOverlapScore:
    def test_full_overlap(self):
        score = _lexical_overlap_score("hello world", "hello world")
        assert score > 0.0

    def test_no_overlap(self):
        score = _lexical_overlap_score("hello world", "goodbye universe")
        assert score == 0.0

    def test_partial_overlap(self):
        score = _lexical_overlap_score("hello world foo", "hello bar baz")
        assert 0.0 < score < 1.0

    def test_empty_query(self):
        assert _lexical_overlap_score("", "hello world") == 0.0

    def test_empty_content(self):
        assert _lexical_overlap_score("hello", "") == 0.0

    def test_single_token(self):
        score = _lexical_overlap_score("hello", "hello world")
        assert score > 0.0


class TestHybridScore:
    def test_default_weights(self):
        score = _hybrid_score(1.0, 1.0)
        assert score == 1.0

    def test_all_semantic(self):
        score = _hybrid_score(1.0, 0.0, semantic_weight=1.0, lexical_weight=0.0)
        assert score == 1.0

    def test_all_lexical(self):
        score = _hybrid_score(0.0, 1.0, semantic_weight=0.0, lexical_weight=1.0)
        assert score == 1.0

    def test_half_half(self):
        score = _hybrid_score(0.5, 0.5, semantic_weight=0.5, lexical_weight=0.5)
        assert score == 0.5

    def test_clamped(self):
        score = _hybrid_score(2.0, 2.0)
        assert score <= 1.0


class TestNormalizeForDedupe:
    def test_collapse_spaces(self):
        assert _normalize_for_dedupe("Hello    World") == "hello world"

    def test_case_folding(self):
        assert _normalize_for_dedupe("HELLO WORLD") == "hello world"

    def test_empty(self):
        assert _normalize_for_dedupe("") == ""

    def test_whitespace_only(self):
        assert _normalize_for_dedupe("   ") == ""


# ---------------------------------------------------------------------------
# Reranking tests
# ---------------------------------------------------------------------------

class TestRerankResults:
    def test_basic_rerank(self):
        results = [
            {"document_id": 1, "chunk_index": 0, "score": 0.5, "title": "test", "content": "hello"},
            {"document_id": 1, "chunk_index": 1, "score": 0.3, "title": "other", "content": "world"},
        ]
        reranked = _rerank_results("test", results)
        assert len(reranked) == 2
        assert "rerank_score" in reranked[0]
        assert "token_estimate" in reranked[0]

    def test_respects_max_candidates(self):
        results = [
            {"document_id": i, "chunk_index": 0, "score": 0.5, "title": "t", "content": "c"}
            for i in range(20)
        ]
        reranked = _rerank_results("query", results, max_rerank_candidates=5)
        assert len(reranked) == 5

    def test_empty_results(self):
        assert _rerank_results("query", []) == []

    def test_title_boost(self):
        results = [
            {"document_id": 1, "chunk_index": 0, "score": 0.5, "title": "exact match title", "content": "hello"},
            {"document_id": 2, "chunk_index": 0, "score": 0.5, "title": "unrelated", "content": "world"},
        ]
        reranked = _rerank_results("exact match title", results)
        # The exact-match title should get a boost
        assert reranked[0]["rerank_score"] > results[0]["score"]

    def test_position_decay(self):
        results = [
            {"document_id": 1, "chunk_index": 0, "score": 0.5, "title": "same", "content": "a"},
            {"document_id": 1, "chunk_index": 10, "score": 0.5, "title": "same", "content": "b"},
        ]
        reranked = _rerank_results("same", results)
        # Since scores are equal but position differs, rank may depend
        assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDedupeScoredResults:
    def test_no_duplicates(self):
        results = [
            {"document_id": 1, "content": "hello world"},
            {"document_id": 2, "content": "goodbye world"},
        ]
        deduped = _dedupe_scored_results(results)
        assert len(deduped) == 2

    def test_exact_duplicate_removed(self):
        results = [
            {"document_id": 1, "content": "hello world"},
            {"document_id": 1, "content": "hello world"},
        ]
        deduped = _dedupe_scored_results(results)
        assert len(deduped) == 1

    def test_different_document_same_content(self):
        results = [
            {"document_id": 1, "content": "hello world"},
            {"document_id": 2, "content": "hello world"},
        ]
        deduped = _dedupe_scored_results(results)
        assert len(deduped) == 2  # Same content, different document → kept

    def test_empty_results(self):
        assert _dedupe_scored_results([]) == []


# ---------------------------------------------------------------------------
# Evidence classification tests
# ---------------------------------------------------------------------------

class TestClassifyEvidenceStrength:
    def test_no_results(self):
        assert _classify_evidence_strength([], fallback_used=False) == "none"

    def test_fallback_used(self):
        results = [{"score": 0.9}]
        assert _classify_evidence_strength(results, fallback_used=True) == "fallback"

    def test_grounded(self):
        results = [{"score": 0.5}]
        assert _classify_evidence_strength(results, fallback_used=False) == "grounded"

    def test_weak(self):
        results = [{"score": 0.1}]
        assert _classify_evidence_strength(results, fallback_used=False) == "weak"

    def test_custom_threshold(self):
        results = [{"score": 0.3}]
        assert _classify_evidence_strength(results, fallback_used=False, low_confidence_score=0.5) == "weak"
        assert _classify_evidence_strength(results, fallback_used=False, low_confidence_score=0.2) == "grounded"


# ---------------------------------------------------------------------------
# Snippet building tests
# ---------------------------------------------------------------------------

class TestBuildSnippet:
    def test_short_text(self):
        assert _build_snippet("hello world") == "hello world"

    def test_long_text(self):
        text = "word " * 100
        snippet = _build_snippet(text, max_chars=50)
        assert len(snippet) <= 50 or snippet.endswith("...")

    def test_empty(self):
        assert _build_snippet("") == ""

    def test_whitespace_normalized(self):
        assert _build_snippet("hello    world") == "hello world"

    def test_truncation_with_ellipsis(self):
        text = "a" * 300
        snippet = _build_snippet(text, max_chars=220)
        assert snippet.endswith("...")
        assert len(snippet) <= 220


# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------

class TestEstimateTokenCount:
    def test_explicit_count(self):
        assert _estimate_token_count("some text", explicit_count=42) == 42

    def test_implicit_estimate(self):
        text = "word " * 40  # ~200 chars
        estimate = _estimate_token_count(text)
        # len("word " * 40) = 200, so estimate = 200 // 4 = 50
        assert estimate == max(1, len(" ".join(text.split())) // 4)

    def test_empty_text(self):
        assert _estimate_token_count("") == 0

    def test_short_text(self):
        assert _estimate_token_count("hi") == 1  # max(1, 2//4=0) = 1


# ---------------------------------------------------------------------------
# Embedding compatibility tests
# ---------------------------------------------------------------------------

class TestIsEmbeddingCompatible:
    def test_matching_provider_and_model(self):
        assert _is_embedding_compatible(
            {"embedding_provider": "local", "embedding_model": "local-hash-v1"},
            "local", "local-hash-v1"
        )

    def test_mismatched_provider(self):
        assert not _is_embedding_compatible(
            {"embedding_provider": "ollama", "embedding_model": "nomic"},
            "local", "local-hash-v1"
        )

    def test_legacy_chunk_no_metadata(self):
        assert _is_embedding_compatible({}, "local", "local-hash-v1")

    def test_legacy_chunk_empty_provider(self):
        assert _is_embedding_compatible(
            {"embedding_provider": ""}, "local", "local-hash-v1"
        )

    def test_case_insensitive(self):
        assert _is_embedding_compatible(
            {"embedding_provider": "LOCAL", "embedding_model": "LOCAL-HASH-V1"},
            "local", "local-hash-v1"
        )

    def test_json_string_meta(self):
        assert _is_embedding_compatible(
            '{"embedding_provider": "local", "embedding_model": "local-hash-v1"}',
            "local", "local-hash-v1"
        )
