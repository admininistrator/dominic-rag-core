"""Tests for the answer/prompt module.

Covers:
- ``_determine_answer_policy()``: Answer policy classification logic
"""
from __future__ import annotations

from rag_core.context.prompt import _determine_answer_policy


class TestDetermineAnswerPolicy:
    """Answer policy classification tests."""

    def test_grounded_evidence_no_document(self):
        """Grounded evidence + no knowledge_document_id + strong lexical -> grounded."""
        result = {
            "evidence_strength": "grounded",
            "results": [
                {"score": 0.85, "rerank_score": 0.90, "lexical_score": 0.35, "semantic_score": 0.80}
            ],
        }
        policy = _determine_answer_policy(result)
        assert policy == "grounded"

    def test_grounded_evidence_no_document_weak_lexical_has_results(self):
        """Grounded evidence + no doc + weak lexical but has results -> cautious_general."""
        result = {
            "evidence_strength": "grounded",
            "results": [
                {"score": 0.85, "rerank_score": 0.90, "lexical_score": 0.05, "semantic_score": 0.80}
            ],
        }
        policy = _determine_answer_policy(
            result,
            retrieval_low_confidence_score=0.2,
        )
        assert policy == "cautious_general"

    def test_grounded_evidence_no_document_no_results(self):
        """Grounded evidence + no doc + no results -> insufficient_evidence."""
        result = {"evidence_strength": "grounded", "results": []}
        policy = _determine_answer_policy(result)
        assert policy == "insufficient_evidence"

    def test_grounded_evidence_with_document(self):
        """Grounded evidence + knowledge_document_id -> grounded."""
        result = {
            "evidence_strength": "grounded",
            "results": [{"score": 0.5, "rerank_score": 0.5}],
        }
        policy = _determine_answer_policy(result, knowledge_document_id=42)
        assert policy == "grounded"

    def test_weak_evidence_no_document(self):
        """Weak evidence + no doc -> cautious_general."""
        result = {"evidence_strength": "weak", "results": [{"score": 0.15}]}
        policy = _determine_answer_policy(result)
        assert policy == "cautious_general"

    def test_none_evidence_no_document(self):
        """No evidence + no doc + no results -> insufficient_evidence."""
        result = {"evidence_strength": "none", "results": []}
        policy = _determine_answer_policy(result)
        assert policy == "insufficient_evidence"

    def test_document_scoped_with_low_confidence_but_lexical_support(self):
        """Document scoped + low confidence but direct lexical support -> grounded."""
        result = {
            "evidence_strength": "weak",
            "results": [
                {"score": 0.25, "rerank_score": 0.25, "lexical_score": 0.25, "semantic_score": 0.01}
            ],
        }
        policy = _determine_answer_policy(
            result,
            knowledge_document_id=1,
            retrieval_low_confidence_score=0.2,
            retrieval_min_lexical_score=0.1,
        )
        assert policy == "grounded"

    def test_document_scoped_with_low_confidence_no_lexical(self):
        """Document scoped + low confidence + no lexical support -> insufficient_evidence."""
        result = {
            "evidence_strength": "none",
            "results": [
                {"score": 0.25, "rerank_score": 0.25, "lexical_score": 0.05, "semantic_score": 0.01}
            ],
        }
        policy = _determine_answer_policy(
            result,
            knowledge_document_id=1,
            retrieval_low_confidence_score=0.2,
            retrieval_min_lexical_score=0.1,
        )
        assert policy == "insufficient_evidence"

    def test_document_scoped_with_semantic_support(self):
        """Document scoped + semantic support but no lexical -> grounded."""
        result = {
            "evidence_strength": "weak",
            "results": [
                {"score": 0.25, "rerank_score": 0.25, "lexical_score": 0.05, "semantic_score": 0.35}
            ],
        }
        policy = _determine_answer_policy(
            result,
            knowledge_document_id=1,
            retrieval_low_confidence_score=0.2,
            retrieval_min_lexical_score=0.1,
        )
        assert policy == "grounded"

    def test_fallback_evidence(self):
        """Fallback evidence -> cautious_general."""
        result = {"evidence_strength": "fallback", "results": [{"score": 0.1}]}
        policy = _determine_answer_policy(result)
        assert policy == "cautious_general"

    def test_scoped_doc_strict_grounding_no_evidence(self):
        """Scoped doc + strict grounding + no evidence -> insufficient_evidence."""
        result = {"evidence_strength": "none", "results": []}
        policy = _determine_answer_policy(
            result,
            knowledge_document_id=5,
            retrieval_strict_grounding_for_scoped_docs=True,
        )
        assert policy == "insufficient_evidence"

    def test_scoped_doc_no_strict_grounding_fallback(self):
        """Scoped doc + no strict grounding + fallback -> cautious_general."""
        result = {"evidence_strength": "fallback", "results": [{"score": 0.1}]}
        policy = _determine_answer_policy(
            result,
            knowledge_document_id=5,
            retrieval_strict_grounding_for_scoped_docs=False,
        )
        assert policy == "cautious_general"

    def test_custom_thresholds(self):
        """Custom thresholds affect the policy decision."""
        result = {
            "evidence_strength": "grounded",
            "results": [
                {"score": 0.5, "rerank_score": 0.5, "lexical_score": 0.05, "semantic_score": 0.1}
            ],
        }
        # With very high low_confidence_score, this should be cautious_general even though grounded
        policy = _determine_answer_policy(
            result,
            retrieval_low_confidence_score=0.5,
        )
        assert policy == "cautious_general"

    def test_None_retrieval_result(self):
        """None retrieval_result -> insufficient_evidence."""
        policy = _determine_answer_policy(None)
        assert policy == "insufficient_evidence"

    def test_empty_retrieval_result(self):
        """Empty dict retrieval result -> insufficient_evidence."""
        policy = _determine_answer_policy({})
        assert policy == "insufficient_evidence"

    def test_packed_results_path(self):
        """Uses packed_results if present."""
        result = {
            "evidence_strength": "grounded",
            "packed_results": [
                {"score": 0.9, "rerank_score": 0.9, "lexical_score": 0.5, "semantic_score": 0.8}
            ],
            "results": [],  # results empty but packed_results populated
        }
        policy = _determine_answer_policy(result)
        assert policy == "grounded"
