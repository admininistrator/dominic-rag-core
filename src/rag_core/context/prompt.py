"""Answer policy and prompt-related decision logic.

Provides:
- ``_determine_answer_policy()``: Determine whether retrieval evidence is
  sufficient for a ``grounded``, ``cautious_general``, or
  ``insufficient_evidence`` answer.

Extracted from DominicBE's ``app/services/chat_service.py``.
- ``settings.retrieval_min_lexical_score``,
  ``settings.retrieval_low_confidence_score``, and
  ``settings.retrieval_strict_grounding_for_scoped_docs`` replaced with
  explicit parameters.

.. note::
    ``_compose_system_prompt()`` was **not** extracted — it contains
    app-specific Vietnamese prompt text and is tightly coupled to chat
    session state (``knowledge_base_active``, ``summary_text``,
    ``web_search_result``). See P06-T01 in the feature tasks for the
    full coupling analysis.
"""
from __future__ import annotations


def _determine_answer_policy(
    retrieval_result: dict | None,
    *,
    knowledge_document_id: int | None = None,
    retrieval_min_lexical_score: float = 0.1,
    retrieval_low_confidence_score: float = 0.2,
    retrieval_strict_grounding_for_scoped_docs: bool = True,
) -> str:
    """Determine the answer policy based on retrieval result quality.

    Evaluates evidence strength, top-result scores, and scoped-document
    grounding rules to classify the answer as one of:

    - ``"grounded"`` — sufficient evidence for a grounded answer.
    - ``"cautious_general"`` — partial evidence; answer cautiously.
    - ``"insufficient_evidence"`` — not enough evidence to answer.

    Args:
        retrieval_result: Dict from ``search_knowledge()`` containing
            ``evidence_strength``, ``results`` or ``packed_results``, etc.
        knowledge_document_id: If set, indicates the answer is scoped to
            a specific document.
        retrieval_min_lexical_score: Minimum lexical overlap score for
            lexical support (default: ``0.1``).
        retrieval_low_confidence_score: Threshold below which evidence
            is considered low-confidence (default: ``0.2``).
        retrieval_strict_grounding_for_scoped_docs: When ``True`` and a
            ``knowledge_document_id`` is set, weak evidence defaults to
            ``"insufficient_evidence"`` (default: ``True``).

    Returns:
        One of ``"grounded"``, ``"cautious_general"``, or
        ``"insufficient_evidence"``.
    """
    evidence_strength = (retrieval_result or {}).get("evidence_strength") or "none"
    results = (
        (retrieval_result or {}).get("packed_results")
        or (retrieval_result or {}).get("results")
        or []
    )
    top_result = results[0] if results else {}
    top_confidence = max(
        float(top_result.get("score") or 0.0),
        float(top_result.get("rerank_score") or 0.0),
    )
    top_lexical = float(top_result.get("lexical_score") or 0.0)
    top_semantic = float(top_result.get("semantic_score") or 0.0)
    has_direct_lexical_support = top_lexical >= retrieval_min_lexical_score
    has_strong_lexical_support = top_lexical >= retrieval_low_confidence_score
    has_high_semantic_support = top_semantic >= retrieval_low_confidence_score

    if evidence_strength == "grounded":
        if knowledge_document_id is None:
            if has_strong_lexical_support:
                return "grounded"
            if results:
                return "cautious_general"
            return "insufficient_evidence"
        return "grounded"

    if (
        knowledge_document_id is not None
        and results
        and top_confidence >= retrieval_low_confidence_score
        and (has_direct_lexical_support or has_high_semantic_support)
    ):
        return "grounded"

    if evidence_strength == "weak":
        return "cautious_general"

    if (
        knowledge_document_id is not None
        and retrieval_strict_grounding_for_scoped_docs
    ):
        return "insufficient_evidence"

    if evidence_strength == "fallback":
        return "cautious_general"

    return "insufficient_evidence"
