"""Context building, source citation formatting, and answer-policy logic."""

from rag_core.context.builder import _build_evidence_context, _pack_retrieval_results
from rag_core.context.prompt import _determine_answer_policy
from rag_core.context.sources import _build_sources, _build_web_sources

__all__ = [
    "_build_evidence_context",
    "_pack_retrieval_results",
    "_build_sources",
    "_build_web_sources",
    "_determine_answer_policy",
]
