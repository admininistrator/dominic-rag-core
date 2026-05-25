"""Retrieval — query processing, scoring, reranking, deduplication, evidence."""

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

__all__ = [
    "QUERY_EXPANSION_RULES",
    "_expand_query",
    "_normalize_for_search",
    "_strip_accents",
    "_tokenize",
    "_cosine_similarity",
    "_hybrid_score",
    "_lexical_overlap_score",
    "_normalize_for_dedupe",
    "_rerank_results",
    "_dedupe_scored_results",
    "_build_snippet",
    "_classify_evidence_strength",
    "_estimate_token_count",
    "_is_embedding_compatible",
]
