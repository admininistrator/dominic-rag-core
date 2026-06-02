"""Sparse/lexical retrieval adapters for provider-neutral retrieval pipelines."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag_core.retrieval.contracts import RetrievalCandidate, RetrievalQuery
from rag_core.retrieval.scoring import _lexical_overlap_score


class LexicalCorpusRetriever:
    """In-memory lexical retriever over a caller-provided corpus.

    The adapter is intentionally small and dependency-free so Phase 2 can expose
    a concrete sparse retriever without binding rag-core to a particular BM25 or
    search-service implementation. A later adapter can swap the corpus scan for a
    real sparse backend while preserving the same ``retrieve()`` contract.
    """

    def __init__(
        self,
        corpus: Sequence[Mapping[str, Any]],
        *,
        enabled: bool = True,
        top_k: int | None = None,
        min_score: float = 0.0,
        source_stage: str = "sparse",
    ) -> None:
        self._corpus = list(corpus or [])
        self.enabled = enabled
        self.top_k = top_k
        self.min_score = min_score
        self.source_stage = source_stage

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        """Return bounded lexical candidates for ``query``.

        When disabled, returns an empty list so vector-only retrieval can proceed
        unchanged. Results are sorted deterministically by descending lexical
        score and stable candidate identity fields.
        """

        if not self.enabled:
            return []

        limit = self._resolve_top_k(query.top_k)
        if limit <= 0:
            return []

        scored: list[RetrievalCandidate] = []
        query_text = query.effective_text
        for raw in self._corpus:
            item = dict(raw or {})
            content = str(item.get("content") or item.get("text") or "")
            title = str(item.get("title") or "")
            score = _lexical_overlap_score(query_text, " ".join(part for part in [title, content] if part))
            if score <= self.min_score:
                continue
            item.setdefault("source_type", "text")
            item["score"] = score
            item["lexical_score"] = score
            item["trace"] = {"retriever": "lexical_corpus"}
            scored.append(RetrievalCandidate.from_mapping(item, source_stage=self.source_stage))

        scored.sort(key=_candidate_sort_key)
        return scored[:limit]

    def _resolve_top_k(self, query_top_k: int) -> int:
        raw_limit = self.top_k if self.top_k is not None else query_top_k
        try:
            return max(0, int(raw_limit))
        except (TypeError, ValueError):
            return 0


def _candidate_sort_key(candidate: RetrievalCandidate) -> tuple[Any, ...]:
    return (
        -float(candidate.score or 0.0),
        candidate.document_id if candidate.document_id is not None else 10**18,
        candidate.chunk_id if candidate.chunk_id is not None else 10**18,
        candidate.chunk_index if candidate.chunk_index is not None else 10**18,
        candidate.title,
        candidate.content,
    )
