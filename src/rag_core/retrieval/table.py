"""Table-aware retrieval and query routing primitives."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rag_core.parsing import TableObject
from rag_core.retrieval.contracts import RetrievalCandidate, RetrievalQuery
from rag_core.retrieval.scoring import _lexical_overlap_score

TableRouteName = Literal["text", "table", "mixed"]


@dataclass(frozen=True, slots=True)
class TableQueryRoute:
    """Routing decision for text/table/mixed QA paths."""

    route: TableRouteName
    reason: str
    table_signal_score: float = 0.0
    text_signal_score: float = 0.0
    table_candidate_count: int = 0
    text_candidate_count: int = 0


class TableCorpusRetriever:
    """Dependency-free retriever over extracted table artifacts.

    The retriever emits first-class ``RetrievalCandidate`` objects with
    ``source_stage='table'`` and ``source_type='table'`` while preserving table
    provenance (table_id, page, section, row/column counts, markdown evidence).
    It stays in rag-core and operates only on caller-supplied, pre-authorized
    table objects or mappings; it performs no auth, DB, file, or network work.
    """

    def __init__(
        self,
        tables: Sequence[TableObject | Mapping[str, Any]],
        *,
        enabled: bool = True,
        top_k: int | None = None,
        min_score: float = 0.0,
        source_stage: str = "table",
    ) -> None:
        self._tables = list(tables or [])
        self.enabled = enabled
        self.top_k = top_k
        self.min_score = min_score
        self.source_stage = source_stage

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        """Return table candidates scored by lexical overlap with table evidence."""

        if not self.enabled:
            return []
        limit = self._resolve_top_k(query.top_k)
        if limit <= 0:
            return []

        scored: list[RetrievalCandidate] = []
        for table in self._tables:
            payload = _table_to_candidate_payload(table)
            score = _lexical_overlap_score(
                query.effective_text,
                " ".join(
                    part
                    for part in [
                        payload.get("title"),
                        payload.get("content"),
                        payload.get("text_summary"),
                        payload.get("markdown_table"),
                    ]
                    if part
                ),
            )
            if score <= self.min_score:
                continue
            payload["score"] = score
            payload["lexical_score"] = score
            table_id = str(payload.get("table_id") or "")
            payload["trace"] = {"retriever": "table_corpus", "table_id": table_id}
            scored.append(RetrievalCandidate.from_mapping(payload, source_stage=self.source_stage))

        scored.sort(key=_candidate_sort_key)
        return scored[:limit]

    def _resolve_top_k(self, query_top_k: int) -> int:
        raw_limit = self.top_k if self.top_k is not None else query_top_k
        try:
            return max(0, int(raw_limit))
        except (TypeError, ValueError):
            return 0


def route_table_aware_query(
    query: RetrievalQuery | str,
    *,
    text_candidates: Sequence[RetrievalCandidate | Mapping[str, Any]] | None = None,
    table_candidates: Sequence[RetrievalCandidate | Mapping[str, Any]] | None = None,
    table_score_threshold: float = 0.05,
    text_score_threshold: float = 0.05,
) -> TableQueryRoute:
    """Choose a text, table, or mixed QA path from query intent and evidence.

    This is deliberately heuristic and provider-neutral: callers can use it after
    collecting candidate sets to decide whether generation should use text-only,
    table-only, or mixed evidence. It never performs retrieval itself and never
    interprets backend scope/permission filters.
    """

    query_text = query.effective_text if isinstance(query, RetrievalQuery) else str(query or "")
    text_items = list(text_candidates or [])
    table_items = list(table_candidates or [])
    table_signal = _max_candidate_score(table_items)
    text_signal = _max_candidate_score(text_items)
    has_table_intent = _has_any(query_text, _TABLE_INTENT_TERMS)
    has_text_intent = _has_any(query_text, _TEXT_INTENT_TERMS)

    if has_table_intent:
        table_signal = max(table_signal, table_score_threshold)
    if has_text_intent:
        text_signal = max(text_signal, text_score_threshold)

    table_supported = bool(table_items) and table_signal >= table_score_threshold
    text_supported = bool(text_items) and text_signal >= text_score_threshold

    if table_supported and text_supported and (has_table_intent or has_text_intent):
        return TableQueryRoute(
            route="mixed",
            reason="table and text evidence are both relevant",
            table_signal_score=table_signal,
            text_signal_score=text_signal,
            table_candidate_count=len(table_items),
            text_candidate_count=len(text_items),
        )
    if table_supported and (has_table_intent or not text_supported or table_signal >= text_signal):
        return TableQueryRoute(
            route="table",
            reason="table evidence best matches query intent",
            table_signal_score=table_signal,
            text_signal_score=text_signal,
            table_candidate_count=len(table_items),
            text_candidate_count=len(text_items),
        )
    if text_supported:
        return TableQueryRoute(
            route="text",
            reason="text evidence best matches query intent",
            table_signal_score=table_signal,
            text_signal_score=text_signal,
            table_candidate_count=len(table_items),
            text_candidate_count=len(text_items),
        )
    if has_table_intent:
        return TableQueryRoute(
            route="table",
            reason="query uses table-oriented language but no strong text evidence is available",
            table_signal_score=table_signal,
            text_signal_score=text_signal,
            table_candidate_count=len(table_items),
            text_candidate_count=len(text_items),
        )
    return TableQueryRoute(
        route="text",
        reason="default text QA route",
        table_signal_score=table_signal,
        text_signal_score=text_signal,
        table_candidate_count=len(table_items),
        text_candidate_count=len(text_items),
    )


def _table_to_candidate_payload(table: TableObject | Mapping[str, Any]) -> dict[str, Any]:
    table_dict = table.to_dict() if isinstance(table, TableObject) else dict(table or {})
    metadata = dict(table_dict.get("metadata") or table_dict.get("metadata_json") or {})
    table_id = table_dict.get("table_id")
    markdown_table = str(table_dict.get("markdown_table") or "")
    text_summary = str(table_dict.get("text_summary") or "")
    content = str(table_dict.get("content") or "\n".join(part for part in [text_summary, markdown_table] if part))
    title = str(table_dict.get("title") or table_dict.get("caption") or table_id or "Table")
    metadata_json = {
        **metadata,
        "source_type": "table",
        "table_id": table_id,
        "page_number": table_dict.get("page_number"),
        "section_key": table_dict.get("section_key"),
        "row_count": table_dict.get("row_count"),
        "column_count": table_dict.get("column_count"),
        "caption": table_dict.get("caption"),
        "markdown_table": markdown_table,
        "text_summary": text_summary,
        "table_source_type": table_dict.get("source_type"),
        "source_filename": table_dict.get("source_filename"),
    }
    return {
        "document_id": metadata.get("document_id") or table_dict.get("document_id"),
        "chunk_id": metadata.get("chunk_id") or table_dict.get("chunk_id"),
        "chunk_index": metadata.get("chunk_index") or table_dict.get("chunk_index"),
        "title": title,
        "content": content,
        "snippet": text_summary or content[:240],
        "source_type": "table",
        "source_uri": metadata.get("source_uri") or table_dict.get("source_uri"),
        "table_id": table_id,
        "page_number": table_dict.get("page_number"),
        "section_key": table_dict.get("section_key"),
        "metadata_json": metadata_json,
        "markdown_table": markdown_table,
        "text_summary": text_summary,
    }


def _candidate_sort_key(candidate: RetrievalCandidate) -> tuple[Any, ...]:
    return (
        -float(candidate.score or 0.0),
        candidate.document_id if candidate.document_id is not None else 10**18,
        candidate.chunk_id if candidate.chunk_id is not None else 10**18,
        candidate.metadata.table_id or "",
        candidate.title,
    )


def _max_candidate_score(candidates: Sequence[RetrievalCandidate | Mapping[str, Any]]) -> float:
    values = [_candidate_score(candidate) for candidate in candidates]
    return max(values, default=0.0)


def _candidate_score(candidate: RetrievalCandidate | Mapping[str, Any]) -> float:
    if isinstance(candidate, RetrievalCandidate):
        return max(
            float(candidate.score or 0.0),
            float(candidate.lexical_score or 0.0),
            float(candidate.semantic_score or 0.0),
            float(candidate.rerank_score or 0.0),
        )
    raw = dict(candidate or {})
    scores = [raw.get("score"), raw.get("lexical_score"), raw.get("semantic_score"), raw.get("rerank_score")]
    parsed: list[float] = []
    for value in scores:
        try:
            parsed.append(float(value or 0.0))
        except (TypeError, ValueError):
            parsed.append(0.0)
    return max(parsed, default=0.0)


def _has_any(text: str, terms: set[str]) -> bool:
    normalized = f" {str(text or '').lower()} "
    return any(term in normalized for term in terms)


_TABLE_INTENT_TERMS = {
    " table ",
    " tabular ",
    " row ",
    " rows ",
    " column ",
    " columns ",
    " cell ",
    " cells ",
    " spreadsheet ",
    " csv ",
    " xlsx ",
    " total ",
    " count ",
    " average ",
    " highest ",
    " lowest ",
    " maximum ",
    " minimum ",
    " compare ",
    " comparison ",
    " q1 ",
    " q2 ",
    " q3 ",
    " q4 ",
}

_TEXT_INTENT_TERMS = {
    " explain ",
    " summarize ",
    " summary ",
    " narrative ",
    " prose ",
    " context ",
    " background ",
    " why ",
    " describe ",
    " details ",
}
