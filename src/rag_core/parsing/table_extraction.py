"""Table-aware ingestion primitives and extraction hooks.

This module intentionally stays dependency-free. It gives rag-core a stable MVP
schema for extracted tables plus pluggable extraction hooks that can be extended
for PDF/DOCX/XLSX backends later without moving table parsing logic into
DominicBE.
"""
from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class TableObject:
    """First-class extracted table artifact used by table-aware RAG.

    ``rows`` includes the header row when known. ``markdown_table`` and
    ``text_summary`` are derived by default so downstream ingestion/retrieval can
    index tables without requiring each extractor to duplicate formatting logic.
    """

    rows: list[list[Any]]
    table_id: str | None = None
    page_number: int | None = None
    caption: str | None = None
    section_key: str | None = None
    raw_table: str | None = None
    markdown_table: str | None = None
    text_summary: str | None = None
    source_type: str | None = None
    source_filename: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rows = _normalize_rows(self.rows)
        if self.raw_table is None:
            self.raw_table = "\n".join("\t".join(row) for row in self.rows)
        if self.markdown_table is None:
            self.markdown_table = render_markdown_table(self.rows)
        if self.text_summary is None:
            self.text_summary = summarize_table(self)
        if not self.table_id:
            self.table_id = _stable_table_id(
                self.rows,
                source_type=self.source_type,
                source_filename=self.source_filename,
                page_number=self.page_number,
                caption=self.caption,
                section_key=self.section_key,
            )
        self.metadata = dict(self.metadata or {})

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "caption": self.caption,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "section_key": self.section_key,
            "raw_table": self.raw_table,
            "markdown_table": self.markdown_table,
            "text_summary": self.text_summary,
            "source_type": self.source_type,
            "source_filename": self.source_filename,
            "rows": [list(row) for row in self.rows],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class TableExtractionResult:
    """Result container returned by table extraction hooks."""

    tables: list[TableObject] = field(default_factory=list)
    source_filename: str | None = None
    source_type: str | None = None
    text_without_tables: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tables(self) -> bool:
        return bool(self.tables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_tables": self.has_tables,
            "source_filename": self.source_filename,
            "source_type": self.source_type,
            "text_without_tables": self.text_without_tables,
            "tables": [table.to_dict() for table in self.tables],
            "metadata": dict(self.metadata or {}),
        }


class TableExtractor(Protocol):
    """Pluggable table extractor hook."""

    def extract(
        self,
        content: bytes | str,
        *,
        filename: str,
        mime_type: str | None = None,
    ) -> TableExtractionResult:
        """Extract table artifacts from file content."""


def render_markdown_table(rows: Sequence[Sequence[Any]]) -> str:
    """Render normalized rows as a markdown table."""

    normalized = _normalize_rows(rows)
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    padded = [row + [""] * (width - len(row)) for row in normalized]
    header = padded[0]
    separator = ["---"] * width
    body = padded[1:]
    rendered = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    rendered.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(rendered)


def summarize_table(table: TableObject) -> str:
    """Build a compact textual summary suitable for indexing/retrieval."""

    label = f"Table {table.caption}" if table.caption else "Table"
    parts = [f"{label} with {table.row_count} rows and {table.column_count} columns."]
    if table.rows:
        headers = [cell for cell in table.rows[0] if cell]
        if headers:
            parts.append("Columns: " + ", ".join(headers) + ".")
    if table.section_key:
        parts.append(f"Section: {table.section_key}.")
    if table.page_number is not None:
        parts.append(f"Page: {table.page_number}.")
    return " ".join(parts)


class CsvTableExtractor:
    """Dependency-free CSV table extractor."""

    def extract(
        self,
        content: bytes | str,
        *,
        filename: str,
        mime_type: str | None = None,
    ) -> TableExtractionResult:
        text = _decode_content(content)
        if not _looks_like_csv(filename, mime_type):
            return TableExtractionResult(source_filename=filename, source_type="csv")
        rows = _parse_csv_rows(text)
        if not _is_table_like(rows):
            return TableExtractionResult(source_filename=filename, source_type="csv")
        table = TableObject(
            rows=rows,
            caption=Path(filename).stem or None,
            raw_table=text,
            source_type="csv",
            source_filename=filename,
        )
        return TableExtractionResult(tables=[table], source_filename=filename, source_type="csv")


class MarkdownTableExtractor:
    """Extract GitHub-style pipe tables from markdown/plain text."""

    def extract(
        self,
        content: bytes | str,
        *,
        filename: str,
        mime_type: str | None = None,
    ) -> TableExtractionResult:
        text = _decode_content(content)
        tables = _extract_markdown_tables(text, filename=filename)
        source_type = "markdown" if tables else _source_type_from_filename(filename)
        return TableExtractionResult(tables=tables, source_filename=filename, source_type=source_type)


@dataclass(slots=True)
class CompositeTableExtractor:
    """Try a sequence of extraction hooks and combine any detected tables."""

    extractors: tuple[TableExtractor, ...]

    @classmethod
    def default(cls) -> "CompositeTableExtractor":
        # PDF/DOCX/XLSX-specific hooks can be appended here later. The MVP keeps
        # the hook boundary explicit while avoiding new heavyweight dependencies.
        return cls((CsvTableExtractor(), MarkdownTableExtractor()))

    def extract(
        self,
        content: bytes | str,
        *,
        filename: str,
        mime_type: str | None = None,
    ) -> TableExtractionResult:
        tables: list[TableObject] = []
        source_type = _source_type_from_filename(filename)
        for extractor in self.extractors:
            result = extractor.extract(content, filename=filename, mime_type=mime_type)
            tables.extend(result.tables)
            if result.source_type:
                source_type = result.source_type
        # De-duplicate by stable table id in case multiple hooks identify the
        # same table representation.
        unique: dict[str, TableObject] = {}
        for table in tables:
            unique[str(table.table_id)] = table
        return TableExtractionResult(
            tables=list(unique.values()),
            source_filename=filename,
            source_type=source_type,
        )


def extract_tables_from_content(
    content: bytes | str,
    *,
    filename: str,
    mime_type: str | None = None,
    extractor: TableExtractor | None = None,
) -> TableExtractionResult:
    """Extract first-class table objects from supported content.

    Unsupported/no-table content returns an empty result rather than raising, so
    non-table ingestion can continue unchanged.
    """

    selected = extractor or CompositeTableExtractor.default()
    return selected.extract(content, filename=filename, mime_type=mime_type)


def _normalize_rows(rows: Iterable[Iterable[Any]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in rows or []:
        values = [str(cell).strip() for cell in row]
        if any(values):
            normalized.append(values)
    return normalized


def _decode_content(content: bytes | str) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content or "")


def _source_type_from_filename(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    return suffix or "unknown"


def _looks_like_csv(filename: str, mime_type: str | None) -> bool:
    lower = (filename or "").lower()
    mime = (mime_type or "").lower()
    return lower.endswith(".csv") or mime in {"text/csv", "application/csv"}


def _parse_csv_rows(text: str) -> list[list[str]]:
    try:
        rows = list(csv.reader(text.splitlines()))
    except csv.Error:
        return []
    return _normalize_rows(rows)


def _is_table_like(rows: Sequence[Sequence[str]]) -> bool:
    if len(rows) < 2:
        return False
    return max((len(row) for row in rows), default=0) >= 2


def _extract_markdown_tables(text: str, *, filename: str) -> list[TableObject]:
    lines = text.splitlines()
    tables: list[TableObject] = []
    idx = 0
    while idx < len(lines) - 1:
        current = lines[idx].strip()
        nxt = lines[idx + 1].strip()
        if _is_pipe_row(current) and _is_markdown_separator(nxt):
            block = [current]
            idx += 2
            while idx < len(lines) and _is_pipe_row(lines[idx].strip()):
                block.append(lines[idx].strip())
                idx += 1
            rows = [_split_pipe_row(line) for line in block]
            if _is_table_like(rows):
                tables.append(
                    TableObject(
                        rows=rows,
                        caption=Path(filename).stem or None,
                        raw_table="\n".join(block),
                        source_type="markdown",
                        source_filename=filename,
                    )
                )
            continue
        idx += 1
    return tables


def _is_pipe_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_markdown_separator(line: str) -> bool:
    if not _is_pipe_row(line):
        return False
    cells = _split_pipe_row(line)
    if not cells:
        return False
    return all(cell.replace(":", "").replace("-", "").strip() == "" and "-" in cell for cell in cells)


def _split_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _stable_table_id(
    rows: Sequence[Sequence[str]],
    *,
    source_type: str | None,
    source_filename: str | None,
    page_number: int | None,
    caption: str | None,
    section_key: str | None,
) -> str:
    seed = "\n".join(
        [
            source_type or "table",
            source_filename or "",
            str(page_number or ""),
            caption or "",
            section_key or "",
            repr([list(row) for row in rows]),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    prefix = source_type or "table"
    return f"{prefix}-{digest}"
