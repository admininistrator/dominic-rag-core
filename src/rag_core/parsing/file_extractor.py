"""File text extraction utilities.

Extracted from DominicBE's ``app/services/knowledge_service.py``
(``extract_text_from_file()`` and per-format extractors, lines 256-555).

The image captioning callback is injectable (``caption_fn`` parameter)
rather than hardcoded to ``app.services.llm_provider``.

Behavior preservation: text extraction output must be identical for all
supported formats when captioning is disabled.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_text_from_file(
    content: bytes,
    filename: str,
    mime_type: str | None = None,
    *,
    caption_fn: Callable[[bytes, str, str], str] | None = None,
    image_captioning_enabled: bool = False,
) -> str:
    """Extract plain text from uploaded file.

    Supported formats:
    - txt, md, csv, log, json, py, js, yaml/yml : decode as UTF-8
    - pdf  : PyMuPDF  — text blocks + embedded table heuristics
    - docx : python-docx – paragraphs + tables + text-boxes
    - pptx : python-pptx – all slide shapes, speaker notes, tables
    - xlsx : openpyxl   – all sheets, all non-empty cells (header|value rows)

    Args:
        content: Raw file bytes.
        filename: Original filename (used for format detection).
        mime_type: Optional MIME type (currently unused, reserved).
        caption_fn: Optional callable for image captioning.
            Signature: ``caption_fn(image_bytes, media_type, context_hint) -> str``.
        image_captioning_enabled: Whether to invoke the caption callback.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: For unsupported file types or missing optional dependencies.
    """
    lower = filename.lower()

    # ── Plain text formats ──────────────────────────────────────────────
    if lower.endswith((".txt", ".md", ".csv", ".log", ".json", ".py", ".js", ".yaml", ".yml")):
        return content.decode("utf-8", errors="replace")

    # ── PDF ────────────────────────────────────────────────────────────
    if lower.endswith(".pdf"):
        return _extract_pdf(content, caption_fn=caption_fn, image_captioning_enabled=image_captioning_enabled)

    # ── DOCX ────────────────────────────────────────────────────────────
    if lower.endswith(".docx"):
        return _extract_docx(content, caption_fn=caption_fn, image_captioning_enabled=image_captioning_enabled)

    # ── PPTX ────────────────────────────────────────────────────────────
    if lower.endswith(".pptx"):
        return _extract_pptx(content, caption_fn=caption_fn, image_captioning_enabled=image_captioning_enabled)

    # ── XLSX / XLS ──────────────────────────────────────────────────────
    if lower.endswith((".xlsx", ".xls")):
        return _extract_xlsx(content)

    raise ValueError(f"Unsupported file type: {filename}")


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------


def _extract_pdf(
    content: bytes,
    *,
    caption_fn: Callable[[bytes, str, str], str] | None = None,
    image_captioning_enabled: bool = False,
) -> str:
    """Extract text from PDF using PyMuPDF.

    Uses ``get_text("blocks")`` to preserve reading order (sorted by y, x).
    When image captioning is enabled, embedded images are extracted per page
    and captions injected as ``[Image N: <description>]`` markers.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ValueError("PDF support requires PyMuPDF. Install with: pip install PyMuPDF")

    doc = fitz.open(stream=content, filetype="pdf")
    page_texts: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        blocks_sorted = sorted(
            [b for b in blocks if b[6] == 0 and b[4].strip()],
            key=lambda b: (round(b[1] / 10) * 10, b[0]),
        )
        parts = [f"[Page {page_num}]"] + [b[4].strip() for b in blocks_sorted]

        if image_captioning_enabled and caption_fn:
            for img_idx, img_info in enumerate(page.get_images(full=True), start=1):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    ext = (base_image.get("ext") or "jpeg").lower().replace("jpg", "jpeg")
                    mt = f"image/{ext}" if ext in ("jpeg", "png", "gif", "webp") else "image/jpeg"
                    caption = caption_fn(img_bytes, media_type=mt, context_hint=f"PDF page {page_num}")
                    if caption:
                        parts.append(f"[Image {img_idx}: {caption}]")
                except Exception as exc:
                    logger.debug("PDF image extraction skipped xref=%s: %s", xref, exc)

        page_texts.append("\n".join(parts))

    doc.close()
    return "\n\n".join(page_texts)


def _extract_docx(
    content: bytes,
    *,
    caption_fn: Callable[[bytes, str, str], str] | None = None,
    image_captioning_enabled: bool = False,
) -> str:
    """Extract text from DOCX using python-docx with full XML traversal."""
    try:
        import io
        from docx import Document
    except ImportError:
        raise ValueError("DOCX support requires python-docx. Install with: pip install python-docx")

    WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def _para_full_text(para) -> str:
        texts = []
        for node in para._element.iter(f"{{{WORD_NS}}}t"):
            t = node.text or ""
            if t:
                texts.append(t)
        return "".join(texts)

    def _extract_table(table) -> list[str]:
        rows: list[str] = []
        for row in table.rows:
            row_parts: list[str] = []
            seen: set[str] = set()
            for cell in row.cells:
                cell_texts = [_para_full_text(p).strip() for p in cell.paragraphs]
                cell_text = "\n".join(t for t in cell_texts if t).strip()
                if cell_text and cell_text not in seen:
                    row_parts.append(cell_text)
                    seen.add(cell_text)
                for nested in cell.tables:
                    rows.extend(_extract_table(nested))
            if row_parts:
                rows.append(" | ".join(row_parts))
        return rows

    doc = Document(io.BytesIO(content))
    parts: list[str] = []

    for para in doc.paragraphs:
        text = _para_full_text(para).strip()
        if text:
            parts.append(text)

    for table in doc.tables:
        parts.extend(_extract_table(table))

    if image_captioning_enabled and caption_fn:
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    img_bytes = rel.target_part.blob
                    mt = rel.target_part.content_type or "image/jpeg"
                    caption = caption_fn(img_bytes, media_type=mt, context_hint="DOCX document")
                    if caption:
                        parts.append(f"[Image: {caption}]")
                except Exception as exc:
                    logger.debug("DOCX image extraction skipped: %s", exc)

    return "\n".join(parts)


def _extract_pptx(
    content: bytes,
    *,
    caption_fn: Callable[[bytes, str, str], str] | None = None,
    image_captioning_enabled: bool = False,
) -> str:
    """Extract text from PPTX using python-pptx."""
    try:
        import io
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        raise ValueError("PPTX support requires python-pptx. Install with: pip install python-pptx")

    prs = Presentation(io.BytesIO(content))
    slide_texts: list[str] = []

    def _shape_texts(shape) -> list[str]:
        results: list[str] = []
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for child in shape.shapes:
                results.extend(_shape_texts(child))
            return results
        if shape.has_table:
            table = shape.table
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    results.append(" | ".join(row_cells))
            return results
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                text = "".join(run.text for run in para.runs).strip()
                if text:
                    results.append(text)
        return results

    for slide_num, slide in enumerate(prs.slides, start=1):
        parts: list[str] = []

        title_text = ""
        if slide.shapes.title and slide.shapes.title.has_text_frame:
            title_text = slide.shapes.title.text.strip()

        header = f"[Slide {slide_num}]"
        if title_text:
            header += f" {title_text}"
        parts.append(header)

        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            parts.extend(_shape_texts(shape))

        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                parts.append(f"[Notes] {notes_text}")

        if image_captioning_enabled and caption_fn:
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        img_bytes = shape.image.blob
                        mt = shape.image.content_type or "image/jpeg"
                        caption = caption_fn(img_bytes, media_type=mt, context_hint=f"Slide {slide_num}: {title_text}")
                        if caption:
                            parts.append(f"[Image: {caption}]")
                    except Exception as e:
                        logger.debug("PPTX image extraction skipped slide=%s: %s", slide_num, e)

        slide_texts.append("\n".join(parts))

    return "\n\n".join(slide_texts)


def _extract_xlsx(content: bytes) -> str:
    """Extract text from XLSX/XLS using openpyxl."""
    try:
        import io
        import openpyxl
    except ImportError:
        raise ValueError("XLSX support requires openpyxl. Install with: pip install openpyxl")

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sheet_texts: list[str] = []

    for sheet in wb.worksheets:
        parts: list[str] = [f"[Sheet: {sheet.title}]"]
        rows_data: list[list[str]] = []

        for row in sheet.iter_rows(values_only=True):
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(cells):
                rows_data.append(cells)

        if not rows_data:
            continue

        first_row = rows_data[0]
        is_header_row = any(cell and not _is_numeric(cell) for cell in first_row)

        if is_header_row and len(rows_data) > 1:
            headers = first_row
            for row_cells in rows_data[1:]:
                pairs = []
                for h, v in zip(headers, row_cells):
                    if v:
                        label = h if h and not _is_numeric(h) else ""
                        pairs.append(f"{label}: {v}" if label else v)
                if pairs:
                    parts.append(" | ".join(pairs))
        else:
            for row_cells in rows_data:
                line = " | ".join(c for c in row_cells if c)
                if line:
                    parts.append(line)

        sheet_texts.append("\n".join(parts))

    wb.close()
    return "\n\n".join(sheet_texts)


def _is_numeric(value: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(value)
        return True
    except ValueError:
        return False
