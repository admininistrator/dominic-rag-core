from rag_core.chunking.custom_pipeline import CustomPipeline
from rag_core.chunking.page_markers import page_metadata_for_span, strip_page_markers


def test_strip_exact_page_marker_lines_only():
    cleaned, markers = strip_page_markers("<<PAGE:1>>\nFirst page\nLiteral <<PAGE:9>> content")
    assert "<<PAGE:1>>" not in cleaned
    assert "Literal <<PAGE:9>> content" in cleaned
    assert markers[0].page_number == 1


def test_page_metadata_for_cross_page_span_has_range():
    cleaned, markers = strip_page_markers("<<PAGE:1>>\nFirst\n<<PAGE:2>>\nSecond")
    meta = page_metadata_for_span(markers, 0, len(cleaned))
    assert meta == {"page_number": 1, "page_range": [1, 2]}


def test_custom_pipeline_strips_sentinels_and_assigns_page_number():
    chunks = CustomPipeline().chunk_document("<<PAGE:1>>\nBài thực hành số 4\nNội dung bài một.")
    assert chunks
    assert "<<PAGE:" not in chunks[0].content
    assert chunks[0].page_number == 1

