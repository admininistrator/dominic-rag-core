from rag_core.chunking.custom_pipeline import CustomPipeline


def test_chunks_under_vietnamese_heading_receive_section_metadata():
    chunks = CustomPipeline().chunk_document(
        "Bài thực hành số 4\n1. Nội dung bài một.\n2. Nội dung bài hai.\n3. Nội dung bài ba."
    )
    assert chunks
    metadata = chunks[0].extra_metadata
    assert metadata["section_key"] == "bai-thuc-hanh-so-4"
    assert chunks[0].section_title == "Bài thực hành số 4"
    assert metadata["section_level"] == 2
    assert metadata["section_order"] == 0


def test_chunks_before_first_heading_have_no_false_section():
    chunks = CustomPipeline().chunk_document("Intro body sentence.\n\nBài thực hành số 4\nDetails here.")
    assert chunks
    first_meta = chunks[0].extra_metadata
    if first_meta.get("char_start", 0) < chunks[0].content.find("Bài thực hành số 4"):
        assert "section_key" not in first_meta

