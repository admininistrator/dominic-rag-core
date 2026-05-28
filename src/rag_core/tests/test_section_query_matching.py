from rag_core.retrieval.section_retrieval import detect_context_expansion_intent, detect_section_query_intent, format_section_context, match_section


def test_vietnamese_section_count_summary_query_matches_high_confidence():
    query = "Bài thực hành số 4 có mấy bài, tóm tắt từng bài"
    sections = [{"section_key": "bai-thuc-hanh-so-4", "section_title": "Bài thực hành số 4"}]
    match = match_section(query, sections)
    assert match.section_key == "bai-thuc-hanh-so-4"
    assert match.confidence >= 0.82
    assert match.intent.wants_count is True
    assert match.intent.wants_summary is True


def test_low_confidence_query_falls_back():
    intent = detect_section_query_intent("Nội dung này nói gì?")
    assert intent.is_section_query is False
    assert match_section("Nội dung này nói gì?", []).reason == "no_section_intent"


def test_format_section_context_preserves_order():
    context = format_section_context([
        {"section_title": "Bài thực hành số 4", "page_number": 1, "content": "1. Nội dung bài một."},
        {"section_title": "Bài thực hành số 4", "page_number": 1, "content": "2. Nội dung bài hai."},
    ])
    assert context.index("bài một") < context.index("bài hai")


def test_optional_heading_matching_handles_near_exact_heading_query():
    match = match_section(
        "Tóm tắt bài thực hành 4",
        [{"section_key": "bai-thuc-hanh-so-4", "section_title": "Bài thực hành số 4"}],
    )
    assert match.section_key == "bai-thuc-hanh-so-4"
    assert match.confidence >= 0.82


def test_context_expansion_intent_detects_vietnamese_count_summary():
    assert detect_context_expansion_intent("có mấy mục, tóm tắt từng mục") is True
    assert detect_context_expansion_intent("định nghĩa học phần") is False

