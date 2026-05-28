from rag_core.chunking.heading_detector import detect_headings, normalize_section_key


def test_vietnamese_heading_normalizes_to_stable_key():
    headings = detect_headings("Mở đầu\n\nBài thực hành số 4\n1. Nội dung bài một.")
    assert [heading.section_key for heading in headings] == ["bai-thuc-hanh-so-4"]
    assert headings[0].title == "Bài thực hành số 4"


def test_markdown_and_english_numbered_headings_are_detected():
    headings = detect_headings("# Introduction\nBody text.\n\n2. Methods\nDetails")
    assert [heading.section_key for heading in headings] == ["introduction", "2-methods"]
    assert headings[0].level == 1


def test_body_sentence_and_colon_line_are_not_heading():
    text = "This is a normal body sentence about results.\nNote: this line has a colon but is body text."
    assert detect_headings(text) == []


def test_normalize_section_key_strips_vietnamese_diacritics():
    assert normalize_section_key("Bài thực hành số 4") == "bai-thuc-hanh-so-4"

