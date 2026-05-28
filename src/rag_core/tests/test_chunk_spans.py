from rag_core.chunking.span_mapper import map_chunks_monotonic
from rag_core.chunking.custom_pipeline import CustomPipeline


def test_repeated_chunk_text_maps_monotonically_to_later_occurrence():
    source = "Alpha repeat. Middle text. Alpha repeat."
    spans = map_chunks_monotonic(source, ["Alpha repeat.", "Alpha repeat."])
    assert spans[0].char_start == 0
    assert spans[1].char_start == source.rfind("Alpha repeat.")


def test_custom_pipeline_adds_char_offsets_when_possible():
    chunks = CustomPipeline().chunk_document("Bài thực hành số 4\nNội dung bài một. Nội dung bài hai.")
    assert chunks
    metadata = chunks[0].extra_metadata
    assert metadata["char_start"] == 0
    assert metadata["char_end"] > metadata["char_start"]

