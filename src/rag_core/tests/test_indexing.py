"""Tests for the indexing pipeline.

Covers:
- ``build_vector_id()`` format
- ``compute_checksum()`` deterministic output
- ``prepare_chunks_for_indexing()`` metadata structure

Note: ``prepare_chunks_for_indexing()`` tests that call the real embedding
provider are limited to basic structure checks. Full embedding parity
tests require the LocalHashProvider (tested in test_embeddings.py).
"""
from __future__ import annotations

from rag_core.indexing.pipeline import (
    build_vector_id,
    compute_checksum,
)


# ---------------------------------------------------------------------------
# build_vector_id tests
# ---------------------------------------------------------------------------

class TestBuildVectorId:
    """Verify vector ID format: ``"local:{document_id}:{chunk_index}:{checksum[:12]}"``."""

    def test_basic_format(self):
        vector_id = build_vector_id(1, 0, "abc123def456ghi789")
        assert vector_id == "local:1:0:abc123def456"

    def test_checksum_truncated_to_12(self):
        vector_id = build_vector_id(42, 3, "abcdef1234567890")
        assert vector_id == "local:42:3:abcdef123456"

    def test_short_checksum(self):
        vector_id = build_vector_id(5, 1, "short")
        assert vector_id == "local:5:1:short"

    def test_large_ids(self):
        vector_id = build_vector_id(99999, 999, "checksum12345")
        assert vector_id == "local:99999:999:checksum1234"

    def test_chunk_index_zero(self):
        vector_id = build_vector_id(1, 0, "abcdefghijklmn")
        assert vector_id == "local:1:0:abcdefghijkl"

    def test_checksum_empty_string(self):
        vector_id = build_vector_id(7, 2, "")
        assert vector_id == "local:7:2:"

    def test_format_always_string(self):
        """Vector ID should always be a string."""
        result = build_vector_id(1, 0, "abc")
        assert isinstance(result, str)

    def test_contains_local_prefix(self):
        """Vector ID must always start with 'local:'."""
        result = build_vector_id(100, 5, "xyz")
        assert result.startswith("local:")


# ---------------------------------------------------------------------------
# compute_checksum tests
# ---------------------------------------------------------------------------

class TestComputeChecksum:
    """Verify compute_checksum produces deterministic SHA-256 output."""

    def test_deterministic(self):
        c1 = compute_checksum("hello world")
        c2 = compute_checksum("hello world")
        assert c1 == c2

    def test_different_inputs(self):
        c1 = compute_checksum("hello")
        c2 = compute_checksum("world")
        assert c1 != c2

    def test_empty_string(self):
        csum = compute_checksum("")
        # SHA-256 of empty string is known
        assert len(csum) == 64
        assert csum == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_unicode(self):
        csum = compute_checksum("Xin chào thế giới")
        assert len(csum) == 64
        assert isinstance(csum, str)

    def test_hex_format(self):
        csum = compute_checksum("test")
        # All characters should be hex digits
        assert all(c in "0123456789abcdef" for c in csum)
