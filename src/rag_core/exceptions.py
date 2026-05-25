"""Core exception hierarchy for rag-core.

Base class ``RagCoreError`` is the root of all exceptions raised by this
package.  Specific subcategories follow for each subsystem.

In later phases, existing DominicBE exceptions such as
``EmbeddingProviderError`` and ``IngestionPipelineError`` will be
re-exported or subclassed from this hierarchy.
"""

from __future__ import annotations


class RagCoreError(RuntimeError):
    """Base exception for all rag-core errors."""


# ── Parsing / Text Extraction ──────────────────────────────────────────


class ParsingError(RagCoreError):
    """Raised when text parsing or file extraction fails.

    Attributes:
        source: Source identifier (e.g. filename, content type).
        category: Failure category string (e.g. 'unsupported_format', 'parse_error').
    """

    def __init__(
        self,
        message: str,
        *,
        source: str = "",
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.source = source
        self.category = category


# ── Chunking ────────────────────────────────────────────────────────────


class ChunkingError(RagCoreError):
    """Raised when document chunking fails.

    Attributes:
        pipeline: Short pipeline name.
        category: Failure category string (e.g. 'empty_text', 'chunk_error').
    """

    def __init__(
        self,
        message: str,
        *,
        pipeline: str = "",
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.pipeline = pipeline
        self.category = category


# ── Embedding ───────────────────────────────────────────────────────────


class EmbeddingError(RagCoreError):
    """Raised when an embedding operation fails.

    Attributes:
        provider: Short provider name.
        model: Model name.
        category: Failure category string (e.g. 'timeout', 'invalid_response').
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.category = category


# ── Vector Store ────────────────────────────────────────────────────────


class VectorStoreError(RagCoreError):
    """Raised when a vector store operation fails.

    Attributes:
        provider: Short provider name.
        category: Failure category string (e.g. 'connection', 'upsert').
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.category = category


# ── Retrieval ───────────────────────────────────────────────────────────


class RetrievalError(RagCoreError):
    """Raised when a retrieval operation fails.

    Attributes:
        category: Failure category string (e.g. 'scoring', 'rerank').
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.category = category


# ── Configuration ───────────────────────────────────────────────────────


class ConfigurationError(RagCoreError):
    """Raised when rag-core configuration is invalid or missing."""
