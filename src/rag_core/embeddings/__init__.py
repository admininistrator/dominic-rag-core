"""Embedding provider protocol, implementations, and factory.

Provides:
- ``EmbeddingProvider`` protocol, result types, and metadata
- ``LocalHashProvider`` — deterministic local hash embedding
- ``OllamaProvider`` — Ollama HTTP embedding
- ``GenericAPIProvider`` — OpenAI-compatible API embedding
- Adapters for OpenAI, Cohere, Voyage, HuggingFace, Ollama formats
- API key security utilities (masking, validation, sanitization)
- ``get_embedding_provider()`` factory
- ``suggest_collection_name()`` / ``validate_collection_config()`` utilities

Extracted from DominicBE's ``app/services/embeddings/``.
"""
from rag_core.embeddings.base import (
    EmbeddingDimensionMismatchError,
    EmbeddingMeta,
    EmbeddingProvider,
    EmbeddingProviderCapabilities,
    EmbeddingProviderError,
    EmbedResult,
    QueryEmbedResult,
)
from rag_core.embeddings.factory import get_embedding_provider

__all__ = [
    # Protocol + types
    "EmbeddingProvider",
    "EmbeddingMeta",
    "EmbedResult",
    "QueryEmbedResult",
    "EmbeddingProviderCapabilities",
    # Exceptions
    "EmbeddingProviderError",
    "EmbeddingDimensionMismatchError",
    # Factory
    "get_embedding_provider",
]
