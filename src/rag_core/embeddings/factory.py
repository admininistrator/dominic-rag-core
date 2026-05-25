"""Embedding provider factory.

Centralizes provider selection so that knowledge_service and retrieval_service
never hard-code provider classes or model names.

Usage::

    from rag_core.embeddings.factory import get_embedding_provider

    provider = get_embedding_provider()
    result = provider.embed_texts(["hello world"])

The factory accepts explicit configuration parameters instead of reading
settings directly.  This makes it testable without environment variables.

Extracted from DominicBE's ``app/services/embeddings/factory.py``.
- ``settings`` import replaced with explicit parameters.
- Lazy imports preserved for all provider implementations.

No dependency on CRUD, endpoints, vector_store, chat, or LlamaIndex.
"""
from __future__ import annotations

import json
import logging

from rag_core.embeddings.base import EmbeddingProvider, EmbeddingProviderError

logger = logging.getLogger(__name__)


def get_embedding_provider(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    batch_size: int | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
    api_version: str | None = None,
    custom_headers: dict | None = None,
    # Fallback defaults (used when explicit params are None)
    default_provider: str = "local",
    default_model: str = "local-hash-v1",
    default_dimensions: int = 64,
    default_base_url: str = "http://localhost:11434",
    default_timeout_seconds: float = 60.0,
    default_batch_size: int = 16,
    default_api_key: str = "",
    default_api_type: str = "",
    default_api_version: str = "",
    default_custom_headers: dict | None = None,
) -> EmbeddingProvider:
    """Return the configured embedding provider instance.

    All primary parameters are optional and fall back to the corresponding
    ``default_*`` parameter.  This allows callers to override specific
    values while keeping sensible defaults.

    Args:
        provider_name: Override default_provider (e.g. ``'local'``,
            ``'ollama'``, ``'api'``).
        model: Override default_model.
        dimensions: Override default_dimensions.
        base_url: Override default_base_url.
        timeout_seconds: Override default_timeout_seconds.
        batch_size: Override default_batch_size.
        api_key: Override default_api_key (used when provider is ``'api'``).
        api_type: Override default_api_type (used when provider is ``'api'``).
        api_version: Override default_api_version (used when provider is ``'api'``).
        custom_headers: Override default_custom_headers parsed as dict (used when
            provider is ``'api'``).
        default_provider: Fallback provider type (default: ``"local"``).
        default_model: Fallback model name (default: ``"local-hash-v1"``).
        default_dimensions: Fallback embedding dimensions (default: ``64``).
        default_base_url: Fallback base URL (default: ``"http://localhost:11434"``).
        default_timeout_seconds: Fallback timeout (default: ``60.0``).
        default_batch_size: Fallback batch size (default: ``16``).
        default_api_key: Fallback API key (default: ``""``).
        default_api_type: Fallback API type (default: ``""``).
        default_api_version: Fallback API version (default: ``""``).
        default_custom_headers: Fallback custom headers (default: ``None``).

    Returns:
        An EmbeddingProvider instance ready to call ``embed_texts`` / ``embed_query``.

    Raises:
        EmbeddingProviderError: If provider_name is not a known provider
            (``'local'``, ``'ollama'``, ``'api'``).
    """
    resolved_provider = (provider_name or default_provider or "local").strip().lower()
    resolved_model = (model or default_model or "local-hash-v1").strip()
    resolved_dimensions = dimensions if dimensions is not None else default_dimensions

    if resolved_provider == "local":
        from rag_core.embeddings.local_hash_provider import LocalHashProvider

        return LocalHashProvider(
            model=resolved_model,
            dimensions=resolved_dimensions,
        )

    if resolved_provider == "ollama":
        from rag_core.embeddings.ollama_provider import OllamaProvider

        resolved_base_url = (base_url or default_base_url or "http://localhost:11434").strip()
        resolved_timeout = timeout_seconds if timeout_seconds is not None else default_timeout_seconds
        resolved_batch = batch_size if batch_size is not None else default_batch_size

        return OllamaProvider(
            model=resolved_model,
            base_url=resolved_base_url,
            timeout_seconds=resolved_timeout,
            batch_size=resolved_batch,
            expected_dimensions=resolved_dimensions,
        )

    if resolved_provider == "api":
        from rag_core.embeddings.generic_api_provider import GenericAPIProvider

        resolved_api_key = api_key if api_key is not None else default_api_key
        resolved_api_type = api_type if api_type is not None else default_api_type
        resolved_api_version = api_version if api_version is not None else default_api_version
        resolved_headers = custom_headers if custom_headers is not None else (default_custom_headers or {})
        resolved_base_url = (base_url or default_base_url or "").strip()
        resolved_timeout = timeout_seconds if timeout_seconds is not None else default_timeout_seconds
        resolved_batch = batch_size if batch_size is not None else default_batch_size

        return GenericAPIProvider(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_type=resolved_api_type,
            timeout_seconds=resolved_timeout,
            batch_size=resolved_batch,
            expected_dimensions=resolved_dimensions,
            api_version=resolved_api_version,
            custom_headers=resolved_headers,
        )

    raise EmbeddingProviderError(
        (
            f"Unknown EMBEDDING_PROVIDER={resolved_provider!r}. "
            "Supported values: 'local', 'ollama', 'api'."
        ),
        provider=resolved_provider,
        model=resolved_model,
        category="configuration_error",
    )
