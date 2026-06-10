"""Collection name suggestion and validation utilities.

RCSI-P3-T01/T05 naming convention::

    rag_{tenant}_{provider}_{model}

The generated name is slug-safe, deterministic, and capped at Qdrant's 63
character collection-name limit. If the conventional name would exceed the
limit, it is truncated with an 8-character SHA-256 suffix so long names remain
stable and low-collision.

Backward compatibility is intentionally preserved for the legacy DominicBE
collection names:

* ``knowledge_chunks``
* ``knowledge_{provider}_{model}``

Those names are accepted by validation as legacy names, but new suggestions use
the rag-core-owned ``rag_*`` convention.
"""
from __future__ import annotations

import hashlib
import re

# Qdrant maximum collection name length used by the Phase 2 data model.
QDRANT_MAX_COLLECTION_LENGTH = 63

# Known legacy collection names.
LEGACY_DEFAULT_COLLECTION = "knowledge_chunks"
LEGACY_PREFIX = "knowledge_"

_SAFE_CHARS_RE = re.compile(r"[^a-z0-9]+")
_UNDERSCORE_RE = re.compile(r"_+")


def _slug(value: str, *, fallback: str) -> str:
    """Return a lower-case Qdrant-safe slug.

    The project convention intentionally uses underscores only. Qdrant supports
    broader names, but keeping one safe character set avoids tenant/provider/model
    punctuation leaking into collection identifiers.
    """
    raw = str(value or "").strip().lower()
    sanitized = _SAFE_CHARS_RE.sub("_", raw)
    sanitized = _UNDERSCORE_RE.sub("_", sanitized).strip("_")
    return sanitized or fallback


def _sanitize_model_name(model: str) -> str:
    """Sanitize a model name for use in a collection name.

    Kept for backward compatibility with older imports/tests; new code should use
    ``suggest_collection_name()`` so tenant/provider/model are handled together.
    """
    return _slug(model, fallback="model")


def _legacy_provider_model_name(provider: str, model: str) -> str:
    provider_slug = _slug(provider, fallback="provider")
    model_slug = _slug(model, fallback="model")
    raw = f"{LEGACY_PREFIX}{provider_slug}_{model_slug}"
    return _truncate_with_hash(raw)


def _truncate_with_hash(raw_name: str) -> str:
    if len(raw_name) <= QDRANT_MAX_COLLECTION_LENGTH:
        return raw_name
    digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:8]
    suffix = f"_{digest}"
    prefix_len = QDRANT_MAX_COLLECTION_LENGTH - len(suffix)
    return f"{raw_name[:prefix_len].rstrip('_')}{suffix}"


def suggest_collection_name(
    provider: str,
    model: str,
    tenant_id: str = "default",
    *,
    legacy: bool = False,
) -> str:
    """Generate a deterministic Qdrant collection name.

    Args:
        provider: Provider identifier, e.g. ``local``, ``ollama``, ``api``.
        model: Model name, e.g. ``local-hash-v1`` or ``text-embedding-3-small``.
        tenant_id: Tenant/workspace scope. Defaults to ``default`` for local/dev
            and for legacy two-argument callers.
        legacy: When true, generate the historical ``knowledge_*`` provider/model
            name. This is for compatibility checks only; new collections should
            leave this false.

    Returns:
        A slug-safe name <= 63 chars. New names follow
        ``rag_{tenant}_{provider}_{model}``.
    """
    if legacy:
        return _legacy_provider_model_name(provider, model)

    tenant_slug = _slug(tenant_id, fallback="default")
    provider_slug = _slug(provider, fallback="provider")
    model_slug = _slug(model, fallback="model")
    raw = f"rag_{tenant_slug}_{provider_slug}_{model_slug}"
    return _truncate_with_hash(raw)


def is_legacy_collection_name(collection: str) -> bool:
    """Return true when *collection* is a known/historical DominicBE name."""
    value = str(collection or "").strip()
    return value == LEGACY_DEFAULT_COLLECTION or value.startswith(LEGACY_PREFIX)


def validate_collection_config(
    provider: str,
    model: str,
    collection: str,
    tenant_id: str = "default",
) -> list[str]:
    """Validate that a collection name matches the rag-core convention.

    Returns warning strings rather than raising so callers can preserve old
    deployments while nudging them to the new ``rag_*`` convention.
    """
    warnings: list[str] = []
    expected = suggest_collection_name(provider, model, tenant_id=tenant_id)
    legacy_expected = suggest_collection_name(provider, model, tenant_id=tenant_id, legacy=True)

    if collection == expected:
        return warnings

    if collection == LEGACY_DEFAULT_COLLECTION:
        if provider != "local":
            warnings.append(
                f"Collection {collection!r} is the legacy default collection and may cause "
                f"dimension conflicts for provider={provider!r}. Use rag-core collection "
                f"{expected!r} instead."
            )
        else:
            warnings.append(
                f"Collection {collection!r} is a legacy collection name. New rag-core "
                f"collections should use {expected!r}."
            )
        return warnings

    if collection == legacy_expected or is_legacy_collection_name(collection):
        warnings.append(
            f"Collection {collection!r} uses a legacy knowledge_* naming convention. "
            f"New rag-core collections should use {expected!r}."
        )
        return warnings

    warnings.append(
        f"Collection {collection!r} does not match the rag-core naming convention "
        f"for tenant={tenant_id!r} provider={provider!r} model={model!r}. "
        f"Expected: {expected!r}."
    )
    return warnings
