"""Safe bootstrap for rag-core-owned object storage.

Creates the configured ``RAG_CORE_OBJECT_STORAGE_BUCKET`` (default:
``rag-source-files``) if it does not already exist. This script never deletes
objects, buckets, or prefixes and must not be pointed at DominicBE's
``dominic-knowledge`` ownership namespace.
"""

from __future__ import annotations

import json

from rag_core.api.config import get_service_settings
from rag_core.services.object_storage import bootstrap_object_storage


def main() -> int:
    settings = get_service_settings()
    result = bootstrap_object_storage(settings)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
