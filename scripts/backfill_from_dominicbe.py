#!/usr/bin/env python
"""Phase 9 safe DominicBE -> rag-core migration/backfill entrypoint.

Dry-run is the default. Use --apply for guarded, non-destructive upserts only.
"""

from rag_core.migrations.dominicbe_backfill import main


if __name__ == "__main__":
    raise SystemExit(main())
