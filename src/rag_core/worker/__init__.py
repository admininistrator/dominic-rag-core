"""rag-core Celery worker.

Queue namespace: rag.ingestion, rag.embedding, rag.reindex, rag.cleanup, rag.eval.
Redis isolation: broker DB 2, result backend DB 3 (separate from DominicBE DB 0/1).
Celery app name: rag_core.
"""

from rag_core.worker.celery_app import celery_app

__all__ = ["celery_app"]
