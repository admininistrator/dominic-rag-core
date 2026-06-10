"""Celery application for rag-core worker.

App name: ``rag_core`` (must match RAG_CORE_CELERY_APP_NAME setting).
Redis scope: broker DB 2, result backend DB 3 — isolated from DominicBE (DB 0/1).
Queue namespace: ``rag.ingestion``, ``rag.embedding``, ``rag.reindex``,
``rag.cleanup``, ``rag.eval``.

Settings are read from env vars via :func:`rag_core.api.config.get_service_settings`
and applied at app creation time.  When ``RAG_CORE_CELERY_ENABLED`` is ``False``
the app is still importable (for eager/synchronous task invocation in tests)
but the worker process will not start.
"""

from __future__ import annotations

from celery import Celery
from kombu import Exchange, Queue

from rag_core.api.config import RagCoreServiceSettings, get_service_settings


def _build_queues(settings: RagCoreServiceSettings) -> list[Queue]:
    """Build the five rag.* queues with dead-letter exchange support."""
    default_exchange = Exchange("rag_core", type="direct", durable=True)

    return [
        Queue(
            settings.queue_ingestion,
            exchange=default_exchange,
            routing_key=settings.queue_ingestion,
            durable=True,
        ),
        Queue(
            settings.queue_embedding,
            exchange=default_exchange,
            routing_key=settings.queue_embedding,
            durable=True,
        ),
        Queue(
            settings.queue_reindex,
            exchange=default_exchange,
            routing_key=settings.queue_reindex,
            durable=True,
        ),
        Queue(
            settings.queue_cleanup,
            exchange=default_exchange,
            routing_key=settings.queue_cleanup,
            durable=True,
        ),
        Queue(
            settings.queue_eval,
            exchange=default_exchange,
            routing_key=settings.queue_eval,
            durable=True,
        ),
    ]


def create_celery_app() -> Celery:
    """Build and configure the rag_core Celery application.

    Returns:
        Configured Celery app instance, ready for worker or eager-mode use.
    """
    settings = get_service_settings()

    app = Celery(settings.celery_app_name)

    # ---- Redis isolation ------------------------------------------------------
    # Broker: Redis DB 2 (DominicBE uses DB 0)
    # Result backend: Redis DB 3 (DominicBE uses DB 1)
    app.conf.broker_url = settings.celery_broker_url
    app.conf.result_backend = settings.celery_result_backend

    # ---- Serialization --------------------------------------------------------
    app.conf.task_serializer = settings.celery_task_serializer
    app.conf.result_serializer = settings.celery_result_serializer
    app.conf.accept_content = ["json"]

    # ---- Task execution -------------------------------------------------------
    app.conf.task_acks_late = True
    app.conf.task_reject_on_worker_lost = True
    app.conf.task_track_started = True
    app.conf.task_soft_time_limit = settings.celery_task_soft_time_limit
    app.conf.task_time_limit = settings.celery_task_time_limit
    app.conf.worker_max_tasks_per_child = settings.celery_max_tasks_per_child

    # ---- Queues and routing ---------------------------------------------------
    app.conf.task_default_queue = settings.celery_default_queue
    app.conf.task_queues = _build_queues(settings)

    # Route tasks to their declared queues
    app.conf.task_routes = {
        "rag.ingest_document": {"queue": settings.queue_ingestion},
        "rag.parse_document": {"queue": settings.queue_ingestion},
        "rag.chunk_document": {"queue": settings.queue_ingestion},
        "rag.index_vectors": {"queue": settings.queue_ingestion},
        "rag.delete_document": {"queue": settings.queue_ingestion},
        "rag.embed_chunks": {"queue": settings.queue_embedding},
        "rag.reindex_document": {"queue": settings.queue_reindex},
        "rag.reindex_collection": {"queue": settings.queue_reindex},
        "rag.cleanup_orphaned_jobs": {"queue": settings.queue_cleanup},
        "rag.run_retrieval_eval": {"queue": settings.queue_eval},
    }

    # ---- Autodiscover tasks ---------------------------------------------------
    app.autodiscover_tasks(
        [
            "rag_core.worker.tasks.ingestion",
            "rag_core.worker.tasks.embedding",
            "rag_core.worker.tasks.cleanup",
            "rag_core.worker.tasks.reindex",
        ],
        related_name="tasks",
    )

    return app


# Singleton app instance — used by both the worker process and API/test code.
celery_app = create_celery_app()
