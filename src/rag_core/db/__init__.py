"""rag_core.db — Database engine, session, Base, and model imports.

This package provides the SQLAlchemy 2.x engine and session factory for the
rag_core PostgreSQL metadata database.

OWNERSHIP BOUNDARY:
  - rag-core owns the "rag_core" database exclusively.
  - DominicBE owns "chatbot_db" exclusively.
  - Do NOT import DominicBE models or sessions here.
  - Do NOT import these from DominicBE.

Public API (importable from rag_core.db):
  - engine         — SQLAlchemy Engine bound to rag_core DB
  - SessionLocal   — sessionmaker factory for rag_core DB sessions
  - Base           — DeclarativeBase for all rag_core models
  - get_db()       — FastAPI dependency that yields a DB session
  - check_db_health() — health-check helper (no exception on failure)

Model sub-packages (import models to register them with Base.metadata):
  - rag_core.db.models.collections     -> RagCollection
  - rag_core.db.models.documents       -> RagDocument
  - rag_core.db.models.ingestion_jobs  -> RagIngestionJob
  - rag_core.db.models.chunks          -> RagChunk
  - rag_core.db.models.embedding_models -> RagEmbeddingModel
  - rag_core.db.models.vector_mappings -> RagVectorMapping
"""

from rag_core.db.database import (
    Base,
    SessionLocal,
    check_db_health,
    engine,
    get_db,
)
from rag_core.db.models import (
    RagChunk,
    RagCollection,
    RagDocument,
    RagEmbeddingModel,
    RagIngestionJob,
    RagVectorMapping,
)

__all__ = [
    "Base",
    "SessionLocal",
    "check_db_health",
    "engine",
    "get_db",
    "RagChunk",
    "RagCollection",
    "RagDocument",
    "RagEmbeddingModel",
    "RagIngestionJob",
    "RagVectorMapping",
]
