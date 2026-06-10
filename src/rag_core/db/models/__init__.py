"""rag_core.db.models — All SQLAlchemy ORM models for the rag_core database.

Import this package (or individual sub-modules) to register all models
with Base.metadata. Required by Alembic env.py for autogenerate.

Usage in alembic/env.py:
    from rag_core.db.database import Base
    import rag_core.db.models  # noqa: F401 — registers all models

Individual imports:
    from rag_core.db.models.collections import RagCollection
    from rag_core.db.models.documents import RagDocument
    from rag_core.db.models.ingestion_jobs import RagIngestionJob
    from rag_core.db.models.chunks import RagChunk
    from rag_core.db.models.embedding_models import RagEmbeddingModel
    from rag_core.db.models.vector_mappings import RagVectorMapping
"""

from rag_core.db.models.collections import RagCollection
from rag_core.db.models.documents import RagDocument
from rag_core.db.models.ingestion_jobs import RagIngestionJob
from rag_core.db.models.chunks import RagChunk
from rag_core.db.models.embedding_models import RagEmbeddingModel
from rag_core.db.models.vector_mappings import RagVectorMapping

__all__ = [
    "RagCollection",
    "RagDocument",
    "RagIngestionJob",
    "RagChunk",
    "RagEmbeddingModel",
    "RagVectorMapping",
]
