"""initial rag-core metadata schema

Revision ID: 20260605_0001
Revises:
Create Date: 2026-06-05
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260605_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=63), nullable=False, comment="Qdrant collection name (max 63 chars). Convention: rag_{tenant}_{provider}_{model}"),
        sa.Column("display_name", sa.String(length=255), nullable=True, comment="Human-readable label"),
        sa.Column("tenant_id", sa.String(length=128), nullable=False, comment="Tenant/workspace scope"),
        sa.Column("embedding_provider", sa.String(length=64), nullable=False, comment="e.g. ollama, api, local"),
        sa.Column("embedding_model", sa.String(length=128), nullable=False, comment="e.g. nomic-embed-text, local-hash-v1"),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False, comment="e.g. 768, 1536, 64"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active", comment="active | archived | deleted"),
        sa.Column("document_count", sa.Integer(), nullable=False, server_default="0", comment="Denormalized document count; updated by worker"),
        sa.Column("vector_count", sa.Integer(), nullable=False, server_default="0", comment="Denormalized vector count; updated by worker"),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment="Provider-specific or consumer-provided extra metadata"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_rag_collections"),
        sa.UniqueConstraint("name", name="uq_rag_collections_name"),
    )
    op.create_index("ix_rag_collections_tenant_id", "rag_collections", ["tenant_id"], unique=False)
    op.create_index("ix_rag_collections_status", "rag_collections", ["status"], unique=False)

    op.create_table(
        "rag_embedding_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False, comment="Embedding provider, e.g. local, ollama, api"),
        sa.Column("model_name", sa.String(length=128), nullable=False, comment="Provider-specific model identifier"),
        sa.Column("dimensions", sa.Integer(), nullable=False, comment="Vector dimension count"),
        sa.Column("version", sa.String(length=32), nullable=True, comment="Optional model version tag"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true(), comment="Whether this model is available for new collections"),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment="Provider-specific metadata/configuration"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_rag_embedding_models"),
        sa.UniqueConstraint("provider", "model_name", "dimensions", name="uq_rag_embedding_models_provider_model_dimensions"),
    )
    op.create_index("ix_rag_embedding_models_is_active", "rag_embedding_models", ["is_active"], unique=False)

    op.create_table(
        "rag_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False, comment="Which collection owns this document"),
        sa.Column("tenant_id", sa.String(length=128), nullable=False, comment="Denormalized from collection for fast tenant-scoped queries"),
        sa.Column("external_id", sa.String(length=255), nullable=True, comment="Consumer document ID (e.g. DominicBE knowledge_items.id)"),
        sa.Column("title", sa.String(length=512), nullable=True, comment="Document title"),
        sa.Column("source_type", sa.String(length=32), nullable=False, comment="upload | text | url"),
        sa.Column("source_uri", sa.String(length=2048), nullable=True, comment="Object storage URI or remote URL"),
        sa.Column("mime_type", sa.String(length=128), nullable=True, comment="MIME type of source content"),
        sa.Column("checksum", sa.String(length=128), nullable=True, comment="SHA-256 of source content (for deduplication)"),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True, comment="Original file size in bytes"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending", comment="pending | processing | indexed | failed | deleted"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0", comment="Denormalized chunk count; updated by worker on completion"),
        sa.Column("owner_username", sa.String(length=128), nullable=True, comment="Legacy owner field for Qdrant payload filtering"),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment="Consumer-provided metadata (tags, labels, etc.)"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True, comment="Soft-delete timestamp; hard-purge after retention window"),
        sa.ForeignKeyConstraint(["collection_id"], ["rag_collections.id"], name="fk_rag_documents_collection_id_rag_collections", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_rag_documents"),
    )
    op.create_index("ix_rag_documents_collection_id", "rag_documents", ["collection_id"], unique=False)
    op.create_index("ix_rag_documents_tenant_id", "rag_documents", ["tenant_id"], unique=False)
    op.create_index("ix_rag_documents_external_id", "rag_documents", ["external_id"], unique=False)
    op.create_index("ix_rag_documents_status", "rag_documents", ["status"], unique=False)
    op.create_index("ix_rag_documents_checksum", "rag_documents", ["checksum"], unique=False)
    op.create_index("ix_rag_documents_created_at", "rag_documents", ["created_at"], unique=False)

    op.create_table(
        "rag_ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False, comment="Document being ingested"),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False, comment="Denormalized collection FK for fast queue queries"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued", comment="queued | processing | embedding | indexing | completed | failed | cancelled"),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True, comment="Celery async result ID for status polling"),
        sa.Column("step_current", sa.String(length=64), nullable=True, comment="Current pipeline step name"),
        sa.Column("step_progress", sa.Float(), nullable=True, comment="Progress within current step: 0.0 - 1.0"),
        sa.Column("chunks_total", sa.Integer(), nullable=True, comment="Total chunks to process"),
        sa.Column("chunks_processed", sa.Integer(), nullable=False, server_default="0", comment="Chunks completed so far"),
        sa.Column("error_message", sa.Text(), nullable=True, comment="Human-readable error description on failure"),
        sa.Column("error_code", sa.String(length=64), nullable=True, comment="Structured error code for programmatic handling"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0", comment="Number of retries attempted"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3", comment="Maximum allowed retries before marking failed"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True, comment="When processing actually began"),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True, comment="When the job reached a terminal state"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["rag_collections.id"], name="fk_rag_ingestion_jobs_collection_id_rag_collections", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["rag_documents.id"], name="fk_rag_ingestion_jobs_document_id_rag_documents", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_rag_ingestion_jobs"),
    )
    op.create_index("ix_rag_ingestion_jobs_document_id", "rag_ingestion_jobs", ["document_id"], unique=False)
    op.create_index("ix_rag_ingestion_jobs_collection_id", "rag_ingestion_jobs", ["collection_id"], unique=False)
    op.create_index("ix_rag_ingestion_jobs_status", "rag_ingestion_jobs", ["status"], unique=False)
    op.create_index("ix_rag_ingestion_jobs_celery_task_id", "rag_ingestion_jobs", ["celery_task_id"], unique=False)
    op.create_index("ix_rag_ingestion_jobs_created_at", "rag_ingestion_jobs", ["created_at"], unique=False)

    op.create_table(
        "rag_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False, comment="Parent rag_documents row"),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False, comment="Denormalized collection FK for management queries"),
        sa.Column("chunk_index", sa.Integer(), nullable=False, comment="Zero-based order of this chunk within the document"),
        sa.Column("token_count", sa.Integer(), nullable=True, comment="Approximate token count for the chunk"),
        sa.Column("vector_id", sa.String(length=255), nullable=True, comment="Qdrant point ID when indexed"),
        sa.Column("embedding_model", sa.String(length=128), nullable=True, comment="Embedding model identifier used for this chunk"),
        sa.Column("content_hash", sa.String(length=128), nullable=True, comment="SHA-256 hash of chunk content for idempotency/change detection"),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment="Section/page/span metadata; never raw chunk content"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["rag_collections.id"], name="fk_rag_chunks_collection_id_rag_collections", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["rag_documents.id"], name="fk_rag_chunks_document_id_rag_documents", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_rag_chunks"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_rag_chunks_document_chunk_index"),
    )
    op.create_index("ix_rag_chunks_document_id", "rag_chunks", ["document_id"], unique=False)
    op.create_index("ix_rag_chunks_collection_id", "rag_chunks", ["collection_id"], unique=False)
    op.create_index("ix_rag_chunks_vector_id", "rag_chunks", ["vector_id"], unique=False)

    op.create_table(
        "rag_vector_mappings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False, comment="rag_chunks row mapped to the vector point"),
        sa.Column("collection_name", sa.String(length=63), nullable=False, comment="Qdrant collection name"),
        sa.Column("point_id", sa.String(length=255), nullable=False, comment="Qdrant point UUID/ID"),
        sa.Column("embedding_model_id", postgresql.UUID(as_uuid=True), nullable=True, comment="Embedding model registry row used for this vector"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active", comment="active | deleted | stale"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["rag_chunks.id"], name="fk_rag_vector_mappings_chunk_id_rag_chunks", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["embedding_model_id"], ["rag_embedding_models.id"], name="fk_rag_vector_mappings_embedding_model_id_rag_embedding_models", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_rag_vector_mappings"),
        sa.UniqueConstraint("collection_name", "point_id", name="uq_rag_vector_mappings_collection_point"),
    )
    op.create_index("ix_rag_vector_mappings_chunk_id", "rag_vector_mappings", ["chunk_id"], unique=False)
    op.create_index("ix_rag_vector_mappings_status", "rag_vector_mappings", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_rag_vector_mappings_status", table_name="rag_vector_mappings")
    op.drop_index("ix_rag_vector_mappings_chunk_id", table_name="rag_vector_mappings")
    op.drop_table("rag_vector_mappings")

    op.drop_index("ix_rag_chunks_vector_id", table_name="rag_chunks")
    op.drop_index("ix_rag_chunks_collection_id", table_name="rag_chunks")
    op.drop_index("ix_rag_chunks_document_id", table_name="rag_chunks")
    op.drop_table("rag_chunks")

    op.drop_index("ix_rag_ingestion_jobs_created_at", table_name="rag_ingestion_jobs")
    op.drop_index("ix_rag_ingestion_jobs_celery_task_id", table_name="rag_ingestion_jobs")
    op.drop_index("ix_rag_ingestion_jobs_status", table_name="rag_ingestion_jobs")
    op.drop_index("ix_rag_ingestion_jobs_collection_id", table_name="rag_ingestion_jobs")
    op.drop_index("ix_rag_ingestion_jobs_document_id", table_name="rag_ingestion_jobs")
    op.drop_table("rag_ingestion_jobs")

    op.drop_index("ix_rag_documents_created_at", table_name="rag_documents")
    op.drop_index("ix_rag_documents_checksum", table_name="rag_documents")
    op.drop_index("ix_rag_documents_status", table_name="rag_documents")
    op.drop_index("ix_rag_documents_external_id", table_name="rag_documents")
    op.drop_index("ix_rag_documents_tenant_id", table_name="rag_documents")
    op.drop_index("ix_rag_documents_collection_id", table_name="rag_documents")
    op.drop_table("rag_documents")

    op.drop_index("ix_rag_embedding_models_is_active", table_name="rag_embedding_models")
    op.drop_table("rag_embedding_models")

    op.drop_index("ix_rag_collections_status", table_name="rag_collections")
    op.drop_index("ix_rag_collections_tenant_id", table_name="rag_collections")
    op.drop_table("rag_collections")
