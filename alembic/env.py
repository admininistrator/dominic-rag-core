"""Alembic environment for rag-core metadata migrations.

The migration context is intentionally bound to rag_core service settings only.
Never import DominicBE settings, models, or sessions from this file.
"""

from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure local src/ package imports work when Alembic is run from rag-core/.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_core.api.config import get_service_settings  # noqa: E402
from rag_core.db.database import Base  # noqa: E402
import rag_core.db.models  # noqa: F401,E402  # registers all model tables

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_service_settings()
config.set_main_option("sqlalchemy.url", settings.sqlalchemy_database_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""

    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
