"""SQLAlchemy engine, session factory, and declarative Base for rag-core.

RCSI-P2-T02: rag_core.db.database

This module creates the SQLAlchemy 2.x engine bound to the rag_core PostgreSQL
database. It does NOT connect to or import from chatbot_db or any DominicBE module.

Usage:
    from rag_core.db.database import engine, SessionLocal, Base, get_db

    # FastAPI dependency
    @app.get("/example")
    def example(db: Session = Depends(get_db)):
        ...

    # Alembic env.py: import Base and all model modules so metadata is populated.
"""

from __future__ import annotations

from collections.abc import Generator
from time import perf_counter

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ---------------------------------------------------------------------------
# Lazy import of settings to avoid circular dependencies at import time.
# config.py must not import from rag_core.db.
# ---------------------------------------------------------------------------

def _get_settings():
    from rag_core.api.config import get_service_settings
    return get_service_settings()


def _build_engine():
    """Build the SQLAlchemy engine from rag_core service settings.

    psycopg (v3) is the preferred driver: postgresql+psycopg://...
    The rag_core database is ALWAYS separate from chatbot_db.
    """
    settings = _get_settings()
    url = settings.sqlalchemy_database_url

    connect_args: dict = {}
    # psycopg3 connect_timeout is a connection keyword argument
    if settings.db_connect_timeout:
        connect_args["connect_timeout"] = settings.db_connect_timeout

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_timeout=settings.db_pool_timeout,
        connect_args=connect_args,
        echo=False,
    )


# ---------------------------------------------------------------------------
# Engine and session factory — module-level singletons
# ---------------------------------------------------------------------------

engine = _build_engine()

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Declarative Base — all rag_core models inherit from this
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Declarative base for all rag_core metadata models.

    Do NOT inherit from DominicBE's Base or mix models across services.
    """
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """Yield a database session for use as a FastAPI dependency.

    Example:
        @router.get("/collections")
        def list_collections(db: Session = Depends(get_db)):
            return db.query(RagCollection).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

def check_db_health() -> dict:
    """Check connectivity to the rag_core database.

    Returns a dict with ok=True/False, latency_ms, and detail on failure.
    Never raises — safe to call from /health endpoints.
    """
    started_at = perf_counter()
    settings = _get_settings()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "ok": True,
            "dependency": "postgres",
            "database": settings.db_name,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "dependency": "postgres",
            "database": settings.db_name,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "detail": str(exc),
        }
