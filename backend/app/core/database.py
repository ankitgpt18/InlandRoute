"""
AIDSTL Project — Async Database Configuration
=============================================
SQLAlchemy 2.x async engine with PostGIS support.

Provides:
- Async engine and session factory
- Declarative base with common mixins
- `get_db()` FastAPI dependency
- Health-check helper
- Alembic-compatible sync engine for migrations
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings
from sqlalchemy import DateTime, Integer, String, event, inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _build_engine_kwargs(testing: bool = False) -> dict[str, Any]:
    """Return keyword arguments for the async engine.

    Uses ``NullPool`` during testing to avoid connection leaks across
    test transactions; otherwise uses ``AsyncAdaptedQueuePool`` for
    efficient connection reuse.
    """
    common: dict[str, Any] = {
        "echo": settings.DEBUG,
        "echo_pool": settings.DEBUG,
    }

    if testing:
        common["poolclass"] = NullPool
    else:
        common.update(
            {
                "poolclass": AsyncAdaptedQueuePool,
                "pool_size": settings.DB_POOL_SIZE,
                "max_overflow": settings.DB_MAX_OVERFLOW,
                "pool_timeout": settings.DB_POOL_TIMEOUT,
                # Recycle connections after 30 minutes to survive DB restarts
                "pool_recycle": 1800,
                # Proactively test connections before handing them out
                "pool_pre_ping": True,
            }
        )
    return common


def create_engine(
    database_url: str | None = None, testing: bool = False
) -> AsyncEngine:
    """Create and return the SQLAlchemy async engine.

    Args:
        database_url: Override the URL from settings (useful in tests).
        testing: When ``True``, uses ``NullPool`` to avoid test pollution.

    Returns:
        Configured :class:`AsyncEngine` instance.
    """
    url = database_url or settings.DATABASE_URL
    kwargs = _build_engine_kwargs(testing=testing)
    engine = create_async_engine(url, **kwargs)

    # Attach a connect listener to enable PostGIS and set session params.
    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_conn: Any, connection_record: Any) -> None:  # noqa: ANN001
        """Set per-connection runtime parameters on raw psycopg2/asyncpg connections."""
        # asyncpg exposes a different interface; guard accordingly.
        try:
            if hasattr(dbapi_conn, "cursor"):
                with dbapi_conn.cursor() as cur:
                    cur.execute("SET TIME ZONE 'UTC'")
        except (AttributeError, TypeError):
            pass

    logger.info("Async database engine created: %s", url.split("@")[-1])
    return engine


# ---------------------------------------------------------------------------
# Module-level engine & session factory (initialised lazily)
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the module-level engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the module-level session factory, creating it on first call."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _async_session_factory


# ---------------------------------------------------------------------------
# Declarative base & shared mixins
# ---------------------------------------------------------------------------


class Base(AsyncAttrs, DeclarativeBase):
    """Project-wide declarative base.

    Inheriting from :class:`AsyncAttrs` enables ``await`` on relationship
    accessors inside async contexts without triggering lazy-load errors.
    """

    # Subclasses may override this to set a custom schema.
    __table_args__: dict[str, Any] | tuple[Any, ...] = {}

    def to_dict(self) -> dict[str, Any]:
        """Serialise the model instance to a plain dictionary.

        Only mapped columns are included; relationship attributes are
        *not* eagerly loaded to avoid implicit I/O.
        """
        mapper = inspect(type(self))
        return {col.key: getattr(self, col.key) for col in mapper.column_attrs}

    def __repr__(self) -> str:
        pk_cols = [
            col.key
            for col in inspect(type(self)).mapper.column_attrs
            if col.columns[0].primary_key
        ]
        pk_repr = ", ".join(f"{k}={getattr(self, k)!r}" for k in pk_cols)
        return f"<{type(self).__name__}({pk_repr})>"


class TimestampMixin:
    """Mixin that adds ``created_at`` and ``updated_at`` columns.

    Use this on any model that should track record-level timestamps.
    Both columns store UTC-aware datetimes and are managed automatically.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="UTC timestamp of record creation",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="UTC timestamp of last record update",
    )


class SoftDeleteMixin:
    """Mixin that adds a ``deleted_at`` column for soft-deletion semantics."""

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="UTC timestamp of soft deletion; NULL means the record is active",
    )

    @property
    def is_deleted(self) -> bool:
        """Return ``True`` if this record has been soft-deleted."""
        return self.deleted_at is not None


class IntPKMixin:
    """Mixin that adds a simple auto-increment integer primary key."""

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Surrogate primary key",
    )


class StrPKMixin:
    """Mixin that adds a string primary key (e.g. segment IDs like 'NW-1-042')."""

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="Domain-meaningful string primary key",
    )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a scoped async database session.

    The session is committed automatically on successful completion and
    rolled back if an exception propagates.  The session is always closed
    when the request finishes.

    Usage::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(MyModel))
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Lifecycle helpers (called from main.py lifespan)
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Initialise the database connection pool and verify PostGIS is available.

    This should be called once during application startup (inside the
    FastAPI lifespan context manager).

    Raises:
        RuntimeError: If the PostGIS extension is not installed on the target DB.
    """
    engine = get_engine()

    async with engine.begin() as conn:
        # Verify connectivity
        await conn.execute(text("SELECT 1"))

        # Verify PostGIS extension
        result = await conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'postgis'")
        )
        row = result.fetchone()
        if row is None:
            logger.warning(
                "PostGIS extension not found — attempting to create it. "
                "The database user must have SUPERUSER or CREATE EXTENSION privileges."
            )
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
                await conn.execute(
                    text("CREATE EXTENSION IF NOT EXISTS postgis_topology")
                )
                logger.info("PostGIS extension created successfully.")
            except Exception as exc:
                raise RuntimeError(
                    "PostGIS is not available and could not be created. "
                    "Please install it manually: CREATE EXTENSION postgis;"
                ) from exc
        else:
            logger.info("PostGIS extension confirmed.")

    logger.info("Database initialised successfully.")


async def close_db() -> None:
    """Dispose of the connection pool.

    Call this during application shutdown to release all pooled connections
    gracefully.
    """
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database connection pool disposed.")
        _engine = None
        _async_session_factory = None


async def check_db_health() -> dict[str, str | bool]:
    """Perform a lightweight health check against the database.

    Returns:
        A dictionary with keys ``status``, ``postgis_version``, and
        ``pool_status``.  Used by the ``/health`` endpoint.
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT PostGIS_Full_Version()"))
            row = result.fetchone()
            postgis_version = row[0] if row else "unknown"

        pool = engine.pool
        pool_status = {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }

        return {
            "status": "healthy",
            "postgis_version": postgis_version,
            "pool": pool_status,
        }
    except Exception as exc:
        logger.exception("Database health check failed: %s", exc)
        return {
            "status": "unhealthy",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Alembic-compatible synchronous connection helper
# ---------------------------------------------------------------------------


async def run_sync_in_worker_thread(
    conn: AsyncConnection,
    fn: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a synchronous callable inside an async connection's thread pool.

    This is the recommended pattern for running Alembic migrations
    programmatically from async code.

    Example::

        async with engine.begin() as conn:
            await run_sync_in_worker_thread(conn, Base.metadata.create_all)
    """
    return await conn.run_sync(fn, *args, **kwargs)


async def create_all_tables() -> None:
    """Create all tables defined in the metadata (development / testing only).

    In production use Alembic migrations instead.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await run_sync_in_worker_thread(conn, Base.metadata.create_all)
    logger.info("All tables created via metadata.create_all().")


async def drop_all_tables() -> None:
    """Drop all tables — **DESTRUCTIVE**, only use in tests or CI.

    In production, prefer Alembic ``downgrade`` commands.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await run_sync_in_worker_thread(conn, Base.metadata.drop_all)
    logger.warning("All tables dropped via metadata.drop_all().")
