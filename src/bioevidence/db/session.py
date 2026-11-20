"""Database session management."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from bioevidence.config import get_config
from bioevidence.db.models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        config = get_config()
        if not config.database.url:
            msg = "DATABASE_URL is not configured"
            raise RuntimeError(msg)
        _engine = create_engine(
            config.database.url,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_timeout=config.database.pool_timeout,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def get_session() -> Iterator[Session]:
    """Get a database session."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(get_engine())


def drop_db() -> None:
    """Drop all database tables."""
    Base.metadata.drop_all(get_engine())
