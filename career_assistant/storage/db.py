"""SQLite engine + session factory (Phase 1)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from career_assistant.config import settings


def make_engine(sqlite_path: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for a SQLite database.

    Pass ``":memory:"`` for an ephemeral DB (tests); otherwise the parent directory
    is created on demand. Defaults to ``settings.sqlite_path``.

    SQLite ignores foreign keys unless ``PRAGMA foreign_keys=ON`` is set on each
    connection, so we enable it on connect — otherwise orphan versions/bullets/
    suggestions could be inserted and corrupt the version tree.
    """
    path = sqlite_path or settings.sqlite_path
    if path != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a configured ``sessionmaker`` bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
