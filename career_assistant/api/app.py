"""FastAPI application factory (Phase 14).

``create_app`` builds a fully self-contained app: its SQLite engine/session factory and Chroma
client live on ``app.state``, so a test can stand one up on a temp DB/Chroma dir for full
isolation. Run it with ``uvicorn career_assistant.api.app:app`` — the module-level ``app`` is
created lazily (via ``__getattr__``) so merely importing ``create_app`` in a test doesn't spin up
the default ``./data`` stores.
"""

from __future__ import annotations

from fastapi import FastAPI

from career_assistant.api.errors import register_error_handlers, register_request_logging
from career_assistant.api.routes import router
from career_assistant.config import settings
from career_assistant.storage import chroma, repo
from career_assistant.storage.db import make_engine, make_session_factory


def create_app(*, sqlite_path: str | None = None, chroma_path: str | None = None) -> FastAPI:
    """Build an app bound to the given (or configured) SQLite + Chroma locations."""
    app = FastAPI(title="Career Assistant API", version="0.1.0")

    engine = make_engine(sqlite_path or settings.sqlite_path)
    repo.create_db(engine)  # create + bring an existing schema current (Phase 14 migration path)
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.chroma_client = chroma.get_client(chroma_path or settings.chroma_path)

    register_request_logging(app)
    register_error_handlers(app)
    app.include_router(router)
    return app


def __getattr__(name: str) -> object:
    # Lazily construct the default app only when something actually accesses `app`
    # (e.g. uvicorn), so importing this module for `create_app` has no side effects.
    if name == "app":
        app = create_app()
        globals()["app"] = app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
