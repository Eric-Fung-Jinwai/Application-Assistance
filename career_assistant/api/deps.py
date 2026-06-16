"""FastAPI dependencies: per-request session + shared Chroma collections + client seams.

Everything stateful (the SQLite engine/session factory, the Chroma client) lives on
``app.state``, set by ``create_app`` — so a test can build an app on a temp DB/Chroma dir and
get full isolation. The ``get_llm`` / ``get_embedder`` seams default to the configured clients
but are plain dependencies, so tests override them with deterministic fakes (no network) via
``app.dependency_overrides``.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.llm.factory import get_embedding_client, get_llm_client
from career_assistant.storage import chroma


def get_session(request: Request) -> Iterator[Session]:
    """Yield a request-scoped session; the route commits, this guarantees close."""
    factory = request.app.state.session_factory
    with factory() as session:
        yield session


def get_resume_collection(request: Request):
    return chroma.get_collection(request.app.state.chroma_client, chroma.RESUME_BULLETS)


def get_jd_collection(request: Request):
    return chroma.get_collection(request.app.state.chroma_client, chroma.JD_CHUNKS)


def get_llm() -> LLMClient:
    return get_llm_client()


def get_embedder() -> EmbeddingClient:
    return get_embedding_client()
