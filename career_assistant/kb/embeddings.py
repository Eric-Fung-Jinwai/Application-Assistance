"""Embed resume bullets and upsert them to the ``resume_bullets`` collection (Phase 4).

SQLite owns the bullet text; Chroma stores only the vector, keyed by ``bullet_id``
(see ``storage/chroma.py`` for the contract). All embedding goes through the
``EmbeddingClient`` seam, so vectors are computed here and passed to Chroma explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from career_assistant.domain import ParsedResume
from career_assistant.llm.client import EmbeddingClient
from career_assistant.storage import chroma, repo
from career_assistant.storage.models import BulletRow

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


def embed_and_upsert_bullets(
    bullets: list[BulletRow],
    *,
    embedder: EmbeddingClient,
    collection: Collection,
) -> list[str]:
    """Embed each bullet's ``current_text`` and upsert to ``resume_bullets``.

    Returns the list of upserted ``bullet_id``s. No-op for an empty bullet list.
    """
    if not bullets:
        return []

    texts = [b.current_text for b in bullets]
    vectors = embedder.embed(texts)
    ids = [b.id for b in bullets]
    metadatas = [
        {
            "bullet_id": b.id,
            "resume_version_id": b.resume_version_id,
            # Stable cross-version identity (Phase 9) — lets vectors be filtered/debugged by
            # logical bullet, not just by the per-version row id.
            "lineage_id": b.lineage_id,
            "section": b.section,
        }
        for b in bullets
    ]
    chroma.upsert_embeddings(collection, ids=ids, embeddings=vectors, metadatas=metadatas)
    return ids


def query_bullets(
    text: str,
    *,
    embedder: EmbeddingClient,
    collection: Collection,
    n_results: int = 5,
    resume_version_id: str | None = None,
) -> dict:
    """Nearest-neighbour search over indexed bullets by free text.

    Optionally scope to one resume version. Returns Chroma's result dict; callers
    resolve text from SQLite via the returned ``bullet_id``s.
    """
    vector = embedder.embed([text])[0]
    where = {"resume_version_id": resume_version_id} if resume_version_id else None
    return chroma.query_embeddings(collection, embedding=vector, n_results=n_results, where=where)


def build_resume_kb(
    session: Session,
    *,
    resume_version_id: str,
    parsed: ParsedResume,
    embedder: EmbeddingClient,
    collection: Collection,
) -> list[str]:
    """End-to-end 'on parse' indexing: persist bullets, embed, upsert to Chroma.

    SQLite and Chroma can't share a transaction, so we honour the "SQLite is source of
    truth" contract by **committing the bullet rows before writing their vectors**.
    Unlike the lower-level repo helpers, this function commits. If the Chroma upsert
    fails after the commit, the result is committed bullets with missing vectors —
    recoverable by simply calling this again (see below) — never orphan vectors.

    A resume version is an immutable snapshot, so this is **idempotent**: if the
    version is already persisted (e.g. a retry after a failed embed, or a duplicate
    call), it re-embeds the existing bullets instead of appending new rows. Chroma
    upsert is keyed by ``bullet_id``, so vectors are replaced in place, never
    duplicated.

    Returns the indexed ``bullet_id``s.
    """
    from career_assistant.kb.bullets import persist_bullets

    existing = repo.list_bullets(session, resume_version_id)
    if existing:
        # Already indexed for this (immutable) version → re-embed, don't duplicate.
        return embed_and_upsert_bullets(existing, embedder=embedder, collection=collection)

    rows = persist_bullets(session, resume_version_id=resume_version_id, parsed=parsed)
    session.commit()  # durably persist the source of truth before deriving vectors
    return embed_and_upsert_bullets(rows, embedder=embedder, collection=collection)
