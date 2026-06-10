"""ChromaDB collections + helpers (Phase 1 scaffold; populated in Phase 4).

Bullet ↔ embedding contract
---------------------------
SQLite is the source of truth for **text**; ChromaDB stores only **vectors**. The two
are joined by the bullet's Unique ID (``bullets.id``):

  * ``resume_bullets``   — one vector per resume bullet. Chroma id == ``bullet_id``.
                           metadata: ``{bullet_id, resume_version_id, section}``
  * ``jd_chunks``        — one vector per JD requirement chunk. Chroma id == chunk id.
                           metadata: ``{jd_id, requirement_type}``
  * ``tailored_bullets`` — one vector per tailored suggestion.
                           metadata: ``{suggestion_id, bullet_id}``

Never store the authoritative text in Chroma metadata — look it up in SQLite by id.
Similarity (cosine) results return the id; callers resolve text via the repo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from career_assistant.config import settings

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

# Collection names (single source of truth so call sites don't hardcode strings).
RESUME_BULLETS = "resume_bullets"
JD_CHUNKS = "jd_chunks"
TAILORED_BULLETS = "tailored_bullets"

COLLECTIONS = (RESUME_BULLETS, JD_CHUNKS, TAILORED_BULLETS)


def get_client(chroma_path: str | None = None) -> ClientAPI:
    """Return a persistent Chroma client rooted at ``chroma_path``."""
    import chromadb

    return chromadb.PersistentClient(path=chroma_path or settings.chroma_path)


def get_collection(client: ClientAPI, name: str) -> Collection:
    """Get-or-create a collection. Cosine space matches the embedding similarity used
    throughout fit/integrity scoring."""
    if name not in COLLECTIONS:
        msg = f"Unknown collection {name!r}; expected one of {COLLECTIONS}."
        raise ValueError(msg)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
