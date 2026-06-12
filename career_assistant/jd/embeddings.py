"""Embed JD requirement chunks and upsert them to the ``jd_chunks`` collection (Phase 5).

Mirrors ``kb/embeddings.py``: all embedding goes through the ``EmbeddingClient`` seam,
vectors are passed to Chroma explicitly, and chunk ids are deterministic so the
authoritative text stays in ``jds.parsed_json`` (SQLite), never in Chroma metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from career_assistant.domain import ParsedJD
from career_assistant.jd.chunks import chunks_from_parsed
from career_assistant.llm.client import EmbeddingClient
from career_assistant.storage import chroma

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


def build_jd_kb(
    parsed: ParsedJD,
    *,
    embedder: EmbeddingClient,
    collection: Collection,
    jd_id: str | None = None,
) -> list[str]:
    """Embed each requirement chunk of ``parsed`` and upsert to ``jd_chunks``.

    ``jd_id`` is the **authoritative** id the chunks are keyed under. Pass the SQLite
    ``jds.id`` (the source of truth, like ``resume_version_id`` for ``build_resume_kb``)
    so ``query_jd_chunks(jd_id=...)`` finds them; it defaults to ``parsed.id`` only for
    standalone use before a DB row exists.

    Returns the list of upserted chunk ids. Chunk ids are deterministic, so re-running
    replaces vectors in place (idempotent). Re-indexing the same JD with **fewer**
    requirements also **prunes** the now-orphaned chunks from a previous larger run, so
    the collection never accumulates stale requirements for a JD.
    """
    jid = jd_id or parsed.id
    triples = chunks_from_parsed(parsed, jid)
    ids = [cid for cid, _, _ in triples]

    if triples:
        texts = [text for _, _, text in triples]
        metadatas = [{"jd_id": jid, "requirement_type": rtype} for _, rtype, _ in triples]
        vectors = embedder.embed(texts)
        chroma.upsert_embeddings(collection, ids=ids, embeddings=vectors, metadatas=metadatas)

    # Prune chunks left over from a previous, larger index of this JD. Upsert first so
    # the live vectors are never momentarily missing; only true orphans get deleted.
    keep = set(ids)
    indexed = chroma.existing_ids(collection, where={"jd_id": jid})
    stale = [cid for cid in indexed if cid not in keep]
    chroma.delete_embeddings(collection, ids=stale)
    return ids


def query_jd_chunks(
    text: str,
    *,
    embedder: EmbeddingClient,
    collection: Collection,
    n_results: int = 5,
    jd_id: str | None = None,
) -> dict:
    """Nearest-neighbour search over indexed JD chunks by free text.

    Optionally scope to one JD. Returns Chroma's result dict; callers resolve text from
    ``jds.parsed_json`` via the returned chunk ids.
    """
    vector = embedder.embed([text])[0]
    where = {"jd_id": jd_id} if jd_id else None
    return chroma.query_embeddings(collection, embedding=vector, n_results=n_results, where=where)
