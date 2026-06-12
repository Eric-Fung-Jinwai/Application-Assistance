"""Split a ParsedJD into requirement chunks for embedding (Phase 5).

A "chunk" is one requirement line — a required/preferred skill, a responsibility, or a
named tool. Each chunk gets a **deterministic** id of the form
``{jd_id}::{requirement_type}::{index}`` so the authoritative text stays resolvable from
``jds.parsed_json`` in SQLite (never duplicated into Chroma metadata — see
``storage/chroma.py`` for the contract). Tailoring (Phase 8) maps these JD requirements
onto resume bullets via vector search.
"""

from __future__ import annotations

from career_assistant.domain import ParsedJD

# requirement_type labels (also stored as Chroma metadata on each jd_chunks vector).
REQUIRED_SKILL = "required_skill"
PREFERRED_SKILL = "preferred_skill"
RESPONSIBILITY = "responsibility"
TOOL = "tool"

# Order matters: it fixes each chunk's stable index within its requirement_type.
_REQUIREMENT_FIELDS = (
    (REQUIRED_SKILL, "required_skills"),
    (PREFERRED_SKILL, "preferred_skills"),
    (RESPONSIBILITY, "responsibilities"),
    (TOOL, "tools"),
)


def chunk_id(jd_id: str, requirement_type: str, index: int) -> str:
    """Deterministic Chroma id for a JD requirement chunk."""
    return f"{jd_id}::{requirement_type}::{index}"


def chunks_from_parsed(parsed: ParsedJD, jd_id: str | None = None) -> list[tuple[str, str, str]]:
    """Flatten a ``ParsedJD`` into ``(chunk_id, requirement_type, text)`` triples.

    ``jd_id`` is the **authoritative** id the chunks are keyed under — pass the SQLite
    ``jds.id`` so Chroma ids/metadata match what ``query_jd_chunks(jd_id=...)`` filters
    on. Defaults to ``parsed.id`` for standalone use (no DB row yet).

    Empty/whitespace-only entries are skipped so we never index blank requirements.
    The index in the chunk id counts only the non-blank entries kept, so the id is
    stable for a given ``parsed`` value.
    """
    jid = jd_id or parsed.id
    triples: list[tuple[str, str, str]] = []
    for requirement_type, field in _REQUIREMENT_FIELDS:
        kept = 0
        for text in getattr(parsed, field):
            if text.strip():
                cid = chunk_id(jid, requirement_type, kept)
                triples.append((cid, requirement_type, text.strip()))
                kept += 1
    return triples
