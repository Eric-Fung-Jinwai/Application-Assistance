"""Split a ParsedResume into bullets, assign the Unique ID, persist to SQLite (Phase 4).

Only experience and project bullets are indexed — they're the evidence the tailoring
and integrity stages reason over. Skills/education/certifications stay on the resume
record but aren't bullet-level units.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from career_assistant.domain import ParsedResume
from career_assistant.storage import repo
from career_assistant.storage.models import BulletRow

# Section labels stored on each bullet (also used as Chroma metadata).
SECTION_EXPERIENCE = "experience"
SECTION_PROJECTS = "projects"


def bullets_from_parsed(parsed: ParsedResume) -> list[tuple[str, str]]:
    """Flatten experience + project bullets into ``(section, text)`` pairs, in order.

    Empty/whitespace-only bullets are skipped so we never index blank evidence.
    """
    pairs: list[tuple[str, str]] = []
    for exp in parsed.experience:
        for text in exp.bullets:
            if text.strip():
                pairs.append((SECTION_EXPERIENCE, text.strip()))
    for proj in parsed.projects:
        for text in proj.bullets:
            if text.strip():
                pairs.append((SECTION_PROJECTS, text.strip()))
    return pairs


def persist_bullets(
    session: Session, *, resume_version_id: str, parsed: ParsedResume
) -> list[BulletRow]:
    """Persist the parsed resume's bullets under ``resume_version_id``.

    Each bullet gets a UUID4 Unique ID (``BulletRow.id``) — the join key to its Chroma
    vector. ``order_index`` preserves resume order. Caller commits.
    """
    rows: list[BulletRow] = []
    for order_index, (section, text) in enumerate(bullets_from_parsed(parsed)):
        rows.append(
            repo.add_bullet(
                session,
                resume_version_id=resume_version_id,
                section=section,
                order_index=order_index,
                original_text=text,
            )
        )
    return rows
