"""Resume version tree: immutable bullet snapshots linked by ``parent_version_id`` (Phase 9).

Each ``ResumeVersion`` owns its own copy of every bullet, so a version is a full, immutable
snapshot. Accept / customize **grow** the tree (``create_child_version`` copies the parent's
bullets and applies the edit); reject / rollback / restore-original are **navigation** that
return an existing version — the tree is append-only, so nothing is ever lost and the chain
stays walkable.

Copied bullets keep their true ``original_text`` verbatim across every generation. That's
what lets ``restore_original`` hand back byte-identical original bullets and keeps the
integrity judge grounding rewrites against the real source, however deep the edit history.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from career_assistant.domain import VersionType
from career_assistant.storage import repo
from career_assistant.storage.models import BulletRow, ResumeVersion


def create_child_version(
    session: Session,
    *,
    parent_version_id: str,
    version_type: VersionType | str,
    label: str | None = None,
    edits: dict[str, str] | None = None,
) -> ResumeVersion:
    """Snapshot ``parent_version_id`` into a new child version, applying per-bullet edits.

    ``edits`` maps a bullet **lineage_id** → its new ``current_text`` (used by accept and
    customize). Because lineage is stable across snapshots, a suggestion captured against an
    earlier version still applies to a later one — no id remapping. Unedited bullets are
    copied verbatim; each copy gets a fresh row id but keeps the parent's lineage_id,
    ``original_text``, section, order, and keywords. Caller commits.

    Raises ``ValueError`` if the parent is unknown or an edit targets a lineage that isn't in
    the parent version (catches Phase 10 wiring mistakes early).
    """
    parent = session.get(ResumeVersion, parent_version_id)
    if parent is None:
        raise ValueError(f"unknown parent version {parent_version_id!r}")

    edits = edits or {}
    parent_bullets = repo.list_bullets(session, parent.id)
    unknown = set(edits) - {b.lineage_id for b in parent_bullets}
    if unknown:
        raise ValueError(f"edits target bullet lineages not in parent version: {sorted(unknown)}")

    child = repo.create_version(
        session,
        resume_id=parent.resume_id,
        version_type=version_type,
        parent_version_id=parent.id,
        label=label,
    )
    for bullet in parent_bullets:
        repo.add_bullet(
            session,
            resume_version_id=child.id,
            section=bullet.section,
            order_index=bullet.order_index,
            original_text=bullet.original_text,
            current_text=edits.get(bullet.lineage_id, bullet.current_text),
            keywords=bullet.keywords_json,
            lineage_id=bullet.lineage_id,
        )
    return child


def get_version(session: Session, version_id: str) -> ResumeVersion | None:
    return session.get(ResumeVersion, version_id)


def bullets_by_lineage(session: Session, version_id: str) -> dict[str, BulletRow]:
    """Map ``lineage_id`` → the bullet row carrying it in ``version_id``.

    The lineage_id is stable across snapshots, so this lets a caller resolve "the current
    bullet for this lineage" in any version — e.g. Phase 10 turning a suggestion (captured
    against an earlier version) into an edit against the latest one.
    """
    return {b.lineage_id: b for b in repo.list_bullets(session, version_id)}


def list_versions(session: Session, resume_id: str) -> list[ResumeVersion]:
    """All versions for a resume, oldest first (for a tree/history view)."""
    stmt = (
        select(ResumeVersion)
        .where(ResumeVersion.resume_id == resume_id)
        .order_by(ResumeVersion.created_at)
    )
    return list(session.scalars(stmt))


def version_chain(session: Session, version_id: str) -> list[ResumeVersion]:
    """The ancestry root → ``version_id``, following ``parent_version_id`` links."""
    chain: list[ResumeVersion] = []
    current = session.get(ResumeVersion, version_id)
    while current is not None:
        chain.append(current)
        current = (
            session.get(ResumeVersion, current.parent_version_id)
            if current.parent_version_id
            else None
        )
    return list(reversed(chain))


def find_original_version(session: Session, resume_id: str) -> ResumeVersion | None:
    """The root ``original`` version created at parse time."""
    stmt = (
        select(ResumeVersion)
        .where(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.version_type == VersionType.original.value,
        )
        .order_by(ResumeVersion.created_at)
    )
    return session.scalars(stmt).first()


def latest_accepted_version(session: Session, resume_id: str) -> ResumeVersion | None:
    """The most recent ``accepted`` version anywhere on the resume.

    Resume-global (newest across *all* branches) — use it for "the best accepted resume
    overall" (e.g. an export default), NOT for Reject. Reject must stay on the current
    branch; use ``latest_accepted_ancestor`` for that.
    """
    stmt = (
        select(ResumeVersion)
        .where(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.version_type == VersionType.accepted.value,
        )
        .order_by(ResumeVersion.created_at.desc())
    )
    return session.scalars(stmt).first()


def latest_accepted_ancestor(session: Session, version_id: str) -> ResumeVersion | None:
    """The nearest ``accepted`` version on this version's own lineage — what Reject reverts to.

    Walks ``parent_version_id`` upward starting at ``version_id`` itself, so in a branched
    tree Reject can never jump to an accepted version from an unrelated branch. Returns
    ``None`` when the lineage has no accepted version yet (caller falls back to the original).
    """
    current = session.get(ResumeVersion, version_id)
    while current is not None:
        if current.version_type == VersionType.accepted.value:
            return current
        current = (
            session.get(ResumeVersion, current.parent_version_id)
            if current.parent_version_id
            else None
        )
    return None


def rollback_to(session: Session, version_id: str) -> ResumeVersion:
    """Return the version to roll back to (validating it exists).

    Navigation, not mutation: the returned version becomes the base the caller builds the
    next child from, so the append-only history is preserved.
    """
    version = session.get(ResumeVersion, version_id)
    if version is None:
        raise ValueError(f"unknown version {version_id!r}")
    return version


def restore_original(session: Session, resume_id: str) -> ResumeVersion:
    """Return the original version — its bullets are byte-identical to the parsed resume."""
    original = find_original_version(session, resume_id)
    if original is None:
        raise ValueError(f"resume {resume_id!r} has no original version")
    return original
