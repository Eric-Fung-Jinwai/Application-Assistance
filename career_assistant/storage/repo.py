"""Repository functions: schema init + CRUD over the ORM tables (Phase 1).

Every function takes an explicit ``Session`` so callers control transaction scope.
Helpers ``flush`` (not ``commit``) so several writes can share one transaction; the
caller commits.
"""

from __future__ import annotations

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from career_assistant.domain import IntegrityBand, SuggestionStatus, VersionType
from career_assistant.storage.models import (
    JD,
    Base,
    BulletRow,
    EditHistory,
    FitScoreRow,
    Resume,
    ResumeVersion,
    TailoringSuggestionRow,
    User,
)


def create_db(engine: Engine) -> None:
    """Create all tables on ``engine`` (idempotent)."""
    Base.metadata.create_all(engine)


# --- Users ------------------------------------------------------------------------


def create_user(session: Session, *, preferences: dict | None = None) -> User:
    user = User(preferences_json=preferences)
    session.add(user)
    session.flush()
    return user


# --- Resumes & versions -----------------------------------------------------------


def create_resume(
    session: Session,
    *,
    filename: str,
    raw_text: str,
    user_id: str | None = None,
    parsed: dict | None = None,
) -> Resume:
    resume = Resume(user_id=user_id, filename=filename, raw_text=raw_text, parsed_json=parsed)
    session.add(resume)
    session.flush()
    return resume


def create_version(
    session: Session,
    *,
    resume_id: str,
    version_type: VersionType | str,
    parent_version_id: str | None = None,
    label: str | None = None,
) -> ResumeVersion:
    version = ResumeVersion(
        resume_id=resume_id,
        parent_version_id=parent_version_id,
        version_type=_enum_value(version_type),
        label=label,
    )
    session.add(version)
    session.flush()
    return version


# --- Bullets ----------------------------------------------------------------------


def add_bullet(
    session: Session,
    *,
    resume_version_id: str,
    section: str,
    order_index: int,
    original_text: str,
    current_text: str | None = None,
    keywords: list[str] | None = None,
) -> BulletRow:
    bullet = BulletRow(
        resume_version_id=resume_version_id,
        section=section,
        order_index=order_index,
        original_text=original_text,
        current_text=current_text if current_text is not None else original_text,
        keywords_json=keywords,
    )
    session.add(bullet)
    session.flush()
    return bullet


def get_bullet(session: Session, bullet_id: str) -> BulletRow | None:
    return session.get(BulletRow, bullet_id)


def list_bullets(session: Session, resume_version_id: str) -> list[BulletRow]:
    stmt = (
        select(BulletRow)
        .where(BulletRow.resume_version_id == resume_version_id)
        .order_by(BulletRow.order_index)
    )
    return list(session.scalars(stmt))


# --- Job descriptions -------------------------------------------------------------


def create_jd(
    session: Session,
    *,
    raw_text: str,
    user_id: str | None = None,
    parsed: dict | None = None,
) -> JD:
    jd = JD(user_id=user_id, raw_text=raw_text, parsed_json=parsed)
    session.add(jd)
    session.flush()
    return jd


# --- Fit scores -------------------------------------------------------------------


def create_fit_score(
    session: Session,
    *,
    resume_version_id: str,
    jd_id: str,
    skill: float,
    experience: float,
    seniority: float,
    location: float,
    comp: float,
    overall: float,
    breakdown: dict | None = None,
) -> FitScoreRow:
    row = FitScoreRow(
        resume_version_id=resume_version_id,
        jd_id=jd_id,
        skill=skill,
        experience=experience,
        seniority=seniority,
        location=location,
        comp=comp,
        overall=overall,
        breakdown_json=breakdown,
    )
    session.add(row)
    session.flush()
    return row


# --- Tailoring suggestions --------------------------------------------------------


def create_suggestion(
    session: Session,
    *,
    bullet_id: str,
    jd_id: str,
    suggested_text: str,
    reasoning: str | None = None,
    llm_judgment: float | None = None,
    embedding_similarity: float | None = None,
    integrity_score: float | None = None,
    integrity_band: IntegrityBand | str | None = None,
    status: SuggestionStatus | str = SuggestionStatus.pending,
) -> TailoringSuggestionRow:
    row = TailoringSuggestionRow(
        bullet_id=bullet_id,
        jd_id=jd_id,
        suggested_text=suggested_text,
        reasoning=reasoning,
        llm_judgment=llm_judgment,
        embedding_similarity=embedding_similarity,
        integrity_score=integrity_score,
        integrity_band=_enum_value(integrity_band) if integrity_band is not None else None,
        status=_enum_value(status),
    )
    session.add(row)
    session.flush()
    return row


# --- Edit history -----------------------------------------------------------------


def log_edit(
    session: Session,
    *,
    bullet_id: str,
    action: str,
    actor: str,
    from_text: str | None = None,
    to_text: str | None = None,
) -> EditHistory:
    row = EditHistory(
        bullet_id=bullet_id,
        action=action,
        actor=actor,
        from_text=from_text,
        to_text=to_text,
    )
    session.add(row)
    session.flush()
    return row


def _enum_value(value: object) -> str:
    """Accept either an enum member or its raw string value."""
    return value.value if hasattr(value, "value") else str(value)
