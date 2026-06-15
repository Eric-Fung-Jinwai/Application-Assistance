"""SQLAlchemy ORM tables — SQLite is the relational source of truth (Phase 1).

Entity-store map:
  * relational data (resumes, versions, bullets, JDs, scores, history) → SQLite (here)
  * vectors (bullet / JD-chunk / tailored embeddings) → ChromaDB (``storage/chroma.py``)

JSON-bearing columns use the SQLAlchemy ``JSON`` type, which serialises dict/list to
TEXT on SQLite. Enum-valued columns store the plain string ``value`` of the matching
``domain`` enum (``version_type``, ``status``, ``integrity_band``); the pydantic layer
enforces the allowed set at the boundary.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(default=_now)
    preferences_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    resumes: Mapped[list[Resume]] = relationship(back_populates="user")


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    filename: Mapped[str] = mapped_column(String)
    raw_text: Mapped[str] = mapped_column(Text)
    parsed_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now)

    user: Mapped[User | None] = relationship(back_populates="resumes")
    versions: Mapped[list[ResumeVersion]] = relationship(back_populates="resume")


class ResumeVersion(Base):
    __tablename__ = "resume_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    resume_id: Mapped[str] = mapped_column(ForeignKey("resumes.id"))
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("resume_versions.id"), nullable=True
    )
    # domain.VersionType: original | ai_generated | accepted | rejected | customized
    version_type: Mapped[str] = mapped_column(String)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now)

    resume: Mapped[Resume] = relationship(back_populates="versions")
    bullets: Mapped[list[BulletRow]] = relationship(back_populates="version")


class BulletRow(Base):
    """A resume bullet. ``id`` is the **Unique ID** linking SQLite text to its Chroma
    vector (collection ``resume_bullets``), tailoring suggestions, and edit history."""

    __tablename__ = "bullets"
    # A logical bullet (lineage) appears at most once per version snapshot — guards the
    # version-copy path against accidentally duplicating a lineage within one version.
    __table_args__ = (UniqueConstraint("resume_version_id", "lineage_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    resume_version_id: Mapped[str] = mapped_column(ForeignKey("resume_versions.id"))
    # Stable identity for "the same bullet" across version snapshots: an original bullet
    # self-roots with a fresh id; every copy inherits the parent's. Edits/suggestions target
    # a bullet by lineage, so they survive a version change regardless of position — and
    # remain valid even when bullets are later added/removed/reordered.
    lineage_id: Mapped[str] = mapped_column(String, default=_uuid, index=True)
    section: Mapped[str] = mapped_column(String)
    order_index: Mapped[int] = mapped_column(Integer)
    original_text: Mapped[str] = mapped_column(Text)
    current_text: Mapped[str] = mapped_column(Text)
    keywords_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    version: Mapped[ResumeVersion] = relationship(back_populates="bullets")


class JD(Base):
    __tablename__ = "jds"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text)
    parsed_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class FitScoreRow(Base):
    __tablename__ = "fit_scores"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    resume_version_id: Mapped[str] = mapped_column(ForeignKey("resume_versions.id"))
    jd_id: Mapped[str] = mapped_column(ForeignKey("jds.id"))
    skill: Mapped[float] = mapped_column(Float)
    experience: Mapped[float] = mapped_column(Float)
    seniority: Mapped[float] = mapped_column(Float)
    location: Mapped[float] = mapped_column(Float)
    comp: Mapped[float] = mapped_column(Float)
    overall: Mapped[float] = mapped_column(Float)
    breakdown_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class TailoringSuggestionRow(Base):
    __tablename__ = "tailoring_suggestions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    bullet_id: Mapped[str] = mapped_column(ForeignKey("bullets.id"))
    jd_id: Mapped[str] = mapped_column(ForeignKey("jds.id"))
    suggested_text: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_judgment: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    integrity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # domain.IntegrityBand: safe | moderate | aggressive | high_risk
    integrity_band: Mapped[str | None] = mapped_column(String, nullable=True)
    # Non-empty unsupported-* flag lists by category (IntegrityResult.unsupported());
    # kept so the Phase 10 review panel can explain WHY a suggestion is risky after reload.
    integrity_flags_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # domain.SuggestionStatus: pending | accepted | rejected | customized
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(default=_now)


class EditHistory(Base):
    __tablename__ = "edit_history"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    bullet_id: Mapped[str] = mapped_column(ForeignKey("bullets.id"))
    from_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    to_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    action: Mapped[str] = mapped_column(String)  # accept | reject | customize | ...
    actor: Mapped[str] = mapped_column(String)  # "user" | "ai"
    created_at: Mapped[datetime] = mapped_column(default=_now)
