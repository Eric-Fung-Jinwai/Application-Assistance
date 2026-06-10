"""Phase 1 acceptance: schema builds; resume → version → 3 bullets round-trips."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from career_assistant.domain import (
    Bullet,
    FitScore,
    IntegrityBand,
    IntegrityResult,
    ParsedJD,
    SuggestionStatus,
    VersionType,
)
from career_assistant.storage import repo
from career_assistant.storage.db import make_engine, make_session_factory


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    repo.create_db(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        yield s


def test_create_db_builds_schema():
    engine = make_engine(":memory:")
    repo.create_db(engine)
    from sqlalchemy import inspect

    tables = set(inspect(engine).get_table_names())
    assert {
        "users",
        "resumes",
        "resume_versions",
        "bullets",
        "jds",
        "fit_scores",
        "tailoring_suggestions",
        "edit_history",
    } <= tables


def test_resume_version_bullets_roundtrip(session):
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="…")
    version = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)

    ids = []
    for i in range(3):
        b = repo.add_bullet(
            session,
            resume_version_id=version.id,
            section="experience",
            order_index=i,
            original_text=f"Built thing #{i}",
            keywords=[f"kw{i}"],
        )
        ids.append(b.id)
    session.commit()

    # Read each back by its Unique ID (bullet_id).
    for i, bid in enumerate(ids):
        got = repo.get_bullet(session, bid)
        assert got is not None
        assert got.original_text == f"Built thing #{i}"
        assert got.current_text == f"Built thing #{i}"  # defaults to original
        assert got.keywords_json == [f"kw{i}"]
        assert got.resume_version_id == version.id

    # Ordered listing returns all three.
    listed = repo.list_bullets(session, version.id)
    assert [b.id for b in listed] == ids


def test_suggestion_and_history_persist(session):
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    version = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    bullet = repo.add_bullet(
        session,
        resume_version_id=version.id,
        section="experience",
        order_index=0,
        original_text="Shipped feature",
    )
    jd = repo.create_jd(session, raw_text="We need an engineer")

    sugg = repo.create_suggestion(
        session,
        bullet_id=bullet.id,
        jd_id=jd.id,
        suggested_text="Shipped a high-impact feature",
        integrity_band=IntegrityBand.safe.value,
        status=SuggestionStatus.pending,
    )
    repo.log_edit(session, bullet_id=bullet.id, action="accept", actor="user", to_text="…")
    session.commit()

    assert sugg.status == "pending"
    assert sugg.integrity_band == "safe"


def test_foreign_keys_enforced(session):
    # PRAGMA foreign_keys=ON: an orphan version (no such resume) must be rejected.
    # repo helpers flush() internally, so the violation surfaces here, not at commit.
    with pytest.raises(IntegrityError):
        repo.create_version(
            session, resume_id="does-not-exist", version_type=VersionType.original
        )


def test_score_ranges_validated():
    with pytest.raises(ValidationError):
        FitScore(skill=80, experience=70, seniority=60, location=100, compensation=50, overall=500)
    with pytest.raises(ValidationError):
        FitScore(skill=-1, experience=70, seniority=60, location=100, compensation=50, overall=72)
    with pytest.raises(ValidationError):
        IntegrityResult(llm_judgment=2, embedding_similarity=0.5, score=80, band=IntegrityBand.safe)
    with pytest.raises(ValidationError):
        IntegrityResult(
            llm_judgment=0.9, embedding_similarity=-3, score=80, band=IntegrityBand.safe
        )


def test_integrity_band_normalized_to_str(session):
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    version = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    bullet = repo.add_bullet(
        session,
        resume_version_id=version.id,
        section="experience",
        order_index=0,
        original_text="x",
    )
    jd = repo.create_jd(session, raw_text="jd")
    # Pass the enum member; the stored value must be the plain string, even pre-reload.
    sugg = repo.create_suggestion(
        session,
        bullet_id=bullet.id,
        jd_id=jd.id,
        suggested_text="y",
        integrity_band=IntegrityBand.aggressive,
    )
    assert sugg.integrity_band == "aggressive"
    assert type(sugg.integrity_band) is str


def test_domain_models_construct():
    # Domain contracts instantiate with UUID4 ids and sane defaults.
    bullet = Bullet(section="experience", order_index=0, original_text="x", current_text="x")
    assert len(bullet.id) == 36
    jd = ParsedJD(required_skills=["python"])
    assert jd.years_experience is None
    fit = FitScore(skill=80, experience=70, seniority=60, location=100, compensation=50, overall=72)
    assert 0 <= fit.overall <= 100
    integ = IntegrityResult(
        llm_judgment=0.9, embedding_similarity=0.8, score=88, band=IntegrityBand.moderate
    )
    assert integ.band is IntegrityBand.moderate
