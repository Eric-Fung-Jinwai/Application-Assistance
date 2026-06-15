"""Phase 9 acceptance: accept→customize→reject leaves a navigable version chain, and
restore-original returns byte-identical original bullets."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from career_assistant.domain import VersionType
from career_assistant.storage import repo
from career_assistant.storage.db import make_engine, make_session_factory
from career_assistant.versioning import (
    bullets_by_lineage,
    create_child_version,
    latest_accepted_ancestor,
    latest_accepted_version,
    restore_original,
    rollback_to,
    version_chain,
)

ORIGINAL_BULLETS = [
    ("experience", "Built a Python data pipeline"),
    ("experience", "Led migration to Docker"),
    ("projects", "Designed a Rust caching sidecar"),
]


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    repo.create_db(engine)
    with make_session_factory(engine)() as s:
        yield s


def _seed_original(session):
    """Resume + original version with three bullets (current_text == original_text)."""
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    v1 = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    for i, (section, text) in enumerate(ORIGINAL_BULLETS):
        repo.add_bullet(
            session, resume_version_id=v1.id, section=section, order_index=i, original_text=text
        )
    session.commit()
    return resume, v1


# --- core snapshot behaviour -----------------------------------------------------


def test_create_child_copies_bullets_verbatim_when_no_edits(session):
    _, v1 = _seed_original(session)
    v2 = create_child_version(session, parent_version_id=v1.id, version_type=VersionType.accepted)
    session.commit()

    parent = repo.list_bullets(session, v1.id)
    child = repo.list_bullets(session, v2.id)
    assert [b.current_text for b in child] == [b.current_text for b in parent]
    # Each version owns its own bullet rows (fresh ids).
    assert {b.id for b in child}.isdisjoint({b.id for b in parent})
    assert v2.parent_version_id == v1.id


def test_edits_apply_to_named_bullet_and_preserve_original_text(session):
    _, v1 = _seed_original(session)
    target = repo.list_bullets(session, v1.id)[0]

    v2 = create_child_version(
        session,
        parent_version_id=v1.id,
        version_type=VersionType.accepted,
        edits={target.lineage_id: "Engineered a Python data pipeline cutting latency 30%"},
    )
    session.commit()

    edited = repo.list_bullets(session, v2.id)[0]
    assert edited.current_text == "Engineered a Python data pipeline cutting latency 30%"
    # The true original survives the edit (grounding + restore depend on this).
    assert edited.original_text == "Built a Python data pipeline"
    # Same logical bullet across the copy — the lineage carries, the row id does not.
    assert edited.lineage_id == target.lineage_id
    assert edited.id != target.id


def test_create_child_rejects_edit_for_foreign_bullet(session):
    _, v1 = _seed_original(session)
    with pytest.raises(ValueError, match="not in parent version"):
        create_child_version(
            session,
            parent_version_id=v1.id,
            version_type=VersionType.accepted,
            edits={"not-a-real-bullet-id": "x"},
        )


# --- acceptance: accept → customize → reject -------------------------------------


def test_accept_customize_reject_leaves_navigable_chain(session):
    resume, v1 = _seed_original(session)

    # Accept: edit bullet 0 → accepted version v2.
    b0 = repo.list_bullets(session, v1.id)[0]
    v2 = create_child_version(
        session,
        parent_version_id=v1.id,
        version_type=VersionType.accepted,
        edits={b0.lineage_id: "Accepted text"},
    )
    session.commit()

    # Customize: edit a bullet of v2 → customized version v3.
    c1 = repo.list_bullets(session, v2.id)[1]
    v3 = create_child_version(
        session,
        parent_version_id=v2.id,
        version_type=VersionType.customized,
        edits={c1.lineage_id: "Customized text"},
    )
    session.commit()

    # Reject: revert to the latest accepted version → v2 (v3 is NOT discarded).
    reverted = latest_accepted_version(session, resume.id)
    assert reverted is not None
    assert reverted.id == v2.id

    # The chain root → v3 is navigable and ordered.
    chain = version_chain(session, v3.id)
    assert [v.id for v in chain] == [v1.id, v2.id, v3.id]
    assert [v.version_type for v in chain] == ["original", "accepted", "customized"]


# --- restore original + rollback -------------------------------------------------


def test_restore_original_returns_byte_identical_bullets(session):
    resume, v1 = _seed_original(session)
    # Walk through a couple of edits so the head is far from the original.
    b0 = repo.list_bullets(session, v1.id)[0]
    v2 = create_child_version(
        session,
        parent_version_id=v1.id,
        version_type=VersionType.accepted,
        edits={b0.lineage_id: "changed"},
    )
    create_child_version(session, parent_version_id=v2.id, version_type=VersionType.customized)
    session.commit()

    original = restore_original(session, resume.id)
    assert original.id == v1.id
    restored = [(b.section, b.current_text) for b in repo.list_bullets(session, original.id)]
    assert restored == ORIGINAL_BULLETS  # byte-identical to the parsed resume


def test_rollback_to_returns_requested_version(session):
    resume, v1 = _seed_original(session)
    v2 = create_child_version(session, parent_version_id=v1.id, version_type=VersionType.accepted)
    session.commit()
    assert rollback_to(session, v2.id).id == v2.id
    with pytest.raises(ValueError, match="unknown version"):
        rollback_to(session, "nope")


# --- branch-aware reject ---------------------------------------------------------


def test_latest_accepted_ancestor_stays_on_current_branch(session):
    resume, v1 = _seed_original(session)
    # Branch A: v1 → v2 (accepted) → v3 (customized).
    v2 = create_child_version(session, parent_version_id=v1.id, version_type=VersionType.accepted)
    v3 = create_child_version(session, parent_version_id=v2.id, version_type=VersionType.customized)
    session.commit()
    # Branch B: a *newer* accepted version off the original — unrelated to v3's lineage.
    v4 = create_child_version(session, parent_version_id=v1.id, version_type=VersionType.accepted)
    session.commit()

    # Reject from v3 must revert to v2 (its ancestor), NOT the newer v4 on another branch.
    assert latest_accepted_ancestor(session, v3.id).id == v2.id
    # The resume-global lookup, by contrast, returns the newest accepted anywhere (v4).
    assert latest_accepted_version(session, resume.id).id == v4.id
    # A version that is itself accepted is its own nearest accepted ancestor.
    assert latest_accepted_ancestor(session, v4.id).id == v4.id


def test_latest_accepted_ancestor_none_when_no_accepted_in_lineage(session):
    _, v1 = _seed_original(session)
    v2 = create_child_version(session, parent_version_id=v1.id, version_type=VersionType.customized)
    session.commit()
    assert latest_accepted_ancestor(session, v2.id) is None  # caller falls back to original


# --- stable lineage across version changes ---------------------------------------


def test_lineage_is_stable_across_copies(session):
    _, v1 = _seed_original(session)
    v2 = create_child_version(session, parent_version_id=v1.id, version_type=VersionType.accepted)
    session.commit()

    src = bullets_by_lineage(session, v1.id)
    dst = bullets_by_lineage(session, v2.id)
    assert set(src) == set(dst)  # same lineage ids on both versions
    for lineage, parent_bullet in src.items():
        assert dst[lineage].id != parent_bullet.id  # different row, same lineage
        assert (dst[lineage].section, dst[lineage].order_index) == (
            parent_bullet.section,
            parent_bullet.order_index,
        )


def test_one_lineage_cannot_appear_twice_in_a_version(session):
    # The (resume_version_id, lineage_id) unique constraint guards the copy path against
    # accidentally duplicating a logical bullet within one version snapshot.
    _, v1 = _seed_original(session)
    existing = repo.list_bullets(session, v1.id)[0]
    with pytest.raises(IntegrityError):
        repo.add_bullet(
            session,
            resume_version_id=v1.id,
            section="experience",
            order_index=99,
            original_text="dup",
            lineage_id=existing.lineage_id,
        )  # repo.add_bullet flushes, so the constraint trips here


def test_stable_lineage_lets_second_accept_apply_without_remap(session):
    # The earlier hazard, now gone: a suggestion's lineage_id is the SAME on every version,
    # so a second accept from the original payload applies directly — no id remapping.
    _, v1 = _seed_original(session)
    b0, b1 = repo.list_bullets(session, v1.id)[:2]

    # Accept the first suggestion (by lineage) → v2.
    v2 = create_child_version(
        session,
        parent_version_id=v1.id,
        version_type=VersionType.accepted,
        edits={b0.lineage_id: "first"},
    )
    session.commit()

    # The second suggestion still carries b1's lineage_id, which v2 also has → applies cleanly.
    v3 = create_child_version(
        session,
        parent_version_id=v2.id,
        version_type=VersionType.accepted,
        edits={b1.lineage_id: "second"},
    )
    session.commit()
    texts = [b.current_text for b in repo.list_bullets(session, v3.id)]
    assert "first" in texts and "second" in texts
