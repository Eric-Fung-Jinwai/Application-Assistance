"""Phase 10 acceptance: accept / reject / customize each update the DB + version tree
correctly, and customize-via-chat returns a re-scored suggestion. Every action logs to
edit_history; integrity is re-judged for unvetted edits (direct + chat), never for plain accept."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import select

from career_assistant.domain import IntegrityBand, SuggestionStatus, VersionType
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.review import (
    accept_suggestion,
    build_review_payload,
    customize_suggestion,
    refine_suggestion,
    reject_suggestion,
)
from career_assistant.storage import repo
from career_assistant.storage.db import make_engine, make_session_factory
from career_assistant.storage.models import EditHistory, TailoringSuggestionRow
from career_assistant.versioning import bullets_by_lineage, version_chain

ORIGINAL_BULLETS = [
    ("experience", "Built a Python data pipeline"),
    ("experience", "Led migration to Docker"),
]


class FakeEmbedder(EmbeddingClient):
    """Deterministic bag-of-words embedder — shared tokens → cosine overlap."""

    DIM = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.DIM
            for tok in text.lower().split():
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
            if not any(v):
                v[0] = 1.0
            out.append(v)
        return out


class RoutingFakeLLM(LLMClient):
    """Judge payload for the integrity auditor prompt, else the rewrite/refine payload."""

    def __init__(self, *, rewrite: dict | str | None = None, judge: dict | str | None = None):
        self.rewrite = rewrite
        self.judge = judge

    def complete(self, system, user, *, json_schema=None, temperature=0.2):
        return self.judge if "auditor" in system else self.rewrite


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    repo.create_db(engine)
    with make_session_factory(engine)() as s:
        yield s


def _seed(session):
    """Resume + original version + two bullets + one pending suggestion on bullet 0."""
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    v1 = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    bullets = [
        repo.add_bullet(
            session, resume_version_id=v1.id, section=sec, order_index=i, original_text=text
        )
        for i, (sec, text) in enumerate(ORIGINAL_BULLETS)
    ]
    jd = repo.create_jd(session, raw_text="JD text")
    suggestion = repo.create_suggestion(
        session,
        bullet_id=bullets[0].id,
        jd_id=jd.id,
        suggested_text="Engineered a Python data pipeline cutting batch latency",
        reasoning="stronger verb",
        llm_judgment=0.9,
        embedding_similarity=0.8,
        integrity_score=86.0,
        integrity_band=IntegrityBand.moderate,
        status=SuggestionStatus.pending,
    )
    session.commit()
    return resume, v1, bullets, suggestion


# --- review payload --------------------------------------------------------------


def test_build_review_payload_carries_original_and_integrity(session):
    _, _, bullets, suggestion = _seed(session)
    item = build_review_payload(session, suggestion_id=suggestion.id)
    assert item.original_text == "Built a Python data pipeline"
    assert item.suggested_text == suggestion.suggested_text
    assert item.integrity_band == IntegrityBand.moderate.value
    assert item.integrity_score == 86.0
    assert item.status == "pending"


# --- accept ----------------------------------------------------------------------


def test_accept_applies_edit_and_creates_accepted_version(session):
    _, v1, bullets, suggestion = _seed(session)
    outcome = accept_suggestion(session, suggestion_id=suggestion.id, base_version_id=v1.id)

    assert outcome.created_version
    assert outcome.version.version_type == VersionType.accepted.value
    assert outcome.version.parent_version_id == v1.id
    # The edited text lands on the same lineage in the new version.
    edited = bullets_by_lineage(session, outcome.version.id)[bullets[0].lineage_id]
    assert edited.current_text == suggestion.suggested_text
    assert edited.original_text == "Built a Python data pipeline"  # evidence preserved
    # Status persisted + edit logged.
    assert session.get(TailoringSuggestionRow, suggestion.id).status == "accepted"
    log = session.scalars(select(EditHistory)).all()
    assert [e.action for e in log] == ["accept"]
    assert log[0].from_text == "Built a Python data pipeline"
    assert log[0].to_text == suggestion.suggested_text


def test_accept_defaults_base_to_suggestions_own_version(session):
    _, v1, bullets, suggestion = _seed(session)
    outcome = accept_suggestion(session, suggestion_id=suggestion.id)
    assert outcome.version.parent_version_id == v1.id


def test_sequential_accepts_chain_when_head_is_threaded(session):
    # Two suggestions accepted in a row, threading the new head → a single chain, no branch.
    resume, v1, bullets, s0 = _seed(session)
    s1 = repo.create_suggestion(
        session,
        bullet_id=bullets[1].id,
        jd_id=s0.jd_id,
        suggested_text="Drove the Docker migration",
        status=SuggestionStatus.pending,
        integrity_band=IntegrityBand.safe,
    )
    session.commit()

    v2 = accept_suggestion(session, suggestion_id=s0.id, base_version_id=v1.id).version
    v3 = accept_suggestion(session, suggestion_id=s1.id, base_version_id=v2.id).version

    chain = version_chain(session, v3.id)
    assert [v.id for v in chain] == [v1.id, v2.id, v3.id]
    # Both edits coexist on the head — the lineage made the second accept apply with no remap.
    head_texts = {b.current_text for b in repo.list_bullets(session, v3.id)}
    assert "Engineered a Python data pipeline cutting batch latency" in head_texts
    assert "Drove the Docker migration" in head_texts


def test_accept_reembeds_new_version_when_collection_given(session, tmp_path):
    from career_assistant.kb.embeddings import query_bullets
    from career_assistant.storage import chroma

    _, v1, bullets, suggestion = _seed(session)
    client = chroma.get_client(str(tmp_path / "chroma"))
    collection = chroma.get_collection(client, chroma.RESUME_BULLETS)
    embedder = FakeEmbedder()

    outcome = accept_suggestion(
        session,
        suggestion_id=suggestion.id,
        base_version_id=v1.id,
        embedder=embedder,
        collection=collection,
    )
    # The new version's bullets are searchable in Chroma scoped to that version.
    res = query_bullets(
        "Python data pipeline",
        embedder=embedder,
        collection=collection,
        resume_version_id=outcome.version.id,
    )
    assert res["ids"][0]  # at least one indexed bullet for the new head


# --- reject ----------------------------------------------------------------------


def test_reject_reverts_to_latest_accepted_on_branch(session):
    _, v1, bullets, suggestion = _seed(session)
    # Accept first → v2 (accepted). A later pending suggestion is then rejected.
    v2 = accept_suggestion(session, suggestion_id=suggestion.id, base_version_id=v1.id).version
    s2 = repo.create_suggestion(
        session,
        bullet_id=bullets[1].id,
        jd_id=suggestion.jd_id,
        suggested_text="Owned a fleet of 50 services",
        status=SuggestionStatus.pending,
        integrity_band=IntegrityBand.high_risk,
    )
    session.commit()

    outcome = reject_suggestion(session, suggestion_id=s2.id, base_version_id=v2.id)
    assert not outcome.created_version
    assert outcome.version.id == v2.id  # reverted to the latest accepted version
    assert session.get(TailoringSuggestionRow, s2.id).status == "rejected"
    actions = [e.action for e in session.scalars(select(EditHistory))]
    assert actions[-1] == "reject"


def test_reject_falls_back_to_original_when_no_accepted_yet(session):
    _, v1, bullets, suggestion = _seed(session)
    outcome = reject_suggestion(session, suggestion_id=suggestion.id, base_version_id=v1.id)
    assert outcome.version.id == v1.id  # the original
    assert session.get(TailoringSuggestionRow, suggestion.id).status == "rejected"


# --- customize: direct edit ------------------------------------------------------


def test_customize_direct_edit_rescored_and_versioned(session):
    _, v1, bullets, suggestion = _seed(session)
    # The reviewer's own text introduces a fabricated team → must be re-judged as high-risk.
    llm = RoutingFakeLLM(judge={"grounded": 0.1, "unsupported_claims": ["led a team of 12"]})
    outcome = customize_suggestion(
        session,
        suggestion_id=suggestion.id,
        new_text="Led a team of 12 on a Python data pipeline",
        base_version_id=v1.id,
        llm=llm,
        embedder=FakeEmbedder(),
    )

    assert outcome.version.version_type == VersionType.customized.value
    row = session.get(TailoringSuggestionRow, suggestion.id)
    assert row.status == "customized"
    assert row.suggested_text == "Led a team of 12 on a Python data pipeline"
    # Integrity was recomputed for the unvetted edit, not inherited from the original 86.0.
    assert row.integrity_band == IntegrityBand.high_risk.value
    assert row.integrity_score < 50
    assert row.integrity_flags_json == {"unsupported_claims": ["led a team of 12"]}
    edited = bullets_by_lineage(session, outcome.version.id)[bullets[0].lineage_id]
    assert edited.current_text == "Led a team of 12 on a Python data pipeline"


# --- customize: chat refinement --------------------------------------------------


def test_refine_returns_rescored_suggestion_without_versioning(session):
    _, _, _, suggestion = _seed(session)
    versions_before = session.scalars(select(EditHistory)).all()
    llm = RoutingFakeLLM(
        rewrite={"suggested_text": "Engineered a Python data pipeline", "reasoning": "shorter"},
        judge={"grounded": 0.95},
    )
    refined = refine_suggestion(
        session,
        suggestion_id=suggestion.id,
        instruction="make it shorter",
        llm=llm,
        embedder=FakeEmbedder(),
    )

    # Returned a re-scored suggestion, still pending (no version/edit-history created).
    assert refined.suggested_text == "Engineered a Python data pipeline"
    assert refined.status == SuggestionStatus.pending
    assert refined.integrity is not None
    assert refined.integrity.band != IntegrityBand.high_risk
    assert len(session.scalars(select(EditHistory)).all()) == len(versions_before)
    # The persisted row reflects the refined text + fresh score.
    row = session.get(TailoringSuggestionRow, suggestion.id)
    assert row.suggested_text == "Engineered a Python data pipeline"
    assert row.status == "pending"


def test_refine_then_accept_applies_refined_text(session):
    _, v1, bullets, suggestion = _seed(session)
    llm = RoutingFakeLLM(
        rewrite={"suggested_text": "Streamlined a Python data pipeline", "reasoning": "tighter"},
        judge={"grounded": 0.95},
    )
    refine_suggestion(
        session,
        suggestion_id=suggestion.id,
        instruction="make it shorter",
        llm=llm,
        embedder=FakeEmbedder(),
    )
    outcome = accept_suggestion(session, suggestion_id=suggestion.id, base_version_id=v1.id)
    edited = bullets_by_lineage(session, outcome.version.id)[bullets[0].lineage_id]
    assert edited.current_text == "Streamlined a Python data pipeline"


def test_refine_falls_back_to_current_text_on_garbled_llm(session):
    _, _, _, suggestion = _seed(session)
    original_suggested = suggestion.suggested_text
    llm = RoutingFakeLLM(rewrite="not json", judge={"grounded": 0.9})
    refined = refine_suggestion(
        session,
        suggestion_id=suggestion.id,
        instruction="make it shorter",
        llm=llm,
        embedder=FakeEmbedder(),
    )
    assert refined.suggested_text == original_suggested  # safe no-op, nothing unvetted emitted


# --- error handling --------------------------------------------------------------


def test_actions_raise_on_unknown_suggestion(session):
    _, v1, _, _ = _seed(session)
    with pytest.raises(ValueError, match="unknown suggestion"):
        accept_suggestion(session, suggestion_id="nope", base_version_id=v1.id)


def test_decided_suggestion_cannot_be_re_reviewed(session):
    # Once accepted, a suggestion is terminal — no re-accept (forking versions) or reopen.
    _, v1, _, suggestion = _seed(session)
    accept_suggestion(session, suggestion_id=suggestion.id, base_version_id=v1.id)
    for action in (accept_suggestion, reject_suggestion):
        with pytest.raises(ValueError, match="not pending"):
            action(session, suggestion_id=suggestion.id, base_version_id=v1.id)
    with pytest.raises(ValueError, match="not pending"):
        customize_suggestion(
            session, suggestion_id=suggestion.id, new_text="x", base_version_id=v1.id
        )
    with pytest.raises(ValueError, match="not pending"):
        refine_suggestion(session, suggestion_id=suggestion.id, instruction="shorter")


def test_reject_rejects_base_from_a_foreign_branch(session):
    # A base version whose lineage doesn't contain the suggestion's bullet must be refused,
    # so reject can't mark this suggestion rejected and hand back an unrelated branch's version.
    _, _, _, suggestion = _seed(session)
    other_resume = repo.create_resume(session, filename="other.pdf", raw_text="y")
    other_v1 = repo.create_version(
        session, resume_id=other_resume.id, version_type=VersionType.original
    )
    repo.add_bullet(
        session,
        resume_version_id=other_v1.id,
        section="experience",
        order_index=0,
        original_text="z",
    )
    session.commit()
    with pytest.raises(ValueError, match="not in base version"):
        reject_suggestion(session, suggestion_id=suggestion.id, base_version_id=other_v1.id)
