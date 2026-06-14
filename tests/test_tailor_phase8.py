"""Phase 8 acceptance: a planted fabrication ("led team of 10" with no team evidence)
scores < 50 and flags an unsupported claim. Also covers rewrite guardrails, the integrity
judge's conservative failure mode, candidate mapping, and the persisted tailoring pass."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import select

from career_assistant.domain import (
    ExperienceItem,
    IntegrityBand,
    ParsedJD,
    ParsedResume,
    SuggestionStatus,
    VersionType,
)
from career_assistant.kb import build_resume_kb
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.storage import chroma, repo
from career_assistant.storage.db import make_engine, make_session_factory
from career_assistant.storage.models import TailoringSuggestionRow
from career_assistant.tailor import (
    generate_suggestions,
    judge_integrity,
    rewrite_bullet,
    select_candidate_bullets,
)
from career_assistant.tailor.tailor import _jd_requirements


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
    """Returns the judge payload for the integrity auditor prompt, else the rewrite payload.
    Either may be a dict, a raw string (to exercise salvage), or None if not expected."""

    def __init__(self, *, rewrite: dict | str | None = None, judge: dict | str | None = None):
        self.rewrite = rewrite
        self.judge = judge

    def complete(self, system, user, *, json_schema=None, temperature=0.2):
        return self.judge if "auditor" in system else self.rewrite


# --- integrity judge (acceptance) ------------------------------------------------


def test_planted_fabrication_scores_below_50_and_flags_claim():
    judge_payload = {
        "grounded": 0.1,
        "unsupported_claims": ["led a team of 10 engineers"],
        "unsupported_responsibilities": ["managing a team"],
    }
    result = judge_integrity(
        "Contributed to the backend team's services",
        "Led a team of 10 engineers building backend services",
        llm=RoutingFakeLLM(judge=judge_payload),
        embedder=FakeEmbedder(),
    )
    assert result.score < 50
    assert result.band == IntegrityBand.high_risk
    assert result.unsupported_claims == ["led a team of 10 engineers"]


def test_unsupported_flags_force_high_risk_even_when_grounded_high():
    # Defense in depth: a high `grounded` score must not let a named fabrication read safe.
    result = judge_integrity(
        "Contributed to the backend team",
        "Led a team of 10 engineers",
        llm=RoutingFakeLLM(judge={"grounded": 0.95, "unsupported_claims": ["led a team of 10"]}),
        embedder=FakeEmbedder(),
    )
    assert result.unsupported_claims == ["led a team of 10"]
    assert result.band == IntegrityBand.high_risk
    assert result.score < 50


def test_faithful_rewrite_scores_high_and_has_no_flags():
    result = judge_integrity(
        "Reduced API latency by 30% via query optimization",
        "Optimized database queries to cut API latency by 30%",
        llm=RoutingFakeLLM(judge={"grounded": 0.95}),
        embedder=FakeEmbedder(),
    )
    assert result.band != IntegrityBand.high_risk
    assert result.score > 70
    assert result.unsupported_claims == []
    assert result.unsupported_metrics == []


def test_judge_is_conservative_on_garbled_response():
    # An unreadable judge response must NOT pass — grounded defaults to 0 → high_risk.
    result = judge_integrity(
        "Built REST APIs",
        "Architected planet-scale distributed systems",
        llm=RoutingFakeLLM(judge="not json at all"),
        embedder=FakeEmbedder(),
    )
    assert result.llm_judgment == 0.0
    assert result.band == IntegrityBand.high_risk


def test_judge_clamps_out_of_range_grounded():
    result = judge_integrity(
        "Built data pipelines",
        "Built robust data pipelines",
        llm=RoutingFakeLLM(judge={"grounded": 1.7}),
        embedder=FakeEmbedder(),
    )
    assert result.llm_judgment == 1.0  # clamped into [0, 1]


# --- rewrite ---------------------------------------------------------------------


def test_rewrite_bullet_returns_text_and_reasoning():
    text, reasoning = rewrite_bullet(
        "Built data pipelines",
        targets=["Python"],
        llm=RoutingFakeLLM(
            rewrite={"suggested_text": "Engineered data pipelines", "reasoning": "stronger verb"}
        ),
    )
    assert text == "Engineered data pipelines"
    assert reasoning == "stronger verb"


def test_rewrite_falls_back_to_original_on_garbled_response():
    text, reasoning = rewrite_bullet(
        "Built data pipelines",
        targets=["Python"],
        llm=RoutingFakeLLM(rewrite="not json"),
    )
    assert text == "Built data pipelines"  # safe no-op, nothing unvetted emitted
    assert reasoning == ""


# --- candidate mapping + full pipeline -------------------------------------------


PARSED = ParsedResume(
    experience=[
        ExperienceItem(
            title="Engineer",
            company="Acme",
            bullets=["Built Python data pipelines", "Designed a Rust caching sidecar"],
        )
    ]
)


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    repo.create_db(engine)
    with make_session_factory(engine)() as s:
        yield s


@pytest.fixture
def resume_collection(tmp_path):
    client = chroma.get_client(str(tmp_path / "chroma"))
    return chroma.get_collection(client, chroma.RESUME_BULLETS)


def _seed_resume(session, collection, embedder):
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    version = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    build_resume_kb(
        session,
        resume_version_id=version.id,
        parsed=PARSED,
        embedder=embedder,
        collection=collection,
    )
    session.commit()
    return version


def test_jd_requirements_include_preferred_skills_and_tools():
    jd = ParsedJD(
        required_skills=["Python"],
        preferred_skills=["Docker"],
        responsibilities=["Ship services"],
        tools=["AWS"],
    )
    assert _jd_requirements(jd) == ["Python", "Docker", "Ship services", "AWS"]


def test_jd_requirements_dedupe_case_insensitively():
    jd = ParsedJD(required_skills=["Python"], tools=["python"])
    assert _jd_requirements(jd) == ["Python"]


def test_select_candidate_bullets_maps_requirement_to_bullet(session, resume_collection):
    embedder = FakeEmbedder()
    version = _seed_resume(session, resume_collection, embedder)
    jd = ParsedJD(required_skills=["Python"])

    mapping = select_candidate_bullets(
        jd, embedder=embedder, collection=resume_collection, resume_version_id=version.id
    )
    bullets = repo.list_bullets(session, version.id)
    python_bullet = next(b for b in bullets if "Python" in b.current_text)
    assert python_bullet.id in mapping
    assert mapping[python_bullet.id] == ["Python"]


def test_generate_suggestions_persists_pending_with_integrity(session, resume_collection):
    embedder = FakeEmbedder()
    version = _seed_resume(session, resume_collection, embedder)
    jd_row = repo.create_jd(session, raw_text="JD text")
    session.commit()

    jd = ParsedJD(
        id=jd_row.id, required_skills=["Python"], responsibilities=["Build data pipelines"]
    )
    llm = RoutingFakeLLM(
        rewrite={
            "suggested_text": "Engineered Python data pipelines",
            "reasoning": "stronger verb",
        },
        judge={"grounded": 0.9},
    )

    suggestions = generate_suggestions(
        session,
        resume_version_id=version.id,
        jd=jd,
        jd_id=jd_row.id,
        embedder=embedder,
        collection=resume_collection,
        llm=llm,
    )
    session.commit()

    assert suggestions
    for s in suggestions:
        assert s.status == SuggestionStatus.pending
        assert s.integrity is not None

    rows = list(session.scalars(select(TailoringSuggestionRow)))
    assert len(rows) == len(suggestions)
    row = rows[0]
    assert row.jd_id == jd_row.id  # authoritative SQLite id, not the in-memory ParsedJD id
    assert row.status == "pending"
    assert row.integrity_score is not None
    assert row.integrity_band is not None


def test_generate_suggestions_flags_fabrication_as_high_risk(session, resume_collection):
    embedder = FakeEmbedder()
    version = _seed_resume(session, resume_collection, embedder)
    jd_row = repo.create_jd(session, raw_text="JD text")
    session.commit()

    jd = ParsedJD(id=jd_row.id, required_skills=["Python"])
    # The rewrite invents scope; the judge catches it → every persisted suggestion is high-risk.
    llm = RoutingFakeLLM(
        rewrite={"suggested_text": "Led a team of 10 building Python platforms", "reasoning": "x"},
        judge={"grounded": 0.1, "unsupported_claims": ["led a team of 10"]},
    )
    generate_suggestions(
        session,
        resume_version_id=version.id,
        jd=jd,
        jd_id=jd_row.id,
        embedder=embedder,
        collection=resume_collection,
        llm=llm,
    )
    session.commit()

    rows = list(session.scalars(select(TailoringSuggestionRow)))
    assert rows
    assert all(r.integrity_band == IntegrityBand.high_risk.value for r in rows)
    assert all(r.integrity_score < 50 for r in rows)
    # Flags survive the round-trip so the Phase 10 review panel can explain the risk.
    assert all(r.integrity_flags_json == {"unsupported_claims": ["led a team of 10"]} for r in rows)
