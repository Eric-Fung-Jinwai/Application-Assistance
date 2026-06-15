"""Phase 4 acceptance: embedding a parsed resume populates resume_bullets; a
nearest-neighbour query by text returns the seeded bullet."""

from __future__ import annotations

import hashlib

import pytest

from career_assistant.domain import ExperienceItem, ParsedResume, ProjectItem, VersionType
from career_assistant.kb import build_resume_kb, bullets_from_parsed, query_bullets
from career_assistant.llm.client import EmbeddingClient
from career_assistant.storage import chroma, repo
from career_assistant.storage.db import make_engine, make_session_factory


class FakeEmbedder(EmbeddingClient):
    """Deterministic bag-of-words hashing embedder — identical text → identical vector,
    so an exact-text query is its own nearest neighbour. No model download."""

    DIM = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.DIM] += 1.0
        if not any(v):  # cosine is undefined for the zero vector
            v[0] = 1.0
        return v


PARSED = ParsedResume(
    experience=[
        ExperienceItem(
            title="Software Engineer",
            company="Acme",
            bullets=[
                "Built a Python data pipeline that cut latency by thirty percent",
                "Led migration of services to Docker and Kubernetes",
                "   ",  # blank → must be skipped
            ],
        )
    ],
    projects=[ProjectItem(name="Sidecar", bullets=["Designed a Rust caching sidecar"])],
)


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    repo.create_db(engine)
    with make_session_factory(engine)() as s:
        yield s


@pytest.fixture
def collection(tmp_path):
    client = chroma.get_client(str(tmp_path / "chroma"))
    return chroma.get_collection(client, chroma.RESUME_BULLETS)


def test_bullets_from_parsed_orders_and_skips_blanks():
    pairs = bullets_from_parsed(PARSED)
    assert pairs == [
        ("experience", "Built a Python data pipeline that cut latency by thirty percent"),
        ("experience", "Led migration of services to Docker and Kubernetes"),
        ("projects", "Designed a Rust caching sidecar"),
    ]


def test_build_kb_populates_and_queries(session, collection):
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    version = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    embedder = FakeEmbedder()

    ids = build_resume_kb(
        session,
        resume_version_id=version.id,
        parsed=PARSED,
        embedder=embedder,
        collection=collection,
    )
    session.commit()

    # 3 non-blank bullets persisted to SQLite and embedded into Chroma.
    assert len(ids) == 3
    assert collection.count() == 3
    persisted = repo.list_bullets(session, version.id)
    assert [b.id for b in persisted] == ids

    # Nearest-neighbour query by a bullet's own text returns that bullet first.
    target = persisted[1]  # the Docker/Kubernetes bullet
    res = query_bullets(target.current_text, embedder=embedder, collection=collection, n_results=3)
    assert res["ids"][0][0] == target.id

    # Metadata carries the join keys back to SQLite.
    meta = res["metadatas"][0][0]
    assert meta["bullet_id"] == target.id
    assert meta["resume_version_id"] == version.id
    assert meta["lineage_id"] == target.lineage_id
    assert meta["section"] == "experience"


def test_query_scoped_to_resume_version(session, collection):
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    v1 = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    v2 = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    embedder = FakeEmbedder()
    for v in (v1, v2):
        build_resume_kb(
            session,
            resume_version_id=v.id,
            parsed=PARSED,
            embedder=embedder,
            collection=collection,
        )
    session.commit()

    res = query_bullets(
        "Docker and Kubernetes",
        embedder=embedder,
        collection=collection,
        n_results=5,
        resume_version_id=v1.id,
    )
    returned_versions = {m["resume_version_id"] for m in res["metadatas"][0]}
    assert returned_versions == {v1.id}


def test_build_kb_commits_truth_before_chroma(session, collection):
    # Source of truth is committed before vectors are written, so a later caller
    # rollback cannot leave Chroma vectors orphaned from SQLite rows.
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    version = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    ids = build_resume_kb(
        session,
        resume_version_id=version.id,
        parsed=PARSED,
        embedder=FakeEmbedder(),
        collection=collection,
    )
    session.rollback()  # nothing to undo — build_resume_kb already committed

    persisted = repo.list_bullets(session, version.id)
    assert len(persisted) == 3
    assert collection.count() == 3
    assert {b.id for b in persisted} == set(ids)


def test_build_kb_idempotent(session, collection):
    # Re-indexing an (immutable) version re-embeds existing bullets, never duplicates.
    resume = repo.create_resume(session, filename="cv.pdf", raw_text="x")
    version = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    embedder = FakeEmbedder()
    kw = dict(resume_version_id=version.id, parsed=PARSED, embedder=embedder, collection=collection)

    ids1 = build_resume_kb(session, **kw)
    ids2 = build_resume_kb(session, **kw)

    assert ids1 == ids2  # same bullet IDs, not regenerated
    assert len(repo.list_bullets(session, version.id)) == 3  # not 6
    assert collection.count() == 3  # not 6
