"""Phase 5 acceptance: a JD parses into the 7 fields; the coverage report lists
matched vs missing required skills. Also covers JD-chunk embedding into jd_chunks."""

from __future__ import annotations

import hashlib

import pytest

from career_assistant.domain import (
    CoverageReport,
    ExperienceItem,
    ParsedJD,
    ParsedResume,
    ProjectItem,
)
from career_assistant.jd import (
    analyze_coverage,
    analyze_jd,
    build_jd_kb,
    chunks_from_parsed,
    query_jd_chunks,
)
from career_assistant.jd.chunks import PREFERRED_SKILL, REQUIRED_SKILL, chunk_id
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.storage import chroma


class FakeEmbedder(EmbeddingClient):
    """Deterministic bag-of-words hashing embedder — identical text → identical vector.

    A tiny synonym map lets a single token stand in for its canonical form (``k8s`` →
    ``kubernetes``), so the *non-literal* embedding-overlap path is exercisable without a
    real model: ``k8s`` and a bullet mentioning ``Kubernetes`` land on the same bucket.
    """

    DIM = 64
    SYNONYMS = {"k8s": "kubernetes"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for raw in text.lower().split():
            tok = self.SYNONYMS.get(raw, raw)
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.DIM] += 1.0
        if not any(v):
            v[0] = 1.0
        return v


class FakeLLM(LLMClient):
    """Returns a canned structured extraction without hitting the network."""

    def __init__(self, payload: dict | str) -> None:
        self.payload = payload

    def complete(self, system, user, *, json_schema=None, temperature=0.2):
        return self.payload


_GOOD_JD = {
    "required_skills": ["Python", "Kubernetes", "SQL"],
    "preferred_skills": ["Rust", "GraphQL"],
    "responsibilities": ["Build data pipelines", "Own service reliability"],
    "tools": ["Docker", "Airflow"],
    "domain": "ML infrastructure",
    "seniority": "senior",
    "years_experience": 5,
}

RESUME = ParsedResume(
    skills=["Python", "SQL", "Docker"],
    experience=[
        ExperienceItem(
            title="Software Engineer",
            company="Acme",
            bullets=[
                "Built a Python data pipeline that cut latency by thirty percent",
                "Led migration of services to Docker and Kubernetes",
            ],
        )
    ],
    projects=[ProjectItem(name="Sidecar", bullets=["Designed a Rust caching sidecar"])],
)


# --- extraction ------------------------------------------------------------------


def test_analyze_jd_parses_seven_fields():
    parsed = analyze_jd("some JD text", llm=FakeLLM(_GOOD_JD))
    assert isinstance(parsed, ParsedJD)
    assert parsed.required_skills == ["Python", "Kubernetes", "SQL"]
    assert parsed.preferred_skills == ["Rust", "GraphQL"]
    assert parsed.responsibilities == ["Build data pipelines", "Own service reliability"]
    assert parsed.tools == ["Docker", "Airflow"]
    assert parsed.domain == "ML infrastructure"
    assert parsed.seniority == "senior"
    assert parsed.years_experience == 5


def test_analyze_jd_salvages_partial_on_bad_field():
    # responsibilities malformed (not a list of strings) → dropped; rest survives.
    payload = {**_GOOD_JD, "responsibilities": [{"not": "a string"}]}
    parsed = analyze_jd("x", llm=FakeLLM(payload))
    assert parsed.required_skills == ["Python", "Kubernetes", "SQL"]  # kept
    assert parsed.responsibilities == []  # dropped, defaulted
    assert parsed.seniority == "senior"  # kept


def test_analyze_jd_non_json_string_is_safe():
    parsed = analyze_jd("x", llm=FakeLLM("totally not json"))
    assert parsed.required_skills == []
    assert parsed.responsibilities == []


# --- JD chunk embedding ----------------------------------------------------------


def test_chunks_from_parsed_ids_and_types():
    jd = ParsedJD(
        id="jd1",
        required_skills=["Python", "  "],  # blank skipped
        preferred_skills=["Rust"],
    )
    triples = chunks_from_parsed(jd)
    assert triples == [
        (chunk_id("jd1", REQUIRED_SKILL, 0), REQUIRED_SKILL, "Python"),
        (chunk_id("jd1", PREFERRED_SKILL, 0), PREFERRED_SKILL, "Rust"),
    ]


@pytest.fixture
def jd_collection(tmp_path):
    client = chroma.get_client(str(tmp_path / "chroma"))
    return chroma.get_collection(client, chroma.JD_CHUNKS)


def test_build_jd_kb_populates_and_queries(jd_collection):
    jd = ParsedJD.model_validate(_GOOD_JD)
    embedder = FakeEmbedder()

    ids = build_jd_kb(jd, embedder=embedder, collection=jd_collection)

    # 3 required + 2 preferred + 2 responsibilities + 2 tools = 9 chunks.
    assert len(ids) == 9
    assert jd_collection.count() == 9

    res = query_jd_chunks("Kubernetes", embedder=embedder, collection=jd_collection, n_results=1)
    top_id = res["ids"][0][0]
    assert top_id == chunk_id(jd.id, REQUIRED_SKILL, 1)  # the Kubernetes required skill
    assert res["metadatas"][0][0] == {"jd_id": jd.id, "requirement_type": REQUIRED_SKILL}


def test_build_jd_kb_idempotent(jd_collection):
    jd = ParsedJD.model_validate(_GOOD_JD)
    embedder = FakeEmbedder()
    ids1 = build_jd_kb(jd, embedder=embedder, collection=jd_collection)
    ids2 = build_jd_kb(jd, embedder=embedder, collection=jd_collection)
    assert ids1 == ids2  # deterministic ids → replaced in place
    assert jd_collection.count() == 9  # not 18


def test_build_jd_kb_keys_on_authoritative_jd_id(jd_collection):
    # The SQLite jds.id (passed as jd_id) — not the in-memory parsed.id — is what the
    # chunks are keyed/filtered under, so a query scoped to the DB id finds them.
    jd = ParsedJD.model_validate(_GOOD_JD)
    sql_id = "sqlite-jd-row-id"
    assert sql_id != jd.id

    ids = build_jd_kb(jd, embedder=FakeEmbedder(), collection=jd_collection, jd_id=sql_id)
    assert all(cid.startswith(f"{sql_id}::") for cid in ids)

    res = query_jd_chunks(
        "Kubernetes", embedder=FakeEmbedder(), collection=jd_collection, jd_id=sql_id, n_results=1
    )
    assert res["metadatas"][0][0]["jd_id"] == sql_id
    # The stale in-memory id matches nothing.
    miss = query_jd_chunks(
        "Kubernetes", embedder=FakeEmbedder(), collection=jd_collection, jd_id=jd.id, n_results=1
    )
    assert miss["ids"][0] == []


def test_build_jd_kb_prunes_stale_chunks_on_reindex(jd_collection):
    # Re-indexing a JD that shrank (3 required skills → 1) must drop the orphaned chunks,
    # not leave phantom requirements behind in jd_chunks.
    embedder = FakeEmbedder()
    sql_id = "jd-1"
    big = ParsedJD(required_skills=["Python", "Kubernetes", "SQL"])
    build_jd_kb(big, embedder=embedder, collection=jd_collection, jd_id=sql_id)
    assert jd_collection.count() == 3

    small = ParsedJD(required_skills=["Python"])
    ids = build_jd_kb(small, embedder=embedder, collection=jd_collection, jd_id=sql_id)
    assert ids == [chunk_id(sql_id, REQUIRED_SKILL, 0)]
    assert jd_collection.count() == 1  # ::1 and ::2 pruned

    # Re-parsing to zero requirements clears the JD's chunks entirely.
    empty = ParsedJD(required_skills=[])
    assert build_jd_kb(empty, embedder=embedder, collection=jd_collection, jd_id=sql_id) == []
    assert jd_collection.count() == 0


# --- coverage --------------------------------------------------------------------


def test_coverage_lists_matched_and_missing_required():
    # Python + SQL hit declared skills; Kubernetes is only in a bullet but the keyword
    # scan covers bullet text too; Terraform appears nowhere → missing.
    jd = ParsedJD(required_skills=["Python", "Kubernetes", "SQL", "Terraform"])
    report = analyze_coverage(jd, RESUME, embedder=FakeEmbedder())
    assert isinstance(report, CoverageReport)

    assert report.matched_required_skills == ["Python", "Kubernetes", "SQL"]
    assert report.missing_required_skills == ["Terraform"]
    assert report.required_coverage == 0.75


def test_coverage_embedding_fallback_matches_synonym():
    # "k8s" never appears literally, so the keyword scan misses it; only the embedding
    # overlap (k8s ≈ Kubernetes) can match it against the declared "Kubernetes" skill.
    resume = ParsedResume(skills=["Kubernetes"])
    jd = ParsedJD(required_skills=["k8s"])

    keyword_only = analyze_coverage(jd, resume, embedder=None)
    assert keyword_only.missing_required_skills == ["k8s"]

    with_embeddings = analyze_coverage(jd, resume, embedder=FakeEmbedder())
    assert with_embeddings.matched_required_skills == ["k8s"]
    assert with_embeddings.required_coverage == 1.0


def test_coverage_preferred_and_empty_required():
    jd = ParsedJD(preferred_skills=["Rust", "GraphQL"])
    report = analyze_coverage(jd, RESUME, embedder=FakeEmbedder())
    # Rust appears literally in a project bullet (keyword); GraphQL appears nowhere.
    assert report.matched_preferred_skills == ["Rust"]
    assert report.missing_preferred_skills == ["GraphQL"]
    assert report.preferred_coverage == 0.5
    # No required skills listed → fully covered by convention.
    assert report.required_coverage == 1.0


def test_coverage_domain_match():
    matched = analyze_coverage(ParsedJD(domain="Docker"), RESUME, embedder=FakeEmbedder())
    assert matched.domain_match is True
    none = analyze_coverage(ParsedJD(domain=None), RESUME, embedder=FakeEmbedder())
    assert none.domain_match is None
