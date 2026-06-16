"""Phase 14 acceptance: every endpoint has a passing integration test against a temp
SQLite + Chroma, with deterministic fake LLM/embedder injected (no network). Covers the full
happy path (parse → JD → fit → recommendation → tailor → review → export) plus error mapping."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from career_assistant.api import deps
from career_assistant.api.app import create_app
from career_assistant.llm.client import EmbeddingClient, LLMClient

RESUME_TEXT = """Jane Doe
Senior Software Engineer

Experience
Acme (2020 - present) — Senior Software Engineer
- Built Python data pipelines processing millions of events
- Led migration to Docker across the platform

Skills: Python, Docker, AWS
Education: State University, BSc Computer Science
"""

RESUME_JSON = {
    "education": [{"institution": "State University", "degree": "BSc", "field_of_study": "CS"}],
    "experience": [
        {
            "title": "Senior Software Engineer",
            "company": "Acme",
            "start_date": "2020",
            "end_date": "present",
            "bullets": [
                "Built Python data pipelines processing millions of events",
                "Led migration to Docker across the platform",
            ],
        }
    ],
    "projects": [],
    "skills": ["Python", "Docker", "AWS"],
    "certifications": [],
    "confidence": 1.0,
}

JD_JSON = {
    "required_skills": ["Python"],
    "preferred_skills": ["Docker"],
    "responsibilities": ["Build data pipelines"],
    "tools": ["AWS"],
    "domain": "backend",
    "seniority": "senior",
    "years_experience": 5,
}


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
    """Route by the system prompt to the right canned JSON payload (no network)."""

    def complete(self, system, user, *, json_schema=None, temperature=0.2):
        if "resume parser" in system:
            return RESUME_JSON
        if "job-description parser" in system:
            return JD_JSON
        if "auditor" in system:  # integrity judge
            return {"grounded": 0.9}
        if "refine ONE" in system:
            return {"suggested_text": "Engineered Python data pipelines", "reasoning": "tighter"}
        # rewrite
        return {"suggested_text": "Engineered Python data pipelines", "reasoning": "stronger verb"}


def _chromium_available() -> bool:
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        return True
    except Exception:
        return False


requires_chromium = pytest.mark.skipif(
    not _chromium_available(), reason="Chromium not installed — run `playwright install chromium`"
)


@pytest.fixture
def client(tmp_path):
    app = create_app(sqlite_path=str(tmp_path / "api.db"), chroma_path=str(tmp_path / "chroma"))
    app.dependency_overrides[deps.get_llm] = lambda: RoutingFakeLLM()
    app.dependency_overrides[deps.get_embedder] = lambda: FakeEmbedder()
    with TestClient(app) as c:
        yield c


def _parse_resume(client) -> dict:
    resp = client.post(
        "/parse_resume", files={"file": ("cv.txt", RESUME_TEXT.encode(), "text/plain")}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _analyze_jd(client) -> dict:
    resp = client.post("/analyze_jd", json={"text": "We need a senior Python engineer."})
    assert resp.status_code == 200, resp.text
    return resp.json()


# --- per-endpoint ----------------------------------------------------------------


def test_parse_resume(client):
    body = _parse_resume(client)
    assert body["resume_id"] and body["resume_version_id"]
    assert body["parsed"]["skills"] == ["Python", "Docker", "AWS"]


def test_analyze_jd(client):
    body = _analyze_jd(client)
    assert body["jd_id"]
    assert body["parsed"]["required_skills"] == ["Python"]
    assert body["parsed"]["id"] == body["jd_id"]  # parsed id aligned to the SQLite id


def test_analyze_jd_persists_aligned_id(client):
    # The stored parse must carry the authoritative SQLite id, not the random in-memory one.
    from career_assistant.storage.models import JD

    body = _analyze_jd(client)
    with client.app.state.session_factory() as session:
        row = session.get(JD, body["jd_id"])
        assert row.parsed_json["id"] == body["jd_id"]


def test_fit_score(client):
    resume = _parse_resume(client)
    jd = _analyze_jd(client)
    resp = client.post(
        "/fit_score",
        json={"resume_version_id": resume["resume_version_id"], "jd_id": jd["jd_id"]},
    )
    assert resp.status_code == 200, resp.text
    fit = resp.json()
    assert 0.0 <= fit["overall"] <= 100.0
    assert fit["band"] in {"strong", "moderate", "weak"}


def test_recommendation(client):
    resume = _parse_resume(client)
    jd = _analyze_jd(client)
    resp = client.post(
        "/recommendation",
        json={"resume_version_id": resume["resume_version_id"], "jd_id": jd["jd_id"]},
    )
    assert resp.status_code == 200, resp.text
    rec = resp.json()
    assert rec["band"] in {"Strong Match", "Proceed With Caution", "Stretch", "Low Alignment"}
    assert "Python" in rec["strengths"]


def test_tailor_then_review_accept(client):
    resume = _parse_resume(client)
    jd = _analyze_jd(client)
    tailor = client.post(
        "/tailor",
        json={"resume_version_id": resume["resume_version_id"], "jd_id": jd["jd_id"]},
    )
    assert tailor.status_code == 200, tailor.text
    suggestions = tailor.json()["suggestions"]
    assert suggestions, "expected at least one suggestion"
    suggestion_id = suggestions[0]["id"]

    review = client.post("/review", json={"suggestion_id": suggestion_id, "action": "accept"})
    assert review.status_code == 200, review.text
    out = review.json()
    assert out["status"] == "accepted"
    assert out["version"]["created"] is True
    assert out["version"]["version_type"] == "accepted"


def test_accept_all_applies_into_one_version(client):
    resume = _parse_resume(client)
    jd = _analyze_jd(client)
    tailor = client.post(
        "/tailor",
        json={"resume_version_id": resume["resume_version_id"], "jd_id": jd["jd_id"]},
    )
    ids = [s["id"] for s in tailor.json()["suggestions"]]
    assert ids

    resp = client.post("/accept_all", json={"suggestion_ids": ids})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["accepted_count"] == len(ids)
    assert body["version"]["created"] is True
    assert body["version"]["version_type"] == "accepted"


def test_budget_endpoint_reports_cost(client):
    resp = client.get("/budget", params={"tailored": 8})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["llm_model"]
    assert body["items"]  # per-stage line items present
    # Local HF embedder → its embedding line items are free.
    embed_costs = [it["cost"] for it in body["items"] if it["kind"] == "embedding"]
    assert embed_costs and all(c == 0.0 for c in embed_costs)
    assert body["per_application_latency_s"] > 0


def test_llm_client_factory_is_cached():
    # The factory caches clients so the embedding model loads once per process, not per request.
    from career_assistant.llm.factory import get_llm_client

    assert get_llm_client() is get_llm_client()


def test_review_refine_returns_rescored_suggestion(client):
    resume = _parse_resume(client)
    jd = _analyze_jd(client)
    tailor = client.post(
        "/tailor",
        json={"resume_version_id": resume["resume_version_id"], "jd_id": jd["jd_id"]},
    )
    suggestion_id = tailor.json()["suggestions"][0]["id"]
    resp = client.post(
        "/review",
        json={"suggestion_id": suggestion_id, "action": "refine", "instruction": "make it shorter"},
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["suggestion"]["suggested_text"] == "Engineered Python data pipelines"
    assert out["suggestion"]["integrity"] is not None  # re-scored
    assert out["version"] is None  # refine creates no version


def test_generate_html(client):
    resume = _parse_resume(client)
    resp = client.post(
        "/generate_html",
        json={"resume_version_id": resume["resume_version_id"], "template": "modern"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/html")
    assert "Built Python data pipelines processing millions of events" in resp.text


@requires_chromium
def test_generate_pdf(client):
    resume = _parse_resume(client)
    resp = client.post(
        "/generate_pdf",
        json={"resume_version_id": resume["resume_version_id"], "template": "ats"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")


# --- error mapping ---------------------------------------------------------------


def test_unsupported_file_type_returns_415(client):
    resp = client.post(
        "/parse_resume", files={"file": ("cv.xyz", b"some bytes", "application/octet-stream")}
    )
    assert resp.status_code == 415


def test_unknown_jd_returns_404(client):
    resume = _parse_resume(client)
    resp = client.post(
        "/fit_score",
        json={"resume_version_id": resume["resume_version_id"], "jd_id": "nope"},
    )
    assert resp.status_code == 404


def test_unknown_suggestion_review_returns_404(client):
    resp = client.post("/review", json={"suggestion_id": "nope", "action": "accept"})
    assert resp.status_code == 404


def test_customize_without_edit_returns_422(client):
    resume = _parse_resume(client)
    jd = _analyze_jd(client)
    tailor = client.post(
        "/tailor",
        json={"resume_version_id": resume["resume_version_id"], "jd_id": jd["jd_id"]},
    )
    suggestion_id = tailor.json()["suggestions"][0]["id"]
    resp = client.post("/review", json={"suggestion_id": suggestion_id, "action": "customize"})
    assert resp.status_code == 422
