"""Phase 13: the UI's API client drives the full happy path (upload → JD → fit →
recommendation → tailor → review → export) against the ASGI app in-process — same wiring the
browser uses, deterministic fakes, no servers/network. Plus theme-token checks."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from career_assistant.api import deps
from career_assistant.api.app import create_app
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.ui import theme
from career_assistant.ui.api_client import ApiClient, ApiError

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
    def complete(self, system, user, *, json_schema=None, temperature=0.2):
        if "resume parser" in system:
            return RESUME_JSON
        if "job-description parser" in system:
            return JD_JSON
        if "auditor" in system:
            return {"grounded": 0.9}
        return {"suggested_text": "Engineered Python data pipelines", "reasoning": "stronger verb"}


@pytest.fixture
def api_client(tmp_path):
    app = create_app(sqlite_path=str(tmp_path / "ui.db"), chroma_path=str(tmp_path / "chroma"))
    app.dependency_overrides[deps.get_llm] = lambda: RoutingFakeLLM()
    app.dependency_overrides[deps.get_embedder] = lambda: FakeEmbedder()
    # TestClient is an httpx.Client subclass with a sync portal over the ASGI app — the same
    # in-process wiring the API tests use, drop-in for the UI's ApiClient.
    client = ApiClient(client=TestClient(app))
    yield client
    client.close()


def test_full_happy_path_through_client(api_client):
    resume = api_client.parse_resume(filename="cv.txt", content=RESUME_TEXT.encode())
    assert resume["resume_version_id"]

    jd = api_client.analyze_jd("We need a senior Python engineer.")
    assert jd["jd_id"]

    rv, jid = resume["resume_version_id"], jd["jd_id"]

    fit = api_client.fit_score(rv, jid)
    assert 0.0 <= fit["overall"] <= 100.0

    rec = api_client.recommendation(rv, jid)
    assert rec["band"] in {"Strong Match", "Proceed With Caution", "Stretch", "Low Alignment"}

    suggestions = api_client.tailor(rv, jid)
    assert suggestions
    sid = suggestions[0]["id"]

    # Export the original version: it still shows the verbatim parsed bullets.
    original_html = api_client.generate_html(rv, "modern")
    assert "Built Python data pipelines processing millions of events" in original_html

    review = api_client.review(suggestion_id=sid, action="accept")
    assert review["status"] == "accepted"
    head = review["version"]["version_id"]

    # Export the accepted head: it now reflects the applied rewrite.
    head_html = api_client.generate_html(head, "modern")
    assert "Engineered Python data pipelines" in head_html


def test_review_refine_returns_rescored_suggestion(api_client):
    resume = api_client.parse_resume(filename="cv.txt", content=RESUME_TEXT.encode())
    jd = api_client.analyze_jd("Senior Python engineer.")
    suggestions = api_client.tailor(resume["resume_version_id"], jd["jd_id"])
    out = api_client.review(
        suggestion_id=suggestions[0]["id"], action="refine", instruction="make it shorter"
    )
    assert out["version"] is None
    assert out["suggestion"]["integrity"] is not None


def test_api_error_carries_status_and_detail(api_client):
    resume = api_client.parse_resume(filename="cv.txt", content=RESUME_TEXT.encode())
    with pytest.raises(ApiError) as excinfo:
        api_client.fit_score(resume["resume_version_id"], "no-such-jd")
    assert excinfo.value.status_code == 404


# --- theme -----------------------------------------------------------------------


def test_theme_css_carries_design_tokens():
    css = theme.build_css()
    assert theme.COLORS["canvas"] in css  # cream canvas
    assert theme.COLORS["primary"] in css  # coral CTA
    assert "Cormorant Garamond" in css  # serif display substitute
    assert "Inter" in css  # humanist sans body


def test_band_color_maps_integrity_bands():
    assert theme.band_color("safe") == theme.COLORS["success"]
    assert theme.band_color("high_risk") == theme.COLORS["error"]
    assert theme.band_color("unknown") == theme.COLORS["muted"]


def test_meter_clamps_and_fills():
    assert "width:50%" in theme.meter(50)
    assert "width:100%" in theme.meter(250)  # clamped
    assert "width:0%" in theme.meter(-5)


def test_badge_escapes_dynamic_label():
    # A suggestion band/label is LLM/user-derived; a payload must not break out of the badge.
    out = theme.badge("<img src=x onerror=alert(1)>", theme.COLORS["error"])
    assert "<img" not in out
    assert "&lt;img" in out


def test_esc_escapes_html():
    assert theme.esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_esc_md_neutralizes_markdown_metacharacters():
    # A skill like `_NET` or `a*b` must not render as emphasis in a plain st.markdown call.
    assert theme.esc_md("_NET") == r"\_NET"
    assert theme.esc_md("a*b") == r"a\*b"
    assert theme.esc_md("[x](y)") == r"\[x\]\(y\)"
