"""Phase 12 acceptance: a version renders to all 4 templates; the generated PDF has
selectable text (pypdf recovers the bullets) and paginates a 2-page resume. Also covers the
positional bullet→role regrouping and its safe ungrouped fallback, plus HTML escaping."""

from __future__ import annotations

import io

import pytest

from career_assistant.domain import EducationItem, ExperienceItem, ParsedResume, VersionType
from career_assistant.export import (
    TEMPLATES,
    build_resume_document,
    render_html,
)
from career_assistant.export.pdf import html_to_pdf, render_pdf
from career_assistant.kb.bullets import SECTION_EXPERIENCE
from career_assistant.storage import repo
from career_assistant.storage.db import make_engine, make_session_factory


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    repo.create_db(engine)
    with make_session_factory(engine)() as s:
        yield s


PARSED = ParsedResume(
    experience=[
        ExperienceItem(
            title="Senior Engineer",
            company="Acme",
            start_date="2021",
            end_date="present",
            bullets=["Built a Python data pipeline", "Led migration to Docker"],
        ),
        ExperienceItem(
            title="Engineer",
            company="Globex",
            start_date="2018",
            end_date="2021",
            bullets=["Shipped a billing service"],
        ),
    ],
    education=[EducationItem(institution="State University", degree="BSc", field_of_study="CS")],
    skills=["Python", "Docker", "Kubernetes"],
    certifications=["AWS Solutions Architect"],
)

# The version's bullet text, in the same flattened order bullets_from_parsed produces.
VERSION_EXP_BULLETS = [
    "Engineered a Python data pipeline cutting latency 30%",
    "Drove the Docker migration across 12 services",
    "Owned a billing service handling $4M/mo",
]


def _seed(session, *, exp_bullets=VERSION_EXP_BULLETS, parsed=PARSED, filename="jane_doe.pdf"):
    resume = repo.create_resume(
        session,
        filename=filename,
        raw_text="x",
        parsed=parsed.model_dump() if parsed is not None else None,
    )
    v1 = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    for i, text in enumerate(exp_bullets):
        repo.add_bullet(
            session,
            resume_version_id=v1.id,
            section=SECTION_EXPERIENCE,
            order_index=i,
            original_text=text,
        )
    session.commit()
    return resume, v1


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
    not _chromium_available(),
    reason="Chromium not installed — run `playwright install chromium`",
)


# --- document assembly -----------------------------------------------------------


def test_document_regroups_version_bullets_into_roles_positionally(session):
    _, v1 = _seed(session)
    doc = build_resume_document(session, version_id=v1.id)

    assert [r.company for r in doc.experience] == ["Acme", "Globex"]
    # Edited text (not the original parse) lands under the right role, by position.
    assert doc.experience[0].bullets == VERSION_EXP_BULLETS[:2]
    assert doc.experience[1].bullets == VERSION_EXP_BULLETS[2:]
    assert doc.experience[0].end_date == "present"
    assert not doc.extra_experience  # everything mapped cleanly
    assert doc.skills == ["Python", "Docker", "Kubernetes"]
    assert doc.education[0].institution == "State University"
    assert doc.name == "jane_doe"  # filename stem (the parse captures no name)


def test_document_falls_back_to_ungrouped_when_counts_drift(session):
    # The version has 3 experience bullets but the parse expects 2 → don't risk misattribution.
    parsed = ParsedResume(
        experience=[ExperienceItem(title="Eng", company="Acme", bullets=["a", "b"])]
    )
    _, v1 = _seed(session, exp_bullets=["x", "y", "z"], parsed=parsed)
    doc = build_resume_document(session, version_id=v1.id)
    assert doc.experience == []
    assert doc.extra_experience == ["x", "y", "z"]


def test_document_handles_missing_parse(session):
    _, v1 = _seed(session, parsed=None)
    doc = build_resume_document(session, version_id=v1.id)
    assert doc.experience == []
    assert doc.extra_experience == VERSION_EXP_BULLETS  # rendered ungrouped, nothing lost


def test_build_document_raises_on_unknown_version(session):
    with pytest.raises(ValueError, match="unknown version"):
        build_resume_document(session, version_id="nope")


# --- HTML rendering --------------------------------------------------------------


def test_renders_to_all_four_templates_with_bullet_text(session):
    _, v1 = _seed(session)
    assert set(TEMPLATES) == {"ats", "technical", "pm", "modern"}
    for template in TEMPLATES:
        html = render_html(session, version_id=v1.id, template=template)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        for bullet in VERSION_EXP_BULLETS:
            assert bullet in html
        assert "Acme" in html and "State University" in html
        assert "Python · Docker · Kubernetes" in html  # skills joined


def test_render_name_override(session):
    _, v1 = _seed(session)
    html = render_html(session, version_id=v1.id, template="ats", name="Jane Q. Doe")
    assert "<h1>Jane Q. Doe</h1>" in html


def test_render_escapes_user_text(session):
    parsed = ParsedResume(experience=[ExperienceItem(title="Eng", company="Acme", bullets=["a"])])
    _, v1 = _seed(session, exp_bullets=["Built <script>alert(1)</script> tooling"], parsed=parsed)
    html = render_html(session, version_id=v1.id, template="ats")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_unknown_template_raises(session):
    _, v1 = _seed(session)
    with pytest.raises(ValueError, match="unknown template"):
        render_html(session, version_id=v1.id, template="fancy")


# --- PDF (requires Chromium) -----------------------------------------------------


@requires_chromium
def test_pdf_has_selectable_bullet_text(session):
    from pypdf import PdfReader

    _, v1 = _seed(session)
    pdf_bytes = render_pdf(session, version_id=v1.id, template="ats")
    assert pdf_bytes.startswith(b"%PDF")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() for page in reader.pages)
    # Real text layer, not a raster — the bullet content is recoverable.
    assert "Engineered a Python data pipeline" in text


@requires_chromium
def test_pdf_paginates_a_two_page_resume(session):
    from pypdf import PdfReader

    # Enough long bullets to overflow a single A4 page.
    parsed = ParsedResume(
        experience=[
            ExperienceItem(title="Eng", company="Acme", bullets=[f"b{i}" for i in range(60)])
        ]
    )
    long_bullets = [
        f"Delivered initiative {i}: " + ("scaled distributed systems and mentored engineers " * 3)
        for i in range(60)
    ]
    _, v1 = _seed(session, exp_bullets=long_bullets, parsed=parsed)

    pdf_bytes = render_pdf(session, version_id=v1.id, template="ats")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 2


@requires_chromium
def test_html_to_pdf_renders_minimal_html():
    pdf = html_to_pdf("<!DOCTYPE html><html><body><p>hello pdf</p></body></html>")
    assert pdf.startswith(b"%PDF")
