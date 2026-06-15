"""Render a resume version to ATS-friendly HTML via Jinja2 templates (Phase 12).

Autoescaping is on: bullet/skill text is user content and could contain ``<`` / ``&``, so it
is HTML-escaped to keep the markup well-formed (and to avoid injecting markup into the resume).
The output is a self-contained document with embedded CSS — no external assets to fetch.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy.orm import Session

from career_assistant.export.document import ResumeDocument, build_resume_document

TEMPLATES_DIR = Path(__file__).parent / "templates"

# The four selectable themes (Phase 12). Each is a Jinja2 file extending ``_base.html.j2``.
TEMPLATES = ("ats", "technical", "pm", "modern")
DEFAULT_TEMPLATE = "ats"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2", "html.j2"], default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_document_html(doc: ResumeDocument, *, template: str = DEFAULT_TEMPLATE) -> str:
    """Render an already-assembled ``ResumeDocument`` with the named template."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; choose one of {TEMPLATES}")
    rendered = _env().get_template(f"{template}.html.j2").render(doc=asdict(doc))
    return rendered


def render_html(
    session: Session,
    *,
    version_id: str,
    template: str = DEFAULT_TEMPLATE,
    name: str | None = None,
) -> str:
    """Assemble the document for ``version_id`` and render it to HTML."""
    doc = build_resume_document(session, version_id=version_id, name=name)
    return render_document_html(doc, template=template)
