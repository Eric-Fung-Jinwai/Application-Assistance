"""Assemble a render-ready resume document from a ``ResumeVersion`` (Phase 12).

The version's ``BulletRow``s are the source of truth for bullet **text** (they carry every
accept/customize edit); the resume's ``parsed_json`` supplies the surrounding structure —
role titles, companies, dates, education, skills, certifications — that bullets alone lack.

Bullets are stored as a flat, ordered list per section (``bullets_from_parsed`` concatenates
each role's bullets in order), so we re-associate them to their roles **positionally**: walk
the parsed roles in order, consuming the version's experience bullets to refill each role's
slots. This is exact while bullet count/order is stable across versions (today's invariant —
edits only change text). If the counts ever diverge (e.g. a future add/delete feature) we
**fall back to an ungrouped block** rather than risk filing a bullet under the wrong employer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError
from sqlalchemy.orm import Session

from career_assistant.domain import ParsedResume
from career_assistant.kb.bullets import SECTION_EXPERIENCE, SECTION_PROJECTS
from career_assistant.storage import repo
from career_assistant.storage.models import Resume, ResumeVersion


@dataclass
class ExportRole:
    title: str = ""
    company: str = ""
    start_date: str | None = None
    end_date: str | None = None
    bullets: list[str] = field(default_factory=list)


@dataclass
class ExportProject:
    name: str = ""
    description: str = ""
    bullets: list[str] = field(default_factory=list)


@dataclass
class ExportEducation:
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_date: str | None = None
    end_date: str | None = None


@dataclass
class ResumeDocument:
    """Everything a template needs to render one resume version."""

    name: str
    experience: list[ExportRole] = field(default_factory=list)
    projects: list[ExportProject] = field(default_factory=list)
    education: list[ExportEducation] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    # Section bullets that couldn't be grouped to a role/project (ungrouped fallback).
    extra_experience: list[str] = field(default_factory=list)
    extra_projects: list[str] = field(default_factory=list)


def build_resume_document(
    session: Session,
    *,
    version_id: str,
    name: str | None = None,
) -> ResumeDocument:
    """Build the render model for ``version_id``: version bullet text + parsed structure.

    ``name`` overrides the header (the parse captures no name/contact); when omitted it falls
    back to the resume filename stem. Raises ``ValueError`` if the version is unknown.
    """
    version = session.get(ResumeVersion, version_id)
    if version is None:
        raise ValueError(f"unknown version {version_id!r}")
    resume = session.get(Resume, version.resume_id)

    bullets = repo.list_bullets(session, version_id)  # ordered by order_index
    exp_texts = [b.current_text for b in bullets if b.section == SECTION_EXPERIENCE]
    proj_texts = [b.current_text for b in bullets if b.section == SECTION_PROJECTS]

    parsed = _parsed_resume(resume)
    experience, extra_exp = _regroup_experience(parsed, exp_texts)
    projects, extra_proj = _regroup_projects(parsed, proj_texts)

    return ResumeDocument(
        name=name or _default_name(resume),
        experience=experience,
        projects=projects,
        education=[
            ExportEducation(
                institution=e.institution,
                degree=e.degree,
                field_of_study=e.field_of_study,
                start_date=e.start_date,
                end_date=e.end_date,
            )
            for e in parsed.education
        ],
        skills=list(parsed.skills),
        certifications=list(parsed.certifications),
        extra_experience=extra_exp,
        extra_projects=extra_proj,
    )


# --- helpers ------------------------------------------------------------------------


def _parsed_resume(resume: Resume | None) -> ParsedResume:
    """The resume's structured parse, or an empty one when absent/invalid."""
    if resume is None or not resume.parsed_json:
        return ParsedResume()
    try:
        return ParsedResume.model_validate(resume.parsed_json)
    except ValidationError:
        return ParsedResume()


def _default_name(resume: Resume | None) -> str:
    """Best-available header label — the parse captures no name, so use the filename stem."""
    if resume is None or not resume.filename:
        return "Resume"
    stem = resume.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.strip() or "Resume"


def _regroup_experience(
    parsed: ParsedResume, texts: list[str]
) -> tuple[list[ExportRole], list[str]]:
    """Refill each role's bullet slots from ``texts`` in order; ungrouped leftovers returned."""
    expected = [len([b for b in exp.bullets if b.strip()]) for exp in parsed.experience]
    if not parsed.experience or sum(expected) != len(texts):
        # No structure to map onto, or the count drifted → don't risk wrong attribution.
        return [], list(texts)

    roles: list[ExportRole] = []
    cursor = 0
    for exp, count in zip(parsed.experience, expected, strict=True):
        roles.append(
            ExportRole(
                title=exp.title,
                company=exp.company,
                start_date=exp.start_date,
                end_date=exp.end_date,
                bullets=texts[cursor : cursor + count],
            )
        )
        cursor += count
    return roles, []


def _regroup_projects(
    parsed: ParsedResume, texts: list[str]
) -> tuple[list[ExportProject], list[str]]:
    expected = [len([b for b in proj.bullets if b.strip()]) for proj in parsed.projects]
    if not parsed.projects or sum(expected) != len(texts):
        return [], list(texts)

    projects: list[ExportProject] = []
    cursor = 0
    for proj, count in zip(parsed.projects, expected, strict=True):
        projects.append(
            ExportProject(
                name=proj.name,
                description=proj.description,
                bullets=texts[cursor : cursor + count],
            )
        )
        cursor += count
    return projects, []
