"""Pydantic domain models — the in-memory data contracts (Phase 1).

These models are what flows between pipeline stages (parse → embed → fit → tailor →
review). SQLite (``storage/models.py``) is the persistent source of truth for text;
ChromaDB stores vectors keyed by ``bullet_id``. See ``storage/chroma.py`` for the
``bullet_id`` ↔ vector contract.

All primary keys are UUID4 strings — the ``Bullet.id`` "Unique ID" is the linchpin that
ties a resume bullet to its embedding, its tailoring suggestions, and its edit history.
"""

from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field


def new_id() -> str:
    """Generate a fresh UUID4 primary key as a string."""
    return str(uuid.uuid4())


# --- Enums (shared vocabulary; ORM columns store the string values) ---------------


class VersionType(str, Enum):
    original = "original"
    ai_generated = "ai_generated"
    accepted = "accepted"
    rejected = "rejected"
    customized = "customized"


class SuggestionStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"
    customized = "customized"


class IntegrityBand(str, Enum):
    safe = "safe"
    moderate = "moderate"
    aggressive = "aggressive"
    high_risk = "high_risk"


# --- Resume ingest / parse --------------------------------------------------------


class RawResume(BaseModel):
    """Raw extracted text before LLM structuring (output of ``ingest/parser.py``)."""

    id: str = Field(default_factory=new_id)
    filename: str
    raw_text: str
    content_type: str | None = None


class Bullet(BaseModel):
    """A single resume bullet — the linchpin Unique ID across stores."""

    id: str = Field(default_factory=new_id)
    section: str  # e.g. "experience" / "projects"
    order_index: int
    original_text: str
    current_text: str
    keywords: list[str] = Field(default_factory=list)


class ExperienceItem(BaseModel):
    title: str = ""
    company: str = ""
    start_date: str | None = None
    end_date: str | None = None
    bullets: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_date: str | None = None
    end_date: str | None = None


class ProjectItem(BaseModel):
    name: str = ""
    description: str = ""
    bullets: list[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    """Structured resume (output of ``ingest/extract.py``)."""

    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    # Phase 3 low-confidence extractions return partial data with confidence < 1.0
    # instead of crashing.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# --- Job description --------------------------------------------------------------


class ParsedJD(BaseModel):
    """Structured job description — the 7 fields from Phase 5."""

    id: str = Field(default_factory=new_id)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    domain: str | None = None
    seniority: str | None = None
    years_experience: float | None = None  # YOE


class CoverageReport(BaseModel):
    """Which JD requirements the resume already evidences (Phase 5).

    Built by ``jd/coverage.py`` from a ``ParsedJD`` + ``ParsedResume`` using an exact
    keyword set-diff plus an embedding-overlap fallback for semantic matches (e.g. a JD
    asking for ``Kubernetes`` covered by a resume bullet mentioning ``k8s``). Seniority /
    YOE matching is left to the fit engine (Phase 6), which models resume experience.
    """

    matched_required_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    matched_preferred_skills: list[str] = Field(default_factory=list)
    missing_preferred_skills: list[str] = Field(default_factory=list)
    # Fraction of JD skills the resume covers, in [0, 1]; 1.0 when the JD lists none.
    required_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    preferred_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    # None when the JD declares no domain.
    domain_match: bool | None = None


# --- Fit evaluation ---------------------------------------------------------------


class FitScore(BaseModel):
    """Per-dimension and overall fit, all in ``[0, 100]`` (Phase 6)."""

    skill: float = Field(ge=0.0, le=100.0)
    experience: float = Field(ge=0.0, le=100.0)
    seniority: float = Field(ge=0.0, le=100.0)
    location: float = Field(ge=0.0, le=100.0)
    compensation: float = Field(ge=0.0, le=100.0)
    overall: float = Field(ge=0.0, le=100.0)
    band: str | None = None
    breakdown: dict = Field(default_factory=dict)  # per-dimension reasoning


# --- Tailoring + integrity --------------------------------------------------------


class IntegrityResult(BaseModel):
    """Whether a tailored claim is grounded in the original evidence (Phase 8)."""

    llm_judgment: float = Field(ge=0.0, le=1.0)
    embedding_similarity: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=100.0)
    band: IntegrityBand
    unsupported_claims: list[str] = Field(default_factory=list)
    unsupported_technologies: list[str] = Field(default_factory=list)
    unsupported_metrics: list[str] = Field(default_factory=list)
    unsupported_responsibilities: list[str] = Field(default_factory=list)

    def unsupported(self) -> dict[str, list[str]]:
        """Non-empty unsupported-* flag lists, keyed by category — what the reviewer sees.

        Returns an empty dict when nothing was flagged. Persisted with the suggestion so the
        Phase 10 review panel can show *why* a rewrite is risky after reload.
        """
        categories = {
            "unsupported_claims": self.unsupported_claims,
            "unsupported_technologies": self.unsupported_technologies,
            "unsupported_metrics": self.unsupported_metrics,
            "unsupported_responsibilities": self.unsupported_responsibilities,
        }
        return {category: items for category, items in categories.items() if items}


class TailoringSuggestion(BaseModel):
    """An evidence-aware rewrite of a single bullet for a specific JD (Phase 8)."""

    id: str = Field(default_factory=new_id)
    bullet_id: str
    jd_id: str
    suggested_text: str
    reasoning: str = ""
    integrity: IntegrityResult | None = None
    status: SuggestionStatus = SuggestionStatus.pending
