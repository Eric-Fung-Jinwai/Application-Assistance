"""Recommendation: map a fit score + coverage into an apply/don't-apply verdict (Phase 7 MVP).

This is the minimal Phase 7 core the Phase 14 ``/recommendation`` endpoint needs: it turns the
already-computed ``FitScore`` and ``CoverageReport`` into a human-facing band plus the supporting
strengths, missing requirements, and concerns. Richer signals (salary/location, the Phase 2
preference profile) are noted as not-yet-evaluated rather than fabricated — consistent with the
project's "never overstate" stance. No new scoring happens here; it only reshapes existing facts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from career_assistant.domain import CoverageReport, FitScore

# Overall-fit cutoffs → verdict band. Tuned conservatively; the eval harness can sweep later.
BAND_STRONG_MIN = 75.0
BAND_CAUTION_MIN = 60.0
BAND_STRETCH_MIN = 45.0

STRONG_MATCH = "Strong Match"
PROCEED_WITH_CAUTION = "Proceed With Caution"
STRETCH = "Stretch"
LOW_ALIGNMENT = "Low Alignment"

# Stated, not implied: these dimensions aren't modelled yet (need ParsedJD fields + Phase 2).
_UNSCORED_NOTE = "Salary and location fit are not yet evaluated."


class Recommendation(BaseModel):
    """An apply-decision summary for one resume-version × JD pairing."""

    band: str
    overall: float = Field(ge=0.0, le=100.0)
    strengths: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


def band_for_overall(overall: float) -> str:
    """Map an overall fit score in [0, 100] to its recommendation band."""
    if overall >= BAND_STRONG_MIN:
        return STRONG_MATCH
    if overall >= BAND_CAUTION_MIN:
        return PROCEED_WITH_CAUTION
    if overall >= BAND_STRETCH_MIN:
        return STRETCH
    return LOW_ALIGNMENT


def build_recommendation(fit: FitScore, coverage: CoverageReport) -> Recommendation:
    """Reshape a ``FitScore`` + ``CoverageReport`` into a ``Recommendation`` (no new scoring)."""
    strengths = [*coverage.matched_required_skills, *coverage.matched_preferred_skills]

    concerns: list[str] = []
    if coverage.missing_required_skills:
        concerns.append(
            f"Missing {len(coverage.missing_required_skills)} required skill(s): "
            + ", ".join(coverage.missing_required_skills)
        )
    if coverage.domain_match is False:
        concerns.append("Resume domain does not match the role's domain.")
    concerns.append(_UNSCORED_NOTE)

    return Recommendation(
        band=band_for_overall(fit.overall),
        overall=fit.overall,
        strengths=strengths,
        missing_requirements=list(coverage.missing_required_skills),
        concerns=concerns,
    )
