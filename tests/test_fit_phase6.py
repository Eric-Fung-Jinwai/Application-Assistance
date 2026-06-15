"""Phase 6 acceptance: (ParsedResume, ParsedJD) → numeric sub-scores + overall in
[0, 100] with a per-dimension breakdown and a readable band."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from career_assistant.domain import ExperienceItem, FitScore, ParsedJD, ParsedResume
from career_assistant.fit import score_fit
from career_assistant.fit.engine import NEUTRAL_SCORE, _resume_years
from career_assistant.llm.client import EmbeddingClient


class FakeEmbedder(EmbeddingClient):
    """Deterministic bag-of-words embedder with a tiny synonym map (k8s → kubernetes),
    so the semantic skill-coverage path is exercisable without a real model."""

    DIM = 64
    SYNONYMS = {"k8s": "kubernetes"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.DIM
            for raw in text.lower().split():
                tok = self.SYNONYMS.get(raw, raw)
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
            if not any(v):
                v[0] = 1.0
            out.append(v)
        return out


STRONG_RESUME = ParsedResume(
    skills=["Python", "SQL", "Docker", "Kubernetes"],
    experience=[
        ExperienceItem(
            title="Senior Software Engineer",
            company="Acme",
            start_date="2015",
            end_date="2023",  # ~8 years span
            bullets=["Built Python data pipelines", "Led the Kubernetes migration"],
        )
    ],
)

SENIOR_JD = ParsedJD(
    required_skills=["Python", "SQL"],
    preferred_skills=["Docker"],
    seniority="senior",
    years_experience=5,
)


# --- acceptance ------------------------------------------------------------------


def test_score_fit_returns_subscores_overall_and_band():
    fit = score_fit(STRONG_RESUME, SENIOR_JD)
    assert isinstance(fit, FitScore)

    # All five sub-scores plus overall are real numbers in [0, 100] (pydantic also
    # enforces the bounds at construction).
    for value in (
        fit.skill,
        fit.experience,
        fit.seniority,
        fit.location,
        fit.compensation,
        fit.overall,
    ):
        assert 0.0 <= value <= 100.0

    # A strong candidate clears every applicable dimension → strong band.
    assert fit.skill == 100.0
    assert fit.experience == 100.0  # 8y span ≥ 5y required
    assert fit.seniority == 100.0  # ~staff resume vs senior JD
    assert fit.overall == 100.0
    assert fit.band == "strong"

    # The harness reads per-dimension reasoning + which dimensions counted.
    assert fit.breakdown["skill"]["applicable"] is True
    assert "required" in fit.breakdown["skill"]["reason"]
    assert set(fit.breakdown["overall"]["applicable_dimensions"]) == {
        "skill",
        "experience",
        "seniority",
    }


def test_location_and_comp_are_non_applicable_and_excluded():
    fit = score_fit(STRONG_RESUME, SENIOR_JD)
    # No JD/preference data yet → neutral value, held out of the overall average.
    assert fit.location == NEUTRAL_SCORE
    assert fit.compensation == NEUTRAL_SCORE
    assert fit.breakdown["location"]["applicable"] is False
    assert fit.breakdown["compensation"]["applicable"] is False
    assert "location" not in fit.breakdown["overall"]["applicable_dimensions"]


# --- per-dimension behaviour -----------------------------------------------------


def test_weak_candidate_lands_in_weak_band():
    resume = ParsedResume(
        skills=["Excel"],
        experience=[ExperienceItem(start_date="2022", end_date="2023", bullets=["Did tasks"])],
    )
    jd = ParsedJD(
        required_skills=["Python", "Kubernetes", "SQL"], seniority="senior", years_experience=6
    )
    fit = score_fit(resume, jd)
    assert fit.skill == 0.0  # none of the required skills present
    assert fit.seniority == 30.0  # ~1y resume vs senior JD, two ranks short
    assert fit.band == "weak"


def test_experience_scales_linearly_below_requirement():
    resume = ParsedResume(
        experience=[ExperienceItem(start_date="2018", end_date="2023", bullets=["x"])]  # 5y span
    )
    jd = ParsedJD(years_experience=10)  # no skills, no seniority → experience is the only dim
    fit = score_fit(resume, jd)
    assert fit.experience == 50.0  # 5 / 10
    assert fit.overall == 50.0  # renormalised over the single applicable dimension


def test_graceful_degradation_when_jd_has_nothing_to_judge():
    fit = score_fit(STRONG_RESUME, ParsedJD())  # empty JD
    assert fit.overall == NEUTRAL_SCORE
    assert fit.band == "moderate"
    for dim in ("skill", "experience", "seniority", "location", "compensation"):
        assert fit.breakdown[dim]["applicable"] is False
    assert fit.breakdown["overall"]["applicable_dimensions"] == []


def test_embedder_enables_semantic_skill_coverage():
    resume = ParsedResume(skills=["Kubernetes"])
    jd = ParsedJD(required_skills=["k8s"])  # non-literal synonym

    assert score_fit(resume, jd, embedder=None).skill == 0.0
    assert score_fit(resume, jd, embedder=FakeEmbedder()).skill == 100.0


# --- experience-span helper ------------------------------------------------------


def test_resume_years_spans_earliest_start_to_latest_end():
    resume = ParsedResume(
        experience=[
            ExperienceItem(start_date="Jan 2015", end_date="Dec 2018", bullets=[]),
            ExperienceItem(start_date="2020", end_date="2024", bullets=[]),
        ]
    )
    assert _resume_years(resume) == 9.0  # 2024 - 2015


def test_resume_years_none_when_no_parseable_dates():
    resume = ParsedResume(experience=[ExperienceItem(start_date=None, end_date=None, bullets=[])])
    assert _resume_years(resume) is None


def test_resume_years_skips_role_with_missing_end_date():
    # A missing end is NOT assumed to be "now" — that would inflate tenure on a parse miss.
    resume = ParsedResume(experience=[ExperienceItem(start_date="2015", end_date=None, bullets=[])])
    assert _resume_years(resume) is None  # no usable start+end pair → non-applicable


def test_resume_years_counts_only_dated_roles_not_undated_ones():
    resume = ParsedResume(
        experience=[
            ExperienceItem(start_date="2018", end_date="2021", bullets=[]),  # counts
            ExperienceItem(start_date="2022", end_date="garbage", bullets=[]),  # skipped
        ]
    )
    assert _resume_years(resume) == 3.0  # only the 2018–2021 role


def test_resume_years_treats_explicit_present_as_ongoing():
    now = datetime.now(UTC).year
    resume = ParsedResume(
        experience=[ExperienceItem(start_date="2020", end_date="Present", bullets=[])]
    )
    assert _resume_years(resume) == float(now - 2020)
