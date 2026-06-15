"""Fit evaluation: ``(ParsedResume, ParsedJD)`` → ``FitScore`` (Phase 6).

Five sub-scores — skill / experience / seniority / location / compensation — each in
``[0, 100]``, combined into a weighted ``overall`` and a coarse ``band``. Every weight and
cutoff is a named constant (no magic numbers), so the Phase 16 harness can read and, if
needed, sweep them.

Graceful degradation is the rule: a dimension with no data to judge (e.g. a JD that
states no YOE requirement, or location/comp with no user preferences yet) is marked
**not applicable** — it still appears in ``breakdown`` with a neutral value and a reason,
but is held *out* of the overall average so it neither rewards nor penalises the fit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from career_assistant.domain import FitScore, ParsedJD, ParsedResume
from career_assistant.jd.coverage import analyze_coverage
from career_assistant.llm.client import EmbeddingClient

# --- Tunable constants (named, not magic — readable by the Phase 16 harness) --------

# Overall weights; renormalised over the applicable dimensions before averaging.
W_SKILL = 0.40
W_EXPERIENCE = 0.25
W_SENIORITY = 0.15
W_LOCATION = 0.10
W_COMPENSATION = 0.10

# Within the skill dimension, required coverage outweighs preferred.
SKILL_W_REQUIRED = 0.7
SKILL_W_PREFERRED = 0.3

# Value recorded for a dimension we can't judge; kept out of the overall average.
NEUTRAL_SCORE = 70.0

# Seniority score by (resume_rank - jd_rank): meets/exceeds, one short, two-plus short.
SENIORITY_MEETS = 100.0
SENIORITY_ONE_SHORT = 65.0
SENIORITY_TWO_PLUS_SHORT = 30.0

# Overall → band cutoffs.
BAND_STRONG_MIN = 75.0
BAND_MODERATE_MIN = 50.0


@dataclass
class _Dim:
    """One sub-score: its value, its overall weight, whether it counts, and why."""

    value: float
    weight: float
    applicable: bool
    reason: str


def score_fit(
    resume: ParsedResume,
    jd: ParsedJD,
    *,
    embedder: EmbeddingClient | None = None,
) -> FitScore:
    """Score how well ``resume`` fits ``jd``.

    ``embedder`` enables semantic skill coverage (degrades to keyword-only without it).
    Location/compensation are not yet scored — they need ``ParsedJD`` location/comp fields
    and the Phase 2 user-preference profile, neither of which exists; they're reported as
    non-applicable. A ``preferences`` parameter returns with Phase 2, when it can do real
    work.
    """
    dims = {
        "skill": _score_skill(resume, jd, embedder),
        "experience": _score_experience(resume, jd),
        "seniority": _score_seniority(resume, jd),
        "location": _score_location(),
        "compensation": _score_compensation(),
    }

    overall = _weighted_overall(dims)
    breakdown = {
        name: {"score": round(d.value, 1), "applicable": d.applicable, "reason": d.reason}
        for name, d in dims.items()
    }
    breakdown["overall"] = {
        "applicable_dimensions": [n for n, d in dims.items() if d.applicable],
    }

    return FitScore(
        skill=dims["skill"].value,
        experience=dims["experience"].value,
        seniority=dims["seniority"].value,
        location=dims["location"].value,
        compensation=dims["compensation"].value,
        overall=overall,
        band=_band(overall),
        breakdown=breakdown,
    )


def _weighted_overall(dims: dict[str, _Dim]) -> float:
    """Weighted average over *applicable* dimensions, with weights renormalised."""
    applicable = [d for d in dims.values() if d.applicable]
    total_w = sum(d.weight for d in applicable)
    if not applicable or total_w == 0:
        return NEUTRAL_SCORE
    return sum(d.value * d.weight for d in applicable) / total_w


def _band(overall: float) -> str:
    if overall >= BAND_STRONG_MIN:
        return "strong"
    if overall >= BAND_MODERATE_MIN:
        return "moderate"
    return "weak"


# --- Skill --------------------------------------------------------------------------


def _score_skill(resume: ParsedResume, jd: ParsedJD, embedder: EmbeddingClient | None) -> _Dim:
    has_req, has_pref = bool(jd.required_skills), bool(jd.preferred_skills)
    if not has_req and not has_pref:
        return _Dim(NEUTRAL_SCORE, W_SKILL, False, "JD lists no skills to match")

    cov = analyze_coverage(jd, resume, embedder=embedder)
    # Renormalise so absent preferred skills don't hand out free points.
    parts: list[tuple[float, float]] = []
    if has_req:
        parts.append((SKILL_W_REQUIRED, cov.required_coverage))
    if has_pref:
        parts.append((SKILL_W_PREFERRED, cov.preferred_coverage))
    wsum = sum(w for w, _ in parts)
    value = 100.0 * sum(w * c for w, c in parts) / wsum

    n_req = len(cov.matched_required_skills) + len(cov.missing_required_skills)
    n_pref = len(cov.matched_preferred_skills) + len(cov.missing_preferred_skills)
    reason = (
        f"{len(cov.matched_required_skills)}/{n_req} required, "
        f"{len(cov.matched_preferred_skills)}/{n_pref} preferred skills covered"
    )
    return _Dim(value, W_SKILL, True, reason)


# --- Experience (years) -------------------------------------------------------------


def _score_experience(resume: ParsedResume, jd: ParsedJD) -> _Dim:
    required = jd.years_experience
    if required is None or required <= 0:
        return _Dim(NEUTRAL_SCORE, W_EXPERIENCE, False, "JD states no YOE requirement")
    span = _resume_years(resume)
    if span is None:
        return _Dim(NEUTRAL_SCORE, W_EXPERIENCE, False, "resume work dates unparseable")
    value = 100.0 * min(1.0, span / required)
    return _Dim(value, W_EXPERIENCE, True, f"~{span:.0f}y experience vs {required:.0f}y required")


# Only these explicit markers mean "still in this role" — a *missing* end date does not.
_PRESENT_MARKERS = ("present", "current", "ongoing", "to date", "till date", "till now")


def _resume_years(resume: ParsedResume) -> float | None:
    """Best-effort career span in years: latest end minus earliest start across roles.

    A span (not a sum of durations) avoids double-counting overlapping jobs. A role only
    counts when *both* its start and end are determinable: an end that is missing or
    unparseable — and not an explicit "present"/"ongoing" — is **not** assumed to be the
    current year, since that would silently inflate tenure (and thus seniority) on a parse
    failure. Returns ``None`` when no role yields a usable start+end pair.
    """
    now = datetime.now(UTC).year
    starts: list[int] = []
    ends: list[int] = []
    for exp in resume.experience:
        start_year = _year(exp.start_date)
        end_year = _end_year(exp.end_date, now)
        if start_year is None or end_year is None:
            continue  # can't place this role on the timeline — skip rather than guess
        starts.append(start_year)
        ends.append(end_year)
    if not starts:
        return None
    return float(max(0, max(ends) - min(starts)))


def _year(value: str | None) -> int | None:
    """Pull the first 4-digit year out of a free-form date string."""
    if not value:
        return None
    match = re.search(r"(?:19|20)\d{2}", value)
    return int(match.group()) if match else None


def _end_year(value: str | None, now: int) -> int | None:
    """Resolve an end date to a year.

    An explicit present/ongoing marker → the current year; a parseable year → that year;
    anything else (missing or unparseable) → ``None`` so the caller skips the role instead
    of overstating it.
    """
    if value and any(marker in value.lower() for marker in _PRESENT_MARKERS):
        return now
    return _year(value)


# --- Seniority ----------------------------------------------------------------------

# Checked high-rank first so "senior staff" resolves to staff, not senior.
_SENIORITY_RANKS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (4, ("staff", "principal", "lead", "director", "head of")),
    (3, ("senior", "sr")),
    (2, ("mid", "intermediate")),
    (1, ("junior", "entry", "jr", "graduate", "associate")),
    (0, ("intern",)),
)


def _score_seniority(resume: ParsedResume, jd: ParsedJD) -> _Dim:
    jd_rank = _seniority_rank(jd.seniority)
    if jd_rank is None:
        return _Dim(NEUTRAL_SCORE, W_SENIORITY, False, "JD states no seniority level")
    span = _resume_years(resume)
    resume_rank = _rank_from_years(span)
    if resume_rank is None:
        return _Dim(NEUTRAL_SCORE, W_SENIORITY, False, "resume seniority can't be inferred")

    diff = resume_rank - jd_rank
    if diff >= 0:
        value = SENIORITY_MEETS
    elif diff == -1:
        value = SENIORITY_ONE_SHORT
    else:
        value = SENIORITY_TWO_PLUS_SHORT
    return _Dim(value, W_SENIORITY, True, f"resume level ~{resume_rank} vs JD level {jd_rank}")


def _seniority_rank(text: str | None) -> int | None:
    if not text:
        return None
    lowered = text.lower()
    for rank, keywords in _SENIORITY_RANKS:
        if any(kw in lowered for kw in keywords):
            return rank
    return None


def _rank_from_years(span: float | None) -> int | None:
    """Infer a seniority rank from career span (intern can't be derived from years)."""
    if span is None:
        return None
    if span < 2:
        return 1  # junior
    if span < 5:
        return 2  # mid
    if span < 8:
        return 3  # senior
    return 4  # staff+


# --- Location & compensation (Phase 2 preferences; neutral until then) ---------------
# Both stay non-applicable until ``ParsedJD`` gains location/comp fields *and* the Phase 2
# user-preference profile exists. They take no inputs today rather than accept a
# ``preferences`` arg that would be silently ignored.


_NOT_SCORED_YET = "not scored yet (needs JD field + Phase 2 prefs)"


def _score_location() -> _Dim:
    return _Dim(NEUTRAL_SCORE, W_LOCATION, False, f"location {_NOT_SCORED_YET}")


def _score_compensation() -> _Dim:
    return _Dim(NEUTRAL_SCORE, W_COMPENSATION, False, f"compensation {_NOT_SCORED_YET}")
