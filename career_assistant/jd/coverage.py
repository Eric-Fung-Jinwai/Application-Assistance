"""Coverage analysis: which JD requirements the resume already evidences (Phase 5).

Two complementary signals, as in the roadmap:
  * **keyword set-diff** — a word-boundary scan for the JD skill term anywhere in the
    resume corpus (declared skills *and* bullet text). Cheap and precise: it catches
    ``Kubernetes`` whether it is a listed skill or only named in a bullet.
  * **embedding overlap** — for JD skills the keyword scan misses, fall back to cosine
    similarity against the resume corpus, catching *non-literal* matches (``k8s`` ≈
    ``Kubernetes``) that a string scan cannot.

Seniority / YOE matching is intentionally left to the fit engine (Phase 6), which models
resume experience; here we cover skills and the JD domain.
"""

from __future__ import annotations

import math
import re

from career_assistant.domain import CoverageReport, ParsedJD, ParsedResume
from career_assistant.llm.client import EmbeddingClient

# Cosine cutoff for the embedding-overlap fallback. Tunable by the Phase 16 eval sweep.
COVERAGE_SIM_THRESHOLD = 0.75


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace for keyword comparison."""
    return " ".join(text.lower().split())


def _keyword_hit(skill_norm: str, blob: str) -> bool:
    """Whole-token match of ``skill_norm`` in the normalised resume ``blob``.

    Boundaries are alphanumeric-aware so short skills don't over-match (``Go`` ≠
    ``Google``) while symbol-bearing names still work (``C++``, ``Node.js``).
    """
    if not skill_norm:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(skill_norm)}(?![a-z0-9])"
    return re.search(pattern, blob) is not None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _resume_corpus(resume: ParsedResume) -> list[str]:
    """Skill phrases + experience/project bullets — the text a JD skill can match against."""
    corpus = [s for s in resume.skills if s.strip()]
    corpus += [b for exp in resume.experience for b in exp.bullets if b.strip()]
    corpus += [b for proj in resume.projects for b in proj.bullets if b.strip()]
    return corpus


def _classify(
    skills: list[str],
    *,
    blob: str,
    corpus_vecs: list[list[float]],
    embedder: EmbeddingClient | None,
    threshold: float,
) -> tuple[list[str], list[str]]:
    """Split ``skills`` into (matched, missing), preserving input order.

    A skill matches on a word-boundary keyword hit in the resume ``blob``, or — failing
    that — when its embedding is within ``threshold`` cosine of any resume corpus vector.
    """
    skills = [s for s in skills if _norm(s)]
    pending = [s for s in skills if not _keyword_hit(_norm(s), blob)]

    semantic: dict[str, bool] = {}
    if pending and corpus_vecs and embedder is not None:
        for skill, vec in zip(pending, embedder.embed(pending), strict=True):
            semantic[skill] = any(_cosine(vec, cv) >= threshold for cv in corpus_vecs)

    matched, missing = [], []
    for skill in skills:
        if _keyword_hit(_norm(skill), blob) or semantic.get(skill, False):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def analyze_coverage(
    jd: ParsedJD,
    resume: ParsedResume,
    *,
    embedder: EmbeddingClient | None = None,
    threshold: float = COVERAGE_SIM_THRESHOLD,
) -> CoverageReport:
    """Report matched vs missing JD skills against the resume.

    ``embedder`` is optional: without it, coverage degrades gracefully to keyword-only
    matching (no embedding fallback). With it, the resume corpus is embedded once and
    reused across all comparisons.
    """
    corpus = _resume_corpus(resume)
    blob = "\n".join(_norm(c) for c in corpus)
    corpus_vecs = embedder.embed(corpus) if (embedder is not None and corpus) else []

    matched_req, missing_req = _classify(
        jd.required_skills,
        blob=blob,
        corpus_vecs=corpus_vecs,
        embedder=embedder,
        threshold=threshold,
    )
    matched_pref, missing_pref = _classify(
        jd.preferred_skills,
        blob=blob,
        corpus_vecs=corpus_vecs,
        embedder=embedder,
        threshold=threshold,
    )

    domain_match = _match_domain(
        jd.domain,
        blob=blob,
        corpus_vecs=corpus_vecs,
        embedder=embedder,
        threshold=threshold,
    )

    return CoverageReport(
        matched_required_skills=matched_req,
        missing_required_skills=missing_req,
        matched_preferred_skills=matched_pref,
        missing_preferred_skills=missing_pref,
        required_coverage=_coverage(len(matched_req), len(missing_req)),
        preferred_coverage=_coverage(len(matched_pref), len(missing_pref)),
        domain_match=domain_match,
    )


def _coverage(matched: int, missing: int) -> float:
    """Fraction matched; 1.0 when the JD lists no skills of this kind (nothing to miss)."""
    total = matched + missing
    return 1.0 if total == 0 else matched / total


def _match_domain(
    domain: str | None,
    *,
    blob: str,
    corpus_vecs: list[list[float]],
    embedder: EmbeddingClient | None,
    threshold: float,
) -> bool | None:
    """Best-effort domain presence: keyword hit anywhere in the resume, else embedding."""
    if not domain or not domain.strip():
        return None
    if _keyword_hit(_norm(domain), blob):
        return True
    if corpus_vecs and embedder is not None:
        dvec = embedder.embed([domain])[0]
        return any(_cosine(dvec, cv) >= threshold for cv in corpus_vecs)
    return False
