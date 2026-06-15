"""Integrity scoring (Phase 8) — the evidence-grounding check that gates every rewrite.

Two layers:
  * **Pure primitives** (``compute_integrity_score`` / ``band_for_score``) — the
    config-driven score formula + band cutoffs the Phase 16 harness sweeps. Named
    parameters default to ``config.settings`` rather than magic numbers.
  * **The judge** (``judge_integrity``) — runs an LLM grounding audit + embedding
    similarity over ``(source evidence, tailored rewrite)`` and assembles a full
    ``IntegrityResult`` (score, band, and the unsupported claim/tech/metric/responsibility
    flags shown to the reviewer). Failure modes default *conservative*: an unreadable judge
    response yields ``grounded = 0`` so the rewrite lands in ``high_risk``, never a free pass.
"""

from __future__ import annotations

import json
import logging
import math

from career_assistant.config import settings
from career_assistant.domain import IntegrityBand, IntegrityResult
from career_assistant.llm.client import EmbeddingClient, LLMClient

logger = logging.getLogger("career_assistant.tailor")

# Float tolerance for the "weights sum to 1" check.
_WEIGHT_SUM_TOL = 1e-6


def compute_integrity_score(
    llm_judgment: float,
    embedding_similarity: float,
    *,
    w_llm: float | None = None,
    w_embed: float | None = None,
) -> float:
    """Combine the two grounding signals into a 0–100 integrity score.

    ``score = 100 * (w_llm * llm_judgment + w_embed * embedding_similarity)``.
    Weights default to the tuned ``config`` values; pass explicit ones to sweep.

    Inputs are validated so a malformed judge output or a bad sweep value can't silently
    produce an out-of-range score: both signals must be in ``[0, 1]`` and the weights must
    be non-negative and sum to 1.0 — keeping the result in ``[0, 100]``.
    """
    signals = (("llm_judgment", llm_judgment), ("embedding_similarity", embedding_similarity))
    for name, value in signals:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    w_llm = settings.integrity_w_llm if w_llm is None else w_llm
    w_embed = settings.integrity_w_embed if w_embed is None else w_embed
    if w_llm < 0 or w_embed < 0:
        raise ValueError("integrity weights must be non-negative")
    if abs(w_llm + w_embed - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(f"integrity weights must sum to 1.0, got {w_llm} + {w_embed}")

    return 100.0 * (w_llm * llm_judgment + w_embed * embedding_similarity)


def band_for_score(
    score: float,
    *,
    safe: int | None = None,
    moderate: int | None = None,
    aggressive: int | None = None,
) -> IntegrityBand:
    """Map a score to its band using the ``>= safe / moderate / aggressive`` cutoffs.

    Below the ``aggressive`` cutoff is ``high_risk`` — the block-worthy band a planted
    fabrication should land in. Cutoffs default to the ``config`` values.
    """
    safe = settings.integrity_band_safe if safe is None else safe
    moderate = settings.integrity_band_moderate if moderate is None else moderate
    aggressive = settings.integrity_band_aggressive if aggressive is None else aggressive
    if not aggressive <= moderate <= safe:
        raise ValueError(
            f"cutoffs must satisfy aggressive <= moderate <= safe, "
            f"got {aggressive} <= {moderate} <= {safe}"
        )

    if score >= safe:
        return IntegrityBand.safe
    if score >= moderate:
        return IntegrityBand.moderate
    if score >= aggressive:
        return IntegrityBand.aggressive
    return IntegrityBand.high_risk


# --- The judge ----------------------------------------------------------------------

JUDGE_SYSTEM = """You are a strict resume-integrity auditor. You receive the ORIGINAL \
evidence (verbatim resume bullet text) and a TAILORED rewrite. Decide whether every claim \
in the tailored text is supported by the original evidence.

Return ONLY a JSON object with exactly these keys:
- grounded: number in [0,1] — 1.0 = every claim is supported by the evidence; 0.0 = fabricated
- unsupported_claims: [string] — claims with no basis in the evidence
- unsupported_technologies: [string] — tools/tech in the rewrite but absent from the evidence
- unsupported_metrics: [string] — numbers/metrics not present in the evidence
- unsupported_responsibilities: [string] — duties or scope the evidence does not support

Rules:
- Rewording, stronger verbs, reordering, and surfacing skills the evidence ALREADY shows are fine.
- Adding a NEW skill, technology, metric, or responsibility absent from the evidence is NOT.
- Be conservative: if a claim has no supporting evidence, flag it and lower `grounded`.
"""

JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "grounded": {"type": "number"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "unsupported_technologies": {"type": "array", "items": {"type": "string"}},
        "unsupported_metrics": {"type": "array", "items": {"type": "string"}},
        "unsupported_responsibilities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded"],
}

_FLAG_FIELDS = (
    "unsupported_claims",
    "unsupported_technologies",
    "unsupported_metrics",
    "unsupported_responsibilities",
)


def judge_integrity(
    source: str | list[str],
    tailored: str,
    *,
    llm: LLMClient | None = None,
    embedder: EmbeddingClient | None = None,
) -> IntegrityResult:
    """Score whether ``tailored`` is grounded in ``source`` evidence.

    ``source`` is the original bullet text (or several supporting bullets). Combines the
    LLM grounding judgment with the max cosine similarity of the rewrite to any source
    bullet, then applies the config-driven score + band. A failed/garbled LLM response is
    treated as ungrounded (``grounded = 0``) so it can never silently pass.
    """
    from career_assistant.llm.factory import get_embedding_client, get_llm_client

    sources = [source] if isinstance(source, str) else [s for s in source if s.strip()]
    client = llm or get_llm_client()
    embed = embedder or get_embedding_client()

    raw = client.complete(JUDGE_SYSTEM, _judge_user(sources, tailored), json_schema=JUDGE_SCHEMA)
    data = _coerce_judgment(raw)

    llm_judgment = _clamp01(data.get("grounded", 0.0))
    embedding_similarity = _max_cosine(tailored, sources, embed)
    score = compute_integrity_score(llm_judgment, embedding_similarity)

    flags = {field: _str_list(data.get(field)) for field in _FLAG_FIELDS}
    if any(flags.values()):
        # The judge named specific unsupported content — a fabrication by its own finding.
        # A high `grounded`/similarity score must not let it read as safe: cap into high-risk.
        score = _cap_to_high_risk(score)

    return IntegrityResult(
        llm_judgment=llm_judgment,
        embedding_similarity=embedding_similarity,
        score=score,
        band=band_for_score(score),
        unsupported_claims=flags["unsupported_claims"],
        unsupported_technologies=flags["unsupported_technologies"],
        unsupported_metrics=flags["unsupported_metrics"],
        unsupported_responsibilities=flags["unsupported_responsibilities"],
    )


def _cap_to_high_risk(score: float) -> float:
    """Cap a flagged rewrite's score just below the aggressive cutoff → ``high_risk`` band."""
    ceiling = max(0.0, float(settings.integrity_band_aggressive) - 1.0)
    return min(score, ceiling)


def _judge_user(sources: list[str], tailored: str) -> str:
    evidence = "\n".join(f"- {s}" for s in sources) if sources else "(none)"
    return f"ORIGINAL EVIDENCE:\n{evidence}\n\nTAILORED:\n{tailored}"


def _coerce_judgment(raw: object) -> dict:
    """Parse the judge response into a dict; conservative empty default on failure."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("integrity.judge non-JSON response; treating as ungrounded")
            return {"grounded": 0.0}
    return raw if isinstance(raw, dict) else {"grounded": 0.0}


def _clamp01(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v).strip()]


def _max_cosine(tailored: str, sources: list[str], embedder: EmbeddingClient) -> float:
    """Best cosine similarity of the rewrite to any source bullet (0.0 if no source)."""
    if not sources:
        return 0.0
    vectors = embedder.embed([tailored, *sources])
    tailored_vec, source_vecs = vectors[0], vectors[1:]
    return max(_cosine(tailored_vec, sv) for sv in source_vecs)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))
