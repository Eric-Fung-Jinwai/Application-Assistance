"""Evidence-aware tailoring: rewrite resume bullets toward a JD, then gate on integrity.

Pipeline (Phase 8):
  1. **Map** JD requirements → candidate bullets via vector search over ``resume_bullets``.
  2. **Rewrite** each candidate with *allowed operations only* (stronger wording, surfacing
     skills the bullet already shows, reordering) — the prompt forbids inventing skills,
     tech, metrics, or responsibilities.
  3. **Judge** every rewrite with ``judge_integrity`` and persist a ``tailoring_suggestion``
     (status ``pending``) carrying its score, band, and unsupported-claim flags for review.

The guardrails are advisory (a prompt can still slip); integrity scoring is the *hard*
backstop, which is why every suggestion is judged before it reaches a human.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from career_assistant.domain import ParsedJD, TailoringSuggestion
from career_assistant.kb.embeddings import query_bullets
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.storage import repo
from career_assistant.tailor.integrity import judge_integrity

logger = logging.getLogger("career_assistant.tailor")

# How many top bullets each JD requirement pulls from the vector index.
CANDIDATES_PER_REQUIREMENT = 1

REWRITE_SYSTEM = """You rewrite ONE resume bullet to better fit a target job, using ALLOWED \
operations ONLY:
- stronger, more active wording
- surfacing keywords/skills the bullet ALREADY demonstrates
- reordering or condensing for impact

You MUST NOT add anything not already supported by the original bullet:
- no new skills, technologies, or tools
- no new metrics or numbers
- no new responsibilities or scope
If a target keyword is not supported by the bullet, leave it out.

Return ONLY a JSON object with exactly these keys:
- suggested_text: string — the rewritten bullet
- reasoning: string — one sentence on what you changed and why
"""

REWRITE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "suggested_text": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["suggested_text"],
}


def rewrite_bullet(
    source_text: str,
    *,
    targets: list[str],
    llm: LLMClient | None = None,
) -> tuple[str, str]:
    """Rewrite one bullet, emphasising the ``targets`` it already supports.

    Returns ``(suggested_text, reasoning)``. On a garbled response it falls back to the
    original text (a safe no-op rewrite) rather than emitting something unvetted.
    """
    from career_assistant.llm.factory import get_llm_client

    client = llm or get_llm_client()
    raw = client.complete(
        REWRITE_SYSTEM, _rewrite_user(source_text, targets), json_schema=REWRITE_SCHEMA
    )
    data = _coerce_rewrite(raw)
    suggested = data.get("suggested_text") or source_text
    reasoning = data.get("reasoning") or ""
    return str(suggested), str(reasoning)


REFINE_SYSTEM = """You refine ONE already-tailored resume bullet to follow a user's \
REFINEMENT INSTRUCTION (e.g. "make it shorter", "more PM-oriented", "emphasize Docker", \
"highlight leadership"), using ALLOWED operations ONLY:
- stronger, more active wording
- surfacing keywords/skills the ORIGINAL EVIDENCE already demonstrates
- reordering, condensing, or shifting emphasis for impact

The instruction NEVER licenses adding anything the original evidence does not support:
- no new skills, technologies, or tools
- no new metrics or numbers
- no new responsibilities or scope
If the instruction asks to emphasise something the evidence does not support, do NOT invent it.

Return ONLY a JSON object with exactly these keys:
- suggested_text: string — the refined bullet
- reasoning: string — one sentence on what you changed and why
"""


def refine_bullet(
    source_text: str,
    current_text: str,
    instruction: str,
    *,
    llm: LLMClient | None = None,
) -> tuple[str, str]:
    """Refine an existing tailored bullet per a chat ``instruction``, staying grounded.

    ``source_text`` is the original evidence (the integrity anchor); ``current_text`` is the
    suggestion being refined. Returns ``(suggested_text, reasoning)``. On a garbled response it
    falls back to ``current_text`` (a safe no-op) rather than reverting or emitting anything
    unvetted. The caller is expected to re-run integrity scoring on the result before persisting.
    """
    from career_assistant.llm.factory import get_llm_client

    client = llm or get_llm_client()
    raw = client.complete(
        REFINE_SYSTEM,
        _refine_user(source_text, current_text, instruction),
        json_schema=REWRITE_SCHEMA,
    )
    data = _coerce_rewrite(raw)
    suggested = data.get("suggested_text") or current_text
    reasoning = data.get("reasoning") or ""
    return str(suggested), str(reasoning)


def tailor_bullet(
    *,
    bullet_id: str,
    source_text: str,
    jd_id: str,
    targets: list[str],
    llm: LLMClient | None = None,
    embedder: EmbeddingClient | None = None,
) -> TailoringSuggestion:
    """Rewrite a bullet toward its target requirements and attach an integrity verdict."""
    suggested, reasoning = rewrite_bullet(source_text, targets=targets, llm=llm)
    integrity = judge_integrity(source_text, suggested, llm=llm, embedder=embedder)
    return TailoringSuggestion(
        bullet_id=bullet_id,
        jd_id=jd_id,
        suggested_text=suggested,
        reasoning=reasoning,
        integrity=integrity,
    )


def select_candidate_bullets(
    jd: ParsedJD,
    *,
    embedder: EmbeddingClient,
    collection,
    resume_version_id: str | None = None,
    per_requirement: int = CANDIDATES_PER_REQUIREMENT,
) -> dict[str, list[str]]:
    """Map ``bullet_id`` → the JD requirement texts it best evidences (vector search).

    A bullet that surfaces for several requirements is rewritten once, emphasising all of
    them. Optionally scope the search to one resume version.
    """
    mapping: dict[str, list[str]] = {}
    for requirement in _jd_requirements(jd):
        res = query_bullets(
            requirement,
            embedder=embedder,
            collection=collection,
            n_results=per_requirement,
            resume_version_id=resume_version_id,
        )
        for bullet_id in res["ids"][0]:
            mapping.setdefault(bullet_id, []).append(requirement)
    return mapping


def generate_suggestions(
    session: Session,
    *,
    resume_version_id: str,
    jd: ParsedJD,
    jd_id: str | None = None,
    embedder: EmbeddingClient,
    collection,
    llm: LLMClient | None = None,
) -> list[TailoringSuggestion]:
    """Full tailoring pass: map → rewrite → judge → persist ``pending`` suggestions.

    ``jd_id`` is the authoritative SQLite ``jds.id`` (defaults to ``jd.id`` for standalone
    use). Persists each suggestion via the repo (which flushes); the caller commits.
    """
    jid = jd_id or jd.id
    mapping = select_candidate_bullets(
        jd, embedder=embedder, collection=collection, resume_version_id=resume_version_id
    )

    suggestions: list[TailoringSuggestion] = []
    for bullet_id, targets in mapping.items():
        row = repo.get_bullet(session, bullet_id)
        if row is None:
            logger.warning("tailor: bullet %s indexed in Chroma but missing from SQLite", bullet_id)
            continue
        suggestion = tailor_bullet(
            bullet_id=bullet_id,
            source_text=row.current_text,
            jd_id=jid,
            targets=targets,
            llm=llm,
            embedder=embedder,
        )
        integrity = suggestion.integrity
        row_db = repo.create_suggestion(
            session,
            bullet_id=bullet_id,
            jd_id=jid,
            suggested_text=suggestion.suggested_text,
            reasoning=suggestion.reasoning,
            llm_judgment=integrity.llm_judgment if integrity else None,
            embedding_similarity=integrity.embedding_similarity if integrity else None,
            integrity_score=integrity.score if integrity else None,
            integrity_band=integrity.band if integrity else None,
            integrity_flags=integrity.unsupported() if integrity else None,
            status=suggestion.status,
        )
        # Return the persisted row's id as the canonical one, so callers (API /review,
        # Phase 13 UI) act on the suggestion that actually exists in the DB.
        suggestions.append(suggestion.model_copy(update={"id": row_db.id}))
    return suggestions


# --- helpers ------------------------------------------------------------------------


def _jd_requirements(jd: ParsedJD) -> list[str]:
    """Every JD signal a bullet should be matched against — required *and* preferred skills,
    responsibilities, and named tools (the four ``jd_chunks`` requirement types). De-duped,
    preserving order, so a term repeated across fields drives one search."""
    requirements = (
        *jd.required_skills,
        *jd.preferred_skills,
        *jd.responsibilities,
        *jd.tools,
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in requirements:
        text = raw.strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            ordered.append(text)
    return ordered


def _refine_user(source_text: str, current_text: str, instruction: str) -> str:
    return (
        f"ORIGINAL EVIDENCE (never exceed what this supports):\n{source_text}\n\n"
        f"CURRENT BULLET:\n{current_text}\n\n"
        f"REFINEMENT INSTRUCTION:\n{instruction}"
    )


def _rewrite_user(source_text: str, targets: list[str]) -> str:
    target_block = "\n".join(f"- {t}" for t in targets) if targets else "(none)"
    return (
        f"ORIGINAL BULLET:\n{source_text}\n\n"
        f"TARGETS (emphasise only what the bullet supports):\n{target_block}"
    )


def _coerce_rewrite(raw: object) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("tailor.rewrite non-JSON response; keeping original bullet")
            return {}
    return raw if isinstance(raw, dict) else {}
