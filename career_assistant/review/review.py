"""Human-in-the-loop review (Phase 10): turn a tailoring suggestion into a decision.

A reviewer sees a payload (``ReviewItem``: original text, AI suggestion, reasoning, integrity
score/band, and the unsupported-claim flags) and takes one of three actions:

  * **Accept**  → apply the suggestion as an edit, growing the version tree with an
    ``accepted`` child; the prior integrity verdict stands (the human is the gate).
  * **Reject**  → revert to the latest accepted version *on this branch* (or the original if
    none); navigation only, no new version.
  * **Customize** → either a *direct edit* (the reviewer's own text) or a *chat refinement*
    (``make shorter`` / ``emphasize Docker`` …). Both re-run the integrity judge so the edited
    text is re-scored before it can be persisted — a fresh edit never inherits a stale score.

Edits are keyed by **lineage_id** (Phase 9), so a suggestion captured against an earlier
version still applies to whatever ``base_version_id`` is the current head — no id remapping.
Every action also logs to ``edit_history``. The caller owns the session; these helpers commit.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from career_assistant.domain import (
    IntegrityResult,
    SuggestionStatus,
    TailoringSuggestion,
    VersionType,
)
from career_assistant.kb.embeddings import embed_and_upsert_bullets
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.storage import repo
from career_assistant.storage.models import (
    BulletRow,
    ResumeVersion,
    TailoringSuggestionRow,
)
from career_assistant.tailor.integrity import judge_integrity
from career_assistant.tailor.tailor import refine_bullet
from career_assistant.versioning import (
    bullets_by_lineage,
    create_child_version,
    latest_accepted_ancestor,
    restore_original,
)

# Edit-history action labels + the actor for human-driven review.
_ACTOR_USER = "user"
_ACTION_ACCEPT = "accept"
_ACTION_REJECT = "reject"
_ACTION_CUSTOMIZE = "customize"


@dataclass
class ReviewItem:
    """The per-suggestion payload shown in the review panel."""

    suggestion_id: str
    bullet_id: str
    jd_id: str
    original_text: str
    suggested_text: str
    reasoning: str
    integrity_score: float | None
    integrity_band: str | None
    integrity_flags: dict[str, list[str]]
    status: str


@dataclass
class ReviewOutcome:
    """The result of a review action: the resulting head version + the decided suggestion."""

    action: str
    suggestion: TailoringSuggestionRow
    version: ResumeVersion
    created_version: bool


def build_review_payload(session: Session, *, suggestion_id: str) -> ReviewItem:
    """Assemble the review panel payload for one suggestion (its original evidence included)."""
    row = _require_suggestion(session, suggestion_id)
    bullet = _require_bullet(session, row.bullet_id)
    return ReviewItem(
        suggestion_id=row.id,
        bullet_id=row.bullet_id,
        jd_id=row.jd_id,
        original_text=bullet.original_text,
        suggested_text=row.suggested_text,
        reasoning=row.reasoning or "",
        integrity_score=row.integrity_score,
        integrity_band=row.integrity_band,
        integrity_flags=row.integrity_flags_json or {},
        status=row.status,
    )


def accept_suggestion(
    session: Session,
    *,
    suggestion_id: str,
    base_version_id: str | None = None,
    embedder: EmbeddingClient | None = None,
    collection=None,
) -> ReviewOutcome:
    """Apply a suggestion as an edit, creating an ``accepted`` child of ``base_version_id``.

    ``base_version_id`` defaults to the version the suggestion was generated against; thread the
    current head through it when accepting several suggestions in a row so each builds on the
    last (rather than branching off the original). The suggestion's existing integrity verdict
    stands — the human reviewer is the gate, so even a flagged rewrite can be accepted on
    purpose. Re-embeds the new version's bullets when ``embedder`` and ``collection`` are given.
    """
    row = _require_pending(_require_suggestion(session, suggestion_id))
    bullet = _require_bullet(session, row.bullet_id)
    base_id = base_version_id or bullet.resume_version_id
    target = _resolve_lineage_in_base(session, bullet, base_id)

    new_version = create_child_version(
        session,
        parent_version_id=base_id,
        version_type=VersionType.accepted,
        edits={bullet.lineage_id: row.suggested_text},
    )
    _log_applied_edit(session, new_version, bullet.lineage_id, _ACTION_ACCEPT, target.current_text)
    row.status = SuggestionStatus.accepted.value
    _reembed(session, new_version, embedder=embedder, collection=collection)
    session.commit()
    return ReviewOutcome(_ACTION_ACCEPT, row, new_version, created_version=True)


def reject_suggestion(
    session: Session,
    *,
    suggestion_id: str,
    base_version_id: str,
) -> ReviewOutcome:
    """Reject a suggestion and revert to the latest accepted version on its branch.

    Navigation only — the append-only tree is untouched. Falls back to the original version when
    no accepted version exists yet on the branch. ``base_version_id`` is the current head, so the
    revert stays on this branch and never jumps to an accepted version from an unrelated one.
    """
    row = _require_pending(_require_suggestion(session, suggestion_id))
    bullet = _require_bullet(session, row.bullet_id)
    base = _require_version(session, base_version_id)
    # Confirm the suggestion belongs to this branch before reverting it — otherwise a wrong head
    # would mark it rejected and hand back an unrelated resume/branch's accepted version.
    _resolve_lineage_in_base(session, bullet, base_version_id)

    target = latest_accepted_ancestor(session, base_version_id)
    if target is None:
        target = restore_original(session, base.resume_id)

    repo.log_edit(
        session,
        bullet_id=row.bullet_id,
        action=_ACTION_REJECT,
        actor=_ACTOR_USER,
        from_text=row.suggested_text,
        to_text=None,
    )
    row.status = SuggestionStatus.rejected.value
    session.commit()
    return ReviewOutcome(_ACTION_REJECT, row, target, created_version=False)


def customize_suggestion(
    session: Session,
    *,
    suggestion_id: str,
    new_text: str,
    base_version_id: str | None = None,
    llm: LLMClient | None = None,
    embedder: EmbeddingClient | None = None,
    collection=None,
) -> ReviewOutcome:
    """Apply the reviewer's own edited text, re-scoring its integrity before persisting.

    A direct edit is unvetted text, so it is judged against the original evidence (not trusted
    like an accept) and the suggestion's integrity fields are updated to match. Creates a
    ``customized`` child of ``base_version_id`` (default: the suggestion's own version).
    """
    row = _require_pending(_require_suggestion(session, suggestion_id))
    bullet = _require_bullet(session, row.bullet_id)
    base_id = base_version_id or bullet.resume_version_id
    target = _resolve_lineage_in_base(session, bullet, base_id)

    _rescore(session, row, bullet.original_text, new_text, llm=llm, embedder=embedder)
    row.suggested_text = new_text
    row.status = SuggestionStatus.customized.value

    new_version = create_child_version(
        session,
        parent_version_id=base_id,
        version_type=VersionType.customized,
        edits={bullet.lineage_id: new_text},
    )
    _log_applied_edit(
        session, new_version, bullet.lineage_id, _ACTION_CUSTOMIZE, target.current_text
    )
    _reembed(session, new_version, embedder=embedder, collection=collection)
    session.commit()
    return ReviewOutcome(_ACTION_CUSTOMIZE, row, new_version, created_version=True)


def refine_suggestion(
    session: Session,
    *,
    suggestion_id: str,
    instruction: str,
    llm: LLMClient | None = None,
    embedder: EmbeddingClient | None = None,
) -> TailoringSuggestion:
    """Chat-refine a suggestion (``make shorter`` …) and return the re-scored result.

    Refines the current suggested text per ``instruction`` within the allowed operations (no
    fabrication), re-runs the integrity judge against the original evidence, and persists the new
    text + score onto the still-``pending`` suggestion so the panel reflects it. The reviewer
    then accepts or customizes the refined suggestion — refinement itself creates no version.

    Only a ``pending`` suggestion can be refined; refining never changes status (it stays pending).
    """
    row = _require_pending(_require_suggestion(session, suggestion_id))
    bullet = _require_bullet(session, row.bullet_id)

    suggested, reasoning = refine_bullet(
        bullet.original_text, row.suggested_text, instruction, llm=llm
    )
    integrity = _rescore(session, row, bullet.original_text, suggested, llm=llm, embedder=embedder)
    row.suggested_text = suggested
    row.reasoning = reasoning
    session.commit()

    return TailoringSuggestion(
        id=row.id,
        bullet_id=row.bullet_id,
        jd_id=row.jd_id,
        suggested_text=suggested,
        reasoning=reasoning,
        integrity=integrity,
        status=SuggestionStatus.pending,
    )


# --- helpers ------------------------------------------------------------------------


def _rescore(
    session: Session,
    row: TailoringSuggestionRow,
    source_text: str,
    new_text: str,
    *,
    llm: LLMClient | None,
    embedder: EmbeddingClient | None,
) -> IntegrityResult:
    """Judge ``new_text`` against the original evidence and write the verdict onto ``row``."""
    integrity = judge_integrity(source_text, new_text, llm=llm, embedder=embedder)
    row.llm_judgment = integrity.llm_judgment
    row.embedding_similarity = integrity.embedding_similarity
    row.integrity_score = integrity.score
    row.integrity_band = integrity.band.value
    row.integrity_flags_json = integrity.unsupported() or None
    return integrity


def _resolve_lineage_in_base(
    session: Session, bullet: BulletRow, base_version_id: str
) -> BulletRow:
    """Return the bullet carrying ``bullet``'s lineage in the base version (validates both)."""
    _require_version(session, base_version_id)
    head = bullets_by_lineage(session, base_version_id)
    target = head.get(bullet.lineage_id)
    if target is None:
        raise ValueError(
            f"suggestion's bullet lineage {bullet.lineage_id!r} is not in base version "
            f"{base_version_id!r}"
        )
    return target


def _log_applied_edit(
    session: Session,
    new_version: ResumeVersion,
    lineage_id: str,
    action: str,
    from_text: str,
) -> None:
    """Log an applied edit against the bullet that now carries ``lineage_id`` in the new version."""
    new_bullet = bullets_by_lineage(session, new_version.id)[lineage_id]
    repo.log_edit(
        session,
        bullet_id=new_bullet.id,
        action=action,
        actor=_ACTOR_USER,
        from_text=from_text,
        to_text=new_bullet.current_text,
    )


def _reembed(
    session: Session,
    version: ResumeVersion,
    *,
    embedder: EmbeddingClient | None,
    collection,
) -> None:
    """Re-embed the new version's bullets so later fit/tailoring sees the edited head."""
    if embedder is None or collection is None:
        return
    embed_and_upsert_bullets(
        repo.list_bullets(session, version.id), embedder=embedder, collection=collection
    )


def _require_suggestion(session: Session, suggestion_id: str) -> TailoringSuggestionRow:
    row = session.get(TailoringSuggestionRow, suggestion_id)
    if row is None:
        raise ValueError(f"unknown suggestion {suggestion_id!r}")
    return row


def _require_pending(row: TailoringSuggestionRow) -> TailoringSuggestionRow:
    """Guard a review action to the ``pending`` state.

    A suggestion is decided once — accept/reject/customize are terminal, and refine only edits a
    still-open one. Re-acting on a decided suggestion would fork extra versions or silently reopen
    a closed decision, so block it. (Reopening, if ever wanted, belongs in its own explicit action.)
    """
    if row.status != SuggestionStatus.pending.value:
        raise ValueError(
            f"suggestion {row.id!r} is {row.status!r}, not pending; a decided suggestion "
            f"cannot be re-reviewed"
        )
    return row


def _require_bullet(session: Session, bullet_id: str) -> BulletRow:
    bullet = repo.get_bullet(session, bullet_id)
    if bullet is None:
        raise ValueError(f"suggestion references unknown bullet {bullet_id!r}")
    return bullet


def _require_version(session: Session, version_id: str) -> ResumeVersion:
    version = session.get(ResumeVersion, version_id)
    if version is None:
        raise ValueError(f"unknown version {version_id!r}")
    return version
