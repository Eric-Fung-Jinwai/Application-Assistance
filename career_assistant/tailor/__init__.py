"""Evidence-aware tailoring + integrity scoring (Phase 8)."""

from career_assistant.tailor.integrity import (
    band_for_score,
    compute_integrity_score,
    judge_integrity,
)
from career_assistant.tailor.tailor import (
    generate_suggestions,
    refine_bullet,
    rewrite_bullet,
    select_candidate_bullets,
    tailor_bullet,
)

__all__ = [
    "band_for_score",
    "compute_integrity_score",
    "generate_suggestions",
    "judge_integrity",
    "refine_bullet",
    "rewrite_bullet",
    "select_candidate_bullets",
    "tailor_bullet",
]
