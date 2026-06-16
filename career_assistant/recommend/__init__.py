"""Recommendation engine: fit + coverage → apply verdict (Phase 7 MVP)."""

from career_assistant.recommend.recommend import (
    Recommendation,
    band_for_overall,
    build_recommendation,
)

__all__ = ["Recommendation", "band_for_overall", "build_recommendation"]
