"""Request/response models for the FastAPI backend (Phase 14).

Responses reuse the domain pydantic models (``ParsedResume``, ``ParsedJD``, ``FitScore``,
``TailoringSuggestion``, ``Recommendation``) directly — the API contract is the domain contract,
so there is one source of truth and no hand-copied field drift.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from career_assistant.domain import FitScore, ParsedJD, ParsedResume, TailoringSuggestion
from career_assistant.recommend import Recommendation


class ReviewAction(StrEnum):
    accept = "accept"
    reject = "reject"
    customize = "customize"
    refine = "refine"


# --- requests --------------------------------------------------------------------


class AnalyzeJDRequest(BaseModel):
    text: str = Field(min_length=1)


class PairRequest(BaseModel):
    """A resume version paired with a JD — shared by fit / recommendation / tailor."""

    resume_version_id: str
    jd_id: str


class ReviewRequest(BaseModel):
    suggestion_id: str
    action: ReviewAction
    # base_version_id defaults (server-side) to the suggestion's own version when omitted.
    base_version_id: str | None = None
    edit: str | None = None  # required for action=customize (the reviewer's text)
    instruction: str | None = None  # required for action=refine (the chat instruction)


class AcceptAllRequest(BaseModel):
    suggestion_ids: list[str] = Field(min_length=1)
    base_version_id: str | None = None


class GenerateRequest(BaseModel):
    resume_version_id: str
    template: str = "ats"
    name: str | None = None


# --- responses -------------------------------------------------------------------


class ParseResumeResponse(BaseModel):
    resume_id: str
    resume_version_id: str
    parsed: ParsedResume


class AnalyzeJDResponse(BaseModel):
    jd_id: str
    parsed: ParsedJD


class TailorResponse(BaseModel):
    suggestions: list[TailoringSuggestion]


class VersionRef(BaseModel):
    version_id: str
    version_type: str
    created: bool  # True when a new version was created (accept/customize), False for reject


class ReviewResponse(BaseModel):
    action: ReviewAction
    status: str
    # accept/reject/customize return the resulting head version; refine returns none + a suggestion.
    version: VersionRef | None = None
    suggestion: TailoringSuggestion | None = None


class AcceptAllResponse(BaseModel):
    version: VersionRef  # the single accepted child holding every applied edit
    accepted_count: int


class BudgetLineItem(BaseModel):
    label: str
    kind: str  # "completion" | "embedding"
    calls: int
    cost: float | None  # None when the model's price is unknown
    latency_s: float


class BudgetResponse(BaseModel):
    """Projected per-application cost/latency — estimated from token assumptions, not metered."""

    llm_model: str
    embedding: str
    per_application_cost: float | None
    per_application_latency_s: float
    benchmark_jds: int
    benchmark_cost: float | None
    pricing_known: bool  # False → the cost is a placeholder until the model is priced
    items: list[BudgetLineItem]


# Re-exported so routers can annotate with the domain types as response models.
__all__ = [
    "AcceptAllRequest",
    "AcceptAllResponse",
    "AnalyzeJDRequest",
    "AnalyzeJDResponse",
    "BudgetLineItem",
    "BudgetResponse",
    "FitScore",
    "GenerateRequest",
    "PairRequest",
    "ParseResumeResponse",
    "Recommendation",
    "ReviewAction",
    "ReviewRequest",
    "ReviewResponse",
    "TailorResponse",
    "VersionRef",
]
