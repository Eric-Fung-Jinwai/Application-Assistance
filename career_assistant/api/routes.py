"""API routes wiring each pipeline stage to an HTTP endpoint (Phase 14).

Endpoints are sync ``def`` so FastAPI runs the blocking DB / LLM / embedding work in a thread
pool rather than on the event loop. Each route owns its commit; the session dependency guarantees
close. Domain text (resume/JD parses) is the source of truth, so responses return the domain
models directly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from career_assistant.api import deps
from career_assistant.api.schemas import (
    AcceptAllRequest,
    AcceptAllResponse,
    AnalyzeJDRequest,
    AnalyzeJDResponse,
    BudgetLineItem,
    BudgetResponse,
    GenerateRequest,
    PairRequest,
    ParseResumeResponse,
    Recommendation,
    ReviewAction,
    ReviewRequest,
    ReviewResponse,
    TailorResponse,
    VersionRef,
)
from career_assistant.config import settings
from career_assistant.domain import FitScore, ParsedJD, ParsedResume, VersionType
from career_assistant.eval.budget import (
    ApplicationProfile,
    build_summary,
    embedding_label,
    is_local_embedding,
)
from career_assistant.export import render_html, render_pdf
from career_assistant.fit import score_fit
from career_assistant.ingest.extract import extract_resume
from career_assistant.ingest.parser import parse_file
from career_assistant.jd import analyze_coverage, analyze_jd, build_jd_kb
from career_assistant.kb import build_resume_kb
from career_assistant.llm.pricing import COMPLETION_PRICING
from career_assistant.recommend import build_recommendation
from career_assistant.review import (
    accept_suggestion,
    accept_suggestions,
    customize_suggestion,
    refine_suggestion,
    reject_suggestion,
)
from career_assistant.storage import repo
from career_assistant.storage.models import JD, Resume, ResumeVersion, TailoringSuggestionRow
from career_assistant.tailor import generate_suggestions

router = APIRouter()


# --- ingest ----------------------------------------------------------------------


@router.post("/parse_resume", response_model=ParseResumeResponse)
def parse_resume(
    file: UploadFile = File(...),
    session: Session = Depends(deps.get_session),
    llm=Depends(deps.get_llm),
    embedder=Depends(deps.get_embedder),
    collection=Depends(deps.get_resume_collection),
) -> ParseResumeResponse:
    """Upload → parsed resume + an ``original`` version with its bullets embedded."""
    suffix = Path(file.filename or "").suffix
    content = file.file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        raw = parse_file(tmp_path)  # typed ingest errors → handlers map to 4xx
    finally:
        os.unlink(tmp_path)
    raw.filename = file.filename or raw.filename  # keep the user's name, not the temp path

    parsed = extract_resume(raw, llm=llm)
    resume = repo.create_resume(
        session, filename=raw.filename, raw_text=raw.raw_text, parsed=parsed.model_dump()
    )
    version = repo.create_version(session, resume_id=resume.id, version_type=VersionType.original)
    build_resume_kb(
        session,
        resume_version_id=version.id,
        parsed=parsed,
        embedder=embedder,
        collection=collection,
    )  # commits
    return ParseResumeResponse(resume_id=resume.id, resume_version_id=version.id, parsed=parsed)


@router.post("/analyze_jd", response_model=AnalyzeJDResponse)
def analyze_jd_endpoint(
    body: AnalyzeJDRequest,
    session: Session = Depends(deps.get_session),
    llm=Depends(deps.get_llm),
    embedder=Depends(deps.get_embedder),
    collection=Depends(deps.get_jd_collection),
) -> AnalyzeJDResponse:
    """JD text → structured ``ParsedJD`` with its requirement chunks embedded."""
    parsed = analyze_jd(body.text, llm=llm)
    # Create the row first to mint the authoritative id, then store the parse aligned to it —
    # otherwise jds.parsed_json keeps the random in-memory ParsedJD.id.
    jd = repo.create_jd(session, raw_text=body.text)
    parsed = parsed.model_copy(update={"id": jd.id})
    jd.parsed_json = parsed.model_dump()
    build_jd_kb(parsed, embedder=embedder, collection=collection, jd_id=jd.id)
    session.commit()
    return AnalyzeJDResponse(jd_id=jd.id, parsed=parsed)


# --- fit / recommendation --------------------------------------------------------


@router.post("/fit_score", response_model=FitScore)
def fit_score_endpoint(
    body: PairRequest,
    session: Session = Depends(deps.get_session),
    embedder=Depends(deps.get_embedder),
) -> FitScore:
    """Score a resume version against a JD, persisting the breakdown."""
    _, resume_parsed = _load_parsed_resume(session, body.resume_version_id)
    _, jd_parsed = _load_parsed_jd(session, body.jd_id)
    fit = score_fit(resume_parsed, jd_parsed, embedder=embedder)
    repo.create_fit_score(
        session,
        resume_version_id=body.resume_version_id,
        jd_id=body.jd_id,
        skill=fit.skill,
        experience=fit.experience,
        seniority=fit.seniority,
        location=fit.location,
        comp=fit.compensation,
        overall=fit.overall,
        breakdown=fit.breakdown,
    )
    session.commit()
    return fit


@router.post("/recommendation", response_model=Recommendation)
def recommendation_endpoint(
    body: PairRequest,
    session: Session = Depends(deps.get_session),
    embedder=Depends(deps.get_embedder),
) -> Recommendation:
    """Map fit + coverage into an apply verdict (Phase 7 MVP)."""
    _, resume_parsed = _load_parsed_resume(session, body.resume_version_id)
    _, jd_parsed = _load_parsed_jd(session, body.jd_id)
    fit = score_fit(resume_parsed, jd_parsed, embedder=embedder)
    coverage = analyze_coverage(jd_parsed, resume_parsed, embedder=embedder)
    return build_recommendation(fit, coverage)


# --- tailor / review -------------------------------------------------------------


@router.post("/tailor", response_model=TailorResponse)
def tailor_endpoint(
    body: PairRequest,
    session: Session = Depends(deps.get_session),
    llm=Depends(deps.get_llm),
    embedder=Depends(deps.get_embedder),
    collection=Depends(deps.get_resume_collection),
) -> TailorResponse:
    """Generate evidence-graded tailoring suggestions for a resume version × JD."""
    if session.get(ResumeVersion, body.resume_version_id) is None:
        raise HTTPException(404, f"unknown resume version {body.resume_version_id!r}")
    _, jd_parsed = _load_parsed_jd(session, body.jd_id)
    suggestions = generate_suggestions(
        session,
        resume_version_id=body.resume_version_id,
        jd=jd_parsed,
        jd_id=body.jd_id,
        embedder=embedder,
        collection=collection,
        llm=llm,
    )
    session.commit()
    return TailorResponse(suggestions=suggestions)


@router.post("/review", response_model=ReviewResponse)
def review_endpoint(
    body: ReviewRequest,
    session: Session = Depends(deps.get_session),
    llm=Depends(deps.get_llm),
    embedder=Depends(deps.get_embedder),
    collection=Depends(deps.get_resume_collection),
) -> ReviewResponse:
    """Apply a review decision (accept / reject / customize) or chat-refine a suggestion."""
    row = session.get(TailoringSuggestionRow, body.suggestion_id)
    if row is None:
        raise HTTPException(404, f"unknown suggestion {body.suggestion_id!r}")

    if body.action == ReviewAction.refine:
        if not body.instruction:
            raise HTTPException(422, "refine requires 'instruction'")
        suggestion = refine_suggestion(
            session,
            suggestion_id=body.suggestion_id,
            instruction=body.instruction,
            llm=llm,
            embedder=embedder,
        )
        return ReviewResponse(
            action=body.action, status=suggestion.status.value, suggestion=suggestion
        )

    if body.action == ReviewAction.accept:
        outcome = accept_suggestion(
            session,
            suggestion_id=body.suggestion_id,
            base_version_id=body.base_version_id,
            embedder=embedder,
            collection=collection,
        )
    elif body.action == ReviewAction.customize:
        if body.edit is None:
            raise HTTPException(422, "customize requires 'edit'")
        outcome = customize_suggestion(
            session,
            suggestion_id=body.suggestion_id,
            new_text=body.edit,
            base_version_id=body.base_version_id,
            llm=llm,
            embedder=embedder,
            collection=collection,
        )
    else:  # reject — needs a branch head; default to the suggestion's own version
        bullet = repo.get_bullet(session, row.bullet_id)
        base = body.base_version_id or (bullet.resume_version_id if bullet else None)
        if base is None:
            raise HTTPException(400, "cannot resolve a base version to reject against")
        outcome = reject_suggestion(session, suggestion_id=body.suggestion_id, base_version_id=base)

    return ReviewResponse(
        action=body.action,
        status=outcome.suggestion.status,
        version=VersionRef(
            version_id=outcome.version.id,
            version_type=outcome.version.version_type,
            created=outcome.created_version,
        ),
    )


@router.post("/accept_all", response_model=AcceptAllResponse)
def accept_all_endpoint(
    body: AcceptAllRequest,
    session: Session = Depends(deps.get_session),
    embedder=Depends(deps.get_embedder),
    collection=Depends(deps.get_resume_collection),
) -> AcceptAllResponse:
    """Apply many pending suggestions in one accepted version (backs 'Accept all')."""
    outcome = accept_suggestions(
        session,
        suggestion_ids=body.suggestion_ids,
        base_version_id=body.base_version_id,
        embedder=embedder,
        collection=collection,
    )
    return AcceptAllResponse(
        version=VersionRef(
            version_id=outcome.version.id,
            version_type=outcome.version.version_type,
            created=outcome.created_version,
        ),
        accepted_count=len(body.suggestion_ids),
    )


# --- budget ----------------------------------------------------------------------


@router.get("/budget", response_model=BudgetResponse)
def budget_endpoint(
    bullets: int = 20,
    jd_chunks: int = 10,
    tailored: int = 8,
    jds_per_resume: int = 1,
    jds: int = 100,
) -> BudgetResponse:
    """Projected cost/latency for one application (estimate from token assumptions, not actuals).

    Reflects the configured models, so a local embedder shows $0 for embeddings and an unpriced
    completion model surfaces ``pricing_known=False`` instead of a misleading $0.
    """
    profile = ApplicationProfile(
        n_bullets=bullets,
        n_jd_chunks=jd_chunks,
        n_tailored=tailored,
        jds_per_resume=jds_per_resume,
    )
    summary = build_summary(
        profile,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_is_local=is_local_embedding(settings.embedding_provider),
    )
    per = summary.per_run_cost
    return BudgetResponse(
        llm_model=settings.llm_model,
        embedding=embedding_label(settings.embedding_provider, settings.embedding_model),
        per_application_cost=per,
        per_application_latency_s=summary.per_run_latency_s,
        benchmark_jds=jds,
        benchmark_cost=None if per is None else per * jds,
        pricing_known=settings.llm_model in COMPLETION_PRICING,
        items=[
            BudgetLineItem(
                label=it.label, kind=it.kind, calls=it.n_calls, cost=it.cost, latency_s=it.latency_s
            )
            for it in summary.items
        ],
    )


# --- export ----------------------------------------------------------------------


@router.post("/generate_html", response_class=HTMLResponse)
def generate_html_endpoint(
    body: GenerateRequest,
    session: Session = Depends(deps.get_session),
) -> HTMLResponse:
    """Render a resume version to ATS-friendly HTML in the requested template."""
    html = render_html(
        session, version_id=body.resume_version_id, template=body.template, name=body.name
    )
    return HTMLResponse(content=html)


@router.post("/generate_pdf")
def generate_pdf_endpoint(
    body: GenerateRequest,
    session: Session = Depends(deps.get_session),
) -> Response:
    """Render a resume version to a PDF (selectable text). 503 if Chromium isn't installed."""
    pdf_bytes = render_pdf(
        session, version_id=body.resume_version_id, template=body.template, name=body.name
    )
    return Response(content=pdf_bytes, media_type="application/pdf")


# --- helpers ---------------------------------------------------------------------


def _load_parsed_resume(
    session: Session, resume_version_id: str
) -> tuple[ResumeVersion, ParsedResume]:
    version = session.get(ResumeVersion, resume_version_id)
    if version is None:
        raise HTTPException(404, f"unknown resume version {resume_version_id!r}")
    resume = session.get(Resume, version.resume_id)
    parsed = (
        ParsedResume.model_validate(resume.parsed_json)
        if resume and resume.parsed_json
        else ParsedResume()
    )
    return version, parsed


def _load_parsed_jd(session: Session, jd_id: str) -> tuple[JD, ParsedJD]:
    jd = session.get(JD, jd_id)
    if jd is None:
        raise HTTPException(404, f"unknown jd {jd_id!r}")
    parsed = ParsedJD.model_validate(jd.parsed_json) if jd.parsed_json else ParsedJD(id=jd.id)
    return jd, parsed
