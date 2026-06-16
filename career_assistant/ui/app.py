"""Streamlit MVP for the Evidence-Aware Career Application Assistant (Phase 13).

A single warm-editorial page (see ``theme.py`` / DESIGN.md) walking the full happy path against
the FastAPI backend: upload → JD → fit → recommendation → tailor → review → export. The UI holds
no business logic — every step calls the API via ``ApiClient`` — so the browser exercises the same
contract as any other client. Run the backend (``uvicorn career_assistant.api.app:app``) and then
``streamlit run career_assistant/ui/app.py``.
"""

from __future__ import annotations

import streamlit as st

from career_assistant.ui import theme
from career_assistant.ui.api_client import ApiClient, ApiError
from career_assistant.ui.theme import esc, esc_md


def get_client() -> ApiClient:
    if "api_client" not in st.session_state:
        st.session_state.api_client = ApiClient()
    return st.session_state.api_client


def header() -> None:
    st.markdown(
        '<div class="ca-eyebrow"><span class="ca-spike">✳</span> Evidence-Aware</div>',
        unsafe_allow_html=True,
    )
    st.title("Tailor your resume — without overstating it.")
    st.markdown(
        "Upload a resume, paste a job description, and get evidence-grounded rewrites that an "
        "integrity check refuses to let drift into fabrication.",
        unsafe_allow_html=True,
    )
    st.divider()


def section_upload() -> None:
    st.subheader("1 · Resume")
    if st.session_state.get("resume_version_id"):
        _render_parsed_resume(st.session_state.resume_parsed)
        if st.button("Upload a different resume"):
            _reset_from("resume_version_id")
            st.rerun()
        return

    uploaded = st.file_uploader("Resume file", type=["txt", "pdf", "docx"])
    with st.expander("Search preferences (optional — captured locally for now)"):
        st.session_state.setdefault("prefs", {})
        st.session_state.prefs["goal"] = st.text_input("Target role / career goal", "")
        st.session_state.prefs["locations"] = st.text_input("Preferred locations", "")
        st.caption(
            "Preferences aren't fed to scoring yet — that arrives with onboarding (Phase 2)."
        )

    if uploaded and st.button("Parse resume", type="primary"):
        with st.spinner("Parsing and indexing resume…"):
            try:
                body = get_client().parse_resume(
                    filename=uploaded.name, content=uploaded.getvalue()
                )
            except ApiError as exc:
                st.error(f"Could not parse resume: {exc.detail}")
                return
        st.session_state.resume_id = body["resume_id"]
        st.session_state.resume_version_id = body["resume_version_id"]
        st.session_state.head_version_id = body["resume_version_id"]
        st.session_state.resume_parsed = body["parsed"]
        st.rerun()


def _render_parsed_resume(parsed: dict) -> None:
    """Show the full parsed resume so the user can verify the extraction (not just skills)."""
    exp = parsed.get("experience", [])
    proj = parsed.get("projects", [])
    edu = parsed.get("education", [])
    skills = parsed.get("skills", [])
    certs = parsed.get("certifications", [])

    confidence = parsed.get("confidence", 1.0)
    if confidence < 1.0 or not exp:
        st.warning(
            f"Parse confidence {confidence:.0%}. Some sections look empty — if this is a PDF, a "
            "**text-based** (not scanned) file extracts best; a .docx or .txt is most reliable."
        )

    cols = st.columns(4)
    cols[0].metric("Roles", len(exp))
    cols[1].metric("Projects", len(proj))
    cols[2].metric("Education", len(edu))
    cols[3].metric("Skills", len(skills))

    if exp:
        st.markdown("**Experience**")
        for role in exp:
            head = " — ".join(p for p in [role.get("title", ""), role.get("company", "")] if p)
            dates = " – ".join(p for p in [role.get("start_date"), role.get("end_date")] if p)
            st.markdown(f"- {esc_md(head)}" + (f"  ·  _{esc_md(dates)}_" if dates else ""))
            for bullet in role.get("bullets", []):
                st.markdown(f"    - {esc_md(bullet)}")

    if proj:
        st.markdown("**Projects**")
        for p in proj:
            st.markdown(f"- {esc_md(p.get('name', ''))}")
            for bullet in p.get("bullets", []):
                st.markdown(f"    - {esc_md(bullet)}")

    if edu:
        st.markdown("**Education**")
        for e in edu:
            line = ", ".join(
                part
                for part in [e.get("institution"), e.get("degree"), e.get("field_of_study")]
                if part
            )
            st.markdown(f"- {esc_md(line)}")

    if skills:
        st.markdown("**Skills**: " + esc_md(", ".join(skills)))
    if certs:
        st.markdown("**Certifications**: " + esc_md(", ".join(certs)))

    with st.expander("Raw parsed JSON (debug)"):
        st.json(parsed)


def section_jd() -> None:
    st.subheader("2 · Job description")
    if st.session_state.get("jd_id"):
        jd = st.session_state.jd_parsed
        required = esc(", ".join(jd.get("required_skills", [])) or "—")
        preferred = esc(", ".join(jd.get("preferred_skills", [])) or "—")
        st.markdown(
            f'<div class="ca-card">Required: {required}<br>Preferred: {preferred}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Analyze a different JD"):
            _reset_jd()
            st.rerun()
        return

    text = st.text_area("Paste the job description", height=200)
    if text.strip() and st.button("Analyze JD", type="primary"):
        with st.spinner("Analyzing job description…"):
            try:
                body = get_client().analyze_jd(text)
            except ApiError as exc:
                st.error(f"Could not analyze JD: {exc.detail}")
                return
        st.session_state.jd_id = body["jd_id"]
        st.session_state.jd_parsed = body["parsed"]
        st.rerun()


def section_fit() -> None:
    st.subheader("3 · Fit & recommendation")
    rv, jd = st.session_state.head_version_id, st.session_state.jd_id
    client = get_client()
    try:
        fit = client.fit_score(rv, jd)
        rec = client.recommendation(rv, jd)
    except ApiError as exc:
        st.error(f"Could not score fit: {exc.detail}")
        return

    left, right = st.columns([3, 2])
    with left:
        overall = f'{fit["overall"]:.0f}<span style="font-size:1rem">/100</span>'
        st.markdown(
            f'<div class="ca-eyebrow">Overall fit · {esc(fit["band"] or "—")}</div>'
            f'<h2 style="margin:0">{overall}</h2>',
            unsafe_allow_html=True,
        )
        for dim in ("skill", "experience", "seniority", "location", "compensation"):
            st.markdown(
                f'<div style="margin-top:8px"><span class="ca-eyebrow">{dim}</span>'
                f"{theme.meter(fit[dim])}</div>",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            f'<div class="ca-card"><span class="ca-eyebrow">Recommendation</span><br>'
            f'<b style="font-size:1.1rem">{esc(rec["band"])}</b></div>',
            unsafe_allow_html=True,
        )
        if rec["strengths"]:
            st.markdown("**Strengths**: " + esc_md(", ".join(rec["strengths"])))
        if rec["missing_requirements"]:
            st.markdown("**Missing**: " + esc_md(", ".join(rec["missing_requirements"])))
        for concern in rec["concerns"]:
            st.caption(esc_md(concern))


def section_tailor_review() -> None:
    st.subheader("4 · Tailoring & review")
    client = get_client()

    if st.button("Generate tailoring suggestions", type="primary"):
        with st.spinner("Rewriting bullets and grading integrity…"):
            try:
                st.session_state.suggestions = client.tailor(
                    st.session_state.head_version_id, st.session_state.jd_id
                )
                st.session_state.done_ids = set()
            except ApiError as exc:
                st.error(f"Could not generate suggestions: {exc.detail}")
                return

    suggestions = st.session_state.get("suggestions")
    if not suggestions:
        st.caption("No suggestions yet — generate them to start reviewing.")
        return

    _integrity_dashboard(suggestions)
    done = st.session_state.setdefault("done_ids", set())
    pending_ids = [s["id"] for s in suggestions if s["id"] not in done]
    if pending_ids and st.button(f"Accept all ({len(pending_ids)})", type="primary"):
        with st.spinner("Applying all suggestions…"):
            try:
                out = client.accept_all(
                    pending_ids, base_version_id=st.session_state.head_version_id
                )
            except ApiError as exc:
                st.error(f"Accept all failed: {exc.detail}")
            else:
                # One new version holds every edit; mark them all decided and advance the head.
                st.session_state.head_version_id = out["version"]["version_id"]
                st.session_state.done_ids.update(pending_ids)
                st.rerun()

    for sug in suggestions:
        _review_card(client, sug, decided=sug["id"] in done)


def _integrity_dashboard(suggestions: list[dict]) -> None:
    counts: dict[str, int] = {}
    for sug in suggestions:
        band = (sug.get("integrity") or {}).get("band", "unknown")
        counts[band] = counts.get(band, 0) + 1
    chips = " ".join(
        theme.badge(f"{band} · {n}", theme.band_color(band)) for band, n in sorted(counts.items())
    )
    st.markdown(
        f'<div class="ca-card"><span class="ca-eyebrow">Integrity overview</span><br>{chips}</div>',
        unsafe_allow_html=True,
    )


def _review_card(client: ApiClient, sug: dict, *, decided: bool) -> None:
    integrity = sug.get("integrity") or {}
    band = integrity.get("band", "unknown")
    score = integrity.get("score")
    badge_html = theme.badge(
        f"{band}{f' · {score:.0f}' if score is not None else ''}", theme.band_color(band)
    )
    flags = _unsupported_flags(integrity)
    flags_html = (
        f'<div class="ca-eyebrow" style="margin-top:6px;color:{theme.COLORS["error"]}">'
        f"⚠ unsupported: {esc(', '.join(flags))}</div>"
        if flags
        else ""
    )
    st.markdown(
        f'<div class="ca-card">{badge_html}'
        f'<div class="ca-suggested" style="margin-top:8px">{esc(sug["suggested_text"])}</div>'
        f'<div class="ca-eyebrow" style="margin-top:6px">{esc(sug.get("reasoning", ""))}</div>'
        f"{flags_html}</div>",
        unsafe_allow_html=True,
    )
    if decided:
        st.caption("✓ decided")
        return

    sid = sug["id"]
    c1, c2, c3 = st.columns(3)
    if c1.button("Accept", key=f"acc_{sid}", type="primary"):
        _do_review(client, suggestion_id=sid, action="accept")
    if c2.button("Reject", key=f"rej_{sid}"):
        _do_review(client, suggestion_id=sid, action="reject")
    with c3.popover("Customize / Refine"):
        edit = st.text_input("Your edited text", key=f"edit_{sid}")
        if st.button("Save edit", key=f"saveedit_{sid}") and edit.strip():
            _do_review(client, suggestion_id=sid, action="customize", edit=edit)
        instr = st.text_input("Refine instruction (e.g. 'make shorter')", key=f"instr_{sid}")
        if st.button("Refine", key=f"refine_{sid}") and instr.strip():
            _do_review(client, suggestion_id=sid, action="refine", instruction=instr)


def _unsupported_flags(integrity: dict) -> list[str]:
    """Flatten the integrity result's non-empty unsupported-* lists for display."""
    fields = (
        "unsupported_claims",
        "unsupported_technologies",
        "unsupported_metrics",
        "unsupported_responsibilities",
    )
    flags: list[str] = []
    for field in fields:
        flags.extend(integrity.get(field) or [])
    return flags


def _do_review(client: ApiClient, **kwargs) -> None:
    try:
        out = client.review(base_version_id=st.session_state.head_version_id, **kwargs)
    except ApiError as exc:
        st.error(f"Review failed: {exc.detail}")
        return
    if out.get("version"):  # accept/customize created a new head
        st.session_state.head_version_id = out["version"]["version_id"]
        st.session_state.done_ids.add(kwargs["suggestion_id"])
    elif kwargs["action"] == "reject":
        st.session_state.done_ids.add(kwargs["suggestion_id"])
    elif out.get("suggestion"):  # refine → replace in place with the re-scored suggestion
        st.session_state.suggestions = [
            out["suggestion"] if s["id"] == kwargs["suggestion_id"] else s
            for s in st.session_state.suggestions
        ]
    st.rerun()


def section_export() -> None:
    st.subheader("5 · Export")
    client = get_client()
    rv = st.session_state.head_version_id
    template = st.selectbox("Template", ["ats", "technical", "pm", "modern"])

    col1, col2 = st.columns(2)
    if col1.button("Preview HTML", type="primary"):
        try:
            html = client.generate_html(rv, template)
        except ApiError as exc:
            st.error(f"Could not render HTML: {exc.detail}")
        else:
            st.components.v1.html(html, height=600, scrolling=True)

    if col2.button("Build PDF"):
        try:
            pdf = client.generate_pdf(rv, template)
        except ApiError as exc:
            # 503 = Chromium not installed; surface the setup hint, don't crash.
            st.warning(f"PDF unavailable: {exc.detail}")
        else:
            st.download_button(
                "Download PDF", data=pdf, file_name=f"resume_{template}.pdf", mime="application/pdf"
            )

    # Close the loop: tailor the SAME resume for the next job without re-uploading.
    st.divider()
    if st.button("📋 Tailor for another job — keep my resume"):
        _reset_jd()
        st.rerun()


def _reset_from(*keys: str) -> None:
    """Clear a step and everything downstream of it so the flow stays consistent."""
    downstream = [
        "resume_version_id",
        "resume_id",
        "resume_parsed",
        "head_version_id",
        "jd_id",
        "jd_parsed",
        "suggestions",
        "done_ids",
    ]
    start = downstream.index(keys[0])
    for key in downstream[start:]:
        st.session_state.pop(key, None)


# Session keys scoped to one JD pass — cleared when starting another job.
_JD_STATE_KEYS = ("jd_id", "jd_parsed", "suggestions", "done_ids")


def _apply_new_jd(state) -> None:
    """Start a fresh JD against the SAME uploaded resume.

    Clears the JD + tailoring state and rewinds the working head back to the *original* uploaded
    resume version, so every job tailors from the clean resume rather than the previous job's
    accepted edits. Pure (operates on the given mapping) so it is unit-testable without Streamlit.
    """
    for key in _JD_STATE_KEYS:
        state.pop(key, None)
    original = state.get("resume_version_id")
    if original:
        state["head_version_id"] = original


def _reset_jd() -> None:
    _apply_new_jd(st.session_state)


def sidebar_budget() -> None:
    """Show the projected per-application cost (Phase 0 estimator) in the sidebar."""
    with st.sidebar:
        st.markdown("### 💸 Cost estimate")
        # Use the real tailored-bullet count once suggestions exist, else the default profile.
        tailored = len(st.session_state.get("suggestions") or []) or 8
        try:
            b = get_client().budget(tailored=tailored)
        except ApiError as exc:
            st.caption(f"Estimate unavailable: {exc.detail}")
            return
        cost = b["per_application_cost"]
        st.metric("Per application", f"${cost:.4f}" if cost is not None else "—")
        st.caption(f"≈ {b['per_application_latency_s']:.0f}s · {b['llm_model']} · {b['embedding']}")
        if not b["pricing_known"]:
            st.caption("⚠ pricing for this model is a placeholder estimate.")
        with st.expander("Breakdown"):
            for it in b["items"]:
                c = it["cost"]
                price = f"${c:.4f}" if c is not None else "?"
                st.write(f"{it['label']} — {price} ({it['calls']} calls)")
        st.caption("Estimate from token assumptions, not metered usage.")


def main() -> None:
    # set_page_config must be the first Streamlit call; keep it here (not at import) so the
    # module stays import-safe for tests. Streamlit reruns main() top-to-bottom each interaction.
    st.set_page_config(page_title="Career Assistant", page_icon="✳", layout="wide")
    st.markdown(theme.build_css(), unsafe_allow_html=True)
    sidebar_budget()
    header()
    section_upload()
    if not st.session_state.get("resume_version_id"):
        return
    st.divider()
    section_jd()
    if not st.session_state.get("jd_id"):
        return
    st.divider()
    section_fit()
    st.divider()
    section_tailor_review()
    st.divider()
    section_export()


if __name__ == "__main__":  # Streamlit runs the script as __main__; import stays side-effect-free.
    main()
