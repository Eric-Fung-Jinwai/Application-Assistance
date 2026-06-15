"""Render resume HTML to PDF with headless Chromium via Playwright (Phase 12).

Chromium prints the HTML's real text layer, so the PDF has **selectable text** (ATS-safe,
``pdftotext``/``pypdf`` recover the bullets) and paginates naturally — no rasterization. The
browser binary is a one-time setup step: ``playwright install chromium`` (see README/setup).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from career_assistant.export.html import DEFAULT_TEMPLATE, render_html

# A4 with comfortable margins; print_background keeps template accent colors/rules.
_PDF_OPTIONS = {"format": "A4", "print_background": True}


def html_to_pdf(html: str) -> bytes:
    """Render a self-contained HTML string to PDF bytes via headless Chromium.

    Raises ``RuntimeError`` with the setup hint if the Chromium binary isn't installed.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # browser not installed / launch failure
            raise RuntimeError(
                "Chromium is required for PDF export. Run `playwright install chromium`."
            ) from exc
        try:
            page = browser.new_page()
            # The HTML is self-contained (embedded CSS, no external assets), so "load" suffices.
            page.set_content(html, wait_until="load")
            return page.pdf(**_PDF_OPTIONS)
        finally:
            browser.close()


def render_pdf(
    session: Session,
    *,
    version_id: str,
    template: str = DEFAULT_TEMPLATE,
    name: str | None = None,
) -> bytes:
    """Assemble + render ``version_id`` to HTML, then to a PDF byte string."""
    html = render_html(session, version_id=version_id, template=template, name=name)
    return html_to_pdf(html)
