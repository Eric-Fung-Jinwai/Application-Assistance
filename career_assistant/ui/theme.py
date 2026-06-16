"""Warm-editorial theme for the Streamlit UI, per DESIGN.md (Phase 13).

Translates the design tokens — cream canvas, coral CTA, warm ink, hairline cards, and the
serif-display / humanist-sans split — into CSS injected once at app start. Copernicus/StyreneB
are licensed, so we use the documented open substitutes: **Cormorant Garamond** (serif display)
and **Inter** (body), loaded from Google Fonts.
"""

from __future__ import annotations

import re
from html import escape

# CommonMark inline punctuation — backslash-escaping any of these renders it literally.
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+.!|<>~-])")

# DESIGN.md color tokens (the cream + coral + dark-navy trinity).
COLORS = {
    "primary": "#cc785c",
    "primary_active": "#a9583e",
    "ink": "#141413",
    "body": "#3d3d3a",
    "muted": "#6c6a64",
    "hairline": "#e6dfd8",
    "canvas": "#faf9f5",
    "surface_soft": "#f5f0e8",
    "surface_card": "#efe9de",
    "success": "#5db872",
    "warning": "#d4a017",
    "amber": "#e8a55a",
    "error": "#c64545",
}

# Integrity band → semantic color (safe→success, high_risk→error, etc.).
BAND_COLORS = {
    "safe": COLORS["success"],
    "moderate": COLORS["amber"],
    "aggressive": COLORS["warning"],
    "high_risk": COLORS["error"],
}


def build_css() -> str:
    """Return the full ``<style>`` block applying the DESIGN.md tokens to Streamlit."""
    c = COLORS
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');

:root {{
  --primary: {c["primary"]};
  --ink: {c["ink"]};
  --canvas: {c["canvas"]};
  --hairline: {c["hairline"]};
}}

.stApp {{ background-color: {c["canvas"]}; color: {c["body"]}; }}
.block-container {{ max-width: 1100px; padding-top: 2.5rem; }}

/* Editorial split: serif display headings, humanist sans body. */
html, body, [class*="css"], .stMarkdown, p, label, input, textarea, button {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
h1, h2, h3 {{
  font-family: 'Cormorant Garamond', 'EB Garamond', Garamond, serif !important;
  color: {c["ink"]};
  font-weight: 600;
  letter-spacing: -0.5px;
}}
h1 {{ font-size: 3rem; line-height: 1.05; }}
h2 {{ font-size: 2rem; line-height: 1.1; }}

/* Coral primary CTA; cream hairline secondary. */
.stButton > button, .stDownloadButton > button {{
  border-radius: 8px; font-weight: 500; border: 1px solid {c["hairline"]};
  background: {c["canvas"]}; color: {c["ink"]};
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: {c["primary"]}; color: #fff; border: none;
}}
.stButton > button[kind="primary"]:hover {{ background: {c["primary_active"]}; color: #fff; }}

/* Inputs on cream with hairline + coral focus. */
.stTextInput input,
.stTextArea textarea,
.stFileUploader,
.stSelectbox div[data-baseweb="select"] > div {{
  background: {c["canvas"]}; border-radius: 8px;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
  border-color: {c["primary"]}; box-shadow: 0 0 0 3px rgba(204,120,92,0.15);
}}

/* Tabs: active reads as a cream card chip. */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [aria-selected="true"] {{ background: {c["surface_card"]}; border-radius: 8px; }}

hr {{ border-color: {c["hairline"]}; }}

/* Reusable building blocks for the panels below. */
.ca-card {{
  background: {c["surface_card"]}; border-radius: 12px; padding: 24px; margin-bottom: 16px;
}}
.ca-eyebrow {{
  font-size: 12px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;
  color: {c["muted"]};
}}
.ca-badge {{
  display: inline-block; padding: 3px 12px; border-radius: 9999px; font-size: 12px;
  font-weight: 600; letter-spacing: 0.5px; color: #fff;
}}
.ca-spike {{ color: {c["primary"]}; font-weight: 700; }}
.ca-meter {{ height: 8px; border-radius: 9999px; background: {c["hairline"]}; overflow: hidden; }}
.ca-meter > span {{ display: block; height: 100%; background: {c["primary"]}; }}
.ca-suggested {{ color: {c["ink"]}; font-weight: 500; }}
.ca-original {{ color: {c["muted"]}; text-decoration: line-through; }}
</style>
"""


def esc(value: object) -> str:
    """HTML-escape dynamic (resume/JD/LLM-derived) text before it enters an ``unsafe_allow_html``
    block — Streamlit renders that markup verbatim, so unescaped user/model content is an XSS
    vector. Mirrors the export layer's Jinja autoescaping."""
    return escape(str(value))


def esc_md(value: object) -> str:
    """Escape Markdown metacharacters in dynamic text rendered via a plain ``st.markdown`` /
    ``st.caption`` call. Not a security control (Streamlit sanitizes HTML there) — it just keeps
    a skill like ``_NET`` or ``a*b`` from rendering as stray emphasis."""
    return _MD_SPECIAL.sub(r"\\\1", str(value))


def badge(label: str, color: str) -> str:
    """An inline pill badge (HTML) in the given fill color. The label is HTML-escaped, so this is
    safe to call with dynamic (LLM/user-derived) text inside an ``unsafe_allow_html`` block."""
    return f'<span class="ca-badge" style="background:{color}">{escape(str(label))}</span>'


def band_color(band: str) -> str:
    """Color for an integrity band (defaults to muted for unknown values)."""
    return BAND_COLORS.get(band, COLORS["muted"])


def meter(value: float, *, maximum: float = 100.0) -> str:
    """A coral progress meter (HTML) filled to ``value`` of ``maximum``."""
    pct = max(0.0, min(100.0, (value / maximum) * 100.0)) if maximum else 0.0
    return f'<div class="ca-meter"><span style="width:{pct:.0f}%"></span></div>'
