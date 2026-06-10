"""PDF → raw text via pypdf, with scanned/image-PDF detection (Phase 3)."""

from __future__ import annotations

from pypdf import PdfReader

from career_assistant.ingest.errors import ScannedPDFError

# A real text resume yields hundreds of chars. A scanned/image PDF yields ~0 (pypdf
# can't OCR). If pages exist but extractable text is below this, treat as scanned.
MIN_EXTRACTABLE_CHARS = 20


def extract_text(path: str) -> str:
    """Extract text from a PDF. Raise ``ScannedPDFError`` if it looks image-only."""
    reader = PdfReader(path)
    pages = reader.pages
    text = "\n".join(page.extract_text() or "" for page in pages).strip()

    if pages and len(text) < MIN_EXTRACTABLE_CHARS:
        msg = (
            "This PDF has no extractable text — it looks scanned or image-only. "
            "Please upload a text-based PDF, DOCX, or TXT resume."
        )
        raise ScannedPDFError(msg)

    return text
