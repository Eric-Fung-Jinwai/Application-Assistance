"""Dispatch a file to the right extractor → ``RawResume`` (Phase 3)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from career_assistant.domain import RawResume
from career_assistant.ingest import docx, pdf, txt
from career_assistant.ingest.errors import EmptyResumeError, UnsupportedResumeError

logger = logging.getLogger("career_assistant.ingest")

# Shorter than this and it isn't a real resume (empty/garbage upload guard).
MIN_RESUME_CHARS = 50
# A real resume has plenty of letters; binary garbage that happens to decode has few.
MIN_RESUME_LETTERS = 30
# More than this fraction of (non-whitespace) control chars => binary garbage, not text.
MAX_CONTROL_RATIO = 0.10
# Below this fraction of ASCII letters, the resume is probably not English.
ENGLISH_ASCII_RATIO = 0.7

_EXTRACTORS: dict[str, Callable[[str], str]] = {
    ".pdf": pdf.extract_text,
    ".docx": docx.extract_text,
    ".txt": txt.extract_text,
}


def parse_file(path: str) -> RawResume:
    """Extract raw text from a resume file, dispatching by extension.

    Raises ``UnsupportedResumeError`` for unknown types, ``ScannedPDFError`` for
    image-only PDFs (from ``pdf.extract_text``), and ``EmptyResumeError`` for
    empty/too-short text. Non-English text is flagged via a warning, not an error.
    """
    p = Path(path)
    ext = p.suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        supported = ", ".join(sorted(_EXTRACTORS))
        msg = f"Unsupported resume type {ext!r}. Supported: {supported}."
        raise UnsupportedResumeError(msg)

    # Corrupt files make pypdf/python-docx raise their own exception types; map any
    # such parser-library failure to our typed error so the API/UI can handle it
    # cleanly. ScannedPDFError (an UnsupportedResumeError) passes straight through.
    try:
        raw_text = extractor(path)
    except UnsupportedResumeError:
        raise
    except (FileNotFoundError, IsADirectoryError):
        raise  # genuine IO/programming error, not a bad upload
    except Exception as exc:
        msg = f"Could not read {p.name!r}; the file may be corrupt or not a valid {ext} file."
        raise UnsupportedResumeError(msg) from exc

    text = raw_text.strip()
    if len(text) < MIN_RESUME_CHARS:
        msg = (
            f"Extracted only {len(text)} characters from {p.name!r}; "
            "the file appears empty or is not a readable resume."
        )
        raise EmptyResumeError(msg)
    if not _looks_like_text(text):
        msg = (
            f"{p.name!r} does not look like readable text (too few letters or too "
            "many control characters); it may be binary/garbage rather than a resume."
        )
        raise EmptyResumeError(msg)

    if _ascii_letter_ratio(raw_text) < ENGLISH_ASCII_RATIO:
        logger.warning(
            "ingest.parse_file file=%s may not be English (low ASCII-letter ratio); "
            "extraction quality may be reduced.",
            p.name,
        )

    return RawResume(filename=p.name, raw_text=raw_text, content_type=ext)


def _looks_like_text(text: str) -> bool:
    """Reject binary/garbage that slipped past the length gate (e.g. NUL bytes).

    Requires a minimum number of actual letters and a low share of non-whitespace
    control characters.
    """
    letters = sum(1 for c in text if c.isalpha())
    if letters < MIN_RESUME_LETTERS:
        return False
    control = sum(1 for c in text if ord(c) < 32 and c not in "\t\n\r\f\v")
    return control / len(text) <= MAX_CONTROL_RATIO


def _ascii_letter_ratio(text: str) -> float:
    """Fraction of alphabetic chars that are ASCII. 1.0 when there are no letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if c.isascii()) / len(letters)
