"""Typed errors for resume ingestion (Phase 3 failure modes)."""

from __future__ import annotations


class UnsupportedResumeError(Exception):
    """Resume cannot be ingested (unknown type, scanned/image, empty, or garbage).

    The API/UI layer maps this to a clear user-facing message rather than a 500.
    """


class ScannedPDFError(UnsupportedResumeError):
    """PDF has pages but ~no extractable text — almost certainly scanned/image-only."""


class EmptyResumeError(UnsupportedResumeError):
    """Extracted text is empty or too short to be a real resume."""
