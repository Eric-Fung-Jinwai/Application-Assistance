"""Plain-text → raw text (Phase 3)."""

from __future__ import annotations

from pathlib import Path


def extract_text(path: str) -> str:
    # errors="replace" so an odd byte never crashes ingestion.
    return Path(path).read_text(encoding="utf-8", errors="replace").strip()
