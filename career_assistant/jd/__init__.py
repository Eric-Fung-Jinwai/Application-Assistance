"""Job-description analysis: extraction, chunk embeddings, coverage (Phase 5)."""

from career_assistant.jd.analyzer import analyze_jd
from career_assistant.jd.chunks import chunks_from_parsed
from career_assistant.jd.coverage import COVERAGE_SIM_THRESHOLD, analyze_coverage
from career_assistant.jd.embeddings import build_jd_kb, query_jd_chunks

__all__ = [
    "COVERAGE_SIM_THRESHOLD",
    "analyze_coverage",
    "analyze_jd",
    "build_jd_kb",
    "chunks_from_parsed",
    "query_jd_chunks",
]
