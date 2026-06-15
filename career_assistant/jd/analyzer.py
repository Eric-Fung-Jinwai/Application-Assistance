"""LLM structured extraction: raw JD text → ``ParsedJD`` (Phase 5).

Mirrors ``ingest/extract.py``: goes through the ``LLMClient`` seam (no raw provider
calls) and salvages partially valid model output into a partial ``ParsedJD`` rather than
crashing the pipeline. Unlike resume extraction, a JD has no ``confidence`` field — a
salvaged JD simply carries the fields that validated and empty lists for the rest.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from career_assistant.domain import ParsedJD
from career_assistant.llm.client import LLMClient
from career_assistant.llm.factory import get_llm_client

logger = logging.getLogger("career_assistant.jd")

# The 7 fields from the Phase 5 spec.
_FIELDS = (
    "required_skills",
    "preferred_skills",
    "responsibilities",
    "tools",
    "domain",
    "seniority",
    "years_experience",
)

JD_EXTRACT_SYSTEM = """You are a precise job-description parser. Extract the posting into JSON.

Return ONLY a JSON object with exactly these keys:
- required_skills: [string] — skills/technologies the posting requires (must-haves)
- preferred_skills: [string] — nice-to-have / preferred skills
- responsibilities: [string] — what the role does day to day
- tools: [string] — concrete tools, frameworks, languages, platforms named
- domain: string or null — the business/technical domain (e.g. "fintech", "ML infrastructure")
- seniority: string or null — e.g. "junior", "mid", "senior", "staff", "intern"
- years_experience: number or null — minimum years of experience required (YOE)

Rules:
- Use ONLY information present in the posting. Do NOT invent requirements.
- Split a required vs preferred skill by the posting's own language
  ("required"/"must have" vs "preferred"/"nice to have"/"bonus").
- Use null for unknown scalars and [] for empty lists.
"""

# Passed to LLMClient.complete to trigger JSON mode; forward-compatible with providers
# that honor a real JSON schema.
JD_EXTRACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "preferred_skills": {"type": "array", "items": {"type": "string"}},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "string"}},
        "domain": {"type": ["string", "null"]},
        "seniority": {"type": ["string", "null"]},
        "years_experience": {"type": ["number", "null"]},
    },
    "required": ["required_skills", "responsibilities"],
}


def analyze_jd(raw_text: str, llm: LLMClient | None = None) -> ParsedJD:
    """Parse raw job-description text into a structured ``ParsedJD``."""
    client = llm or get_llm_client()
    result = client.complete(JD_EXTRACT_SYSTEM, raw_text, json_schema=JD_EXTRACT_SCHEMA)

    if isinstance(result, str):
        # Provider returned text despite JSON mode; try to recover.
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            logger.warning("jd.analyze non-JSON response")
            return ParsedJD()

    return _coerce_parsed(result)


def _coerce_parsed(data: object) -> ParsedJD:
    """Validate model output; on failure, keep the fields that do validate."""
    if not isinstance(data, dict):
        logger.warning("jd.analyze non-object response")
        return ParsedJD()

    try:
        return ParsedJD.model_validate(data)
    except ValidationError:
        pass  # fall through to per-field salvage

    kept: dict = {}
    for field in _FIELDS:
        if field not in data:
            continue
        try:
            ParsedJD.model_validate({field: data[field]})
        except ValidationError:
            logger.warning("jd.analyze dropped invalid field %r", field)
            continue
        kept[field] = data[field]

    return ParsedJD.model_validate(kept)
