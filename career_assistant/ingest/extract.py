"""LLM structured extraction: ``RawResume`` → ``ParsedResume`` (Phase 3).

Goes through the ``LLMClient`` seam (no raw provider calls). Malformed or partially
valid model output is salvaged into a partial ``ParsedResume`` with a lowered
``confidence`` rather than crashing the pipeline.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from career_assistant.domain import ParsedResume, RawResume
from career_assistant.llm.client import LLMClient
from career_assistant.llm.factory import get_llm_client

logger = logging.getLogger("career_assistant.ingest")

# Confidence ceiling applied when we had to drop fields / salvage partial output.
LOW_CONFIDENCE = 0.5

_SECTIONS = ("education", "experience", "projects", "skills", "certifications")

EXTRACT_SYSTEM = """You are a precise resume parser. Extract the resume into JSON.

Return ONLY a JSON object with exactly these keys:
- education: list of {institution, degree, field_of_study, start_date, end_date}
- experience: list of {title, company, start_date, end_date, bullets: [string]}
- projects: list of {name, description, bullets: [string]}
- skills: [string]
- certifications: [string]
- confidence: number in [0,1] — your confidence in this extraction

Rules:
- Use ONLY information present in the resume. Do NOT invent or infer facts that are
  not written. This grounding is critical downstream.
- Each bullet is one accomplishment line, verbatim or only lightly cleaned.
- Use null for unknown dates and [] for empty lists.
"""

# Passed to LLMClient.complete to trigger JSON mode; also forward-compatible with
# providers that honor a real JSON schema.
EXTRACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "education": {"type": "array"},
        "experience": {"type": "array"},
        "projects": {"type": "array"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "certifications": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": list(_SECTIONS),
}


def extract_resume(raw: RawResume, llm: LLMClient | None = None) -> ParsedResume:
    """Parse raw resume text into a structured ``ParsedResume``."""
    client = llm or get_llm_client()
    result = client.complete(EXTRACT_SYSTEM, raw.raw_text, json_schema=EXTRACT_SCHEMA)

    if isinstance(result, str):
        # Provider returned text despite JSON mode; try to recover.
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            logger.warning("ingest.extract non-JSON response for %s", raw.filename)
            return ParsedResume(confidence=0.0)

    return _coerce_parsed(result, filename=raw.filename)


def _coerce_parsed(data: object, *, filename: str) -> ParsedResume:
    """Validate model output; on failure, keep the fields that do validate."""
    if not isinstance(data, dict):
        logger.warning("ingest.extract non-object response for %s", filename)
        return ParsedResume(confidence=0.0)

    try:
        return ParsedResume.model_validate(data)
    except ValidationError:
        pass  # fall through to per-field salvage

    kept: dict = {}
    for field in _SECTIONS:
        if field not in data:
            continue
        try:
            ParsedResume.model_validate({field: data[field]})
        except ValidationError:
            logger.warning("ingest.extract dropped invalid field %r for %s", field, filename)
            continue
        kept[field] = data[field]

    parsed = ParsedResume.model_validate(kept)
    # Signal reduced trust so downstream phases (fit/tailor) can be cautious.
    parsed.confidence = LOW_CONFIDENCE
    return parsed
