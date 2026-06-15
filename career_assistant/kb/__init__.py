"""Resume knowledge base: bullets → embeddings → ChromaDB (Phase 4)."""

from career_assistant.kb.bullets import bullets_from_parsed, persist_bullets
from career_assistant.kb.embeddings import (
    build_resume_kb,
    embed_and_upsert_bullets,
    query_bullets,
)

__all__ = [
    "build_resume_kb",
    "bullets_from_parsed",
    "embed_and_upsert_bullets",
    "persist_bullets",
    "query_bullets",
]
