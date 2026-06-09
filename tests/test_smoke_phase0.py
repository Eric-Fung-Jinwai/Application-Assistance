"""Phase 0 acceptance: a live complete() + embed() smoke.

Integration test — skipped unless deps are installed and OPENAI_API_KEY is set,
so the default `pytest` run stays offline/free. Opt in with the real env loaded.
"""

from __future__ import annotations

import logging

import pytest

from career_assistant.config import settings

openai = pytest.importorskip("openai", reason="openai not installed")
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

pytestmark = pytest.mark.skipif(
    not settings.openai_api_key.strip(),
    reason="OPENAI_API_KEY not set; skipping live Phase 0 smoke",
)


def test_complete_and_embed_smoke(caplog):
    from career_assistant.llm.factory import get_embedding_client, get_llm_client

    with caplog.at_level(logging.INFO, logger="career_assistant.llm"):
        reply = get_llm_client().complete(
            system="Answer in one short sentence.",
            user="In one sentence, what is a resume?",
        )
        vectors = get_embedding_client().embed(["Shipped an evidence-aware resume tailor."])

    assert isinstance(reply, str) and reply.strip()
    assert len(vectors) == 1 and len(vectors[0]) > 0

    # The cost log must print per-call token counts + an estimated cost.
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "llm.complete" in messages
    assert "cost_est=" in messages
    assert "llm.embed" in messages
