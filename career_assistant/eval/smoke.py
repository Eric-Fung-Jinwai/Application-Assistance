"""Phase 0 smoke test: one real complete() + one real embed(), with cost logging.

Acceptance for Phase 0: a smoke run completes one ``complete()`` and one
``embed()`` call and the per-call cost log prints token counts + estimated cost.

Run:  python -m career_assistant.eval.smoke

Requires OPENAI_API_KEY (for the completion) and project deps installed. The
local HuggingFace embedding model downloads on first run.
"""

from __future__ import annotations

import logging

from career_assistant.config import settings
from career_assistant.llm.factory import get_embedding_client, get_llm_client


def main() -> int:
    # INFO so the per-call "cost_est=" log lines from the clients are visible.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if not settings.openai_api_key.strip():
        print("OPENAI_API_KEY is not set — cannot run the completion smoke test.")
        return 1

    print(f"-> complete() via {settings.llm_model}")
    llm = get_llm_client()
    reply = llm.complete(
        system="You are a terse assistant. Answer in one short sentence.",
        user="In one sentence, what is a resume?",
    )
    assert isinstance(reply, str) and reply.strip(), "empty completion"
    print(f"   reply: {reply.strip()}")

    print(f"-> embed() via {settings.embedding_provider}:{settings.embedding_model}")
    embedder = get_embedding_client()
    vectors = embedder.embed(["Built and shipped an evidence-aware resume tailoring tool."])
    assert len(vectors) == 1 and len(vectors[0]) > 0, "empty embedding"
    print(f"   embedding dim: {len(vectors[0])}")

    print("\nPhase 0 smoke OK — see the INFO cost log lines above for tokens + cost_est.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
