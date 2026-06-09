"""OpenAI implementation of the LLM/embedding interfaces (Phase 0).

Includes per-call logging (model, tokens, latency) to feed the Phase 0 cost budget.
"""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from career_assistant.config import settings
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.llm.pricing import (
    estimate_completion_cost,
    estimate_embedding_cost,
    format_cost,
)

logger = logging.getLogger("career_assistant.llm")


class OpenAIClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.llm_model

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        temperature: float = 0.2,
    ) -> str | dict:
        kwargs: dict = {
            "model": self._model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t0

        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = estimate_completion_cost(self._model, prompt_tokens, completion_tokens)
        logger.info(
            "llm.complete model=%s prompt_tokens=%s completion_tokens=%s latency=%.2fs cost_est=%s",
            self._model,
            prompt_tokens,
            completion_tokens,
            dt,
            format_cost(cost),
        )

        content = resp.choices[0].message.content or ""
        return json.loads(content) if json_schema is not None else content


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str | None = None) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        resp = self._client.embeddings.create(model=self._model, input=texts)
        dt = time.perf_counter() - t0
        total_tokens = getattr(resp.usage, "total_tokens", 0) or 0
        cost = estimate_embedding_cost(self._model, total_tokens)
        logger.info(
            "llm.embed model=%s n=%d tokens=%s latency=%.2fs cost_est=%s",
            self._model,
            len(texts),
            total_tokens,
            dt,
            format_cost(cost),
        )
        return [d.embedding for d in resp.data]
