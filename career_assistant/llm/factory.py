"""Factories for configured LLM and embedding clients.

Both factories are ``lru_cache``d so the client — and, critically, the local embedding model it
loads (~80 MB into torch) — is built **once per process** and reused. Without this, a fresh
client per request reloaded the model on every call (parse / fit / tailor / each accept), which
dominated request latency. Cached singletons are reused across the API's threadpool workers.
"""

from __future__ import annotations

from functools import lru_cache

from career_assistant.config import settings
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.llm.huggingface_client import HuggingFaceEmbeddingClient
from career_assistant.llm.openai_client import OpenAIClient, OpenAIEmbeddingClient


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return the (cached) configured completion client."""
    return OpenAIClient()


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    """Return the (cached) configured embedding client — loads the model once per process."""
    provider = settings.embedding_provider.lower().strip()
    if provider in {"huggingface", "hf", "sentence-transformers", "sentence_transformers"}:
        return HuggingFaceEmbeddingClient()
    if provider == "openai":
        return OpenAIEmbeddingClient()

    msg = (
        f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
        "Use 'huggingface' or 'openai'."
    )
    raise ValueError(msg)
