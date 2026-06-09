"""Factories for configured LLM and embedding clients."""

from __future__ import annotations

from career_assistant.config import settings
from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.llm.huggingface_client import HuggingFaceEmbeddingClient
from career_assistant.llm.openai_client import OpenAIClient, OpenAIEmbeddingClient


def get_llm_client() -> LLMClient:
    """Return the configured completion client."""
    return OpenAIClient()


def get_embedding_client() -> EmbeddingClient:
    """Return the configured embedding client."""
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
