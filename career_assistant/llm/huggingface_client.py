"""Hugging Face implementation of the embedding interface."""

from __future__ import annotations

import logging
import time

from career_assistant.config import settings
from career_assistant.llm.client import EmbeddingClient

logger = logging.getLogger("career_assistant.llm")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}


class HuggingFaceEmbeddingClient(EmbeddingClient):
    """Embedding client backed by a local sentence-transformers model."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for HuggingFaceEmbeddingClient. "
                "Install project dependencies or run `pip install sentence-transformers`."
            )
            raise RuntimeError(msg) from exc

        configured_model = model or settings.embedding_model
        if model is None and configured_model in OPENAI_EMBEDDING_MODELS:
            configured_model = DEFAULT_MODEL

        self._model_name = configured_model
        self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        dt = time.perf_counter() - t0
        logger.info(
            "llm.embed provider=huggingface model=%s n=%d latency=%.2fs cost_est=$0.000000",
            self._model_name,
            len(texts),
            dt,
        )
        return embeddings.tolist()
