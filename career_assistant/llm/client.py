"""Swappable LLM + embedding interfaces (Phase 0).

The integrity judge and tailoring are quality-critical; keeping the provider behind
these ABCs lets us A/B providers on the eval harness without touching call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        temperature: float = 0.2,
    ) -> str | dict:
        """Return a completion. If json_schema is given, return parsed JSON (dict)."""
        ...


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...
