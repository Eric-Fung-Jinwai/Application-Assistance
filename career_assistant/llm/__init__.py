"""LLM abstraction layer. No raw provider calls may live outside this package."""

from career_assistant.llm.client import EmbeddingClient, LLMClient
from career_assistant.llm.factory import get_embedding_client, get_llm_client
from career_assistant.llm.huggingface_client import HuggingFaceEmbeddingClient
from career_assistant.llm.openai_client import OpenAIClient, OpenAIEmbeddingClient

__all__ = [
    "EmbeddingClient",
    "HuggingFaceEmbeddingClient",
    "LLMClient",
    "OpenAIClient",
    "OpenAIEmbeddingClient",
    "get_embedding_client",
    "get_llm_client",
]
