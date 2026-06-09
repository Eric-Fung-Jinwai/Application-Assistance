"""Application settings loaded from environment / .env (Phase 0)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider (MVP: OpenAI, behind the LLMClient seam)
    openai_api_key: str = ""
    llm_model: str = "gpt-5.4-mini"

    # Embeddings. Default to a local Hugging Face sentence-transformers model.
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Storage
    sqlite_path: str = "./data/career_assistant.db"
    chroma_path: str = "./data/chroma"

    # Integrity scoring — tuned by the Phase 16 eval harness, NOT hardcoded blindly.
    integrity_w_llm: float = 0.8
    integrity_w_embed: float = 0.2
    # Band cutoffs: >=safe Safe, >=moderate Moderate, >=aggressive Aggressive, else High-Risk
    integrity_band_safe: int = 90
    integrity_band_moderate: int = 70
    integrity_band_aggressive: int = 50


settings = Settings()
