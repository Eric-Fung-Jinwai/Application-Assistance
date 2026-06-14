"""Application settings loaded from environment / .env (Phase 0)."""

from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Float tolerance for the "weights sum to 1" check (avoids 0.1+0.9 != 1.0 surprises).
_WEIGHT_SUM_TOL = 1e-6


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

    @model_validator(mode="after")
    def _validate_integrity(self) -> Settings:
        """Reject bad .env integrity config at startup — invalid weights/cutoffs could push
        scores outside [0,100] or invert the bands, silently letting unsafe rewrites pass."""
        if self.integrity_w_llm < 0 or self.integrity_w_embed < 0:
            raise ValueError("integrity weights must be non-negative")
        if abs(self.integrity_w_llm + self.integrity_w_embed - 1.0) > _WEIGHT_SUM_TOL:
            raise ValueError("integrity_w_llm + integrity_w_embed must sum to 1.0")
        cutoffs = (
            self.integrity_band_aggressive,
            self.integrity_band_moderate,
            self.integrity_band_safe,
        )
        if not 0 <= cutoffs[0] <= cutoffs[1] <= cutoffs[2] <= 100:
            raise ValueError(
                "integrity band cutoffs must satisfy 0 <= aggressive <= moderate <= safe <= 100"
            )
        return self


settings = Settings()
