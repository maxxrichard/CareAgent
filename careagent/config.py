"""Runtime configuration, loaded from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

Backend = Literal["anthropic", "openai", "ollama"]
Effort = Literal["low", "medium", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CAREAGENT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM backend ---------------------------------------------------------------
    llm_backend: Backend = "anthropic"
    anthropic_model: str = "claude-opus-5-5"
    openai_model: str = "gpt-5"
    ollama_model: str = "llama3.3"
    ollama_base_url: str = "http://localhost:11434/v1"
    effort: Effort = "medium"
    anthropic_fallbacks: bool = True
    max_output_tokens: int = 2048
    request_timeout_s: float = 60.0

    # Agent behaviour -----------------------------------------------------------
    strict_grounding: bool = True
    escalation_threshold: float = Field(0.6, ge=0.0, le=1.0)
    max_tool_calls: int = 8

    # Retrieval -----------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 8
    rerank_top_n: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 80
    policies_dir: Path = DATA_DIR / "policies"
    index_dir: Path = DATA_DIR / "index"

    # Data ----------------------------------------------------------------------
    orders_db: Path = DATA_DIR / "orders" / "orders.db"
    seed_orders: int = 400
    seed: int = 42

    # Tracking ------------------------------------------------------------------
    mlflow_experiment: str = "careagent"
    delta_table: str = "main.careagent.eval_results"

    # Cost ----------------------------------------------------------------------
    usd_to_eur: float = 0.92

    @property
    def model_name(self) -> str:
        return {
            "anthropic": self.anthropic_model,
            "openai": self.openai_model,
            "ollama": self.ollama_model,
        }[self.llm_backend]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
