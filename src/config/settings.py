from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # ── Required ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    # HF Spaces doesn't allow underscores in secret names,
    # so also read GROQAPIKEY as a fallback
    GROQAPIKEY: str = ""

    JWT_SECRET: str = "neuroresearch-change-this-in-production"
    # HF Spaces fallback (no underscores)
    JWTSECRET: str = ""
    JWT_SECRET: str = "neuroresearch-change-this-in-production"

    # ── Optional HF token (only needed if using HF embeddings locally) ────────
    HF_API_TOKEN: str = ""

    # ── Model defaults ─────────────────────────────────────────────────────────
    HF_MODEL_ID: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Pipeline ───────────────────────────────────────────────────────────────
    MAX_ITERATIONS: int = 3
    QUALITY_THRESHOLD: float = 0.95

    # ── LangSmith (optional) ──────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "neuroresearch"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ── Langfuse (optional) ───────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ── Eval ──────────────────────────────────────────────────────────────────
    EVAL_OUTPUT_DIR: str = "data/eval_results"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
