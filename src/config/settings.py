import logging
import os
from functools import lru_cache
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_INSECURE_DEFAULT = "neuroresearch-change-this-in-production"


class Settings(BaseSettings):
    # ── Required ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    # HF Spaces doesn't allow underscores in secret names
    GROQAPIKEY: str = ""

    JWT_SECRET: str = _INSECURE_DEFAULT
    # HF Spaces fallback (no underscores in secret names)
    JWTSECRET: str = ""

    # ── Optional HF token ─────────────────────────────────────────────────────
    HF_API_TOKEN: str = ""

    # ── Model defaults — aligned with llm.py DEFAULT_MODEL ───────────────────
    HF_MODEL_ID: str = "openai/gpt-oss-20b"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Pipeline ───────────────────────────────────────────────────────────────
    MAX_ITERATIONS: int = 3
    # Fix #6: lowered from 0.95 (unreachable) to 0.75 so the loop can converge
    QUALITY_THRESHOLD: float = 0.85

    # ── CORS — Fix #13: configurable instead of hardcoded "*" ────────────────
    # Comma-separated list of allowed origins, e.g. "https://myapp.com,http://localhost:3000"
    # Defaults to "*" for local dev / HF Spaces; override in production.
    CORS_ORIGINS: str = "*"

    # ── LangSmith (optional) ──────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "neuroresearch"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ── Langfuse (optional) ───────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ── Tavily Search (optional) ──────────────────────────────────────────
    TAVILY_API_KEY: str = ""

    # ── OpenRouter (optional) ─────────────────────────────────────────────
    OPENROUTER_API_KEY: str = ""

    # ── Eval ──────────────────────────────────────────────────────────────────
    EVAL_OUTPUT_DIR: str = "data/eval_results"

    model_config = ConfigDict(env_file=".env")

    def effective_jwt_secret(self) -> str:
        """Return the active JWT secret, preferring JWTSECRET (HF Spaces) over JWT_SECRET."""
        return self.JWTSECRET or self.JWT_SECRET

    def cors_origins_list(self) -> list[str]:
        """Return CORS_ORIGINS as a list."""
        raw = self.CORS_ORIGINS.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


def check_security(s: "Settings") -> None:
    """
    Fix #13: Warn loudly (or raise in production) if insecure defaults are in use.
    Checks JWT secret and logs a warning — does not crash HF Spaces free-tier deploys.
    """
    jwt = s.effective_jwt_secret()
    if jwt == _INSECURE_DEFAULT:
        logger.warning(
            "[Security] JWT_SECRET is set to the insecure default. "
            "Set JWT_SECRET (local) or JWTSECRET (HF Spaces) to a random secret."
        )
    if not s.GROQ_API_KEY and not s.GROQAPIKEY:
        logger.warning("[Security] GROQ_API_KEY is not set — LLM calls will fail.")


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    check_security(s)
    return s
