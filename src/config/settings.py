from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    HF_API_TOKEN: str
    HF_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_ITERATIONS: int = 3
    QUALITY_THRESHOLD: float = 0.8
    JWT_SECRET: str = "neuroresearch-change-this-in-production"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()