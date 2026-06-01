from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    secret_key: str = "dev-secret-change-me"
    database_url: str = "sqlite:///./smartqa_v3.db"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "smartqa_chunks"
    storage_root: Path = Path("./storage")
    frontend_origin: str = "http://localhost:5173"
    cookie_secure: bool = False
    sync_indexing: bool = True
    max_upload_mb: int = 20
    rate_limiting_enabled: bool = False
    ai_provider: str = "mock"
    chat_model: str = "meta/llama-3.1-70b-instruct"
    embedding_provider: str = "nvidia"
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    nvidia_api_key: str = ""
    openai_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
