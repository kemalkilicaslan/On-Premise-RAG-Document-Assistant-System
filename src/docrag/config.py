"""Settings read from environment variables and `.env`.

JWT_SECRET and ADMIN_PASSWORD have no defaults, so the service refuses to start without them.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    ollama_timeout: float = 300.0

    jwt_secret: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    admin_username: str = "admin"
    admin_password: str = Field(min_length=8, max_length=72)

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    max_upload_mb: int = 25
    upload_dir: str = "/data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
