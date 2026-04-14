from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://dft:dft@db:5432/dftjobs"
    sync_database_url: str = "postgresql+psycopg2://dft:dft@db:5432/dftjobs"
    redis_url: str = "redis://redis:6379"
    rq_queue: str = "dft"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
