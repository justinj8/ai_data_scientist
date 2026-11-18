"""Application settings, loaded from environment variables prefixed with ``ADSCI_``."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ADSCI_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    max_agent_steps: int = 12
    kernel_timeout_s: int = 60
    sql_row_limit: int = 10_000
    sql_timeout_s: int = 30
    preview_rows: int = 50
    model_summary_chars: int = 4000
    session_idle_minutes: int = 60
    bigquery_max_bytes_billed: int = 1_000_000_000
    public_base_url: str = "http://127.0.0.1:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
