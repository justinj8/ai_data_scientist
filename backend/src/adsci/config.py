"""Application settings, loaded from environment variables prefixed with ``ADSCI_``."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ADSCI_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    max_agent_steps: int = 12
    history_char_budget: int = 600_000
    verifier_critic: bool = True
    kernel_timeout_s: int = 60
    sql_row_limit: int = 10_000  # default rows returned to the UI / model
    analysis_row_limit: int = 1_000_000  # max rows loaded for analysis (tools, kernel sql())
    sql_timeout_s: int = 30
    preview_rows: int = 50
    model_summary_chars: int = 4000
    session_idle_minutes: int = 60
    bigquery_max_bytes_billed: int = 1_000_000_000
    public_base_url: str = "http://127.0.0.1:8000"
    sample_dir: Path | None = None
    internal_base_url: str = "http://127.0.0.1:8000"
    xai_oauth_client_id: str | None = None  # only an ID xAI issued to you; enables SuperGrok sign-in
    secret_key_file: Path | None = None
    # Run analysis kernels as another OS user (Docker sets these) so they can't read credentials.
    kernel_user: str | None = None
    kernel_group: str | None = None
    kernel_work_root: Path | None = None
    static_dir: Path | None = None  # built frontend (frontend/dist); served at / when set
    host: str = "127.0.0.1"
    # Host headers the API answers to; blocks DNS-rebinding attacks from web pages.
    allowed_hosts: list[str] = ["127.0.0.1", "localhost", "::1"]
    port: int = 8000  # set in Docker; otherwise the OS keychain holds the key


@lru_cache
def get_settings() -> Settings:
    return Settings()
