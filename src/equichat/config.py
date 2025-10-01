from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator  # <-- add this import


class Settings(BaseSettings):
    """
    Runtime configuration for the EquiChat agent.

    Load order: env vars > .env file > defaults below.
    Prefix all env vars with `EQUICHAT_` (e.g., EQUICHAT_DB_PATH).
    """

    model_config = SettingsConfigDict(env_prefix="EQUICHAT_", env_file=".env", extra="ignore")

    # --- Core ---
    db_path: str = Field(
        default=":memory:",
        description="DuckDB database path (use a file path for persistence, ':memory:' for tests).",
    )
    base_currency: Literal["INR", "USD"] = Field(default="INR")
    base_unit: Literal["cr", "mn", "bn"] = Field(
        default="cr", description="Canonical unit for numeric normalization."
    )
    ignore_last_page: bool = Field(
        default=True, description="If true, ingestion skips the last page (disclaimers/charts)."
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Min confidence needed before returning a value as 'ok'.",
    )

    # --- LLM / OpenAI ---
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-5",
        description="Model used for routing/tool selection & reasoning. Override via env.",
    )
    openai_timeout_seconds: int = Field(default=60)
    openai_max_retries: int = Field(default=4, ge=0, le=12)

    # --- PDF parsing backends ---
    table_backend: Literal["camelot", "none"] = Field(
        default="camelot",
        description="Table extraction engine. Set 'none' to disable tables (text-only ingest).",
    )
    use_openai_extraction: bool = Field(
        default=False,
        description="Use OpenAI GPT for fact extraction instead of regex parsing. Requires openai_api_key.",
    )
    max_pages: Optional[int] = Field(
        default=None, description="For debugging: limit number of pages ingested per PDF."
    )

    # --- Server (FastAPI) ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    enable_metrics: bool = Field(default=True)
    enable_rate_limits: bool = Field(default=True)
    cors_allow_origins: str = Field(
        default="*",
        description="Comma-separated origins for CORS. Use '*' for dev only.",
    )

    # --- Logging / Observability ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    json_logs: bool = Field(default=True, description="If true, output structured JSON logs.")

    # --- Scaling ---
    enable_ray: bool = Field(default=False, description="Enable Ray for parallel ingestion.")
    ray_num_workers: int = Field(default=2, ge=1)

    @field_validator("cors_allow_origins")
    @classmethod
    def _strip_spaces(cls, v: str) -> str:
        return ",".join([s.strip() for s in v.split(",")]) if v else v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


CONFIG = get_settings()
