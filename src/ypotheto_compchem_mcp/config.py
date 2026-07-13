from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COMPCHEM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    api_token: str = ""
    data_dir: Path = Path("~/.compchem-mcp").expanduser()
    port: int = 8348
    auth_mode: str = "token"  # "token" | "none" | "keys"
    public_base_url: str = "http://localhost:8348"
    database_url: str = ""
    spaces_bucket: str | None = None
    spaces_endpoint: str | None = None
    spaces_key: str | None = None
    spaces_secret: str | None = None
    spaces_region: str = "nyc3"
    spaces_prefix: str = "compchem-mcp"
    artifact_url_expiry_seconds: int = 7 * 24 * 3600
    request_timeout_seconds: int = 120
    allowed_origins: list[str] = []

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_comma_separated_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

settings = Settings()

