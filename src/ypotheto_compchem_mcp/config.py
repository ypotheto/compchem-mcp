from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMPCHEM_")
    
    api_token: str = ""
    data_dir: Path = Path("~/.compchem-mcp").expanduser()
    port: int = 8348
    auth_mode: str = "token"  # "token" | "none" | "keys"
    public_base_url: str = "http://localhost:8348"

settings = Settings()
