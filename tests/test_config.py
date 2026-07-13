from ypotheto_compchem_mcp.config import Settings, settings


def test_poisoned_env_file_never_reaches_spaces_backend(tmp_path, monkeypatch):
    """A real .env with live DB/Spaces credentials sits in the working tree
    (statistician-mcp had tests silently hit production this way once). The
    autouse fixture in conftest.py neutralizes settings.spaces_bucket/
    database_url/api_token for the whole test session - this proves that
    neutralization actually holds by (a) confirming a poisoned .env WOULD load
    production-looking values if nothing intervened, then (b) confirming the
    real `settings` singleton used by the app never lets SpacesBackend get
    constructed during a test run regardless of what's in some .env on disk."""
    poisoned_env = tmp_path / ".env"
    poisoned_env.write_text(
        "COMPCHEM_SPACES_BUCKET=prod-bucket\n"
        "COMPCHEM_SPACES_ENDPOINT=https://prod.example.com\n"
        "COMPCHEM_SPACES_KEY=prod-key\n"
        "COMPCHEM_SPACES_SECRET=prod-secret\n"
        "COMPCHEM_API_TOKEN=prod-secret-token\n"
        "COMPCHEM_DATABASE_URL=postgres://prod/db\n",
        encoding="utf-8",
    )
    poisoned_settings = Settings(_env_file=str(poisoned_env))
    assert poisoned_settings.spaces_bucket == "prod-bucket"  # the poison is real

    from ypotheto_compchem_mcp import storage as storage_module

    def _fail_if_constructed(*args, **kwargs):
        raise AssertionError("SpacesBackend must never be constructed during test runs")

    monkeypatch.setattr(storage_module, "SpacesBackend", _fail_if_constructed)

    assert settings.spaces_bucket is None
    assert settings.database_url == ""
    assert settings.api_token == ""
    backend = storage_module._build_storage_backend()
    assert isinstance(backend, storage_module.LocalDirBackend)

def test_allowed_origins_parses_comma_separated_env_string():
    settings = Settings(allowed_origins="https://a.example.com, https://b.example.com")
    assert settings.allowed_origins == ["https://a.example.com", "https://b.example.com"]

def test_allowed_origins_defaults_to_empty_list():
    settings = Settings()
    assert settings.allowed_origins == []

def test_allowed_origins_accepts_actual_list():
    settings = Settings(allowed_origins=["https://a.example.com"])
    assert settings.allowed_origins == ["https://a.example.com"]
