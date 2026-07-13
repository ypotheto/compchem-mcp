import os
from pathlib import Path

import psycopg2
import pytest

from ypotheto_compchem_mcp.apikeys import KeyStore, PostgresKeyStore, SqliteKeyStore


def _load_test_database_url() -> str:
    """Same lookup convention as tests/test_durable_jobs.py: prefer the
    env var, fall back to parsing .env directly (pytest-dotenv isn't wired
    up), skip cleanly if neither is set - matching HUMAN_TASKS.md item 6
    (a disposable test Postgres is optional, never point at prod)."""
    db_url = os.environ.get("COMPCHEM_DATABASE_URL")
    if db_url:
        return db_url
    try:
        dotenv_path = Path(__file__).parents[1] / ".env"
        if dotenv_path.exists():
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("COMPCHEM_DATABASE_URL="):
                    return line.split("=", 1)[1].strip('"').strip("'")
    except Exception:
        pass
    return ""


@pytest.fixture(params=["sqlite", "postgres"])
def key_store(request, tmp_path):
    """Runs every test in this file against both backends, so a divergence
    between SqliteKeyStore and PostgresKeyStore shows up as a normal test
    failure. The postgres param is skipped (not errored) when no test
    database is configured, so sqlite tests still run everywhere."""
    if request.param == "sqlite":
        yield SqliteKeyStore(tmp_path / "keys.db")
        return

    db_url = _load_test_database_url()
    if not db_url:
        pytest.skip("PostgreSQL database URL is not configured.")
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS compchem.api_keys")
        conn.commit()
        conn.close()
    except psycopg2.OperationalError as e:
        pytest.skip(f"PostgreSQL is not reachable for this test run: {e}")

    yield PostgresKeyStore(db_url)


def test_issue_then_verify_roundtrips(key_store: KeyStore):
    raw_key = key_store.issue_key("ws_acme", plan="pro")
    assert raw_key.startswith("sk_")
    assert key_store.verify_key(raw_key) == "ws_acme"


def test_verify_unknown_key_returns_none(key_store: KeyStore):
    assert key_store.verify_key("sk_not_a_real_key") is None


def test_disable_key_then_verify_returns_none(key_store: KeyStore):
    raw_key = key_store.issue_key("ws_acme")
    assert key_store.disable_key(raw_key) is True
    assert key_store.verify_key(raw_key) is None


def test_disable_unknown_key_returns_false(key_store: KeyStore):
    assert key_store.disable_key("sk_not_a_real_key") is False


def test_list_keys_reports_all_issued_keys(key_store: KeyStore):
    key_store.issue_key("ws_a", plan="free")
    key_store.issue_key("ws_b", plan="pro")

    entries = key_store.list_keys()

    assert {(e["workspace_id"], e["plan"], e["disabled"]) for e in entries} == {
        ("ws_a", "free", False),
        ("ws_b", "pro", False),
    }


def test_list_keys_reflects_disabled_status(key_store: KeyStore):
    raw_key = key_store.issue_key("ws_acme")
    key_store.disable_key(raw_key)

    entries = key_store.list_keys()

    assert entries[0]["disabled"] is True
