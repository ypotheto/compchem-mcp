import hashlib
import secrets
import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ypotheto_compchem_mcp.config import Settings


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class KeyStore(ABC):
    """Per-tenant API-key table. `SqliteKeyStore` is a single local file (dev/
    single-Droplet default); `PostgresKeyStore` is for a hosted deployment backed
    by the same shared Postgres database the durable job queue/molecule archive
    already use (COMPCHEM_DATABASE_URL)."""

    @abstractmethod
    def issue_key(self, workspace_id: str, plan: str = "default") -> str:
        """Generate a new API key, store only its hash, and return the raw key -
        the raw value is shown once, at issuance, and is not recoverable afterward."""

    @abstractmethod
    def verify_key(self, raw_key: str) -> str | None:
        """Return the key's workspace_id if it exists and is not disabled, else None."""

    @abstractmethod
    def disable_key(self, raw_key: str) -> bool: ...

    @abstractmethod
    def list_keys(self) -> list[dict[str, Any]]: ...


class SqliteKeyStore(KeyStore):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    plan TEXT NOT NULL DEFAULT 'default',
                    created_at REAL NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()
            yield conn
        finally:
            conn.close()

    def issue_key(self, workspace_id: str, plan: str = "default") -> str:
        raw_key = "sk_" + secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_hash, workspace_id, plan, created_at, disabled) "
                "VALUES (?, ?, ?, ?, 0)",
                (hash_key(raw_key), workspace_id, plan, time.time()),
            )
            conn.commit()
        return raw_key

    def verify_key(self, raw_key: str) -> str | None:
        if not self._db_path.exists():
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT workspace_id FROM api_keys WHERE key_hash = ? AND disabled = 0",
                (hash_key(raw_key),),
            ).fetchone()
        return row[0] if row else None

    def disable_key(self, raw_key: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET disabled = 1 WHERE key_hash = ?", (hash_key(raw_key),)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_keys(self) -> list[dict[str, Any]]:
        if not self._db_path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key_hash, workspace_id, plan, created_at, disabled FROM api_keys "
                "ORDER BY created_at"
            ).fetchall()
        return [
            {
                "key_hash_prefix": key_hash[:12],
                "workspace_id": workspace_id,
                "plan": plan,
                "created_at": created_at,
                "disabled": bool(disabled),
            }
            for key_hash, workspace_id, plan, created_at, disabled in rows
        ]


class PostgresKeyStore(KeyStore):
    """Uses `database.get_connection()` (a fresh connection per call, with
    retry-on-saturated-slots) rather than a persistent pool, matching this
    project's existing Postgres access pattern in `jobs.py`/`database.py` -
    not the connection-pool approach statistician-mcp's own PostgresKeyStore
    uses, since that would be a second, inconsistent way of talking to
    Postgres in the same codebase. Table lives in the `compchem` schema,
    schema-qualified like every other table this project creates (see
    `database.initialize_database`), rather than relying on `search_path`."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        conn = self._connect()
        try:
            cur = conn.cursor()
            # Only attempt schema creation if it's actually missing: `CREATE
            # SCHEMA IF NOT EXISTS` still requires database-level CREATE
            # privilege even when the schema already exists, in at least some
            # Postgres versions/ACL configurations - a role can perfectly
            # validly have CREATE scoped to an existing schema (enough to
            # create this table) without having database-level CREATE (enough
            # to create a NEW schema). Checking first avoids failing outright
            # in that configuration, which is exactly this project's own
            # production database's role setup (discovered while writing this
            # class - `database.initialize_database()` has the same
            # unconditional `CREATE SCHEMA IF NOT EXISTS compchem` call and
            # has been silently failing this same way on every call, masked
            # by its broad except-and-log error handling).
            cur.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'compchem'")
            if cur.fetchone() is None:
                cur.execute("CREATE SCHEMA IF NOT EXISTS compchem;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS compchem.api_keys (
                    key_hash TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    plan TEXT NOT NULL DEFAULT 'default',
                    created_at DOUBLE PRECISION NOT NULL,
                    disabled BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self._database_url)

    def issue_key(self, workspace_id: str, plan: str = "default") -> str:
        raw_key = "sk_" + secrets.token_urlsafe(32)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO compchem.api_keys (key_hash, workspace_id, plan, created_at, disabled) "
                "VALUES (%s, %s, %s, %s, FALSE)",
                (hash_key(raw_key), workspace_id, plan, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
        return raw_key

    def verify_key(self, raw_key: str) -> str | None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT workspace_id FROM compchem.api_keys WHERE key_hash = %s AND disabled = FALSE",
                (hash_key(raw_key),),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def disable_key(self, raw_key: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE compchem.api_keys SET disabled = TRUE WHERE key_hash = %s", (hash_key(raw_key),)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_keys(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT key_hash, workspace_id, plan, created_at, disabled FROM compchem.api_keys "
                "ORDER BY created_at"
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "key_hash_prefix": key_hash[:12],
                "workspace_id": workspace_id,
                "plan": plan,
                "created_at": created_at,
                "disabled": bool(disabled),
            }
            for key_hash, workspace_id, plan, created_at, disabled in rows
        ]


@lru_cache(maxsize=8)
def _cached_key_store(database_url: str, data_dir: str) -> KeyStore:
    if database_url:
        return PostgresKeyStore(database_url)
    return SqliteKeyStore(Path(data_dir) / "keys.db")


def build_key_store(settings: "Settings") -> KeyStore:
    """Cached by (database_url, data_dir) so the request-hot path (auth_mode=
    "keys") doesn't reconstruct a store on every single request, while still
    picking up a settings change (e.g. a test flipping `settings.database_url`)
    the moment either value actually changes."""
    return _cached_key_store(settings.database_url, str(settings.data_dir))
