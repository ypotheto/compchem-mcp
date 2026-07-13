import time
from unittest.mock import MagicMock, patch

from ypotheto_compchem_mcp.jobs import JobManager


def _make_envelope():
    return {
        "ok": True,
        "results": {"energy_ev": -76.4},
        "warnings": [{"type": "TEST_WARNING", "message": "just a test"}],
        "interpretation": "Calculation finished.",
        "artifacts": [{"kind": "report", "description": "Test report", "url": "http://example/test.json"}],
        "meta": {"server_version": "0.6.0"}
    }

def test_thread_fallback_and_db_path_produce_identical_job_shape():
    """Regression test for the JobManager shape-inconsistency bug: the DB-backed
    path used to store the *whole* envelope as JobState.results while the
    thread-fallback path stored only the inner results dict, so get_job_status
    returned differently-nested structures depending on whether
    COMPCHEM_DATABASE_URL was configured. Both paths must now extract the same
    fields from the same envelope shape."""
    envelope = _make_envelope()

    # --- Thread-fallback path: no DB configured, submit_job falls back to a
    # local ThreadPoolExecutor worker (see conftest.py's autouse fixture that
    # neutralizes settings.database_url for the whole test session). ---
    manager = JobManager(max_workers=1)
    try:
        job = manager.submit_job("ws1", lambda: envelope, 5)
        for _ in range(100):
            if job.status in ("completed", "failed"):
                break
            time.sleep(0.05)
        thread_dict = job.to_dict()
    finally:
        manager.stop()

    assert thread_dict["status"] == "completed"
    assert thread_dict["results"] == envelope["results"]
    assert thread_dict["warnings"] == envelope["warnings"]
    assert thread_dict["interpretation"] == envelope["interpretation"]
    assert thread_dict["artifacts"] == envelope["artifacts"]

    # --- Fake DB-backed path: mock get_connection so _load_job_from_db can be
    # exercised without a real Postgres instance. The mocked row's `results`
    # column holds the *whole* envelope, exactly as _execute_job_with_conn
    # actually persists it. ---
    manager2 = JobManager(max_workers=1)
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (
        "job_fake1", "ws1", "completed", "Calculation completed successfully.", 5,
        None, None,
        envelope,
        envelope["warnings"],
        None
    )
    mock_conn.cursor.return_value = mock_cursor

    with patch("ypotheto_compchem_mcp.database.get_connection", return_value=mock_conn):
        db_job = manager2._load_job_from_db("ws1", "job_fake1")

    db_dict = db_job.to_dict()

    assert db_dict["status"] == thread_dict["status"]
    assert db_dict["results"] == thread_dict["results"]
    assert db_dict["warnings"] == thread_dict["warnings"]
    assert db_dict["interpretation"] == thread_dict["interpretation"]
    assert db_dict["artifacts"] == thread_dict["artifacts"]
