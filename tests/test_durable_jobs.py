import pytest
import time
import json
import uuid
import psycopg2
from unittest.mock import patch, MagicMock
from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.database import get_connection, initialize_database
from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords, _load_index
from ypotheto_compchem_mcp.jobs import JobManager, JobState

@pytest.fixture(autouse=True)
def setup_db_for_test():
    import os
    db_url = os.environ.get("COMPCHEM_DATABASE_URL")
    if not db_url:
        try:
            from pathlib import Path
            dotenv_path = Path(__file__).parents[1] / ".env"
            if dotenv_path.exists():
                for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("COMPCHEM_DATABASE_URL="):
                        db_url = line.split("=", 1)[1].strip('"').strip("'")
                        break
        except Exception:
            pass
            
    if not db_url:
        pytest.skip("PostgreSQL database URL is not configured.")
        
    original_url = settings.database_url
    settings.database_url = db_url
    from ypotheto_compchem_mcp.storage import storage
    storage.reset()
    
    # Re-initialize database tables for the test run
    try:
        initialize_database()
    except psycopg2.OperationalError as e:
        if "remaining connection slots" in str(e):
            pytest.skip("PostgreSQL connection slots are saturated.")
        raise e
    
    yield
    
    settings.database_url = original_url
    storage.reset()

def test_database_initialization():
    # If settings.database_url is set, verify we can connect and tables exist
    if not settings.database_url:
        pytest.skip("PostgreSQL database URL is not configured.")
        
    initialize_database()
    
    conn = get_connection()
    assert conn is not None
    cur = conn.cursor()
    
    # Check molecules table
    cur.execute("SELECT to_regclass('compchem.molecules');")
    res = cur.fetchone()[0]
    assert res is not None
    
    # Check jobs table
    cur.execute("SELECT to_regclass('compchem.jobs');")
    res = cur.fetchone()[0]
    assert res is not None
    
    cur.close()
    conn.close()

def test_searchable_archive_db():
    if not settings.database_url:
        pytest.skip("PostgreSQL database URL is not configured.")
        
    workspace_id = f"test_ws_{uuid.uuid4().hex[:6]}"
    molecule_id = f"mol_{uuid.uuid4().hex[:6]}"
    
    meta = {
        "molecule_id": molecule_id,
        "name": "TestMolecule",
        "formula": "H2O",
        "smiles": "O",
        "num_atoms": 3,
        "method": "MMFF94",
        "custom_key": "custom_val"
    }
    
    # Save coordinates
    # We patch storage.write_file to avoid uploading to Spaces
    with patch("ypotheto_compchem_mcp.storage.storage.write_file") as mock_write:
        save_molecule_coords(workspace_id, molecule_id, "sdf block", "xyz block", meta)
        
    # Load index from database
    index = _load_index(workspace_id)
    assert molecule_id in index
    assert index[molecule_id]["name"] == "TestMolecule"
    assert index[molecule_id]["formula"] == "H2O"
    assert index[molecule_id]["custom_key"] == "custom_val"
    
    # Clean up test molecule from database
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM compchem.molecules WHERE workspace_id = %s;", (workspace_id,))
    conn.commit()
    cur.close()
    conn.close()

def test_durable_jobs_db():
    if not settings.database_url:
        pytest.skip("PostgreSQL database URL is not configured.")
        
    workspace_id = f"test_ws_{uuid.uuid4().hex[:6]}"
    
    # Register dummy function
    from ypotheto_compchem_mcp.jobs import _FUNCTIONS_REGISTRY
    
    def dummy_func(*args, **kwargs):
        if "progress_callback" in kwargs:
            kwargs["progress_callback"]("Step 1 done")
        return {"ok": True, "results": {"output": "val"}, "interpretation": "done"}
        
    _FUNCTIONS_REGISTRY["dummy_func"] = dummy_func
    
    # Initialize JobManager
    manager = JobManager(max_workers=2)
    
    try:
        # Submit job
        try:
            job = manager.submit_job(workspace_id, dummy_func, 10, "arg1", kwarg1="val")
        except psycopg2.OperationalError as e:
            if "remaining connection slots" in str(e):
                pytest.skip("PostgreSQL connection slots are saturated.")
            raise e
            
        assert job.status == "queued"
        
        # Wait for worker thread to pick up and complete
        retries = 15
        completed_job = None
        while retries > 0:
            try:
                completed_job = manager.get_job(workspace_id, job.job_id)
            except psycopg2.OperationalError as e:
                if "remaining connection slots" in str(e):
                    pytest.skip("PostgreSQL connection slots are saturated.")
                raise e
                
            if completed_job and completed_job.status in ("completed", "failed"):
                break
            time.sleep(1.5)
            retries -= 1
            
        assert completed_job is not None
        assert completed_job.status == "completed"
        assert completed_job.progress_message == "Calculation completed successfully."
        assert completed_job.results.get("results") == {"output": "val"}
        
    finally:
        manager.stop()
        
        # Clean up database jobs
        try:
            conn = get_connection()
            if conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM compchem.jobs WHERE workspace_id = %s;", (workspace_id,))
                conn.commit()
                cur.close()
                conn.close()
        except Exception:
            pass

def test_job_recovery_db():
    if not settings.database_url:
        pytest.skip("PostgreSQL database URL is not configured.")
        
    workspace_id = f"test_ws_{uuid.uuid4().hex[:6]}"
    job_id = f"job_reco_{uuid.uuid4().hex[:6]}"
    
    # Insert a stale running job directly
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO compchem.jobs (job_id, workspace_id, status, progress_message, estimated_time_seconds, func_name, lease_timeout)
        VALUES (%s, %s, %s, %s, %s, %s, NOW() - INTERVAL '1 minute');
        """,
        (job_id, workspace_id, "running", "Stale calculation...", 10, "dummy_func")
    )
    conn.commit()
    cur.close()
    conn.close()
    
    # Trigger recovery via manager creation and startup
    manager = JobManager(max_workers=1)
    manager.start_workers()
    manager.stop()
    
    # Retrieve job status and check if failed
    job = manager.get_job(workspace_id, job_id)
    assert job is not None
    assert job.status == "failed"
    assert "crashed or server restarted" in job.error["message"]
    
    # Clean up
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM compchem.jobs WHERE workspace_id = %s;", (workspace_id,))
    conn.commit()
    cur.close()
    conn.close()
