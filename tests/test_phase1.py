import time

from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.jobs import job_manager
from ypotheto_compchem_mcp.storage import LocalDirBackend
from ypotheto_compchem_mcp.workspace import (
    current_workspace_id,
    get_workspace_id,
    workspace_manager,
)


def test_storage_backend(tmp_path):
    backend = LocalDirBackend(tmp_path)
    workspace_id = "test_ws"
    
    # Write file
    backend.write_file(workspace_id, "test.txt", b"hello world")
    assert backend.file_exists(workspace_id, "test.txt")
    
    # Read file
    data = backend.read_file(workspace_id, "test.txt")
    assert data == b"hello world"
    
    # List files
    files = backend.list_files(workspace_id)
    assert "test.txt" in files
    
    # Delete file
    backend.delete_file(workspace_id, "test.txt")
    assert not backend.file_exists(workspace_id, "test.txt")

def test_workspace_context():
    # Verify default context
    assert get_workspace_id() == "local"
    
    # Set context
    token = current_workspace_id.set("custom_workspace")
    try:
        assert get_workspace_id() == "custom_workspace"
    finally:
        current_workspace_id.reset(token)

def test_register_artifact(tmp_path):
    # Temporarily override workspace_manager data directory
    original_dir = workspace_manager.data_dir
    workspace_manager.data_dir = tmp_path
    
    try:
        # Register artifact
        art = register_artifact("structure.xyz", b"C 0 0 0", "structure", "Methane coord")
        assert art.kind == "structure"
        assert art.description == "Methane coord"
        assert "structure.xyz" in art.url
        
        # Verify file exists on disk
        workspace_id = get_workspace_id()
        workspace_artifacts_dir = workspace_manager.get_artifacts_dir(workspace_id)
        # Search for file
        files = list(workspace_artifacts_dir.glob("**/structure.xyz"))
        assert len(files) == 1
        assert files[0].read_bytes() == b"C 0 0 0"
    finally:
        workspace_manager.data_dir = original_dir

def test_job_submission():
    # Simple calculation function
    def calculate_energy(val):
        time.sleep(0.1)
        return {
            "ok": True,
            "results": {"energy": val * 2},
            "interpretation": f"Calculated energy is {val * 2}"
        }
        
    job = job_manager.submit_job("test_workspace", calculate_energy, estimated_time=5, val=10)
    assert job.status in ("running", "completed")
    
    # Wait for completion (max 2 seconds)
    for _ in range(20):
        time.sleep(0.1)
        checked_job = job_manager.get_job("test_workspace", job.job_id)
        if checked_job and checked_job.status == "completed":
            break
            
    final_job = job_manager.get_job("test_workspace", job.job_id)
    assert final_job is not None
    assert final_job.status == "completed"
    assert final_job.results["energy"] == 20
    assert "Calculated energy" in final_job.interpretation
