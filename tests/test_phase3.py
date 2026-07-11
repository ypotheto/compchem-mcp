import pytest
import sys
import time
from unittest.mock import patch, MagicMock
from ypotheto_compchem_mcp.modules.quantum_tools import estimate_calculation_time, run_single_point, optimize_geometry, get_job_status
from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.jobs import job_manager

def test_estimate_calculation_time():
    # Build a test molecule (Water)
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    envelope = estimate_calculation_time(molecule_id, "DFT", "sto-3g")
    assert envelope["ok"] is True
    assert envelope["results"]["molecule_id"] == molecule_id
    assert envelope["results"]["estimated_time_seconds"] > 0
    assert "estimated to take" in envelope["interpretation"]

def test_qm_pyscf_unavailable_on_windows():
    # Since the tools are decorated with @mcp_tool_decorator, they don't raise RuntimeError to the caller.
    # Instead, the decorator catches it and returns a standard error envelope {"ok": False, "error": {...}}
    with patch("ypotheto_compchem_mcp.modules.quantum_tools.PYSCF_AVAILABLE", False):
        envelope = run_single_point("mol_dummy")
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "INTERNAL_ERROR"
        assert "not installed or available" in envelope["error"]["message"]
        
        envelope = optimize_geometry("mol_dummy")
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "INTERNAL_ERROR"
        assert "not installed or available" in envelope["error"]["message"]

@patch("ypotheto_compchem_mcp.modules.quantum_tools.PYSCF_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.modules.quantum_tools._estimate_time_seconds", return_value=5)
@patch("ypotheto_compchem_mcp.modules.quantum_tools.run_single_point_engine")
def test_run_single_point_sync(mock_engine, mock_est):
    # Mock engine response
    mock_engine.return_value = {
        "ok": True,
        "results": {
            "energy_hartree": -76.01,
            "energy_ev": -2068.3,
            "dipole_moment_debye": [0.0, 0.0, 1.8],
            "homo_ev": -12.0,
            "lumo_ev": 4.0,
            "homo_lumo_gap_ev": 16.0,
            "mulliken_charges": []
        },
        "warnings": []
    }
    
    envelope = run_single_point("mol_test", method="DFT", run_async=False)
    assert envelope["ok"] is True
    # In success response, envelope["results"] contains results dictionary directly
    assert envelope["results"]["energy_hartree"] == -76.01
    assert "Single-point calculation completed" in envelope["interpretation"]
    assert len(envelope["artifacts"]) == 1

@patch("ypotheto_compchem_mcp.modules.quantum_tools.PYSCF_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.modules.quantum_tools._estimate_time_seconds", return_value=15)
@patch("ypotheto_compchem_mcp.modules.quantum_tools.run_single_point_engine")
def test_run_single_point_async(mock_engine, mock_est):
    # Mock engine response
    mock_engine.return_value = {
        "ok": True,
        "results": {
            "energy_ev": -2000.0,
            "energy_hartree": -75.0,
            "dipole_moment_debye": [0.0, 0.0, 0.0],
            "homo_ev": -10.0,
            "lumo_ev": 2.0,
            "homo_lumo_gap_ev": 12.0,
            "mulliken_charges": []
        },
        "warnings": []
    }
    
    # Submit async
    envelope = run_single_point("mol_test", method="DFT", run_async=True)
    assert envelope["ok"] is True
    assert "job_id" in envelope["results"]
    job_id = envelope["results"]["job_id"]
    
    # Check job status via polling tool
    # Wait up to 1 second for background thread to run the mocked engine
    for _ in range(15):
        time.sleep(0.1)
        status_envelope = get_job_status(job_id)
        if status_envelope["results"]["status"] == "completed":
            break
            
    final_status = get_job_status(job_id)
    assert final_status["ok"] is True
    assert final_status["results"]["status"] == "completed"
    assert final_status["results"]["results"]["energy_ev"] == -2000.0
