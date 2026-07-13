import json
from unittest.mock import MagicMock, patch

import pytest

from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.chemistry.qm_engine import run_pyscf_properties_engine
from ypotheto_compchem_mcp.errors import CalculationFailedError
from ypotheto_compchem_mcp.modules.quantum_tools import run_pyscf_properties
from ypotheto_compchem_mcp.workspace import get_workspace_id


@patch("ypotheto_compchem_mcp.chemistry.qm_engine.PYSCF_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.chemistry.qm_engine.get_engine_runner")
def test_run_pyscf_properties_engine_builds_interpretation_without_crashing(mock_get_runner):
    """
    Regression test: run_pyscf_properties_engine used to reference an undefined
    `loew_charges` variable while building the interpretation string, causing a
    NameError on every successful call. This exercises the REAL engine function
    (not a mock of it) so that bug can't hide behind a wrapper-level mock again.
    """
    workspace_id = get_workspace_id()
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]

    def fake_run_command(workspace_dir, job_id, cmd, input_files):
        results_dir = workspace_dir / "jobs" / job_id
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "results.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "energy_hartree": -76.01,
                    "energy_ev": -2068.3,
                    "dipole_moment_debye": [0.0, 0.0, 1.8],
                    "mulliken_charges": [-0.4, 0.2, 0.2],
                    "loewdin_charges": [-0.3, 0.15, 0.15],
                }
            ),
            encoding="utf-8",
        )
        return MagicMock(stderr="")

    mock_runner = MagicMock()
    mock_runner.run_command.side_effect = fake_run_command
    mock_get_runner.return_value = mock_runner

    res = run_pyscf_properties_engine(workspace_id, molecule_id, method="DFT")

    assert res["results"]["energy_ev"] == -2068.3
    assert res["results"]["loewdin_charges"][0]["charge"] == -0.3
    assert "interpretation" in res
    assert "Loewdin charges: O0:-0.3" in res["interpretation"]


@patch("ypotheto_compchem_mcp.chemistry.qm_engine.PYSCF_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.chemistry.qm_engine.get_engine_runner")
def test_run_pyscf_properties_engine_raises_when_results_file_missing(mock_get_runner):
    """When the driver produces no results.json (crash/timeout), the engine
    must raise a typed CalculationFailedError rather than returning an
    {"ok": False, ...} dict for the caller to manually unwrap."""
    workspace_id = get_workspace_id()
    mol_res = build_molecule_from_smiles_engine("O", "Water for missing results")
    molecule_id = mol_res["molecule_id"]

    mock_runner = MagicMock()
    mock_runner.run_command.return_value = MagicMock(stderr="pyscf_driver.py crashed with SegFault")
    mock_get_runner.return_value = mock_runner

    with pytest.raises(CalculationFailedError) as exc:
        run_pyscf_properties_engine(workspace_id, molecule_id, method="DFT")
    assert "results file is missing" in str(exc.value)

    with patch("ypotheto_compchem_mcp.modules.quantum_tools.PYSCF_AVAILABLE", True), \
         patch("ypotheto_compchem_mcp.modules.quantum_tools._estimate_time_seconds", return_value=3):
        tool_res = run_pyscf_properties(molecule_id, method="DFT", run_async=False)
    assert tool_res["ok"] is False
    assert tool_res["error"]["code"] == "CALCULATION_FAILED"
