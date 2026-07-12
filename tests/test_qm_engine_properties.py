import json
from unittest.mock import MagicMock, patch

from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.chemistry.qm_engine import run_pyscf_properties_engine
from ypotheto_compchem_mcp.workspace import get_workspace_id, workspace_manager


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
