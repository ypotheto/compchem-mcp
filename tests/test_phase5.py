
from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.chemistry.md_engine import run_molecular_dynamics_engine
from ypotheto_compchem_mcp.modules.dynamics_tools import run_molecular_dynamics
from ypotheto_compchem_mcp.workspace import get_workspace_id


def test_molecular_dynamics_engine_ff():
    workspace_id = get_workspace_id()
    # Build Water
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Run molecular dynamics with MMFF94 force field (runs natively on Windows)
    res = run_molecular_dynamics_engine(
        workspace_id,
        molecule_id,
        steps=20,
        time_step_fs=0.5,
        temperature_k=300.0,
        ensemble="NVT",
        calculator_type="MMFF94"
    )
    assert res["ok"] is True
    assert res["results"]["steps_run"] == 20
    assert len(res["results"]["energy_history"]) > 0
    assert len(res["trajectory_xyz"]) > 0
    assert len(res["plot_bytes"]) > 0
    assert res["results"]["final_temperature_k"] > 0.0

def test_molecular_dynamics_tool_sync():
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Run tool synchronously (run_async=False)
    envelope = run_molecular_dynamics(
        molecule_id,
        steps=10,
        time_step_fs=0.5,
        temperature_k=300.0,
        ensemble="NVT",
        calculator_type="MMFF94",
        run_async=False
    )
    
    assert envelope["ok"] is True
    assert envelope["results"]["steps_run"] == 10
    assert len(envelope["artifacts"]) == 2
    assert "trajectory.xyz" in envelope["artifacts"][0]["url"]
    assert "md_profile.png" in envelope["artifacts"][1]["url"]
    assert "Molecular Dynamics simulation completed" in envelope["interpretation"]
