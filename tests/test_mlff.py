import pytest
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.mlff_engine import (
    run_mlff_optimization_engine,
    run_mlff_molecular_dynamics_engine
)
from ypotheto_compchem_mcp.modules.mlff_tools import (
    run_mlff_optimization,
    run_mlff_molecular_dynamics
)

def test_mlff_optimization_and_md():
    workspace_id = get_workspace_id()
    
    # 1. Create a dummy structure (Water molecule)
    from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
    mol_res = build_molecule_from_smiles_engine("O", "Water MLFF")
    molecule_id = mol_res["molecule_id"]
    
    # 2. Test Optimization Engine
    opt_res = run_mlff_optimization_engine(workspace_id, molecule_id, model_name="CHGNet", fmax=0.1)
    assert opt_res["ok"] is True
    opt_id = opt_res["results"]["optimized_molecule_id"]
    assert opt_res["results"]["energy_ev"] < 0.0
    
    # 3. Test MD Engine
    md_res = run_mlff_molecular_dynamics_engine(
        workspace_id, opt_id, model_name="MACE", steps=10, timestep_fs=0.5, temperature_k=300.0, ensemble="nvt"
    )
    assert md_res["ok"] is True
    assert "trajectory_file_url" in md_res["results"]
    assert len(md_res["artifacts"]) == 1
    
    # 4. Test MCP Tool wrappers (sync)
    opt_tool = run_mlff_optimization(molecule_id, model_name="CHGNet", fmax=0.1, run_async=False)
    assert opt_tool["ok"] is True
    
    md_tool = run_mlff_molecular_dynamics(
        opt_tool["results"]["optimized_molecule_id"], model_name="MACE", steps=10, timestep_fs=0.5, temperature_k=100.0, ensemble="nvt", run_async=False
    )
    assert md_tool["ok"] is True
    assert "trajectory_file_url" in md_tool["results"]
