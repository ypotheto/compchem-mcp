import pytest
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.kinetics_engine import (
    run_transition_state_search_engine,
    run_neb_calculation_engine
)
from ypotheto_compchem_mcp.modules.kinetics_tools import (
    run_transition_state_search,
    run_neb_calculation
)

def test_transition_state_search():
    workspace_id = get_workspace_id()
    
    # Save a dummy reactant structure to optimize
    from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
    mol_xyz = "3\nWater molecule\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.0 0.9 0.0"
    save_molecule_coords(workspace_id, "mol_h2o_ts_guess", "", mol_xyz, {"molecule_id": "mol_h2o_ts_guess", "name": "water"})
    
    # Run Sella TS optimization engine
    res = run_transition_state_search_engine(
        workspace_id=workspace_id,
        molecule_id="mol_h2o_ts_guess",
        method="xTB"
    )
    assert res["ok"] is True
    assert "ts_molecule_id" in res
    assert res["energy_ev"] is not None
    assert res["num_atoms"] == 3
    
    # Test tool wrapper
    tool_res = run_transition_state_search(
        molecule_id="mol_h2o_ts_guess",
        method="xTB",
        run_async=False
    )
    assert tool_res["ok"] is True
    assert "ts_molecule_id" in tool_res["results"]

def test_neb_pathway_calculation():
    workspace_id = get_workspace_id()
    
    # Save reactant and product structures
    from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
    reactant_xyz = "3\nReactant\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.0 0.9 0.0"
    product_xyz = "3\nProduct\nO 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 1.0 0.0"
    
    save_molecule_coords(workspace_id, "mol_reactant", "", reactant_xyz, {"molecule_id": "mol_reactant", "name": "reactant"})
    save_molecule_coords(workspace_id, "mol_product", "", product_xyz, {"molecule_id": "mol_product", "name": "product"})
    
    # Run NEB engine
    res = run_neb_calculation_engine(
        workspace_id=workspace_id,
        reactant_molecule_id="mol_reactant",
        product_molecule_id="mol_product",
        num_images=3,
        method="xTB",
        interpolation="linear"
    )
    assert res["ok"] is True
    assert "results" in res
    assert "activation_energy_barrier_ev" in res["results"]
    assert len(res["results"]["image_molecule_ids"]) == 5  # 3 intermediate + 2 endpoints = 5
    assert len(res["artifacts"]) == 1
    
    # Test tool wrapper
    tool_res = run_neb_calculation(
        reactant_molecule_id="mol_reactant",
        product_molecule_id="mol_product",
        num_images=3,
        method="xTB",
        interpolation="linear",
        run_async=False
    )
    assert tool_res["ok"] is True
    assert "activation_energy_barrier_kcal_mol" in tool_res["results"]
