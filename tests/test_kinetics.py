import sys
import types
import pytest
from unittest.mock import patch
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.errors import BackendUnavailableError
from ypotheto_compchem_mcp.chemistry.kinetics_engine import (
    run_transition_state_search_engine,
    run_neb_calculation_engine
)
from ypotheto_compchem_mcp.modules.kinetics_tools import (
    run_transition_state_search,
    run_neb_calculation
)


def _save_water_and_product():
    from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
    workspace_id = get_workspace_id()
    mol_xyz = "3\nWater molecule\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.0 0.9 0.0"
    save_molecule_coords(workspace_id, "mol_h2o_ts_guess", "", mol_xyz, {"molecule_id": "mol_h2o_ts_guess", "name": "water"})
    return workspace_id


def test_transition_state_search_raises_when_xtb_unavailable():
    # No xtb binary is installed in the test environment (or in CI); the engine must
    # raise a typed BackendUnavailableError rather than silently substituting LJ.
    workspace_id = _save_water_and_product()

    with patch("shutil.which", return_value=None):
        with pytest.raises(BackendUnavailableError) as exc:
            run_transition_state_search_engine(
                workspace_id=workspace_id,
                molecule_id="mol_h2o_ts_guess",
                method="xTB"
            )
    assert "xTB" in str(exc.value)

    # Through the MCP tool wrapper, this becomes a clean error envelope, not a crash.
    tool_res = run_transition_state_search(
        molecule_id="mol_h2o_ts_guess",
        method="xTB",
        run_async=False
    )
    assert tool_res["ok"] is False
    assert tool_res["error"]["code"] == "BACKEND_UNAVAILABLE"


def test_neb_pathway_raises_when_xtb_unavailable():
    from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
    workspace_id = get_workspace_id()

    reactant_xyz = "3\nReactant\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.0 0.9 0.0"
    product_xyz = "3\nProduct\nO 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 1.0 0.0"

    save_molecule_coords(workspace_id, "mol_reactant", "", reactant_xyz, {"molecule_id": "mol_reactant", "name": "reactant"})
    save_molecule_coords(workspace_id, "mol_product", "", product_xyz, {"molecule_id": "mol_product", "name": "product"})

    with patch("shutil.which", return_value=None):
        with pytest.raises(BackendUnavailableError):
            run_neb_calculation_engine(
                workspace_id=workspace_id,
                reactant_molecule_id="mol_reactant",
                product_molecule_id="mol_product",
                num_images=3,
                method="xTB",
                interpolation="linear"
            )

    tool_res = run_neb_calculation(
        reactant_molecule_id="mol_reactant",
        product_molecule_id="mol_product",
        num_images=3,
        method="xTB",
        interpolation="linear",
        run_async=False
    )
    assert tool_res["ok"] is False
    assert tool_res["error"]["code"] == "BACKEND_UNAVAILABLE"


def test_transition_state_search_succeeds_with_mocked_xtb():
    """Smoke-tests the Sella/optimizer/artifact plumbing using a fake XTB calculator
    explicitly injected by the test (via sys.modules, since ase.calculators.xtb isn't
    installed in this environment) - never a production-side silent substitution."""
    from ase.calculators.lj import LennardJones
    workspace_id = _save_water_and_product()

    fake_xtb_module = types.ModuleType("ase.calculators.xtb")
    fake_xtb_module.XTB = LennardJones

    with patch("shutil.which", return_value="/usr/bin/xtb"), \
         patch.dict(sys.modules, {"ase.calculators.xtb": fake_xtb_module}):
        res = run_transition_state_search_engine(
            workspace_id=workspace_id,
            molecule_id="mol_h2o_ts_guess",
            method="xTB"
        )
    assert res["ok"] is True
    assert "ts_molecule_id" in res
    assert res["energy_ev"] is not None
    assert res["num_atoms"] == 3
    assert res["method_used"] == "GFN2-xTB"
