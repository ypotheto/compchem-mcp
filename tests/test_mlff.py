from unittest.mock import patch

import pytest

from ypotheto_compchem_mcp.chemistry.mlff_engine import (
    run_mlff_molecular_dynamics_engine,
    run_mlff_optimization_engine,
)
from ypotheto_compchem_mcp.errors import BackendUnavailableError
from ypotheto_compchem_mcp.modules.mlff_tools import (
    run_mlff_molecular_dynamics,
    run_mlff_optimization,
)
from ypotheto_compchem_mcp.workspace import get_workspace_id


def test_mlff_optimization_and_md():
    """
    Smoke-tests the optimizer/MD/artifact plumbing using a fake calculator explicitly
    injected by the test (via mocking mace_off), never a production-side silent
    substitution. Real weight downloads are avoided so this test doesn't need network.

    NOTE: uses MACE, not CHGNet - the chgnet package installed in this environment
    lacks the chgnet.calculators.ase submodule (CHGNET_AVAILABLE is False here), so a
    "CHGNet" run would previously have silently used the LJ fallback and passed these
    assertions without ever touching CHGNet.

    Separately: mace_off(default_dtype="float32") is also mocked because the raw
    MACECalculator this code used to call could never construct successfully either
    (it requires an explicit model_paths/models argument that was never supplied) -
    so the "MACE" path was silently falling back to LJ too, prior to this fix.
    """
    from ase.calculators.lj import LennardJones
    workspace_id = get_workspace_id()

    from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
    mol_res = build_molecule_from_smiles_engine("O", "Water MLFF")
    molecule_id = mol_res["molecule_id"]

    with patch("ypotheto_compchem_mcp.chemistry.mlff_engine.mace_off", return_value=LennardJones()):
        opt_res = run_mlff_optimization_engine(workspace_id, molecule_id, model_name="MACE", fmax=0.1)
        assert opt_res["ok"] is True
        opt_id = opt_res["results"]["optimized_molecule_id"]
        assert opt_res["results"]["energy_ev"] < 0.0
        assert opt_res["results"]["method_used"] == "MACE"

        md_res = run_mlff_molecular_dynamics_engine(
            workspace_id, opt_id, model_name="MACE", steps=10, timestep_fs=0.5, temperature_k=300.0, ensemble="nvt"
        )
        assert md_res["ok"] is True
        assert "trajectory_file_url" in md_res["results"]
        assert len(md_res["artifacts"]) == 1

        opt_tool = run_mlff_optimization(molecule_id, model_name="MACE", fmax=0.1, run_async=False)
        assert opt_tool["ok"] is True

        md_tool = run_mlff_molecular_dynamics(
            opt_tool["results"]["optimized_molecule_id"], model_name="MACE", steps=10, timestep_fs=0.5,
            temperature_k=100.0, ensemble="nvt", run_async=False
        )
        assert md_tool["ok"] is True
        assert "trajectory_file_url" in md_tool["results"]


def test_mlff_optimization_raises_when_chgnet_unavailable():
    # CHGNET_AVAILABLE is already False in this environment (missing chgnet.calculators.ase
    # submodule), so this exercises the real, unpatched code path.
    workspace_id = get_workspace_id()
    from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
    mol_res = build_molecule_from_smiles_engine("O", "Water MLFF Unavailable")
    molecule_id = mol_res["molecule_id"]

    with pytest.raises(BackendUnavailableError):
        run_mlff_optimization_engine(workspace_id, molecule_id, model_name="CHGNet", fmax=0.1)

    tool_res = run_mlff_optimization(molecule_id, model_name="CHGNet", fmax=0.1, run_async=False)
    assert tool_res["ok"] is False
    assert tool_res["error"]["code"] == "BACKEND_UNAVAILABLE"


def test_mlff_optimization_raises_when_mace_unavailable():
    workspace_id = get_workspace_id()
    from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
    mol_res = build_molecule_from_smiles_engine("O", "Water MLFF Unavailable MACE")
    molecule_id = mol_res["molecule_id"]

    with patch("ypotheto_compchem_mcp.chemistry.mlff_engine.MACE_AVAILABLE", False):
        with pytest.raises(BackendUnavailableError):
            run_mlff_optimization_engine(workspace_id, molecule_id, model_name="MACE", fmax=0.1)
