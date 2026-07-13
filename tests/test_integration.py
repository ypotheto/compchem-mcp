"""Real-backend integration tests, run only when the backend is actually
available (never mocked). Every heavy backend elsewhere in this suite is
mocked, so CI never verifies a single real calculation - a past bug
(`run_pyscf_properties_engine` referencing an undefined `loew_charges`
variable) survived unnoticed for exactly this reason.

Run with: pytest -m integration
Auto-skipped on a machine without the relevant binary/package (via
shutil.which / importlib.util.find_spec), so `pytest` (without -m) never
depends on these. The project Docker image is the canonical venue that has
every backend installed - see planning/HUMAN_TASKS.md item 4 and the
Dockerfile's `test` build stage (`docker build --target test`).
"""
import importlib.util
import shutil

import pytest

from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.workspace import get_workspace_id

pytestmark = pytest.mark.integration

_XTB_AVAILABLE = shutil.which("xtb") is not None
_CREST_AVAILABLE = shutil.which("crest") is not None
_PACKMOL_AVAILABLE = shutil.which("packmol") is not None
_LAMMPS_AVAILABLE = bool(
    shutil.which("lammps") or shutil.which("lmp")
    or shutil.which("lmp_serial") or shutil.which("lmp_mpi")
)
_PYSCF_AVAILABLE = importlib.util.find_spec("pyscf") is not None


def _build_water(workspace_id: str) -> str:
    res = build_molecule_from_smiles_engine("O", "Water")
    return res["molecule_id"]


@pytest.mark.skipif(not _XTB_AVAILABLE, reason="xtb binary not installed")
def test_xtb_gfn2_single_point_water_matches_known_energy():
    """GFN2-xTB single-point energy for water at its own relaxed geometry is a
    well-known reference value (~-5.070 Ha) - used elsewhere in this suite's
    mocks (test_xtb.py), verified here against the real xtb binary."""
    from ypotheto_compchem_mcp.chemistry.xtb_engine import run_xtb_calculation_engine

    workspace_id = get_workspace_id()
    molecule_id = _build_water(workspace_id)

    res = run_xtb_calculation_engine(
        workspace_id, molecule_id, task="single_point", method="GFN2-xTB"
    )

    assert res["ok"] is True
    energy_hartree = res["results"]["energy_hartree"]
    assert energy_hartree == pytest.approx(-5.070, abs=1e-3)


@pytest.mark.skipif(not _PYSCF_AVAILABLE, reason="pyscf not installed")
def test_pyscf_hf_sto3g_single_point_water_matches_known_energy():
    """HF/STO-3G single-point energy for water at a fixed, standard geometry
    is a textbook reference value (~-74.96 Ha)."""
    from ypotheto_compchem_mcp.chemistry.qm_engine import run_single_point_engine

    workspace_id = get_workspace_id()
    molecule_id = _build_water(workspace_id)

    res = run_single_point_engine(
        workspace_id, molecule_id, method="HF", basis="sto-3g"
    )

    assert res["ok"] is True
    energy_hartree = res["results"]["energy_hartree"]
    assert energy_hartree == pytest.approx(-74.96, abs=0.01)


@pytest.mark.skipif(not (_CREST_AVAILABLE and _XTB_AVAILABLE), reason="crest/xtb binaries not installed")
def test_crest_conformer_search_smoke_run_on_tiny_molecule():
    """Smoke-runs a real CREST conformer search on ethanol (small enough to
    finish quickly) - just confirms the real binary integration produces a
    plausible, non-empty result, not a specific reference energy."""
    from ypotheto_compchem_mcp.chemistry.xtb_engine import run_conformer_search_engine

    workspace_id = get_workspace_id()
    res = build_molecule_from_smiles_engine("CCO", "Ethanol")
    molecule_id = res["molecule_id"]

    result = run_conformer_search_engine(workspace_id, molecule_id, method="GFN2-xTB")

    assert result["ok"] is True
    assert result["results"]["num_conformers"] >= 1


@pytest.mark.skipif(not _PACKMOL_AVAILABLE, reason="packmol binary not installed")
def test_packmol_amorphous_cell_smoke_run_on_tiny_system():
    """Smoke-runs a real Packmol packing of a handful of water molecules -
    confirms the real binary integration produces a plausible, non-empty
    packed structure, not a specific reference density/energy."""
    from ypotheto_compchem_mcp.chemistry.polymer_engine import pack_amorphous_cell_engine

    workspace_id = get_workspace_id()
    molecule_id = _build_water(workspace_id)

    result = pack_amorphous_cell_engine(
        workspace_id, molecule_ids=[molecule_id], counts=[5], density_g_cm3=0.5
    )

    assert result["num_atoms"] > 0


@pytest.mark.skipif(not _LAMMPS_AVAILABLE, reason="lammps binary not installed")
def test_lammps_simulation_smoke_run_on_tiny_system():
    """Smoke-runs a handful of real LAMMPS MD steps on a tiny packed cell -
    confirms the real binary integration produces a plausible, non-empty
    thermo result, not a specific reference value."""
    from ypotheto_compchem_mcp.chemistry.polymer_engine import (
        pack_amorphous_cell_engine,
        run_lammps_simulation_engine,
    )

    workspace_id = get_workspace_id()
    molecule_id = _build_water(workspace_id)
    packed = pack_amorphous_cell_engine(
        workspace_id, molecule_ids=[molecule_id], counts=[10], density_g_cm3=0.5
    )

    result = run_lammps_simulation_engine(
        workspace_id, packed["packed_molecule_id"], steps=10, timestep_fs=1.0
    )

    assert result["ok"] is True
