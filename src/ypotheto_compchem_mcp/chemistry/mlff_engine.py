import logging
import uuid
from typing import Any

from ypotheto_compchem_mcp.errors import BackendUnavailableError

logger = logging.getLogger(__name__)

try:
    from chgnet.calculators.ase import CHGNetCalculator
    CHGNET_AVAILABLE = True
except ImportError:
    CHGNET_AVAILABLE = False

try:
    # mace_off loads a pretrained MACE-OFF23 organic-chemistry foundation model with no
    # explicit weights path required (mirrors CHGNetCalculator()'s zero-arg convenience).
    # The raw MACECalculator class requires an explicit model_paths/models argument that
    # this integration never supplied, so it could never construct successfully.
    from mace.calculators import mace_off
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False

def _load_structure(workspace_id: str, molecule_id: str):
    from ase.io import read

    from ypotheto_compchem_mcp.workspace import workspace_manager
    workspace_dir = workspace_manager.get_workspace_dir(workspace_id)
    cif_path = workspace_dir / "molecules" / f"{molecule_id}.cif"
    if cif_path.exists():
        return read(str(cif_path), format="cif")
    xyz_path = workspace_dir / "molecules" / f"{molecule_id}.xyz"
    if xyz_path.exists():
        return read(str(xyz_path), format="xyz")
    raise FileNotFoundError(f"Structure {molecule_id} not found in workspace.")

def _get_mlff_calculator(model_name: str):
    model_upper = model_name.upper()
    if model_upper == "CHGNET":
        if not CHGNET_AVAILABLE:
            raise BackendUnavailableError(
                "The CHGNet MLFF model could not be loaded (chgnet is not installed).",
                hint="pip install ypotheto-compchem-mcp[mlff], or use run_xtb_calculation / optimize_geometry instead.",
            )
        try:
            return CHGNetCalculator()
        except Exception as e:
            raise BackendUnavailableError(
                f"Failed to load CHGNet model: {str(e)}",
                hint="pip install ypotheto-compchem-mcp[mlff], or use run_xtb_calculation / optimize_geometry instead.",
            ) from e
    elif model_upper == "MACE":
        if not MACE_AVAILABLE:
            raise BackendUnavailableError(
                "The MACE MLFF model could not be loaded (mace-torch is not installed).",
                hint="pip install ypotheto-compchem-mcp[mlff], or use run_xtb_calculation / optimize_geometry instead.",
            )
        try:
            return mace_off(default_dtype="float32")
        except Exception as e:
            raise BackendUnavailableError(
                f"Failed to load MACE model: {str(e)}",
                hint="pip install ypotheto-compchem-mcp[mlff], or use run_xtb_calculation / optimize_geometry instead.",
            ) from e

    raise BackendUnavailableError(
        f"Unknown MLFF model '{model_name}'.",
        hint="Supported model_name values are 'CHGNet' or 'MACE'.",
    )

def run_mlff_optimization_engine(
    workspace_id: str,
    molecule_id: str,
    model_name: str = "CHGNet",
    fmax: float = 0.05
) -> dict[str, Any]:
    """
    Optimize molecular or periodic structures using pre-trained Machine Learning Force Fields.
    """
    atoms = _load_structure(workspace_id, molecule_id)
    atoms.calc = _get_mlff_calculator(model_name)
    
    from ase.optimize import BFGS
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=50)
    
    is_periodic = any(atoms.pbc)
    opt_id = f"crystal_opt_{uuid.uuid4().hex[:8]}" if is_periodic else f"mol_opt_{uuid.uuid4().hex[:8]}"
    formula = atoms.get_chemical_formula()
    name = f"MLFF Optimized {molecule_id} ({model_name})"
    
    import io

    from ase.io import write

    from ypotheto_compchem_mcp.chemistry.builder_engine import (
        get_molecules_dir,
        load_molecule_index,
        save_molecule_index,
    )
    
    f_xyz = io.StringIO()
    write(f_xyz, atoms, format="xyz")
    xyz_block = f_xyz.getvalue()
    
    mol_dir = get_molecules_dir(workspace_id)
    (mol_dir / f"{opt_id}.xyz").write_text(xyz_block, encoding="utf-8")
    
    cif_block = ""
    if is_periodic:
        f_cif = io.BytesIO()
        write(f_cif, atoms, format="cif")
        cif_block = f_cif.getvalue().decode("latin-1")
        (mol_dir / f"{opt_id}.cif").write_text(cif_block, encoding="utf-8")
        
    meta = {
        "molecule_id": opt_id,
        "name": name,
        "formula": formula,
        "num_atoms": len(atoms),
        "is_periodic": is_periodic,
        "cell": atoms.get_cell().tolist() if is_periodic else None,
        "method": f"MLFF Optimization ({model_name})"
    }
    
    index = load_molecule_index(workspace_id)
    index[opt_id] = meta
    save_molecule_index(workspace_id, index)
    
    energy_ev = float(atoms.get_potential_energy())
    
    return {
        "ok": True,
        "results": {
            "optimized_molecule_id": opt_id,
            "formula": formula,
            "energy_ev": energy_ev,
            "num_atoms": len(atoms),
            "method_used": model_name
        },
        "interpretation": f"Structure optimized successfully using {model_name}.\nFinal Energy = {energy_ev:.4f} eV."
    }

def run_mlff_molecular_dynamics_engine(
    workspace_id: str,
    molecule_id: str,
    model_name: str = "CHGNet",
    steps: int = 1000,
    timestep_fs: float = 1.0,
    temperature_k: float = 300.0,
    ensemble: str = "nvt"
) -> dict[str, Any]:
    """
    Run MD simulations driven by MLFF forces.
    """
    atoms = _load_structure(workspace_id, molecule_id)
    atoms.calc = _get_mlff_calculator(model_name)
    
    from ase import units
    from ase.md.langevin import Langevin

    dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature_k, friction=0.01)
    
    traj_out = [f"{len(atoms)}", "MLFF MD step 0"]
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
        traj_out.append(f"{sym} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
        
    n_chunks = 5
    chunk_steps = max(1, steps // n_chunks)
    for step_idx in range(n_chunks):
        dyn.run(chunk_steps)
        traj_out.append(f"{len(atoms)}")
        traj_out.append(f"MLFF MD step {(step_idx+1)*chunk_steps}")
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
            traj_out.append(f"{sym} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
            
    traj_content = "\n".join(traj_out)
    
    traj_filename = f"{molecule_id}_mlff_trajectory.xyz"
    from ypotheto_compchem_mcp.artifacts import register_artifact
    traj_art = register_artifact(traj_filename, traj_content.encode("utf-8"), "trajectory", f"MLFF MD Trajectory for {molecule_id}")
    
    results = {
        "trajectory_file_url": traj_art.url,
        "potential_energy_ev": float(atoms.get_potential_energy()),
        "model_name": model_name,
        "method_used": model_name
    }
    
    interpretation = (
        f"MLFF Molecular Dynamics simulation completed successfully.\n"
        f"Model: {model_name}, Steps: {steps}, Temperature: {temperature_k} K.\n"
        f"Final potential energy = {results['potential_energy_ev']:.4f} eV."
    )
    
    return {
        "ok": True,
        "results": results,
        "interpretation": interpretation,
        "artifacts": [traj_art]
    }
