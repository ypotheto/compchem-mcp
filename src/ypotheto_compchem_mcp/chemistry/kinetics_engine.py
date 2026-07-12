import logging
import uuid
from typing import Any, Dict, List, Optional

from ypotheto_compchem_mcp.errors import BackendUnavailableError

logger = logging.getLogger(__name__)

_XTB_UNAVAILABLE_HINT = "Install the xtb binary and the xtb-python ASE calculator, or rerun with method='DFT'."

def _load_atoms_from_xyz(workspace_id: str, molecule_id: str):
    from ypotheto_compchem_mcp.workspace import workspace_manager
    from ase import Atoms
    workspace_dir = workspace_manager.get_workspace_dir(workspace_id)
    xyz_path = workspace_dir / "molecules" / f"{molecule_id}.xyz"
    if not xyz_path.exists():
        raise FileNotFoundError(f"Molecule {molecule_id} coordinates not found.")
    xyz_content = xyz_path.read_text(encoding="utf-8")
    symbols = []
    positions = []
    lines = xyz_content.strip().split("\n")
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return Atoms(symbols=symbols, positions=positions)

def run_transition_state_search_engine(
    workspace_id: str,
    molecule_id: str,
    method: str = "xTB",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0
) -> Dict[str, Any]:
    """
    Optimize transition state structure using Sella.
    """
    atoms = _load_atoms_from_xyz(workspace_id, molecule_id)
    
    method_upper = method.upper()
    method_used = ""
    if method_upper == "XTB":
        import shutil
        if shutil.which("xtb"):
            from ase.calculators.xtb import XTB
            atoms.calc = XTB(method="GFN2-xTB")
            method_used = "GFN2-xTB"
        else:
            raise BackendUnavailableError(
                "xTB backend is not available for transition-state search.",
                hint=_XTB_UNAVAILABLE_HINT,
            )
    else:
        from ypotheto_compchem_mcp.chemistry.qm_engine import PySCFCalculator
        atoms.calc = PySCFCalculator(method=method, functional=functional, basis=basis, charge=charge, spin=spin)
        method_used = f"{method.upper()}/{functional}/{basis}"

    from sella import Sella
    opt = Sella(atoms, logfile=None)
    opt.run(fmax=0.05, steps=50)
    
    new_xyz_lines = [f"{len(atoms)}", "Transition State Optimized Geometry"]
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        new_xyz_lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    new_xyz_block = "\n".join(new_xyz_lines)
    
    ts_id = f"mol_ts_{uuid.uuid4().hex[:8]}"
    ts_name = f"TS for {molecule_id} ({method})"
    
    from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
    meta = {
        "molecule_id": ts_id,
        "name": ts_name,
        "formula": "",
        "smiles": "",
        "num_atoms": len(atoms),
        "is_transition_state": True,
        "method": f"Sella TS Optimization ({method})"
    }
    save_molecule_coords(workspace_id, ts_id, "", new_xyz_block, meta)
    
    energy_ev = float(atoms.get_potential_energy())
    
    return {
        "ok": True,
        "ts_molecule_id": ts_id,
        "name": ts_name,
        "energy_ev": energy_ev,
        "num_atoms": len(atoms),
        "method_used": method_used
    }

def run_neb_calculation_engine(
    workspace_id: str,
    reactant_molecule_id: str,
    product_molecule_id: str,
    num_images: int = 5,
    method: str = "xTB",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    interpolation: str = "idpp"
) -> Dict[str, Any]:
    """
    Perform NEB path optimization between reactant and product states.
    """
    atoms_r = _load_atoms_from_xyz(workspace_id, reactant_molecule_id)
    atoms_p = _load_atoms_from_xyz(workspace_id, product_molecule_id)
    
    images = [atoms_r]
    for _ in range(num_images):
        images.append(atoms_r.copy())
    images.append(atoms_p)
    
    from ase.mep import NEB
    neb = NEB(images)
    
    if interpolation.lower() == "idpp":
        neb.interpolate(method='idpp')
    else:
        neb.interpolate(method='linear')
        
    method_upper = method.upper()
    method_used = ""
    for img in images:
        if method_upper == "XTB":
            import shutil
            if shutil.which("xtb"):
                from ase.calculators.xtb import XTB
                img.calc = XTB(method="GFN2-xTB")
                method_used = "GFN2-xTB"
            else:
                raise BackendUnavailableError(
                    "xTB backend is not available for NEB pathway calculations.",
                    hint=_XTB_UNAVAILABLE_HINT,
                )
        else:
            from ypotheto_compchem_mcp.chemistry.qm_engine import PySCFCalculator
            img.calc = PySCFCalculator(method=method, functional=functional, basis=basis, charge=charge, spin=spin)
            method_used = f"{method.upper()}/{functional}/{basis}"

    from ase.optimize import BFGS
    opt = BFGS(neb, logfile=None)
    opt.run(fmax=0.05, steps=50)
    
    energies = [float(img.get_potential_energy()) for img in images]
    energy_barrier_ev = max(energies) - energies[0]
    energy_barrier_kcal = energy_barrier_ev * 23.0605
    
    image_ids = []
    for idx, img in enumerate(images):
        new_xyz_lines = [f"{len(img)}", f"NEB Image {idx}"]
        for sym, pos in zip(img.get_chemical_symbols(), img.get_positions()):
            new_xyz_lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        new_xyz_block = "\n".join(new_xyz_lines)
        
        img_id = f"mol_neb_{idx}_{uuid.uuid4().hex[:8]}"
        img_name = f"NEB Image {idx} for reactant {reactant_molecule_id}"
        
        from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
        meta = {
            "molecule_id": img_id,
            "name": img_name,
            "formula": "",
            "smiles": "",
            "num_atoms": len(img),
            "method": f"NEB Image {idx} ({method})",
            "is_neb_image": True,
            "image_index": idx,
            "energy_ev": energies[idx]
        }
        save_molecule_coords(workspace_id, img_id, "", new_xyz_block, meta)
        image_ids.append(img_id)
        
    plot_art = None
    try:
        import matplotlib.pyplot as plt
        import io
        from ypotheto_compchem_mcp.artifacts import register_artifact
        
        plt.figure(figsize=(7, 4.5))
        rel_energies = [(e - energies[0]) * 23.0605 for e in energies]
        plt.plot(range(len(images)), rel_energies, marker='o', color='purple', linestyle='-', linewidth=2)
        plt.xlabel("Reaction Coordinate (Image Index)")
        plt.ylabel("Relative Energy (kcal/mol)")
        plt.title("Reaction Energy Profile (NEB)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        
        plot_art = register_artifact("neb_reaction_profile.png", buf.getvalue(), "plot", "Reaction Coordinate Energy Profile")
    except Exception as e:
        logger.warning(f"Failed to generate reaction profile plot: {str(e)}")
        
    results = {
        "activation_energy_barrier_ev": energy_barrier_ev,
        "activation_energy_barrier_kcal_mol": energy_barrier_kcal,
        "energies_ev": energies,
        "image_molecule_ids": image_ids,
        "method_used": method_used
    }
    
    interpretation = (
        f"NEB Reaction Path optimization completed successfully.\n"
        f"Activation Energy Barrier = {energy_barrier_kcal:.2f} kcal/mol ({energy_barrier_ev:.4f} eV).\n"
        f"Path intermediate images saved (count: {len(image_ids)})."
    )
    
    artifacts = [plot_art] if plot_art else []
    if plot_art:
        results["plot_url"] = plot_art.url
        
    return {
        "ok": True,
        "results": results,
        "interpretation": interpretation,
        "artifacts": artifacts
    }
