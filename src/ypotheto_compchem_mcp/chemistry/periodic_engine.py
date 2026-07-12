import io
import re
import uuid
import logging
import numpy as np
import spglib
from typing import Any, Dict, List, Optional, Tuple

from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell

from ypotheto_compchem_mcp.workspace import workspace_manager
from ypotheto_compchem_mcp.chemistry.builder_engine import _get_molecules_dir, _load_index, _save_index
from ypotheto_compchem_mcp.errors import BackendUnavailableError, CalculationFailedError

logger = logging.getLogger(__name__)

def import_periodic_structure_engine(
    workspace_id: str,
    cif_content: str,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a CIF string, convert it to an ASE Atoms object, calculate crystal
    symmetry properties via spglib, and store it in the workspace directory.
    """
    # 1. Parse CIF content using ASE
    try:
        f = io.StringIO(cif_content)
        atoms = read(f, format="cif")
    except Exception as e:
        logger.error(f"Failed to parse CIF contents: {str(e)}")
        raise ValueError(f"Failed to parse CIF contents: {str(e)}")

    # Ensure system is periodic
    atoms.pbc = [True, True, True]

    # 2. Extract space group and symmetry data using spglib
    lattice = atoms.get_cell()
    scaled_positions = atoms.get_scaled_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    
    cell = (lattice, scaled_positions, atomic_numbers)
    
    spg_symbol = "Unknown"
    spg_number = None
    try:
        spacegroup = spglib.get_spacegroup(cell, symprec=1e-5)
        if spacegroup:
            match = re.search(r"^(.*?)\s*\((\d+)\)$", spacegroup)
            if match:
                spg_symbol = match.group(1).strip()
                spg_number = int(match.group(2))
            else:
                spg_symbol = spacegroup
    except Exception as e:
        logger.warning(f"Failed to extract spacegroup using spglib: {str(e)}")

    # 3. Export structure representations
    molecule_id = f"crystal_{uuid.uuid4().hex[:8]}"
    formula = atoms.get_chemical_formula()
    name_str = name or f"Crystal {formula}"

    # Write CIF and XYZ block strings
    f_cif = io.BytesIO()
    write(f_cif, atoms, format="cif")
    cif_block = f_cif.getvalue().decode("latin-1")

    f_xyz = io.StringIO()
    write(f_xyz, atoms, format="xyz")
    xyz_block = f_xyz.getvalue()

    # 4. Save files to molecules folder
    mol_dir = _get_molecules_dir(workspace_id)
    (mol_dir / f"{molecule_id}.cif").write_text(cif_block, encoding="utf-8")
    (mol_dir / f"{molecule_id}.xyz").write_text(xyz_block, encoding="utf-8")

    # 5. Register in workspace index
    meta = {
        "molecule_id": molecule_id,
        "name": name_str,
        "formula": formula,
        "num_atoms": len(atoms),
        "is_periodic": True,
        "cell": lattice.tolist(),
        "space_group_symbol": spg_symbol,
        "space_group_number": spg_number,
        "method": "CIF Import"
    }
    
    index = _load_index(workspace_id)
    index[molecule_id] = meta
    _save_index(workspace_id, index)

    return {
        "molecule_id": molecule_id,
        "name": name_str,
        "formula": formula,
        "num_atoms": len(atoms),
        "lattice_parameters": {
            "a": float(atoms.cell.cellpar()[0]),
            "b": float(atoms.cell.cellpar()[1]),
            "c": float(atoms.cell.cellpar()[2]),
            "alpha": float(atoms.cell.cellpar()[3]),
            "beta": float(atoms.cell.cellpar()[4]),
            "gamma": float(atoms.cell.cellpar()[5])
        },
        "space_group": {
            "symbol": spg_symbol,
            "number": spg_number
        },
        "cif_block": cif_block,
        "xyz_block": xyz_block
    }

def load_periodic_structure_engine(workspace_id: str, molecule_id: str) -> Atoms:
    """Load a periodic structure from the workspace as an ASE Atoms object."""
    mol_dir = _get_molecules_dir(workspace_id)
    cif_path = mol_dir / f"{molecule_id}.cif"
    if not cif_path.exists():
        raise FileNotFoundError(f"Periodic structure {molecule_id} coordinates (.cif) not found in workspace.")
    return read(str(cif_path), format="cif")

def analyze_crystal_symmetry_engine(workspace_id: str, molecule_id: str) -> Dict[str, Any]:
    """
    Get detailed crystallographic symmetry dataset via spglib for a stored structure.
    """
    atoms = load_periodic_structure_engine(workspace_id, molecule_id)
    
    lattice = atoms.get_cell()
    scaled_positions = atoms.get_scaled_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    cell = (lattice, scaled_positions, atomic_numbers)
    
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
    if not dataset:
        raise RuntimeError("Failed to resolve symmetry dataset using spglib.")

    # Convert numpy arrays to standard python lists for JSON serialization
    symmetry_data = {
        "number": int(dataset.number),
        "international": str(dataset.international),
        "hall": str(dataset.hall),
        "choice": str(dataset.choice),
        "transformation_matrix": dataset.transformation_matrix.tolist(),
        "origin_shift": dataset.origin_shift.tolist(),
        "rotations": dataset.rotations.tolist(),
        "translations": dataset.translations.tolist(),
        "wyckoffs": list(dataset.wyckoffs),
        "equivalent_atoms": dataset.equivalent_atoms.tolist()
    }
    
    return {
        "ok": True,
        "results": symmetry_data,
        "warnings": []
    }

def generate_supercell_engine(
    workspace_id: str,
    molecule_id: str,
    sc_matrix: List[int],
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a supercell expansion using a scaling matrix and save it back to the workspace.
    Supports either diagonal scaling [nx, ny, nz] or a full 3x3 expansion matrix.
    """
    atoms = load_periodic_structure_engine(workspace_id, molecule_id)
    
    # Process scaling input
    if len(sc_matrix) == 3:
        # Diagonal expansion matrix
        P = np.diag(sc_matrix)
    elif len(sc_matrix) == 9:
        # 3x3 transformation matrix flattened
        P = np.array(sc_matrix).reshape(3, 3)
    else:
        raise ValueError("sc_matrix must be either a 3-element list (diagonal) or 9-element list (3x3).")

    # Generate supercell using ASE
    try:
        super_atoms = make_supercell(atoms, P)
    except Exception as e:
        raise RuntimeError(f"ASE supercell generation failed: {str(e)}")

    # Recalculate spacegroup for the supercell
    super_cell = (super_atoms.get_cell(), super_atoms.get_scaled_positions(), super_atoms.get_atomic_numbers())
    
    spg_symbol = "Unknown"
    spg_number = None
    try:
        spacegroup = spglib.get_spacegroup(super_cell, symprec=1e-5)
        if spacegroup:
            match = re.search(r"^(.*?)\s*\((\d+)\)$", spacegroup)
            if match:
                spg_symbol = match.group(1).strip()
                spg_number = int(match.group(2))
            else:
                spg_symbol = spacegroup
    except Exception:
        pass

    # Save supercell
    super_id = f"crystal_{uuid.uuid4().hex[:8]}"
    formula = super_atoms.get_chemical_formula()
    name_str = name or f"Supercell ({'x'.join(map(str, np.diag(P) if len(sc_matrix)==3 else ['matrix']))}) {formula}"

    f_cif = io.BytesIO()
    write(f_cif, super_atoms, format="cif")
    cif_block = f_cif.getvalue().decode("latin-1")

    f_xyz = io.StringIO()
    write(f_xyz, super_atoms, format="xyz")
    xyz_block = f_xyz.getvalue()

    mol_dir = _get_molecules_dir(workspace_id)
    (mol_dir / f"{super_id}.cif").write_text(cif_block, encoding="utf-8")
    (mol_dir / f"{super_id}.xyz").write_text(xyz_block, encoding="utf-8")

    meta = {
        "molecule_id": super_id,
        "name": name_str,
        "formula": formula,
        "num_atoms": len(super_atoms),
        "is_periodic": True,
        "cell": super_atoms.get_cell().tolist(),
        "space_group_symbol": spg_symbol,
        "space_group_number": spg_number,
        "parent_id": molecule_id,
        "method": "Supercell Expansion"
    }

    index = _load_index(workspace_id)
    index[super_id] = meta
    _save_index(workspace_id, index)

    return {
        "ok": True,
        "results": {
            "original_molecule_id": molecule_id,
            "supercell_molecule_id": super_id,
            "formula": formula,
            "num_atoms": len(super_atoms),
            "lattice_parameters": {
                "a": float(super_atoms.cell.cellpar()[0]),
                "b": float(super_atoms.cell.cellpar()[1]),
                "c": float(super_atoms.cell.cellpar()[2]),
            },
            "space_group": {
                "symbol": spg_symbol,
                "number": spg_number
            }
        },
        "cif_block": cif_block,
        "xyz_block": xyz_block,
        "warnings": []
    }


def build_surface_slab_engine(
    workspace_id: str,
    bulk_molecule_id: str,
    miller_indices: List[int],
    layers: int,
    vacuum_size: float = 10.0
) -> Dict[str, Any]:
    """
    Build a surface slab from bulk periodic structure.
    """
    if len(miller_indices) != 3:
        raise ValueError("miller_indices must contain exactly 3 integers (h, k, l)")
    bulk_atoms = load_periodic_structure_engine(workspace_id, bulk_molecule_id)
    
    from ase.build import surface
    slab = surface(bulk_atoms, tuple(miller_indices), layers, vacuum=vacuum_size)
    
    slab_id = f"crystal_slab_{uuid.uuid4().hex[:8]}"
    formula = slab.get_chemical_formula()
    name = f"Slab {tuple(miller_indices)} ({layers} layers) of {bulk_molecule_id}"
    
    f_cif = io.BytesIO()
    write(f_cif, slab, format="cif")
    cif_block = f_cif.getvalue().decode("latin-1")
    
    f_xyz = io.StringIO()
    write(f_xyz, slab, format="xyz")
    xyz_block = f_xyz.getvalue()
    
    mol_dir = _get_molecules_dir(workspace_id)
    (mol_dir / f"{slab_id}.cif").write_text(cif_block, encoding="utf-8")
    (mol_dir / f"{slab_id}.xyz").write_text(xyz_block, encoding="utf-8")
    
    meta = {
        "molecule_id": slab_id,
        "name": name,
        "formula": formula,
        "num_atoms": len(slab),
        "is_periodic": True,
        "cell": slab.get_cell().tolist(),
        "parent_bulk_id": bulk_molecule_id,
        "miller_indices": miller_indices,
        "layers": layers,
        "vacuum_size": vacuum_size,
        "method": "Surface Slab Generation"
    }
    
    index = _load_index(workspace_id)
    index[slab_id] = meta
    _save_index(workspace_id, index)
    
    return {
        "ok": True,
        "results": {
            "original_bulk_molecule_id": bulk_molecule_id,
            "slab_molecule_id": slab_id,
            "formula": formula,
            "num_atoms": len(slab),
            "lattice_parameters": {
                "a": float(slab.cell.cellpar()[0]),
                "b": float(slab.cell.cellpar()[1]),
                "c": float(slab.cell.cellpar()[2]),
            }
        },
        "cif_block": cif_block,
        "xyz_block": xyz_block
    }


def add_adsorbate_to_surface_engine(
    workspace_id: str,
    slab_molecule_id: str,
    adsorbate_molecule_id: str,
    height: float = 1.5,
    position_type: str = "ontop"
) -> Dict[str, Any]:
    """
    Add adsorbate molecule onto a surface slab.
    """
    slab_atoms = load_periodic_structure_engine(workspace_id, slab_molecule_id)
    
    mol_dir = _get_molecules_dir(workspace_id)
    xyz_path = mol_dir / f"{adsorbate_molecule_id}.xyz"
    if not xyz_path.exists():
        raise FileNotFoundError(f"Adsorbate molecule {adsorbate_molecule_id} coordinates (.xyz) not found in workspace.")
        
    adsorbate_atoms = read(str(xyz_path), format="xyz")
    
    from ase.build import add_adsorbate
    pos = position_type.lower()
    if "," in pos:
        try:
            parts = [float(x) for x in pos.split(",")]
            pos_arg = (parts[0], parts[1])
        except Exception:
            pos_arg = "ontop"
    else:
        pos_arg = pos
        
    if isinstance(pos_arg, str) and pos_arg in ["ontop", "bridge", "hollow"]:
        z_coords = slab_atoms.positions[:, 2]
        top_idx = int(np.argmax(z_coords))
        x_atom = float(slab_atoms.positions[top_idx, 0])
        y_atom = float(slab_atoms.positions[top_idx, 1])
        
        if pos_arg == "ontop":
            pos_arg = (x_atom, y_atom)
        elif pos_arg == "bridge":
            pos_arg = (x_atom + 1.0, y_atom + 1.0)
        else:
            pos_arg = (x_atom + 1.5, y_atom + 1.5)
            
    add_adsorbate(slab_atoms, adsorbate_atoms, height=height, position=pos_arg)
    
    combined_id = f"crystal_ads_{uuid.uuid4().hex[:8]}"
    formula = slab_atoms.get_chemical_formula()
    name = f"Adsorbate {adsorbate_molecule_id} on {slab_molecule_id}"
    
    f_cif = io.BytesIO()
    write(f_cif, slab_atoms, format="cif")
    cif_block = f_cif.getvalue().decode("latin-1")
    
    f_xyz = io.StringIO()
    write(f_xyz, slab_atoms, format="xyz")
    xyz_block = f_xyz.getvalue()
    
    (mol_dir / f"{combined_id}.cif").write_text(cif_block, encoding="utf-8")
    (mol_dir / f"{combined_id}.xyz").write_text(xyz_block, encoding="utf-8")
    
    meta = {
        "molecule_id": combined_id,
        "name": name,
        "formula": formula,
        "num_atoms": len(slab_atoms),
        "is_periodic": True,
        "cell": slab_atoms.get_cell().tolist(),
        "slab_id": slab_molecule_id,
        "adsorbate_id": adsorbate_molecule_id,
        "height": height,
        "position_type": position_type,
        "method": "Adsorption Insertion"
    }
    
    index = _load_index(workspace_id)
    index[combined_id] = meta
    _save_index(workspace_id, index)
    
    return {
        "ok": True,
        "results": {
            "combined_molecule_id": combined_id,
            "formula": formula,
            "num_atoms": len(slab_atoms)
        },
        "cif_block": cif_block,
        "xyz_block": xyz_block
    }


def run_periodic_dft_engine(
    workspace_id: str,
    molecule_id: str,
    kpts: List[int] = [1, 1, 1],
    method: str = "xTB"
) -> Dict[str, Any]:
    """
    Run periodic DFT calculation (or GFN-xTB PBC calculation).

    Unit provenance: the xTB path uses ASE's XTB calculator, which reports
    energy natively in eV; the PySCF path solves the SCF natively in Hartree.
    `energy_ev` is always populated (converted from Hartree for the PySCF
    path). `energy_hartree` is only populated when the underlying method
    genuinely computed in atomic units (PySCF); for the xTB path it is left
    as None rather than presented as a native Hartree value, since it would
    otherwise just be a unit-converted copy of `energy_ev`.
    """
    atoms = load_periodic_structure_engine(workspace_id, molecule_id)

    method_upper = method.upper()
    energy_ev = 0.0
    energy_hartree_native = None
    method_used = ""

    if method_upper == "XTB":
        import shutil
        if shutil.which("xtb"):
            from ase.calculators.xtb import XTB
            atoms.calc = XTB(method="GFN2-xTB")
            energy_ev = float(atoms.get_potential_energy())
            method_used = "GFN2-xTB (periodic)"
        else:
            raise BackendUnavailableError(
                "xTB backend is not available for periodic calculations.",
                hint="Install the xtb binary and the xtb-python ASE calculator, or rerun with method='DFT'.",
            )
    else:
        try:
            from pyscf.pbc import gto, dft
        except ImportError as e:
            raise BackendUnavailableError(
                f"PySCF (with PBC support) is not installed: {str(e)}",
                hint="Install pyscf, or rerun with method='xTB'.",
            ) from e

        try:
            cell = gto.Cell()
            cell.atom = []
            for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                cell.atom.append([sym, pos])
            cell.a = atoms.get_cell().tolist()
            cell.basis = "gth-szv"
            cell.pseudo = "gth-pade"
            cell.build()

            if list(kpts) == [1, 1, 1]:
                mf = dft.RKS(cell)
            else:
                kpts_cell = cell.make_kpts(kpts)
                mf = dft.KRKS(cell, kpts_cell)

            mf.xc = 'lda'
            energy_hartree_native = float(mf.kernel())
            energy_ev = energy_hartree_native * 27.211386
            method_used = "PBC-DFT/LDA/gth-szv"
        except Exception as e:
            raise CalculationFailedError(
                f"Periodic DFT calculation failed: {str(e)}",
                hint="Try a smaller k-point grid, a minimal basis, or method='xTB'.",
            ) from e

    results = {
        "energy_ev": energy_ev,
        "energy_hartree": energy_hartree_native,
        "method": method,
        "method_used": method_used
    }

    if energy_hartree_native is not None:
        energy_note = f" ({energy_hartree_native:.6f} Hartree, native SCF units)."
    else:
        energy_note = " (Hartree not reported: xTB computes natively in eV via ASE, so a back-converted value would not reflect a native atomic-unit quantity)."

    interpretation = (
        f"Periodic calculation completed successfully using {method}.\n"
        f"Periodic Potential Energy = {energy_ev:.4f} eV{energy_note}"
    )

    return {
        "ok": True,
        "results": results,
        "interpretation": interpretation
    }
