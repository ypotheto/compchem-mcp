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
