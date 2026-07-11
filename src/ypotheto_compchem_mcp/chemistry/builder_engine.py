import json
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ypotheto_compchem_mcp.workspace import workspace_manager, get_workspace_id

def _get_molecules_dir(workspace_id: str) -> Path:
    """Get the molecules directory for the workspace."""
    path = workspace_manager.get_workspace_dir(workspace_id) / "molecules"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_index_file(workspace_id: str) -> Path:
    """Get the index file path for molecules."""
    return _get_molecules_dir(workspace_id) / "index.json"

def _load_index(workspace_id: str) -> Dict[str, Any]:
    """Load the molecule index from disk."""
    index_file = _get_index_file(workspace_id)
    if not index_file.exists():
        return {}
    try:
        return json.loads(index_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_index(workspace_id: str, index: Dict[str, Any]):
    """Save the molecule index to disk."""
    index_file = _get_index_file(workspace_id)
    index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")

def save_molecule_coords(workspace_id: str, molecule_id: str, sdf_block: str, xyz_block: str, meta: Dict[str, Any]):
    """Helper to save molecule coordinates and update the index."""
    mol_dir = _get_molecules_dir(workspace_id)
    
    # Save SDF and XYZ files
    (mol_dir / f"{molecule_id}.sdf").write_text(sdf_block, encoding="utf-8")
    (mol_dir / f"{molecule_id}.xyz").write_text(xyz_block, encoding="utf-8")
    
    # Update index
    index = _load_index(workspace_id)
    index[molecule_id] = meta
    _save_index(workspace_id, index)

def get_molecule_path(workspace_id: str, molecule_id: str, fmt: str = "sdf") -> Path:
    """Get the file path of a saved molecule."""
    mol_dir = _get_molecules_dir(workspace_id)
    filepath = mol_dir / f"{molecule_id}.{fmt}"
    if not filepath.exists():
        raise FileNotFoundError(f"Molecule {molecule_id} coordinates not found in workspace.")
    return filepath

def load_molecule_from_workspace(workspace_id: str, molecule_id: str) -> Chem.Mol:
    """Load an RDKit Mol object from the workspace SDF file."""
    path = get_molecule_path(workspace_id, molecule_id, "sdf")
    mol = Chem.MolFromMolBlock(path.read_text(encoding="utf-8"), removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to parse stored coordinates for molecule {molecule_id}.")
    return mol

def build_molecule_from_smiles_engine(
    smiles: str,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a SMILES string, generate 3D coordinates using force-field minimization,
    renders 2D SVG layout, saves to workspace molecules, and returns structure data.
    """
    # 1. Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smiles}'")
        
    # Add Hydrogens for realistic 3D coordinates
    mol_3d = Chem.AddHs(mol)
    
    # 2. Embed 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    embed_ok = AllChem.EmbedMolecule(mol_3d, params)
    if embed_ok < 0:
        # Fallback to random coordinates if ETKDGv3 fails
        AllChem.EmbedMolecule(mol_3d, randomCoords=True)
        
    # 3. Minimize using force field
    # Default to MMFF94, fallback to UFF
    if AllChem.MMFFHasAllMoleculeParams(mol_3d):
        AllChem.MMFFOptimizeMolecule(mol_3d)
        method_used = "MMFF94"
    else:
        AllChem.UFFOptimizeMolecule(mol_3d)
        method_used = "UFF"
        
    # 4. Generate 2D SVG
    mol_2d = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol_2d)
    drawer = rdMolDraw2D.MolDraw2DSVG(350, 300)
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText().encode("utf-8")
    
    # 5. Extract formula and generate blocks
    formula = CalcMolFormula(mol_3d)
    xyz_block = Chem.MolToXYZBlock(mol_3d)
    sdf_block = Chem.MolToMolBlock(mol_3d)
    
    molecule_id = f"mol_{uuid.uuid4().hex[:8]}"
    name_str = name or formula
    
    # Save to disk
    workspace_id = get_workspace_id()
    meta = {
        "molecule_id": molecule_id,
        "name": name_str,
        "formula": formula,
        "smiles": smiles,
        "num_atoms": mol_3d.GetNumAtoms(),
        "method": method_used
    }
    save_molecule_coords(workspace_id, molecule_id, sdf_block, xyz_block, meta)
    
    return {
        "molecule_id": molecule_id,
        "name": name_str,
        "formula": formula,
        "smiles": smiles,
        "num_atoms": mol_3d.GetNumAtoms(),
        "method": method_used,
        "svg_data": svg_data,
        "xyz_block": xyz_block,
        "sdf_block": sdf_block
    }
