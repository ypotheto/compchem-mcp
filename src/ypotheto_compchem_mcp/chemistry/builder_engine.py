import json
import uuid
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ypotheto_compchem_mcp.workspace import get_workspace_id, workspace_manager


def get_molecules_dir(workspace_id: str) -> Path:
    """Get the molecules directory for the workspace."""
    path = workspace_manager.get_workspace_dir(workspace_id) / "molecules"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_index_file(workspace_id: str) -> Path:
    """Get the index file path for molecules."""
    return get_molecules_dir(workspace_id) / "index.json"

def load_molecule_index(workspace_id: str) -> dict[str, Any]:
    """Load the molecule index from storage, falling back to database if configured."""
    import logging

    from ypotheto_compchem_mcp.database import get_connection
    from ypotheto_compchem_mcp.storage import storage
    
    conn = get_connection()
    if conn is not None:
        index = {}
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT molecule_id, name, formula, smiles, num_atoms, method, metadata FROM compchem.molecules WHERE workspace_id = %s;",
                (workspace_id,)
            )
            for row in cur.fetchall():
                molecule_id, name, formula, smiles, num_atoms, method, metadata = row
                index[molecule_id] = {
                    "molecule_id": molecule_id,
                    "name": name,
                    "formula": formula,
                    "smiles": smiles,
                    "num_atoms": num_atoms,
                    "method": method,
                    **(metadata or {})
                }
            cur.close()
            conn.close()
            # Also keep local file cache synced in case local tools look at index.json
            index_file = _get_index_file(workspace_id)
            index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
            return index
        except Exception as e:
            logging.error(f"Failed to load index from PostgreSQL: {str(e)}", exc_info=True)
            
    # Fallback to local files / Spaces
    try:
        data = storage.read_file(workspace_id, "molecules/index.json")
        index = json.loads(data.decode("utf-8"))
        # Cache local copy
        index_file = _get_index_file(workspace_id)
        index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return index
    except FileNotFoundError:
        index_file = _get_index_file(workspace_id)
        if index_file.exists():
            try:
                index = json.loads(index_file.read_text(encoding="utf-8"))
                storage.write_file(workspace_id, "molecules/index.json", json.dumps(index).encode("utf-8"))
                return index
            except Exception:
                pass
        return {}

def save_molecule_index(workspace_id: str, index: dict[str, Any]):
    """Save the molecule index to disk and storage."""
    from ypotheto_compchem_mcp.storage import storage
    index_file = _get_index_file(workspace_id)
    index_text = json.dumps(index, indent=2)
    index_file.write_text(index_text, encoding="utf-8")
    storage.write_file(workspace_id, "molecules/index.json", index_text.encode("utf-8"))

    # Invalidate MoleculeStore's cached index (used by list_molecules/
    # describe_molecule) so a molecule saved just now - by this function or
    # any of its several callers (builder/conformer/mlff/periodic engines) -
    # is visible immediately rather than only after the cache's TTL expires.
    # Lazy import: molecules.py imports from this module at call time, so a
    # module-level import here would be circular.
    from ypotheto_compchem_mcp.molecules import molecule_store
    molecule_store.invalidate(workspace_id)

def save_molecule_coords(workspace_id: str, molecule_id: str, sdf_block: str, xyz_block: str, meta: dict[str, Any]):
    """Helper to save molecule coordinates and update the index."""
    import logging

    from ypotheto_compchem_mcp.database import get_connection
    from ypotheto_compchem_mcp.storage import storage
    mol_dir = get_molecules_dir(workspace_id)
    
    # Save SDF and XYZ files locally
    (mol_dir / f"{molecule_id}.sdf").write_text(sdf_block, encoding="utf-8")
    (mol_dir / f"{molecule_id}.xyz").write_text(xyz_block, encoding="utf-8")
    
    # Upload to storage
    storage.write_file(workspace_id, f"molecules/{molecule_id}.sdf", sdf_block.encode("utf-8"))
    storage.write_file(workspace_id, f"molecules/{molecule_id}.xyz", xyz_block.encode("utf-8"))
    
    # Save to PostgreSQL
    conn = get_connection()
    if conn is not None:
        cur = None
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO compchem.molecules (molecule_id, workspace_id, name, formula, smiles, num_atoms, method, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (molecule_id) DO UPDATE 
                SET name = EXCLUDED.name, formula = EXCLUDED.formula, smiles = EXCLUDED.smiles, 
                    num_atoms = EXCLUDED.num_atoms, method = EXCLUDED.method, metadata = EXCLUDED.metadata;
                """,
                (
                    molecule_id,
                    workspace_id,
                    meta.get("name", ""),
                    meta.get("formula", ""),
                    meta.get("smiles", ""),
                    meta.get("num_atoms", 0),
                    meta.get("method", ""),
                    json.dumps({
                        k: v for k, v in meta.items()
                        if k not in ["molecule_id", "workspace_id", "name", "formula", "smiles", "num_atoms", "method"]
                    })
                )
            )
            conn.commit()
        except Exception as e:
            logging.error(f"Failed to save molecule to PostgreSQL: {str(e)}", exc_info=True)
        finally:
            if cur is not None:
                try:
                    cur.close()
                except Exception:
                    pass
            try:
                conn.close()
            except Exception:
                pass
            
    # Update index
    index = load_molecule_index(workspace_id)
    index[molecule_id] = meta
    save_molecule_index(workspace_id, index)

def get_molecule_path(workspace_id: str, molecule_id: str, fmt: str = "sdf") -> Path:
    """Get the file path of a saved molecule, syncing from storage if needed."""
    from ypotheto_compchem_mcp.storage import storage
    mol_dir = get_molecules_dir(workspace_id)
    filepath = mol_dir / f"{molecule_id}.{fmt}"
    if not filepath.exists():
        try:
            data = storage.read_file(workspace_id, f"molecules/{molecule_id}.{fmt}")
            filepath.write_bytes(data)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Molecule {molecule_id} coordinates not found in workspace or storage.") from e
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
    name: str | None = None
) -> dict[str, Any]:
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
