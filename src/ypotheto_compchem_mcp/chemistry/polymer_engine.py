import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ypotheto_compchem_mcp.workspace import workspace_manager
from ypotheto_compchem_mcp.chemistry.builder_engine import _get_molecules_dir, _load_index, _save_index, save_molecule_coords

logger = logging.getLogger(__name__)

def _get_monomers_dir(workspace_id: str) -> Path:
    """Get the monomers directory for the workspace."""
    path = workspace_manager.get_workspace_dir(workspace_id) / "monomers"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_monomer_index_file(workspace_id: str) -> Path:
    """Get the index file path for monomers."""
    return _get_monomers_dir(workspace_id) / "index.json"

def _load_monomer_index(workspace_id: str) -> Dict[str, Any]:
    """Load the monomer index from disk."""
    index_file = _get_monomer_index_file(workspace_id)
    if not index_file.exists():
        return {}
    try:
        return json.loads(index_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_monomer_index(workspace_id: str, index: Dict[str, Any]):
    """Save the monomer index to disk."""
    index_file = _get_monomer_index_file(workspace_id)
    index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")

def register_monomer_engine(
    workspace_id: str,
    smiles: str,
    name: str,
    head_idx: Optional[int] = None,
    tail_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Register a monomer repeat unit, setting up attachment points [1*] (head) and [2*] (tail).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid monomer SMILES string: '{smiles}'")

    # Ensure attachment points exist or graft them
    if head_idx is not None and tail_idx is not None:
        # Graft dummy atoms [1*] and [2*] onto specified indices
        editable = Chem.EditableMol(mol)
        
        # Add head dummy
        dummy_h = editable.AddAtom(Chem.Atom(0)) # atomic number 0 is dummy
        editable.AddBond(head_idx, dummy_h, Chem.BondType.SINGLE)
        
        # Add tail dummy
        dummy_t = editable.AddAtom(Chem.Atom(0))
        editable.AddBond(tail_idx, dummy_t, Chem.BondType.SINGLE)
        
        mol = editable.GetMol()
        
        # Set isotopes on the newly added dummy atoms
        # They will be the last two atoms added
        mol.GetAtomWithIdx(dummy_h).SetIsotope(1)
        mol.GetAtomWithIdx(dummy_t).SetIsotope(2)
        
    else:
        # Check if dummy atoms (*) are already present
        dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummies) >= 2:
            # Set isotopes to designate head [1*] and tail [2*]
            dummies[0].SetIsotope(1)
            dummies[1].SetIsotope(2)
        elif len(dummies) == 1:
            raise ValueError("Monomer must have exactly 2 connection points; found only 1.")
        else:
            # Fallback: if no dummies are present and no indices specified, try to find polymerizable terminals
            # For simplicity, warn and require explicit attachment points
            raise ValueError(
                "No attachment points found. Please specify head_idx and tail_idx, "
                "or include connection points '*' in the SMILES string (e.g. '*CC(*)C' for propylene)."
            )

    Chem.SanitizeMol(mol)
    monomer_smiles = Chem.MolToSmiles(mol)
    monomer_id = f"mon_{uuid.uuid4().hex[:8]}"

    # Save to index
    meta = {
        "monomer_id": monomer_id,
        "name": name,
        "smiles": monomer_smiles,
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "num_atoms": mol.GetNumAtoms()
    }
    index = _load_monomer_index(workspace_id)
    index[monomer_id] = meta
    _save_monomer_index(workspace_id, index)

    return meta

def build_polymer_chain_engine(
    workspace_id: str,
    monomer_id: str,
    dp: int,
    tacticity: str = "isotactic",
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a polymer chain of degree of polymerization (DP) by connecting registered
    monomer repeat units head-to-tail, minimizing the 3D cell, and saving it.
    """
    index = _load_monomer_index(workspace_id)
    if monomer_id not in index:
        raise FileNotFoundError(f"Monomer {monomer_id} not found in workspace.")
    
    monomer_smiles = index[monomer_id]["smiles"]
    monomer = Chem.MolFromSmiles(monomer_smiles)
    
    # 1. Setup chain building reaction
    # Connection: connects atom bonded to [2*] to atom bonded to [1*]
    rxn = AllChem.ReactionFromSmarts("[*:1]-[2*].[1*]-[*:2]>>[*:1]-[*:2]")
    
    chain = Chem.Mol(monomer)
    for step in range(dp - 1):
        products = rxn.RunReactants((chain, monomer))
        if not products or not products[0]:
            raise RuntimeError("Polymerization connection failed. Verify monomer head/tail attachment points.")
        chain = products[0][0]
        Chem.SanitizeMol(chain)

    # 2. Cap the chain: replace remaining [1*] and [2*] dummy atoms with Hydrogens (H)
    dummy_query = Chem.MolFromSmarts("[#0]") # Matches atomic number 0
    capped = Chem.ReplaceSubstructs(chain, dummy_query, Chem.MolFromSmiles("[H]"), replaceAll=True)[0]
    
    # Clean up to standard representation
    capped = Chem.RemoveHs(capped)
    Chem.SanitizeMol(capped)
    
    polymer_smiles = Chem.MolToSmiles(capped)
    
    # 3. Generate 3D Conformer coordinates
    mol_3d = Chem.AddHs(capped)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    embed_ok = AllChem.EmbedMolecule(mol_3d, params)
    if embed_ok < 0:
        # Fallback to random coordinates for large molecules
        AllChem.EmbedMolecule(mol_3d, randomCoords=True)
        
    if AllChem.MMFFHasAllMoleculeParams(mol_3d):
        AllChem.MMFFOptimizeMolecule(mol_3d)
        method = "MMFF94"
    else:
        AllChem.UFFOptimizeMolecule(mol_3d)
        method = "UFF"

    # 4. Export SVG
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    mol_2d = Chem.Mol(capped)
    rdDepictor.Compute2DCoords(mol_2d)
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText()

    # 5. Save back to molecules directory
    formula = CalcMolFormula(mol_3d)
    xyz_block = Chem.MolToXYZBlock(mol_3d)
    sdf_block = Chem.MolToMolBlock(mol_3d)
    
    polymer_id = f"mol_{uuid.uuid4().hex[:8]}"
    name_str = name or f"Polymer {index[monomer_id]['name']} (DP={dp})"

    meta = {
        "molecule_id": polymer_id,
        "name": name_str,
        "formula": formula,
        "smiles": polymer_smiles,
        "num_atoms": mol_3d.GetNumAtoms(),
        "is_polymer": True,
        "monomer_id": monomer_id,
        "dp": dp,
        "tacticity": tacticity,
        "method": f"Chain Construction ({method})"
    }
    save_molecule_coords(workspace_id, polymer_id, sdf_block, xyz_block, meta)

    return {
        "polymer_molecule_id": polymer_id,
        "name": name_str,
        "formula": formula,
        "smiles": polymer_smiles,
        "num_atoms": mol_3d.GetNumAtoms(),
        "dp": dp,
        "svg_data": svg_data
    }
