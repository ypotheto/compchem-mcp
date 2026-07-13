import logging
from typing import Any

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

from ypotheto_compchem_mcp.chemistry.builder_engine import (
    save_molecule_coords,
)

logger = logging.getLogger(__name__)

def standardize_molecule_engine(
    workspace_id: str,
    smiles_or_sdf: str,
    strip_salts: bool = True,
    neutralize: bool = True,
    canonicalize_tautomer: bool = True,
    name: str | None = None
) -> dict[str, Any]:
    """
    Standardize a molecule: parses structure, strips salts, neutralizes charge,
    canonicalizes tautomers, and sanitizes the final molecule.
    Saves the standardized structure back to the workspace.
    """
    # 1. Parse input (SMILES or SDF block)
    if smiles_or_sdf.strip().startswith("$$$$") or "\n" in smiles_or_sdf:
        # Assume SDF block
        mol = Chem.MolFromMolBlock(smiles_or_sdf)
    else:
        # Assume SMILES
        mol = Chem.MolFromSmiles(smiles_or_sdf)

    if mol is None:
        raise ValueError("Failed to parse molecule input.")

    original_smiles = Chem.MolToSmiles(mol)
    steps_taken = []

    # 2. Strip Salts
    if strip_salts:
        remover = SaltRemover()
        stripped_mol = remover.StripMol(mol, dontRemoveEverything=True)
        if Chem.MolToSmiles(stripped_mol) != Chem.MolToSmiles(mol):
            mol = stripped_mol
            steps_taken.append("Stripped salts/counter-ions")

    # 3. Neutralize/Uncharge
    if neutralize:
        uncharger = rdMolStandardize.Uncharger()
        neutral_mol = uncharger.uncharge(mol)
        if Chem.MolToSmiles(neutral_mol) != Chem.MolToSmiles(mol):
            mol = neutral_mol
            steps_taken.append("Neutralized formal charges")

    # 4. Canonicalize Tautomer
    if canonicalize_tautomer:
        enumerator = rdMolStandardize.TautomerEnumerator()
        canon_mol = enumerator.Canonicalize(mol)
        if Chem.MolToSmiles(canon_mol) != Chem.MolToSmiles(mol):
            mol = canon_mol
            steps_taken.append("Standardized to canonical tautomer")

    # 5. Sanitize final structure
    Chem.SanitizeMol(mol)
    standardized_smiles = Chem.MolToSmiles(mol)
    
    # 6. Save back to workspace
    # Generate 3D coords for standardized parent
    mol_3d = Chem.AddHs(mol)
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    if AllChem.MMFFHasAllMoleculeParams(mol_3d):
        AllChem.MMFFOptimizeMolecule(mol_3d)
        method = "MMFF94"
    else:
        AllChem.UFFOptimizeMolecule(mol_3d)
        method = "UFF"

    # Save SVG/SDF/XYZ
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    
    mol_2d = Chem.Mol(mol)
    from rdkit.Chem import rdDepictor
    rdDepictor.Compute2DCoords(mol_2d)
    drawer = rdMolDraw2D.MolDraw2DSVG(350, 300)
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText()

    formula = CalcMolFormula(mol_3d)
    xyz_block = Chem.MolToXYZBlock(mol_3d)
    sdf_block = Chem.MolToMolBlock(mol_3d)

    import uuid
    molecule_id = f"mol_{uuid.uuid4().hex[:8]}"
    name_str = name or f"Standardized {formula}"

    meta = {
        "molecule_id": molecule_id,
        "name": name_str,
        "formula": formula,
        "smiles": standardized_smiles,
        "num_atoms": mol_3d.GetNumAtoms(),
        "is_standardized": True,
        "steps_taken": steps_taken,
        "method": f"Standardization ({method})"
    }
    save_molecule_coords(workspace_id, molecule_id, sdf_block, xyz_block, meta)

    return {
        "molecule_id": molecule_id,
        "name": name_str,
        "formula": formula,
        "original_smiles": original_smiles,
        "standardized_smiles": standardized_smiles,
        "steps_taken": steps_taken,
        "svg_data": svg_data
    }

def enumerate_tautomers_engine(
    workspace_id: str,
    molecule_id: str
) -> dict[str, Any]:
    """
    Enumerate all tautomeric forms for a stored molecule.
    """
    from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
    mol = load_molecule_from_workspace(workspace_id, molecule_id)
    
    # Tautomer enumeration requires hydrogen-depleted molecules
    mol = Chem.RemoveHs(mol)
    
    # RDKit Tautomer Enumerator
    enumerator = rdMolStandardize.TautomerEnumerator()
    tautomers = enumerator.Enumerate(mol)
    
    tautomer_list = []
    for i, t in enumerate(tautomers):
        # Clean and get SMILES representation
        Chem.SanitizeMol(t)
        smiles = Chem.MolToSmiles(t)
        tautomer_list.append({
            "index": i,
            "smiles": smiles,
            "formula": Chem.rdMolDescriptors.CalcMolFormula(t)
        })
        
    return {
        "molecule_id": molecule_id,
        "tautomers_count": len(tautomer_list),
        "tautomers": tautomer_list
    }
