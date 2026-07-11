import logging
import uuid
from typing import Any, Dict, List, Optional
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ypotheto_compchem_mcp.workspace import workspace_manager
from ypotheto_compchem_mcp.chemistry.builder_engine import (
    load_molecule_from_workspace,
    save_molecule_coords,
    _get_molecules_dir,
    _load_index,
    _save_index
)

logger = logging.getLogger(__name__)

def search_conformers_engine(
    workspace_id: str,
    molecule_id: str,
    num_conformers: int = 50,
    rmsd_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Generate multiple conformers for a stored molecule, minimize them using forcefields,
    remove optimized duplicates, and return a ranked list with Boltzmann populations.
    Saves the multi-conformer SDF in the workspace.
    """
    mol = load_molecule_from_workspace(workspace_id, molecule_id)
    
    # Ensure molecule has hydrogens
    mol_3d = Chem.AddHs(mol)
    
    # 1. Embed multiple conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = rmsd_threshold
    
    conf_ids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_conformers, params=params)
    if len(conf_ids) == 0:
        conf_ids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_conformers, randomCoords=True, pruneRmsThresh=rmsd_threshold)
        
    if len(conf_ids) == 0:
        raise RuntimeError("Failed to embed conformers for this molecule.")

    # 2. Minimize and calculate energy of each conformer
    has_mmff = AllChem.MMFFHasAllMoleculeParams(mol_3d)
    confs_energies = []
    
    for cid in conf_ids:
        try:
            if has_mmff:
                ff = AllChem.MMFFGetMoleculeForceField(mol_3d, AllChem.MMFFGetMoleculeProperties(mol_3d), confId=cid)
                ff.Minimize(maxIts=500)
                energy = ff.CalcEnergy()
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol_3d, confId=cid)
                ff.Minimize(maxIts=500)
                energy = ff.CalcEnergy()
            confs_energies.append((cid, energy))
        except Exception as e:
            logger.warning(f"Failed to minimize conformer {cid}: {str(e)}")

    if not confs_energies:
        raise RuntimeError("Failed to minimize any of the embedded conformers.")

    # 3. Sort by energy
    confs_energies.sort(key=lambda x: x[1])
    
    # 4. Prune duplicate conformers after optimization
    unique_confs = []
    for cid, energy in confs_energies:
        is_duplicate = False
        for u_cid, u_energy in unique_confs:
            rmsd = AllChem.GetBestRMS(mol_3d, mol_3d, cid, u_cid)
            if rmsd < rmsd_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_confs.append((cid, energy))

    # 5. Compute Boltzmann populations (T = 298.15 K, RT = 0.59218 kcal/mol)
    lowest_energy = unique_confs[0][1]
    RT = 0.592183  # kcal/mol at 298.15 K
    
    exps = []
    for cid, energy in unique_confs:
        de = energy - lowest_energy
        exps.append(np.exp(-de / RT))
    Z = sum(exps)
    
    ranked_conformers = []
    for idx, (cid, energy) in enumerate(unique_confs):
        de = energy - lowest_energy
        pop = exps[idx] / Z
        ranked_conformers.append({
            "conformer_index": idx,
            "rdkit_conformer_id": int(cid),
            "energy_kcal_mol": float(energy),
            "relative_energy_kcal_mol": float(de),
            "boltzmann_population": float(pop)
        })

    # 6. Save the multi-conformer molecule block in workspace
    mol_dir = _get_molecules_dir(workspace_id)
    multi_sdf_path = mol_dir / f"{molecule_id}_conformers.sdf"
    
    # Write all unique conformers to a single SDF file
    writer = Chem.SDWriter(str(multi_sdf_path))
    for cid, _ in unique_confs:
        # Set property for identification
        mol_3d.SetProp("_ConformerID", str(cid))
        writer.write(mol_3d, confId=cid)
    writer.close()

    return {
        "molecule_id": molecule_id,
        "conformers_found": len(ranked_conformers),
        "lowest_energy_kcal_mol": float(lowest_energy),
        "forcefield_used": "MMFF94" if has_mmff else "UFF",
        "conformers": ranked_conformers,
        "multi_conformer_sdf_path": str(multi_sdf_path)
    }

def save_conformer_as_molecule_engine(
    workspace_id: str,
    parent_molecule_id: str,
    rdkit_conformer_id: int,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract a single conformer from the multi-conformer SDF and save it as a new
    standalone molecule in the workspace.
    """
    mol_dir = _get_molecules_dir(workspace_id)
    multi_sdf_path = mol_dir / f"{parent_molecule_id}_conformers.sdf"
    if not multi_sdf_path.exists():
        raise FileNotFoundError(f"Multi-conformer file for molecule {parent_molecule_id} not found.")

    # Read the SDF and find the matching conformer
    suppl = Chem.SDMolSupplier(str(multi_sdf_path), removeHs=False)
    target_mol = None
    for mol in suppl:
        if mol is not None and mol.HasProp("_ConformerID"):
            if int(mol.GetProp("_ConformerID")) == rdkit_conformer_id:
                target_mol = mol
                break

    if target_mol is None:
        raise ValueError(f"Conformer ID {rdkit_conformer_id} not found in multi-conformer file.")

    # Save as new molecule
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    
    # Generate 2D depict
    mol_2d = Chem.Mol(target_mol)
    rdDepictor.Compute2DCoords(mol_2d)
    drawer = rdMolDraw2D.MolDraw2DSVG(350, 300)
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText()

    formula = CalcMolFormula(target_mol)
    xyz_block = Chem.MolToXYZBlock(target_mol)
    sdf_block = Chem.MolToMolBlock(target_mol)

    new_molecule_id = f"mol_{uuid.uuid4().hex[:8]}"
    name_str = name or f"Conformer {rdkit_conformer_id} of {parent_molecule_id}"

    meta = {
        "molecule_id": new_molecule_id,
        "name": name_str,
        "formula": formula,
        "num_atoms": target_mol.GetNumAtoms(),
        "parent_id": parent_molecule_id,
        "parent_conformer_id": rdkit_conformer_id,
        "method": "Conformer Extraction"
    }
    save_molecule_coords(workspace_id, new_molecule_id, sdf_block, xyz_block, meta)

    return {
        "molecule_id": new_molecule_id,
        "name": name_str,
        "formula": formula,
        "num_atoms": target_mol.GetNumAtoms(),
        "svg_data": svg_data
    }
