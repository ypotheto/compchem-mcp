from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace


def calculate_descriptors_engine(workspace_id: str, molecule_id: str) -> dict[str, Any]:
    """
    Calculate cheminformatics molecular descriptors and filters for a stored molecule.
    """
    # Load molecule
    mol = load_molecule_from_workspace(workspace_id, molecule_id)
    
    # Create a hydrogen-depleted copy for correct SMARTS-based descriptor matches
    mol_no_hs = Chem.RemoveHs(mol)
    
    # Compute base RDKit descriptors on the hydrogen-depleted molecule
    mw = float(rdMolDescriptors.CalcExactMolWt(mol_no_hs))
    logp = float(Crippen.MolLogP(mol_no_hs))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol_no_hs))
    hbd = int(rdMolDescriptors.CalcNumLipinskiHBD(mol_no_hs))
    hba = int(rdMolDescriptors.CalcNumLipinskiHBA(mol_no_hs))
    rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol_no_hs))
    
    # Evaluate Lipinski's Rule of Five violations
    violations = 0
    lipinski_rules = []
    
    # Rule 1: MW <= 500
    mw_ok = mw <= 500.0
    if not mw_ok:
        violations += 1
    lipinski_rules.append({"rule": "Molecular Weight <= 500", "value": f"{mw:.2f}", "status": "pass" if mw_ok else "fail"})
    
    # Rule 2: LogP <= 5
    logp_ok = logp <= 5.0
    if not logp_ok:
        violations += 1
    lipinski_rules.append({"rule": "LogP (Lipophilicity) <= 5", "value": f"{logp:.2f}", "status": "pass" if logp_ok else "fail"})
    
    # Rule 3: HBD <= 5
    hbd_ok = hbd <= 5
    if not hbd_ok:
        violations += 1
    lipinski_rules.append({"rule": "Hydrogen Bond Donors <= 5", "value": str(hbd), "status": "pass" if hbd_ok else "fail"})
    
    # Rule 4: HBA <= 10
    hba_ok = hba <= 10
    if not hba_ok:
        violations += 1
    lipinski_rules.append({"rule": "Hydrogen Bond Acceptors <= 10", "value": str(hba), "status": "pass" if hba_ok else "fail"})
    
    # Passes if no more than 1 violation
    passes_lipinski = violations <= 1
    
    return {
        "molecule_id": molecule_id,
        "descriptors": {
            "molecular_weight": mw,
            "logp": logp,
            "tpsa": tpsa,
            "hydrogen_bond_donors": hbd,
            "hydrogen_bond_acceptors": hba,
            "rotatable_bonds": rot_bonds
        },
        "lipinski_filter": {
            "passes": passes_lipinski,
            "violations_count": violations,
            "details": lipinski_rules
        }
    }
