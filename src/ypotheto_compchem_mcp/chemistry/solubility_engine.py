import logging
from typing import Any

import numpy as np
from rdkit import Chem

from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace

logger = logging.getLogger(__name__)

# Hoftyzer-Van Krevelen (HVK) Group Contributions for Hansen Solubility Parameters (HSP)
# Group values from: D.W. van Krevelen, "Properties of Polymers", 4th Edition.
# Values are:
# - SMARTS: substructure pattern
# - F_d: dispersion attraction constant (MJ^0.5 cm^1.5 / mol)
# - F_p: polar attraction constant (MJ^0.5 cm^1.5 / mol)
# - E_h: hydrogen bonding energy (J / mol)
# - V: molar volume (cm^3 / mol)
HVK_GROUPS = {
    # Alkanes / Aliphatic
    "CH3": ("[CH3;X4]", 420, 0, 0, 33.5),
    "CH2": ("[CH2;X4]", 270, 0, 0, 16.1),
    "CH": ("[CH1;X4]", 80, 0, 0, -1.0),
    "C": ("[C;X4;!H]", -70, 0, 0, -19.2),
    
    # Alkenes
    "double_bond_CH2": ("[CH2;X3]=[C,c]", 400, 0, 0, 28.5),
    "double_bond_CH": ("[CH1;X3]=[C,c]", 220, 0, 0, 13.5),
    "double_bond_C": ("[C;X3;!H]=[C,c]", 70, 0, 0, -5.5),
    
    # Aromatics
    "aromatic_CH": ("[cH]", 190, 170, 0, 16.0),
    "aromatic_C": ("[c;!H]", 190, 110, 0, 9.0),
    
    # Oxygen containing
    "hydroxyl_OH": ("[OX2H;$(O[C,c])]", 210, 500, 20000, 10.0),
    "ether_O": ("[OD2;$(O([C,c])[C,c])]", 100, 400, 3000, 3.8),
    "carbonyl_CO": ("[CX3;$(C(=O)([C,c])[C,c])]=O", 290, 770, 2000, 10.8),
    "carboxyl_COOH": ("[CX3;$(C(=O)[OX2H])](=[OX1])[OX2H]", 530, 420, 10000, 28.5),
    "ester_COO": ("[CX3;$(C(=O)[OX2H0][C,c])](=[OX1])[OX2H0]", 390, 490, 7000, 18.0),
    
    # Nitrogen containing
    "amine_NH2": ("[NX3H2;$(N[C,c])]", 280, 0, 8400, 19.2),
    "amine_NH": ("[NX3H1;$(N([C,c])[C,c])]", 160, 210, 3100, 4.5),
    "amine_N": ("[NX3H0;$(N([C,c])([C,c])[C,c])]", 20, 800, 5000, -9.0),
    "amide_CONH2": ("[CX3;$(C(=O)N)](=[OX1])[NX3H2]", 530, 540, 22400, 28.0),
    "amide_CONH": ("[CX3;$(C(=O)N)](=[OX1])[NX3H1]", 390, 490, 14000, 14.5),
    "nitrile_CN": ("[CX2;$(C#[NX1])]#[NX1]", 430, 1100, 5000, 24.0),
    
    # Halogens
    "fluorine_F": ("[F]", 80, 340, 0, 18.0),
    "chlorine_Cl": ("[Cl]", 450, 550, 400, 24.0),
    "bromine_Br": ("[Br]", 550, 480, 0, 30.0),
    "iodine_I": ("[I]", 660, 420, 0, 37.0),
}

# Match order: largest/most specific functional groups first to prevent sub-fragment overlap
HVK_ORDER = [
    "carboxyl_COOH", "ester_COO", "amide_CONH2", "amide_CONH",
    "carbonyl_CO", "hydroxyl_OH", "ether_O", "nitrile_CN",
    "amine_NH2", "amine_NH", "amine_N",
    "fluorine_F", "chlorine_Cl", "bromine_Br", "iodine_I",
    "aromatic_CH", "aromatic_C",
    "double_bond_CH2", "double_bond_CH", "double_bond_C",
    "CH3", "CH2", "CH", "C"
]

def calculate_hsp_engine(
    workspace_id: str,
    molecule_id: str
) -> dict[str, Any]:
    """
    Calculate Hansen Solubility Parameters (HSP) and Cohesive Energy Density (CED)
    using the Hoftyzer-Van Krevelen (HVK) group contribution method.
    """
    mol = load_molecule_from_workspace(workspace_id, molecule_id)
    
    # Work on hydrogen-depleted graph for substructure matches
    mol_depleted = Chem.RemoveHs(mol)
    
    assigned_atoms = set()
    group_counts = {}
    
    # Greedy substructure assignment to prevent double-counting
    for gname in HVK_ORDER:
        smarts, fd, fp, eh, v = HVK_GROUPS[gname]
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
            
        matches = mol_depleted.GetSubstructMatches(pattern)
        
        count = 0
        for match in matches:
            # Check if any atom in this match has already been claimed by a larger group
            if any(idx in assigned_atoms for idx in match):
                continue
                
            # Claim all atoms in the match
            for idx in match:
                assigned_atoms.add(idx)
            count += 1
            
        if count > 0:
            group_counts[gname] = count

    # Detect unassigned non-hydrogen atoms
    unassigned = []
    for atom in mol_depleted.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() > 1 and idx not in assigned_atoms:
            unassigned.append(f"{atom.GetSymbol()}(idx={idx})")

    # Sum contributions
    tot_fd = 0.0
    tot_fp2 = 0.0
    tot_eh = 0.0
    tot_v = 0.0
    
    for gname, count in group_counts.items():
        _, fd, fp, eh, v = HVK_GROUPS[gname]
        tot_fd += fd * count
        tot_fp2 += (fp ** 2) * count
        tot_eh += eh * count
        tot_v += v * count

    # Safety bounds check
    warnings = []
    if unassigned:
        warnings.append({
            "type": "HVK_UNASSIGNED_GROUPS",
            "message": f"Some atoms could not be mapped to HVK groups: {', '.join(unassigned)}. HSP values may be underestimated."
        })
        
    if tot_v <= 0.0:
        # Fallback: estimate volume using molecular mass and typical organic density (~1 g/cm^3)
        mw = Chem.rdMolDescriptors.CalcMW(mol_depleted)
        tot_v = mw / 1.0
        warnings.append({
            "type": "HVK_VOLUME_FALLBACK",
            "message": f"Calculated molar volume was non-positive. Used molecular weight fallback: V = {tot_v:.1f} cm^3/mol."
        })

    # Hansen parameters calculation
    delta_d = tot_fd / tot_v
    delta_p = np.sqrt(tot_fp2) / tot_v
    delta_h = np.sqrt(tot_eh / tot_v)
    delta_total = np.sqrt(delta_d**2 + delta_p**2 + delta_h**2)
    
    # Cohesive energy density (CED) in J/cm^3 (equivalent to MPa)
    # CED = delta_total^2
    ced = delta_total ** 2
    
    # Molar cohesive energy (E_coh) in J/mol
    e_coh = ced * tot_v

    return {
        "molecule_id": molecule_id,
        "molar_volume_cm3_mol": float(tot_v),
        "hansen_parameters": {
            "dispersion_delta_d": float(delta_d),
            "polar_delta_p": float(delta_p),
            "hydrogen_bonding_delta_h": float(delta_h),
            "total_solubility_delta": float(delta_total),
            "unit": "MPa^0.5"
        },
        "cohesive_energy": {
            "cohesive_energy_density_j_cm3": float(ced),
            "molar_cohesive_energy_j_mol": float(e_coh),
            "unit_ced": "J/cm^3 (or MPa)",
            "unit_molar": "J/mol"
        },
        "mapped_groups": group_counts,
        "unassigned_atoms": unassigned,
        "warnings": warnings
    }

def calculate_hsp_distance_engine(
    workspace_id: str,
    molecule_id_1: str,
    molecule_id_2: str
) -> dict[str, Any]:
    """
    Calculate the Hansen Solubility Parameter (HSP) distance (Ra) between two molecules.
    A smaller distance Ra suggests higher miscibility or solubility.
    """
    res1 = calculate_hsp_engine(workspace_id, molecule_id_1)
    res2 = calculate_hsp_engine(workspace_id, molecule_id_2)
    
    hsp1 = res1["hansen_parameters"]
    hsp2 = res2["hansen_parameters"]
    
    # Ra^2 = 4*(d_d1 - d_d2)^2 + (d_p1 - d_p2)^2 + (d_h1 - d_h2)^2
    d_d = hsp1["dispersion_delta_d"] - hsp2["dispersion_delta_d"]
    d_p = hsp1["polar_delta_p"] - hsp2["polar_delta_p"]
    d_h = hsp1["hydrogen_bonding_delta_h"] - hsp2["hydrogen_bonding_delta_h"]
    
    ra2 = 4 * (d_d ** 2) + (d_p ** 2) + (d_h ** 2)
    ra = np.sqrt(ra2)
    
    # Estimate relative energy difference classification
    if ra < 4.0:
        miscibility = "Highly Compatible"
    elif ra < 8.0:
        miscibility = "Moderately Compatible"
    else:
        miscibility = "Poorly Compatible / Insoluble"

    return {
        "ok": True,
        "results": {
            "molecule_id_1": molecule_id_1,
            "molecule_id_2": molecule_id_2,
            "hsp_1": hsp1,
            "hsp_2": hsp2,
            "hansen_distance_ra": float(ra),
            "miscibility_estimate": miscibility
        },
        "warnings": res1["warnings"] + res2["warnings"]
    }
