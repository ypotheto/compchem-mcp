import logging
import math
from typing import Any, Dict, List, Optional
from rdkit import Chem
from ase import Atoms
from ase.thermochemistry import IdealGasThermo

from ypotheto_compchem_mcp.chemistry.builder_engine import (
    load_molecule_from_workspace,
    save_molecule_coords
)
from ypotheto_compchem_mcp.chemistry.xtb_engine import (
    run_conformer_search_engine,
    run_xtb_calculation_engine,
    XTB_AVAILABLE,
    CREST_AVAILABLE
)

logger = logging.getLogger(__name__)

# Constants
KB = 8.617333262145e-5  # eV/K
HARTREE_TO_EV = 27.211386245988
EV_TO_KCAL = 23.0609

def compute_gibbs_correction(
    atoms: Atoms,
    frequencies_cm1: List[float],
    spin: int = 1,
    T: float = 298.15,
    P: float = 101325.0
) -> Dict[str, float]:
    """
    Calculate zero-point energy and Gibbs free energy thermal correction using ASE IdealGasThermo.
    """
    # Convert frequencies to vibrational energies in eV (1 cm^-1 = 0.00012398 eV)
    vib_energies = [f * 1.23984193e-4 for f in frequencies_cm1 if f > 10.0]
    
    # Calculate ZPE: ZPE = 0.5 * sum(h * nu)
    zpe = 0.5 * sum(vib_energies)
    
    # Check linear geometry via moments of inertia
    moments = atoms.get_moments_of_inertia()
    is_linear = moments[0] < 1e-4
    geom = "linear" if is_linear else "nonlinear"
    
    # Setup IdealGasThermo
    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        geometry=geom,
        atoms=atoms,
        symmetrynumber=1,
        spin=(spin - 1) / 2.0
    )
    
    enthalpy_corr = float(thermo.get_enthalpy(temperature=T))
    entropy_corr = float(thermo.get_entropy(temperature=T, pressure=P))
    gibbs_corr = float(thermo.get_gibbs_energy(temperature=T, pressure=P))
    
    return {
        "zpe_ev": zpe,
        "enthalpy_corr_ev": enthalpy_corr,
        "entropy_corr_ev_k": entropy_corr,
        "gibbs_corr_ev": gibbs_corr
    }

def run_ensemble_thermochemistry_engine(
    workspace_id: str,
    molecule_id: str,
    method: str = "GFN2-xTB",
    solvent: Optional[str] = None,
    energy_window_kcal: float = 6.0,
    max_conformers_to_optimize: int = 5,
    energy_threshold_kcal: float = 3.0,
    charge: int = 0,
    spin: int = 1
) -> Dict[str, Any]:
    """
    Ensemble Thermochemistry Pipeline:
    1. Generate conformers using CREST.
    2. Filter lowest-energy conformers.
    3. Run geometry optimization and vibrational frequency calculations on each.
    4. Compute Boltzmann populations and ensemble-averaged free energies.
    """
    if not CREST_AVAILABLE or not XTB_AVAILABLE:
        raise RuntimeError("CREST and xTB binaries are required to run the ensemble thermochemistry pipeline.")
        
    # Step 1: Run CREST conformer search
    logger.info(f"Starting CREST conformer search for {molecule_id}")
    crest_res = run_conformer_search_engine(
        workspace_id,
        molecule_id,
        method=method,
        solvent=solvent,
        energy_window_kcal=energy_window_kcal
    )
    
    if not crest_res["ok"]:
        return crest_res
        
    conformers = crest_res["results"]["conformers"]
    if not conformers:
        return {
            "ok": False,
            "error": {
                "code": "NO_CONFORMERS_FOUND",
                "message": "CREST conformer search returned no structures."
            }
        }
        
    # Sort conformers by relative energy
    conformers = sorted(conformers, key=lambda c: c["relative_energy_kcal"])
    
    # Filter conformers to optimize based on max_conformers_to_optimize and energy_threshold_kcal
    filtered_conformers = []
    for idx, c in enumerate(conformers):
        if idx >= max_conformers_to_optimize:
            break
        if c["relative_energy_kcal"] > energy_threshold_kcal:
            break
        filtered_conformers.append(c)
        
    logger.info(f"Optimizing top {len(filtered_conformers)} conformers (out of {len(conformers)})")
    
    refined_results = []
    
    # Step 2 & 3: Run optimization and frequency calculations on selected conformers
    for i, c in enumerate(filtered_conformers):
        conf_mol = Chem.MolFromXYZBlock(c["xyz_block"])
        if not conf_mol:
            continue
            
        # Register a temporary molecule in workspace to run calculations on
        conf_mol_id = f"{molecule_id}_conf_ref_{i}"
        
        # Safely compute formula from atom symbols to avoid implicit valence errors
        from collections import Counter
        symbols = [atom.GetSymbol() for atom in conf_mol.GetAtoms()]
        counts = Counter(symbols)
        formula = "".join(f"{el}{count if count > 1 else ''}" for el, count in sorted(counts.items()))
        
        # Save temporary conformer coordinates
        save_molecule_coords(
            workspace_id,
            conf_mol_id,
            Chem.MolToMolBlock(conf_mol),
            c["xyz_block"],
            {"formula": formula, "num_atoms": conf_mol.GetNumAtoms()}
        )
        
        # Geometry Optimization
        opt_res = run_xtb_calculation_engine(
            workspace_id,
            conf_mol_id,
            task="geometry_optimization",
            method=method,
            solvent=solvent,
            charge=charge,
            spin=spin
        )
        
        if not opt_res["ok"]:
            logger.warning(f"Optimization failed for conformer {i}: {opt_res.get('error')}")
            continue
            
        # Get optimized energy
        opt_energy_ev = opt_res["results"]["energy_ev"]
        
        # Vibrations / Hess calculation
        vib_res = run_xtb_calculation_engine(
            workspace_id,
            conf_mol_id,
            task="vibrations",
            method=method,
            solvent=solvent,
            charge=charge,
            spin=spin
        )
        
        if not vib_res["ok"]:
            logger.warning(f"Vibration check failed for conformer {i}: {vib_res.get('error')}")
            continue
            
        freqs = vib_res["results"]["frequencies_cm1"]
        
        # Convert RDKit structure to ASE Atoms for thermochemistry properties
        symbols = [atom.GetSymbol() for atom in conf_mol.GetAtoms()]
        opt_mol_loaded = load_molecule_from_workspace(workspace_id, conf_mol_id)
        positions = opt_mol_loaded.GetConformer().GetPositions()
        atoms = Atoms(symbols=symbols, positions=positions)
        
        # Calculate Gibbs correction
        thermo_corr = compute_gibbs_correction(atoms, freqs, spin=spin)
        
        # Total Gibbs energy = Electronic Energy + Gibbs Thermal Correction
        gibbs_energy_ev = opt_energy_ev + thermo_corr["gibbs_corr_ev"]
        
        refined_results.append({
            "conformer_index": i,
            "electronic_energy_ev": opt_energy_ev,
            "zpe_ev": thermo_corr["zpe_ev"],
            "gibbs_correction_ev": thermo_corr["gibbs_corr_ev"],
            "total_gibbs_energy_ev": gibbs_energy_ev,
            "imaginary_frequencies_count": sum(1 for f in freqs if f < 0.0),
            "frequencies_cm1": freqs
        })
        
    if not refined_results:
        return {
            "ok": False,
            "error": {
                "code": "REFINEMENT_FAILED",
                "message": "Failed to optimize and run frequency checks on any conformers."
            }
        }
        
    # Step 4: Calculate Boltzmann weighting on the refined Gibbs energies (T = 298.15 K)
    T = 298.15
    kbT_ev = KB * T  # approx 0.02568 eV
    
    min_gibbs = min(r["total_gibbs_energy_ev"] for r in refined_results)
    
    total_q = 0.0
    for r in refined_results:
        delta_g = r["total_gibbs_energy_ev"] - min_gibbs
        r["relative_gibbs_ev"] = delta_g
        r["relative_gibbs_kcal"] = round(delta_g * EV_TO_KCAL, 4)
        weight = math.exp(-delta_g / kbT_ev)
        r["boltzmann_weight"] = weight
        total_q += weight
        
    # Boltzmann population and ensemble totals
    ensemble_gibbs_ev = 0.0
    for r in refined_results:
        r["boltzmann_population"] = round(r["boltzmann_weight"] / total_q, 4)
        ensemble_gibbs_ev += r["boltzmann_population"] * r["total_gibbs_energy_ev"]
        del r["boltzmann_weight"]
        
    results = {
        "molecule_id": molecule_id,
        "temperature_k": T,
        "ensemble_gibbs_free_energy_ev": round(ensemble_gibbs_ev, 6),
        "ensemble_gibbs_free_energy_kcal": round(ensemble_gibbs_ev * EV_TO_KCAL, 4),
        "refined_conformers": refined_results
    }
    
    interpretation = (
        f"Ensemble thermochemistry pipeline completed for {molecule_id}.\n"
        f"Ensemble Gibbs Free Energy = {results['ensemble_gibbs_free_energy_kcal']} kcal/mol.\n"
        f"Refined {len(refined_results)} conformers.\n"
        f"Lowest energy conformer relative Gibbs = 0.0 kcal/mol (Boltzmann pop: {refined_results[0]['boltzmann_population'] * 100:.1f}%)."
    )
    
    return {
        "ok": True,
        "results": results,
        "interpretation": interpretation
    }
