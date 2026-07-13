import logging
import math
from typing import Any

from ase import Atoms
from ase.thermochemistry import IdealGasThermo
from rdkit import Chem

from ypotheto_compchem_mcp.chemistry.builder_engine import (
    load_molecule_from_workspace,
    save_molecule_coords,
)
from ypotheto_compchem_mcp.chemistry.xtb_engine import (
    CREST_AVAILABLE,
    XTB_AVAILABLE,
    run_conformer_search_engine,
    run_xtb_calculation_engine,
)
from ypotheto_compchem_mcp.errors import (
    BackendUnavailableError,
    CalculationFailedError,
    CompchemError,
)

logger = logging.getLogger(__name__)

# Constants
KB = 8.617333262145e-5  # eV/K
HARTREE_TO_EV = 27.211386245988
EV_TO_KCAL = 23.0609

def compute_gibbs_correction(
    atoms: Atoms,
    frequencies_cm1: list[float],
    spin: int = 1,
    T: float = 298.15,
    P: float = 101325.0
) -> dict[str, float]:
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
    solvent: str | None = None,
    energy_window_kcal: float = 6.0,
    max_conformers_to_optimize: int = 5,
    energy_threshold_kcal: float = 3.0,
    charge: int = 0,
    spin: int = 1
) -> dict[str, Any]:
    """
    Ensemble Thermochemistry Pipeline:
    1. Generate conformers using CREST.
    2. Filter lowest-energy conformers.
    3. Run geometry optimization and vibrational frequency calculations on each.
    4. Compute Boltzmann populations and ensemble-averaged free energies.
    """
    if not CREST_AVAILABLE or not XTB_AVAILABLE:
        raise BackendUnavailableError(
            "CREST and xTB binaries are required to run the ensemble thermochemistry pipeline.",
            hint="Install the crest and xtb binaries to run ensemble thermochemistry."
        )

    # Step 1: Run CREST conformer search
    logger.info(f"Starting CREST conformer search for {molecule_id}")
    crest_res = run_conformer_search_engine(
        workspace_id,
        molecule_id,
        method=method,
        solvent=solvent,
        energy_window_kcal=energy_window_kcal
    )

    conformers = crest_res["results"]["conformers"]
    if not conformers:
        raise CalculationFailedError(
            "CREST conformer search returned no structures.",
            hint="Try a larger energy_window_kcal, or check that the input geometry is valid."
        )
        
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
        
        # Save temporary conformer coordinates. Every other save_molecule_coords
        # call site in this codebase includes molecule_id/name/smiles/method in
        # its meta dict; this one didn't, which left an inconsistently-shaped
        # entry (just formula+num_atoms) in the shared workspace molecule index
        # - invisible until MoleculeStore.list()/describe_molecule() (Phase 8)
        # started assuming every entry has a molecule_id. smiles is left empty
        # rather than computed via bond perception from the raw XYZ parse
        # (nontrivial and not needed for this temporary reference structure).
        save_molecule_coords(
            workspace_id,
            conf_mol_id,
            Chem.MolToMolBlock(conf_mol),
            c["xyz_block"],
            {
                "molecule_id": conf_mol_id,
                "name": f"{molecule_id} conformer {i} (ensemble reference)",
                "formula": formula,
                "smiles": "",
                "num_atoms": conf_mol.GetNumAtoms(),
                "method": "xtb_ensemble_reference_conformer",
            }
        )
        
        # Geometry Optimization. run_xtb_calculation_engine now raises on failure
        # instead of returning {"ok": False, ...} - catch it here so one bad
        # conformer doesn't abort the whole ensemble; skip and continue as before.
        try:
            opt_res = run_xtb_calculation_engine(
                workspace_id,
                conf_mol_id,
                task="geometry_optimization",
                method=method,
                solvent=solvent,
                charge=charge,
                spin=spin
            )
        except CompchemError as e:
            logger.warning(f"Optimization failed for conformer {i}: {e}")
            continue

        # Get optimized energy
        opt_energy_ev = opt_res["results"]["energy_ev"]

        # Vibrations / Hess calculation
        try:
            vib_res = run_xtb_calculation_engine(
                workspace_id,
                conf_mol_id,
                task="vibrations",
                method=method,
                solvent=solvent,
                charge=charge,
                spin=spin
            )
        except CompchemError as e:
            logger.warning(f"Vibration check failed for conformer {i}: {e}")
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
        raise CalculationFailedError(
            "Failed to optimize and run frequency checks on any conformers.",
            hint="Check server logs for the per-conformer xtb failures logged above."
        )
        
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
        
    # A full frequencies_cm1 list per conformer is unbounded (3N-6 per conformer,
    # times however many conformers were refined). Write the full table
    # (every conformer, every frequency) to a JSON artifact; inline results keep
    # only summary stats per conformer, plus a first-20-frequency preview for
    # the lowest-Gibbs conformer (the one most callers actually want next).
    json_art = None
    try:
        import json

        from ypotheto_compchem_mcp.artifacts import register_artifact
        full_table_bytes = json.dumps(
            {"molecule_id": molecule_id, "refined_conformers": refined_results}, indent=2
        ).encode("utf-8")
        json_art = register_artifact(
            f"{molecule_id}_ensemble_thermochemistry.json",
            full_table_bytes,
            "report",
            "Full per-conformer ensemble thermochemistry table (all frequencies)"
        )
    except Exception as e:
        logger.warning(f"Failed to generate ensemble thermochemistry JSON artifact: {str(e)}")

    lowest = min(refined_results, key=lambda r: r["total_gibbs_energy_ev"])
    conformers_summary = []
    for r in refined_results:
        freqs = r["frequencies_cm1"]
        entry = {k: v for k, v in r.items() if k != "frequencies_cm1"}
        entry["frequency_summary"] = {
            "count": len(freqs),
            "min_cm1": min(freqs) if freqs else None,
            "max_cm1": max(freqs) if freqs else None
        }
        if r is lowest:
            entry["frequencies_cm1_preview"] = freqs[:20]
            entry["frequencies_truncated"] = len(freqs) > 20
        conformers_summary.append(entry)

    results = {
        "molecule_id": molecule_id,
        "temperature_k": T,
        "ensemble_gibbs_free_energy_ev": round(ensemble_gibbs_ev, 6),
        "ensemble_gibbs_free_energy_kcal": round(ensemble_gibbs_ev * EV_TO_KCAL, 4),
        "refined_conformers": conformers_summary
    }

    interpretation = (
        f"Ensemble thermochemistry pipeline completed for {molecule_id}.\n"
        f"Ensemble Gibbs Free Energy = {results['ensemble_gibbs_free_energy_kcal']} kcal/mol.\n"
        f"Refined {len(refined_results)} conformers.\n"
        f"Lowest energy conformer relative Gibbs = 0.0 kcal/mol (Boltzmann pop: {refined_results[0]['boltzmann_population'] * 100:.1f}%).\n"
        f"Full per-conformer frequency table (all {len(refined_results)} conformers) is in the attached JSON artifact; "
        f"only the lowest-Gibbs conformer's first 20 frequencies are shown inline."
    )

    return {
        "ok": True,
        "results": results,
        "interpretation": interpretation,
        "artifacts": [json_art] if json_art else []
    }
