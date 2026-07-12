import os
import shutil
import subprocess
import tempfile
import logging
import re
import math
from typing import Any, Dict, List, Optional
from rdkit import Chem
from ypotheto_compchem_mcp.chemistry.builder_engine import (
    load_molecule_from_workspace,
    save_molecule_coords
)
from ypotheto_compchem_mcp.errors import BackendUnavailableError

logger = logging.getLogger(__name__)

# Check binary availability
XTB_AVAILABLE = shutil.which("xtb") is not None
CREST_AVAILABLE = shutil.which("crest") is not None

def run_xtb_calculation_engine(
    workspace_id: str,
    molecule_id: str,
    task: str = "single_point",
    method: str = "GFN2-xTB",
    solvent: Optional[str] = None,
    charge: int = 0,
    spin: int = 1  # Spin multiplicity (2S + 1)
) -> Dict[str, Any]:
    """
    Run semi-empirical GFN-xTB calculations via subprocess.
    """
    if not XTB_AVAILABLE:
        raise BackendUnavailableError(
            "xtb executable is not available on this system host.",
            hint="Install the xtb binary, or use run_single_point / optimize_geometry (PySCF) instead."
        )
        
    mol = load_molecule_from_workspace(workspace_id, molecule_id)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write coordinates to XYZ format
        xyz_path = os.path.join(tmpdir, "input.xyz")
        Chem.MolToXYZFile(mol, xyz_path)
        
        # Build command-line arguments
        args = ["xtb", "input.xyz"]
        
        # Method configuration
        method_upper = method.upper()
        if "GFN1" in method_upper:
            args.extend(["--gfn", "1"])
        elif "GFN2" in method_upper:
            args.extend(["--gfn", "2"])
        elif "GFN-FF" in method_upper:
            args.append("--gfnff")
            
        # Task configuration
        if task == "geometry_optimization":
            args.append("--opt")
        elif task == "vibrations":
            args.append("--hess")
            
        # Charge & Multiplicity
        # xtb UHF flag takes number of unpaired spins (multiplicity - 1)
        unpaired_spins = max(0, spin - 1)
        args.extend(["--chrg", str(charge)])
        args.extend(["--uhf", str(unpaired_spins)])
        
        # Solvent ALPB GBSA
        if solvent:
            args.extend(["--alpb", solvent.lower()])
            
        # Run subprocess
        result = subprocess.run(
            args,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return {
                "ok": False,
                "error": {
                    "code": "XTB_EXECUTION_FAILED",
                    "message": f"xTB calculation failed with exit code {result.returncode}.",
                    "details": result.stderr or result.stdout
                }
            }
            
        # Parse stdout
        stdout = result.stdout
        energy = None
        dipole = None
        
        # Parse energy: | TOTAL ENERGY              -14.834918731558 Eh   |
        energy_match = re.search(r"TOTAL ENERGY\s+([\-\d\.]+)", stdout, re.IGNORECASE)
        if energy_match:
            energy = float(energy_match.group(1))
            
        # Parse dipole moment
        #        x           y           z         tot (Debye)
        #  -0.00010     0.00000    -1.85420       1.85420
        dipole_section = re.search(r"dipole:\s*\n.*tot \(Debye\)\n\s*([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)", stdout)
        if dipole_section:
            dipole = [
                float(dipole_section.group(1)),
                float(dipole_section.group(2)),
                float(dipole_section.group(3))
            ]
            
        # If optimization task, retrieve final coordinates and update molecule
        warnings = []
        if task == "geometry_optimization":
            opt_xyz_path = os.path.join(tmpdir, "xtbopt.xyz")
            if os.path.exists(opt_xyz_path):
                # Load optimized coordinates into molecule
                opt_mol = Chem.MolFromXYZFile(opt_xyz_path)
                if opt_mol:
                    # RDKit MolFromXYZFile does not preserve bonds/connectivity.
                    # We transfer coordinates to the original structured molecule.
                    conf = mol.GetConformer()
                    opt_conf = opt_mol.GetConformer()
                    for i in range(mol.GetNumAtoms()):
                        pos = opt_conf.GetAtomPosition(i)
                        conf.SetAtomPosition(i, pos)
                        
                    # Save updated structure to workspace
                    sdf_block = Chem.MolToMolBlock(mol)
                    xyz_block = Chem.MolToXYZBlock(mol)
                    meta = {
                        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                        "num_atoms": mol.GetNumAtoms(),
                        "method": f"xTB optimized ({method})"
                    }
                    save_molecule_coords(workspace_id, molecule_id, sdf_block, xyz_block, meta)
                else:
                    warnings.append({"code": "COORD_TRANSFER_FAILED", "message": "Optimized XYZ was generated but could not be parsed by RDKit."})
            else:
                warnings.append({"code": "OPTIMIZED_COORDS_MISSING", "message": "Geometry optimization completed but optimized coordinate file is missing."})
                
        # If vibrations task, parse frequencies
        frequencies = []
        if task == "vibrations":
            # Extract vibrational frequencies from stdout:
            #  #   frequency/cm-1
            #  1      -20.34
            #  2       12.45
            freq_lines = re.findall(r"\d+\s+([\-\d\.]+)\s+(?:i\s+)?(?:cm-1|cm\^-1)", stdout)
            if freq_lines:
                frequencies = [float(f) for f in freq_lines]
                
        results = {
            "energy_hartree": energy,
            "energy_ev": round(energy * 27.211386, 4) if energy else None,
            "dipole_moment_debye": dipole,
            "frequencies_cm1": frequencies
        }
        
        return {
            "ok": True,
            "results": results,
            "warnings": warnings,
            "interpretation": (
                f"xTB {task} ({method}) completed successfully. "
                f"Total Energy = {results['energy_ev']} eV. "
                f"Dipole Moment (tot) = {round(math.sqrt(sum(x**2 for x in dipole)), 4) if dipole else 0.0} Debye."
            )
        }

def run_conformer_search_engine(
    workspace_id: str,
    molecule_id: str,
    method: str = "GFN2-xTB",
    solvent: Optional[str] = None,
    energy_window_kcal: float = 6.0
) -> Dict[str, Any]:
    """
    Generate conformer ensembles using CREST.
    """
    if not CREST_AVAILABLE or not XTB_AVAILABLE:
        raise BackendUnavailableError(
            "crest or xtb executable is not available on this system host.",
            hint="Install the crest and xtb binaries to run conformer ensemble searches."
        )
        
    mol = load_molecule_from_workspace(workspace_id, molecule_id)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "input.xyz")
        Chem.MolToXYZFile(mol, xyz_path)
        
        # Build command: crest input.xyz --gfn2 --ewin 6.0
        args = ["crest", "input.xyz"]
        
        method_upper = method.upper()
        if "GFN1" in method_upper:
            args.append("--gfn1")
        elif "GFN2" in method_upper:
            args.append("--gfn2")
        elif "GFN-FF" in method_upper:
            args.append("--gfnff")
            
        if solvent:
            args.extend(["--alpb", solvent.lower()])
            
        args.extend(["--ewin", str(energy_window_kcal)])
        
        result = subprocess.run(
            args,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return {
                "ok": False,
                "error": {
                    "code": "CREST_EXECUTION_FAILED",
                    "message": f"CREST conformer search failed with exit code {result.returncode}.",
                    "details": result.stderr or result.stdout
                }
            }
            
        # Parse conformer ensemble
        # CREST writes all conformers to 'crest_conformers.xyz'
        conformers_xyz_path = os.path.join(tmpdir, "crest_conformers.xyz")
        energies_path = os.path.join(tmpdir, "crest.energies")
        
        if not os.path.exists(conformers_xyz_path) or not os.path.exists(energies_path):
            return {
                "ok": False,
                "error": {
                    "code": "CREST_OUTPUT_MISSING",
                    "message": "CREST completed but conformer output files are missing."
                }
            }
            
        # Read energies
        energies = []
        with open(energies_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    energies.append(float(parts[0]))
                    
        # Parse crest_conformers.xyz to extract structures
        # A multi-structure XYZ file has formatting:
        # <num_atoms>
        # Comment line containing energy
        # Coordinate lines...
        conformers = []
        with open(conformers_xyz_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        idx = 0
        natoms = mol.GetNumAtoms()
        conformer_index = 0
        
        while idx < len(lines):
            # Read header
            num_atoms_str = lines[idx].strip()
            if not num_atoms_str:
                break
            num_atoms = int(num_atoms_str)
            comment = lines[idx+1].strip()
            
            coord_block = lines[idx+2 : idx+2+num_atoms]
            xyz_block = f"{num_atoms}\n{comment}\n" + "".join(coord_block)
            
            # Compute Boltzmann weights
            # Delta E in kcal/mol (crest energies are usually relative or absolute Eh)
            # Eh to kcal/mol = 627.509
            energy_hartree = energies[conformer_index] if conformer_index < len(energies) else 0.0
            
            conformers.append({
                "conformer_id": f"{molecule_id}_conf_{conformer_index}",
                "energy_hartree": energy_hartree,
                "energy_ev": round(energy_hartree * 27.211386, 4),
                "xyz_block": xyz_block
            })
            
            idx += 2 + num_atoms
            conformer_index += 1
            
        # Calculate Boltzmann populations (T = 298.15 K)
        # R = 1.9872e-3 kcal/(mol*K)
        # kbT = 0.59248 kcal/mol at 298.15K
        kbT_hartree = 0.59248 / 627.509
        min_energy = min(c["energy_hartree"] for c in conformers)
        
        total_q = 0.0
        for c in conformers:
            delta_e = c["energy_hartree"] - min_energy
            c["relative_energy_kcal"] = round(delta_e * 627.509, 4)
            weight = math.exp(-delta_e / kbT_hartree)
            c["boltzmann_weight"] = weight
            total_q += weight
            
        for c in conformers:
            c["boltzmann_population"] = round(c["boltzmann_weight"] / total_q, 4)
            # Clean up temporary weight variable from final dictionary
            del c["boltzmann_weight"]
            
        return {
            "ok": True,
            "results": {
                "molecule_id": molecule_id,
                "num_conformers": len(conformers),
                "conformers": conformers
            },
            "interpretation": (
                f"CREST conformer search completed successfully for {molecule_id}. "
                f"Generated {len(conformers)} conformers within {energy_window_kcal} kcal/mol energy window."
            )
        }
