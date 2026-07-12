import os
import sys
import json
import uuid
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, LBFGS

from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace, save_molecule_coords
from ypotheto_compchem_mcp.workspace import workspace_manager, get_workspace_id
from ypotheto_compchem_mcp.chemistry.runner import get_engine_runner, DockerContainerRunner
from ypotheto_compchem_mcp.chemistry.parser import parse_qm_log_with_cclib
from ypotheto_compchem_mcp.errors import CalculationFailedError

# Hartree to eV conversion factor
HARTREE_TO_EV = 27.211386245988
# Bohr to Angstrom
BOHR_TO_ANGSTROM = 0.529177210903
# Hartree/Bohr to eV/Angstrom force factor: -27.211386245988 / 0.529177210903 = -51.4220674683
FORCE_CONVERSION = -HARTREE_TO_EV / BOHR_TO_ANGSTROM

# Determine PySCF availability: either local package import or docker execution runner
try:
    import pyscf
    LOCAL_PYSCF_AVAILABLE = True
except ImportError:
    LOCAL_PYSCF_AVAILABLE = False

PYSCF_AVAILABLE = LOCAL_PYSCF_AVAILABLE or (os.getenv("COMPCHEM_ENGINE_RUNNER_TYPE", "").lower() == "docker")

def get_pyscf_driver_code() -> str:
    """Helper to read pyscf_driver.py code from the same directory."""
    driver_path = Path(__file__).parent / "pyscf_driver.py"
    return driver_path.read_text(encoding="utf-8")

class PySCFCalculator(Calculator):
    """
    ASE Calculator wrapping isolated PySCF runs for energy and force evaluations.
    Calculations are run in isolated subprocesses/containers to prevent pollution.
    """
    implemented_properties = ['energy', 'forces', 'dipole']
    
    def __init__(self, method: str = "DFT", functional: str = "B3LYP", basis: str = "sto-3g", charge: int = 0, spin: int = 0, solvent: Optional[str] = None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.method = method.upper()
        self.functional = functional
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.solvent = solvent
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        if not PYSCF_AVAILABLE:
            raise RuntimeError("PySCF is not installed or available on this system.")
            
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # 1. Convert ASE atoms to XYZ block format
        atom_list = []
        for sym, pos in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions()):
            atom_list.append(f"{sym} {pos[0]} {pos[1]} {pos[2]}")
        xyz_content = f"{len(self.atoms)}\n\n" + "\n".join(atom_list)
        
        # 2. Get active workspace environment
        workspace_id = get_workspace_id()
        workspace_dir = workspace_manager.get_workspace_dir(workspace_id)
        
        job_id = f"job_calc_{uuid.uuid4().hex[:8]}"
        
        config = {
            "xyz_path": "atoms.xyz",
            "method": self.method,
            "functional": self.functional,
            "basis": self.basis,
            "charge": self.charge,
            "spin": self.spin,
            "solvent": self.solvent,
            "calculate_forces": "forces" in properties
        }
        
        input_files = {
            "atoms.xyz": xyz_content,
            "config.json": json.dumps(config, indent=2),
            "pyscf_driver.py": get_pyscf_driver_code()
        }
        
        # 3. Choose runner and run
        runner = get_engine_runner()
        if isinstance(runner, DockerContainerRunner):
            cmd = ["python", "pyscf_driver.py", "config.json"]
        else:
            cmd = [sys.executable, "pyscf_driver.py", "config.json"]
            
        run_res = runner.run_command(workspace_dir, job_id, cmd, input_files)
        
        # 4. Extract outputs
        res_file = workspace_dir / "jobs" / job_id / "results.json"
        if res_file.exists():
            with open(res_file, encoding="utf-8") as f:
                res_data = json.load(f)
                
            self.results['energy'] = res_data["energy_ev"]
            if 'forces' in properties:
                self.results['forces'] = np.array(res_data["forces_ev_angstrom"])
            if 'dipole' in properties:
                # Convert Debye to e * Angstrom (1 Debye = 0.2081943 e * Angstrom)
                self.results['dipole'] = np.array(res_data["dipole_moment_debye"]) * 0.2081943
        else:
            # Fallback to cclib log file parsing
            parsed = parse_qm_log_with_cclib(run_res.log_file)
            if parsed.ok:
                self.results['energy'] = parsed.energy_ev
                if 'forces' in properties and parsed.forces_ev_angstrom:
                    self.results['forces'] = np.array(parsed.forces_ev_angstrom)
                if 'dipole' in properties:
                    self.results['dipole'] = np.array(parsed.dipole_moment_debye) * 0.2081943
            else:
                raise RuntimeError(f"Calculation failed with exit code {run_res.return_code}: {run_res.stderr}")


class RDKitCalculator(Calculator):
    """
    ASE Calculator wrapping RDKit forcefields (MMFF94 and UFF) for fast local
    calculations of energy and forces, enabling MD/Geometry optimization on Windows.
    """
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, mol_rdkit, forcefield: str = "MMFF94", **kwargs):
        Calculator.__init__(self, **kwargs)
        self.mol_rdkit = mol_rdkit
        self.forcefield = forcefield.upper()
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Update conformer positions from ASE atoms
        conf = self.mol_rdkit.GetConformer()
        positions = self.atoms.get_positions()
        for i in range(self.mol_rdkit.GetNumAtoms()):
            pos = positions[i]
            conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
            
        from rdkit.Chem import AllChem
        if self.forcefield == "MMFF94":
            ff = AllChem.MMFFGetMoleculeForceField(self.mol_rdkit, AllChem.MMFFGetMoleculeProperties(self.mol_rdkit))
        else:
            ff = AllChem.UFFGetMoleculeForceField(self.mol_rdkit)
            
        energy_kcal = ff.CalcEnergy()
        # Convert kcal/mol to eV (1 kcal/mol = 0.043364115359 eV)
        self.results['energy'] = energy_kcal * 0.043364115359
        
        if 'forces' in properties:
            grad = ff.CalcGrad()
            # Force F = -gradient (converted from kcal/mol/Angstrom to eV/Angstrom)
            self.results['forces'] = -np.array(grad).reshape(-1, 3) * 0.043364115359


def run_single_point_engine(
    workspace_id: str,
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    solvent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform a single-point quantum or force-field calculation on a stored molecule.
    """
    method_upper = method.upper()
    
    # Route to RDKit forcefield calculation if requested
    if method_upper in ("MMFF94", "UFF"):
        from rdkit.Chem import AllChem
        mol_rdkit = load_molecule_from_workspace(workspace_id, molecule_id)
        if method_upper == "MMFF94":
            ff = AllChem.MMFFGetMoleculeForceField(mol_rdkit, AllChem.MMFFGetMoleculeProperties(mol_rdkit))
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol_rdkit)
            
        energy_kcal = ff.CalcEnergy()
        energy_ev = energy_kcal * 0.043364115359
        
        return {
            "ok": True,
            "results": {
                "energy_ev": float(energy_ev),
                "energy_hartree": float(energy_ev / HARTREE_TO_EV),
                "dipole_moment_debye": [0.0, 0.0, 0.0],
                "homo_ev": 0.0,
                "lumo_ev": 0.0,
                "homo_lumo_gap_ev": 0.0,
                "mulliken_charges": []
            },
            "warnings": []
        }

    if not PYSCF_AVAILABLE:
        raise RuntimeError("PySCF is not available on this Windows host. Please run inside Linux/Docker.")
        
    # Load coordinates from workspace
    mol_rdkit = load_molecule_from_workspace(workspace_id, molecule_id)
    workspace_dir = workspace_manager.get_workspace_dir(workspace_id)
    xyz_path = workspace_dir / "molecules" / f"{molecule_id}.xyz"
    xyz_content = xyz_path.read_text(encoding="utf-8")
    
    job_id = f"job_qm_{uuid.uuid4().hex[:8]}"
    
    config = {
        "xyz_path": f"{molecule_id}.xyz",
        "method": method,
        "functional": functional,
        "basis": basis,
        "charge": charge,
        "spin": spin,
        "solvent": solvent,
        "calculate_forces": False
    }
    
    input_files = {
        f"{molecule_id}.xyz": xyz_content,
        "config.json": json.dumps(config, indent=2),
        "pyscf_driver.py": get_pyscf_driver_code()
    }
    
    runner = get_engine_runner()
    if isinstance(runner, DockerContainerRunner):
        cmd = ["python", "pyscf_driver.py", "config.json"]
    else:
        cmd = [sys.executable, "pyscf_driver.py", "config.json"]
        
    run_res = runner.run_command(workspace_dir, job_id, cmd, input_files)
    
    res_file = workspace_dir / "jobs" / job_id / "results.json"
    if res_file.exists():
        with open(res_file, encoding="utf-8") as f:
            res_data = json.load(f)
            
        converged = res_data["ok"]
        energy_ev = res_data["energy_ev"]
        energy_hartree = res_data["energy_hartree"]
        dipole_list = res_data["dipole_moment_debye"]
        
        mo_energies = res_data["mo_energies_ev"]
        nocc = res_data["nocc"]
        try:
            if isinstance(nocc, list) and len(nocc) == 2:
                homo_a = mo_energies[0][nocc[0] - 1]
                lumo_a = mo_energies[0][nocc[0]]
                homo_b = mo_energies[1][nocc[1] - 1]
                lumo_b = mo_energies[1][nocc[1]]
                homo_ev = max(homo_a, homo_b)
                lumo_ev = min(lumo_a, lumo_b)
            else:
                homo_ev = mo_energies[0][nocc - 1]
                lumo_ev = mo_energies[0][nocc]
            homo_lumo_gap = max(0.0, lumo_ev - homo_ev)
        except Exception:
            homo_ev = 0.0
            lumo_ev = 0.0
            homo_lumo_gap = 0.0
            
        charges_list = res_data["mulliken_charges"]
        atom_charges = []
        for i in range(mol_rdkit.GetNumAtoms()):
            atom_charges.append({
                "index": i,
                "element": mol_rdkit.GetAtomWithIdx(i).GetSymbol(),
                "charge": float(charges_list[i]) if i < len(charges_list) else 0.0
            })
    else:
        # Fallback to cclib log file parsing
        parsed = parse_qm_log_with_cclib(run_res.log_file)
        converged = parsed.ok
        energy_ev = parsed.energy_ev
        energy_hartree = parsed.energy_hartree
        dipole_list = parsed.dipole_moment_debye
        homo_ev = parsed.homo_ev
        lumo_ev = parsed.lumo_ev
        homo_lumo_gap = parsed.homo_lumo_gap_ev
        
        atom_charges = []
        for charge_item in parsed.mulliken_charges:
            atom_charges.append({
                "index": charge_item.index,
                "element": charge_item.element,
                "charge": charge_item.charge
            })
            
    warnings = []
    if not converged:
        warnings.append({"type": "SCF_CONVERGENCE", "message": "SCF did not converge."})
    if run_res.return_code != 0:
        warnings.append({"type": "RUN_ERROR", "message": run_res.stderr})
        
    return {
        "ok": converged and run_res.return_code == 0,
        "results": {
            "energy_hartree": float(energy_hartree),
            "energy_ev": float(energy_ev),
            "dipole_moment_debye": dipole_list,
            "homo_ev": homo_ev,
            "lumo_ev": lumo_ev,
            "homo_lumo_gap_ev": homo_lumo_gap,
            "mulliken_charges": atom_charges
        },
        "warnings": warnings
    }


def optimize_geometry_engine(
    workspace_id: str,
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    max_steps: int = 50,
    progress_callback: Optional[Callable[[str], None]] = None,
    solvent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform geometry optimization using ASE LBFGS optimizer coupled with PySCF or RDKit calculator.
    """
    # 1. Load starting coordinate structure from workspace
    mol_rdkit = load_molecule_from_workspace(workspace_id, molecule_id)
    
    method_upper = method.upper()
    
    # 2. Build ASE Atoms object
    symbols = []
    positions = []
    conf = mol_rdkit.GetConformer()
    for i in range(mol_rdkit.GetNumAtoms()):
        atom = mol_rdkit.GetAtomWithIdx(i)
        symbols.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
        
    atoms = Atoms(symbols=symbols, positions=positions)
    
    # 3. Attach Calculator
    if method_upper in ("MMFF94", "UFF"):
        calc = RDKitCalculator(mol_rdkit, forcefield=method_upper)
    else:
        if not PYSCF_AVAILABLE:
            raise RuntimeError("PySCF is not available on this Windows host. Please run inside Linux/Docker.")
        calc = PySCFCalculator(method=method, functional=functional, basis=basis, charge=charge, spin=spin, solvent=solvent)
        
    atoms.calc = calc
    
    # 4. Run LBFGS optimization with callback updates
    opt = LBFGS(atoms, logfile=None)
    
    step = 0
    def _step_callback():
        nonlocal step
        step += 1
        energy = atoms.get_potential_energy()
        msg = f"Optimization step {step}: Energy = {energy:.4f} eV"
        if progress_callback:
            progress_callback(msg)
            
    opt.attach(_step_callback)
    
    # Run optimization (fmax: maximum force threshold, standard is 0.05 eV/Angstrom)
    opt.run(fmax=0.05, steps=max_steps)
    
    # 5. Extract optimized coordinates and save back to workspace
    from rdkit import Chem
    mol_optimized = Chem.Mol(mol_rdkit)
    conf_opt = mol_optimized.GetConformer()
    
    final_positions = atoms.get_positions()
    for i in range(mol_optimized.GetNumAtoms()):
        pos = final_positions[i]
        conf_opt.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
        
    xyz_block = Chem.MolToXYZBlock(mol_optimized)
    sdf_block = Chem.MolToMolBlock(mol_optimized)
    
    # Generate new molecule_id for optimized geometry
    import uuid
    opt_molecule_id = f"mol_{uuid.uuid4().hex[:8]}"
    formula = atoms.get_chemical_formula()
    
    meta = {
        "molecule_id": opt_molecule_id,
        "name": f"Optimized {molecule_id}",
        "formula": formula,
        "parent_id": molecule_id,
        "num_atoms": len(atoms),
        "method": f"{method}/{functional}/{basis}"
    }
    save_molecule_coords(workspace_id, opt_molecule_id, sdf_block, xyz_block, meta)
    
    return {
        "ok": opt.converged(),
        "results": {
            "original_molecule_id": molecule_id,
            "optimized_molecule_id": opt_molecule_id,
            "final_energy_ev": float(atoms.get_potential_energy()),
            "steps": step,
            "converged": bool(opt.converged())
        },
        "warnings": [] if opt.converged() else [{"type": "GEOM_CONVERGENCE", "message": "Geometry optimization did not converge."}],
        "xyz_block": xyz_block,
        "sdf_block": sdf_block
    }


def estimate_time_seconds(workspace_id: str, molecule_id: str, method: str, basis: str) -> int:
    """Estimate execution time based on molecule size, method, and basis set."""
    try:
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
        natoms = mol.GetNumAtoms()
    except Exception:
        natoms = 5  # Fallback
        
    method_upper = method.upper()
    if method_upper in ("MMFF94", "UFF"):
        return 2
        
    basis_clean = basis.lower().strip()
    if "6-31g" in basis_clean:
        factor = 0.4
    elif "sto-3g" in basis_clean:
        factor = 0.08
    else:
        factor = 1.2
        
    # DFT/HF scaling is O(N^3)
    est = int(factor * (natoms ** 3))
    return max(5, min(3600, est))


def run_pyscf_properties_engine(
    workspace_id: str,
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    properties: List[str] = None,
    solvent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform a PySCF calculation and compute expanded properties (ESP, Loewdin pop, orbital cubes).
    """
    if properties is None:
        properties = ["mulliken", "loewdin", "esp", "homo_lumo_cubes"]
        
    if not PYSCF_AVAILABLE:
        raise RuntimeError("PySCF is not available on this Windows host. Please run inside Linux/Docker.")
        
    mol_rdkit = load_molecule_from_workspace(workspace_id, molecule_id)
    workspace_dir = workspace_manager.get_workspace_dir(workspace_id)
    xyz_path = workspace_dir / "molecules" / f"{molecule_id}.xyz"
    xyz_content = xyz_path.read_text(encoding="utf-8")
    
    job_id = f"job_qm_prop_{uuid.uuid4().hex[:8]}"
    
    config = {
        "xyz_path": f"{molecule_id}.xyz",
        "method": method,
        "functional": functional,
        "basis": basis,
        "charge": charge,
        "spin": spin,
        "solvent": solvent,
        "calculate_forces": False,
        "properties": properties
    }
    
    input_files = {
        f"{molecule_id}.xyz": xyz_content,
        "config.json": json.dumps(config, indent=2),
        "pyscf_driver.py": get_pyscf_driver_code()
    }
    
    runner = get_engine_runner()
    if isinstance(runner, DockerContainerRunner):
        cmd = ["python", "pyscf_driver.py", "config.json"]
    else:
        cmd = [sys.executable, "pyscf_driver.py", "config.json"]
        
    run_res = runner.run_command(workspace_dir, job_id, cmd, input_files)
    
    res_file = workspace_dir / "jobs" / job_id / "results.json"
    if not res_file.exists():
        raise CalculationFailedError(
            "Calculation failed or results file is missing.",
            hint=run_res.stderr or "Check server logs for the underlying PySCF driver error."
        )

    with open(res_file, encoding="utf-8") as f:
        res_data = json.load(f)
        
    # Read back results
    converged = res_data["ok"]
    energy_ev = res_data["energy_ev"]
    energy_hartree = res_data["energy_hartree"]
    dipole_list = res_data["dipole_moment_debye"]
    
    # Mulliken
    charges_list = res_data["mulliken_charges"]
    mulliken_charges = []
    for i in range(mol_rdkit.GetNumAtoms()):
        mulliken_charges.append({
            "index": i,
            "element": mol_rdkit.GetAtomWithIdx(i).GetSymbol(),
            "charge": float(charges_list[i]) if i < len(charges_list) else 0.0
        })
        
    # Loewdin
    loewdin_list = res_data.get("loewdin_charges", [])
    loewdin_charges = []
    for i in range(mol_rdkit.GetNumAtoms()):
        loewdin_charges.append({
            "index": i,
            "element": mol_rdkit.GetAtomWithIdx(i).GetSymbol(),
            "charge": float(loewdin_list[i]) if i < len(loewdin_list) else 0.0
        })
        
    # Register cube files
    registered_artifacts = []
    from ypotheto_compchem_mcp.artifacts import register_artifact
    for cube_name, title in [("homo.cube", "HOMO Orbital"), ("lumo.cube", "LUMO Orbital"), ("esp.cube", "Electrostatic Potential")]:
        cube_file = workspace_dir / "jobs" / job_id / cube_name
        if cube_file.exists():
            cube_bytes = cube_file.read_bytes()
            art = register_artifact(f"{molecule_id}_{cube_name}", cube_bytes, "structure", f"{title} Volume (Cube)")
            registered_artifacts.append(art)
            
    results = {
        "converged": converged,
        "energy_hartree": energy_hartree,
        "energy_ev": energy_ev,
        "dipole_moment_debye": dipole_list,
        "mulliken_charges": mulliken_charges,
        "loewdin_charges": loewdin_charges,
        "artifacts": registered_artifacts
    }
    
    mull_str = ", ".join(f"{c['element']}{c['index']}:{round(c['charge'], 3)}" for c in mulliken_charges[:4])
    loew_str = ", ".join(f"{c['element']}{c['index']}:{round(c['charge'], 3)}" for c in loewdin_charges[:4])
    
    interpretation = (
        f"Calculated expanded properties for {molecule_id}.\n"
        f"Energy = {energy_ev} eV.\n"
        f"Mulliken charges: {mull_str}...\n"
        f"Loewdin charges: {loew_str}...\n"
        f"Generated {len(registered_artifacts)} volumetric cube artifacts."
    )
    
    return {
        "ok": True,
        "results": results,
        "warnings": [],
        "interpretation": interpretation
    }

