import sys
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, LBFGS

from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace, save_molecule_coords

# Check if PySCF is installed
try:
    import pyscf
    from pyscf import gto, scf, grad
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

# Hartree to eV conversion factor
HARTREE_TO_EV = 27.211386245988
# Bohr to Angstrom
BOHR_TO_ANGSTROM = 0.529177210903
# Hartree/Bohr to eV/Angstrom force factor: -27.211386245988 / 0.529177210903 = -51.4220674683
FORCE_CONVERSION = -HARTREE_TO_EV / BOHR_TO_ANGSTROM

class PySCFCalculator(Calculator):
    """
    ASE Calculator wrapping PySCF for energy and force evaluations.
    Allows coupling PySCF with ASE's geometry optimizers and MD runners.
    """
    implemented_properties = ['energy', 'forces', 'dipole']
    
    def __init__(self, method: str = "DFT", functional: str = "B3LYP", basis: str = "sto-3g", charge: int = 0, spin: int = 0, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.method = method.upper()
        self.functional = functional
        self.basis = basis
        self.charge = charge
        self.spin = spin
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        if not PYSCF_AVAILABLE:
            raise RuntimeError("PySCF is not installed or available on this system.")
            
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # 1. Convert ASE atoms to PySCF atom coordinate format
        atom_list = []
        for sym, pos in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions()):
            atom_list.append(f"{sym} {pos[0]} {pos[1]} {pos[2]}")
        atom_str = "; ".join(atom_list)
        
        # 2. Build PySCF Molecule
        mol_pyscf = gto.M(
            atom=atom_str,
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            verbose=0
        )
        
        # 3. Setup SCF calculations
        if self.method == "HF":
            if self.spin == 0:
                mf = scf.RHF(mol_pyscf)
            else:
                mf = scf.UHF(mol_pyscf)
        else:
            if self.spin == 0:
                mf = scf.RKS(mol_pyscf)
            else:
                mf = scf.UKS(mol_pyscf)
            mf.xc = self.functional
            
        # 4. Calculate energy
        energy_hartree = mf.kernel()
        self.results['energy'] = energy_hartree * HARTREE_TO_EV
        
        # 5. Calculate forces
        if 'forces' in properties:
            grad_obj = mf.nuc_grad_method()
            g = grad_obj.kernel()
            # Force F = -dE/dx (converted from Hartree/Bohr to eV/Angstrom)
            self.results['forces'] = g * FORCE_CONVERSION
            
        # 6. Calculate dipole (convert Debye to e * Angstrom)
        if 'dipole' in properties:
            try:
                dip_debye = mf.dip_moment(verbose=0)
                self.results['dipole'] = dip_debye * 0.2081943
            except Exception:
                self.results['dipole'] = np.array([0.0, 0.0, 0.0])


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
    spin: int = 0
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
    xyz_path = workspace_manager.get_workspace_dir(workspace_id) / "molecules" / f"{molecule_id}.xyz"
    
    # 1. Setup PySCF Molecule
    mol_pyscf = gto.M(
        atom=str(xyz_path),
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=0
    )
    
    # 2. Setup SCF
    method_upper = method.upper()
    if method_upper == "HF":
        if spin == 0:
            mf = scf.RHF(mol_pyscf)
        else:
            mf = scf.UHF(mol_pyscf)
    elif method_upper == "DFT":
        if spin == 0:
            mf = scf.RKS(mol_pyscf)
        else:
            mf = scf.UKS(mol_pyscf)
        mf.xc = functional
    else:
        raise ValueError(f"Unsupported quantum method: '{method}'")
        
    # Run calculation
    energy_hartree = mf.kernel()
    converged = bool(mf.converged)
    
    # 3. Dipole Moment (in Debye)
    try:
        dipole = mf.dip_moment()
        if hasattr(dipole, "tolist"):
            dipole_list = dipole.tolist()
        else:
            dipole_list = list(dipole)
    except Exception:
        dipole_list = [0.0, 0.0, 0.0]
        
    # 4. HOMO / LUMO analysis (converted to eV)
    mo_energy = mf.mo_energy
    if spin == 0:
        # Restricted Closed Shell
        homo_idx = mol_pyscf.nelectron // 2 - 1
        homo_ev = float(mo_energy[homo_idx] * HARTREE_TO_EV)
        lumo_ev = float(mo_energy[homo_idx + 1] * HARTREE_TO_EV)
        homo_lumo_gap = lumo_ev - homo_ev
    else:
        # Unrestricted Open Shell (Alpha and Beta energies)
        # alpha
        homo_idx_a = mol_pyscf.nelec[0] - 1
        homo_ev_a = float(mo_energy[0][homo_idx_a] * HARTREE_TO_EV)
        lumo_ev_a = float(mo_energy[0][homo_idx_a + 1] * HARTREE_TO_EV)
        # beta
        homo_idx_b = mol_pyscf.nelec[1] - 1
        homo_ev_b = float(mo_energy[1][homo_idx_b] * HARTREE_TO_EV)
        lumo_ev_b = float(mo_energy[1][homo_idx_b + 1] * HARTREE_TO_EV)
        
        homo_ev = max(homo_ev_a, homo_ev_b)
        lumo_ev = min(lumo_ev_a, lumo_ev_b)
        homo_lumo_gap = max(0.0, lumo_ev - homo_ev)
        
    # 5. Mulliken population charges
    try:
        _, mulliken_charges = mf.pop(verbose=0)
        charges_list = mulliken_charges.tolist()
    except Exception:
        charges_list = [0.0] * mol_pyscf.natm
        
    atom_charges = []
    for i in range(mol_pyscf.natm):
        atom_charges.append({
            "index": i,
            "element": mol_pyscf.atom_symbol(i),
            "charge": float(charges_list[i])
        })
        
    return {
        "ok": converged,
        "results": {
            "energy_hartree": float(energy_hartree),
            "energy_ev": float(energy_hartree * HARTREE_TO_EV),
            "dipole_moment_debye": dipole_list,
            "homo_ev": homo_ev,
            "lumo_ev": lumo_ev,
            "homo_lumo_gap_ev": homo_lumo_gap,
            "mulliken_charges": atom_charges
        },
        "warnings": [] if converged else [{"type": "SCF_CONVERGENCE", "message": "SCF did not converge."}]
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
    progress_callback: Optional[Callable[[str], None]] = None
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
        calc = PySCFCalculator(method=method, functional=functional, basis=basis, charge=charge, spin=spin)
        
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
