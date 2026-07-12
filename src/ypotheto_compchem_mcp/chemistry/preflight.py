import logging
from typing import Any, Dict, Tuple
from rdkit import Chem
from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace

def validate_charge_spin_multiplicity(mol: Chem.Mol, charge: int, spin: int) -> Tuple[bool, str]:
    """
    Validate that the requested charge and spin multiplicity are mathematically consistent.
    Returns (is_valid, error_message).
    """
    try:
        # Sum of atomic numbers of all atoms in the molecule
        total_nuclear_charge = sum(atom.GetAtomicNum() for atom in mol.GetAtoms())
    except Exception as e:
        return False, f"Failed to inspect molecular atomic numbers: {str(e)}"
        
    n_electrons = total_nuclear_charge - charge
    
    # 1. Parity validation:
    # Let S be the spin. Spin multiplicity M = 2S + 1.
    # 2S (M - 1) represents the number of unpaired spins.
    # The parity of 2S must match the parity of the total number of electrons.
    if (n_electrons % 2) != ((spin - 1) % 2):
        if n_electrons % 2 == 0:
            return False, (
                f"Invalid spin multiplicity. Molecule has an even number of electrons ({n_electrons}) "
                f"based on nuclear charge {total_nuclear_charge} and charge {charge}. "
                f"Spin multiplicity must be ODD (e.g. singlet=1, triplet=3, quintet=5); got {spin}."
            )
        else:
            return False, (
                f"Invalid spin multiplicity. Molecule has an odd number of electrons ({n_electrons}) "
                f"based on nuclear charge {total_nuclear_charge} and charge {charge}. "
                f"Spin multiplicity must be EVEN (e.g. doublet=2, quartet=4, sextet=6); got {spin}."
            )
            
    # 2. Lower bound validation
    if spin < 1:
        return False, f"Spin multiplicity must be at least 1; got {spin}."
        
    # 3. Upper bound validation:
    # Unpaired electrons (spin - 1) cannot exceed total electrons.
    if (spin - 1) > n_electrons:
        return False, (
            f"Invalid spin multiplicity. The requested multiplicity of {spin} "
            f"requires {spin - 1} unpaired electrons, which exceeds the total "
            f"number of electrons ({n_electrons}) in the system."
        )
        
    return True, ""

def validate_basis_set_coverage(mol: Chem.Mol, basis: str) -> Tuple[bool, str]:
    """
    Validate that all element types in the molecule are supported by the requested basis set.
    Returns (is_valid, error_message).
    """
    basis_clean = basis.lower().strip()
    
    # Define maximum atomic number supported by basis families
    if "sto-3g" in basis_clean:
        max_z = 54  # H to Xe supported
    elif "def2-" in basis_clean:
        max_z = 86  # H to Rn supported
    elif "lanl2dz" in basis_clean:
        max_z = 86  # Effectively covers all common metals and heavy atoms
    elif "cc-pv" in basis_clean:
        max_z = 36  # H to Kr (Z=36)
    else:
        # Standard organic basis sets (6-31g, 3-21g, etc.) typically support H to Ar
        max_z = 18
        
    unsupported_atoms = []
    try:
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            symbol = atom.GetSymbol()
            if z > max_z:
                unsupported_atoms.append((symbol, z))
    except Exception as e:
        return False, f"Failed to inspect molecule element types: {str(e)}"
        
    if unsupported_atoms:
        atoms_str = ", ".join(f"{sym}(Z={z})" for sym, z in sorted(set(unsupported_atoms)))
        return False, (
            f"Basis set '{basis}' does not support heavy elements present in this molecule: {atoms_str}. "
            f"Please select a transition-metal/heavy-element compatible basis set like 'def2-svp' or 'lanl2dz'."
        )
        
    return True, ""

def _estimate_basis_functions(mol: Chem.Mol, basis: str) -> int:
    """Helper to estimate the number of basis functions in the system."""
    basis_clean = basis.lower().strip()
    total_funcs = 0
    
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z <= 2:  # H, He
            if "sto-3g" in basis_clean:
                total_funcs += 1
            elif "6-31g" in basis_clean:
                total_funcs += 2
            else:
                total_funcs += 5
        elif z <= 10:  # Li to Ne
            if "sto-3g" in basis_clean:
                total_funcs += 5
            elif "6-31g" in basis_clean:
                total_funcs += 9
            else:
                total_funcs += 15
        elif z <= 18:  # Na to Ar
            if "sto-3g" in basis_clean:
                total_funcs += 9
            elif "6-31g" in basis_clean:
                total_funcs += 13
            else:
                total_funcs += 25
        else:  # K to Rn (Transition metals & heavy atoms)
            if "lanl2dz" in basis_clean:
                total_funcs += 15
            else:
                total_funcs += 35
                
    return max(2, total_funcs)

def estimate_computational_resources(mol: Chem.Mol, method: str, basis: str, task: str) -> Dict[str, Any]:
    """
    Estimate computational wall time, memory, disk scratch space, and credit billing cost.
    """
    method_upper = method.upper()
    natoms = mol.GetNumAtoms()
    
    # 1. Force Field methods (synchronous, extremely fast)
    if method_upper in ("MMFF94", "UFF"):
        est_time = 1
        ram_mb = 100
        disk_mb = 2
        credits = 0.05
        run_mode = "sync"
    else:
        # 2. Semi-empirical / xTB
        n_basis = _estimate_basis_functions(mol, basis)
        
        if "XTB" in method_upper:
            est_time = int(0.01 * (natoms ** 2.0))
            ram_mb = 256 + natoms
            disk_mb = 10
            run_mode = "sync" if est_time < 10 else "async"
        else:
            # 3. Hartree-Fock / DFT
            # Estimate scaling coefficients
            if "sto-3g" in basis.lower():
                base_factor = 0.0001
            elif "6-31g" in basis.lower():
                base_factor = 0.0004
            else:
                base_factor = 0.001
                
            # O(N^3) single point energy scaling
            sp_time = base_factor * (n_basis ** 3.0)
            
            if task == "geometry_optimization":
                # Optimizations typically take ~15 steps
                est_time = int(15 * sp_time)
            elif task == "vibrations":
                # Frequencies require displacements (3 * Natoms single points)
                est_time = int(3 * natoms * sp_time)
            else:
                est_time = int(sp_time)
                
            ram_mb = int(512 + 0.05 * (n_basis ** 2.0))
            disk_mb = int(50 + 0.02 * (n_basis ** 2.0))
            run_mode = "sync" if est_time < 10 else "async"
            
        # Clamp values to safe boundaries
        est_time = max(2, min(7200, est_time))
        ram_mb = max(256, min(16384, ram_mb))
        disk_mb = max(5, min(50000, disk_mb))
        credits = round((est_time * 0.01) + (ram_mb / 1024.0 * 0.05), 2)
        
    return {
        "estimated_wall_time_seconds": est_time,
        "estimated_ram_mb": ram_mb,
        "estimated_disk_scratch_mb": disk_mb,
        "compute_credits_cost": credits,
        "recommended_run_mode": run_mode
    }
