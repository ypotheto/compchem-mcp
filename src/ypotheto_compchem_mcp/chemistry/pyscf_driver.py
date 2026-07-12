import sys
import json
import numpy as np
from pyscf import gto, scf, grad

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_CONVERSION = -HARTREE_TO_EV / BOHR_TO_ANGSTROM

def main():
    if len(sys.argv) < 2:
        print("Usage: pyscf_driver.py input.json")
        sys.exit(1)
        
    with open(sys.argv[1], encoding="utf-8") as f:
        params = json.load(f)
        
    mol = gto.M(
        atom=params["xyz_path"],
        basis=params["basis"],
        charge=params["charge"],
        spin=params["spin"],
        verbose=4  # cclib requires high verbosity to parse orbital energies, dipole, etc.
    )
    
    if params["method"].upper() == "HF":
        mf = scf.RHF(mol) if params["spin"] == 0 else scf.UHF(mol)
    else:
        mf = scf.RKS(mol) if params["spin"] == 0 else scf.UKS(mol)
        mf.xc = params["functional"]
        
    if params.get("solvent"):
        from pyscf import solvent
        mf = solvent.ddCOSMO(mf)
        solv_lower = params["solvent"].lower()
        eps_map = {
            "water": 78.3,
            "methanol": 32.6,
            "ethanol": 24.3,
            "acetone": 20.7,
            "benzene": 2.27,
            "chloroform": 4.81,
            "dichloromethane": 8.93,
            "acetonitrile": 36.6,
            "thf": 7.58,
            "dmso": 46.7,
            "hexane": 1.88,
            "toluene": 2.38
        }
        if solv_lower in eps_map:
            mf.with_solvent.eps = eps_map[solv_lower]
        
    energy_hartree = mf.kernel()
    converged = bool(mf.converged)
    
    # Dipole
    dipole_list = [0.0, 0.0, 0.0]
    try:
        dipole = mf.dip_moment(verbose=0)
        if hasattr(dipole, "tolist"):
            dipole_list = dipole.tolist()
        else:
            dipole_list = list(dipole)
    except Exception:
        pass
        
    # HOMO / LUMO / MO Energies
    mo_energy = mf.mo_energy
    mo_energy_list = []
    if hasattr(mo_energy, "tolist"):
        mo_energy_list = mo_energy.tolist()
    elif isinstance(mo_energy, (list, tuple)):
        mo_energy_list = [x.tolist() if hasattr(x, "tolist") else list(x) for x in mo_energy]
    else:
        # Single element or other shape
        mo_energy_list = [float(mo_energy)]
        
    nocc = mol.nelectron // 2 if params["spin"] == 0 else [int(x) for x in mol.nelec]
    
    # Mulliken
    charges_list = []
    try:
        _, mulliken_charges = mf.pop(verbose=0)
        charges_list = mulliken_charges.tolist()
    except Exception:
        charges_list = [0.0] * mol.natm
        
    # Forces / Gradients
    forces_list = []
    if params.get("calculate_forces", False):
        try:
            grad_obj = mf.nuc_grad_method()
            g = grad_obj.kernel()
            forces = g * FORCE_CONVERSION
            forces_list = forces.tolist()
        except Exception as e:
            sys.stderr.write(f"Failed to calculate forces: {str(e)}\n")
            forces_list = [[0.0, 0.0, 0.0]] * mol.natm

    # Loewdin population analysis
    loewdin_charges = []
    if "loewdin" in params.get("properties", []):
        try:
            import scipy.linalg
            dm = mf.make_rdm1()
            s = mf.get_ovlp()
            if isinstance(dm, tuple) or (isinstance(dm, np.ndarray) and dm.ndim == 3):
                dm_tot = dm[0] + dm[1]
            else:
                dm_tot = dm
            s12 = scipy.linalg.sqrtm(s)
            dm_lowdin = np.dot(np.dot(s12, dm_tot), s12)
            for i in range(mol.natm):
                ao_slice = mol.aoslice(i)
                start_idx, end_idx = ao_slice[2], ao_slice[3]
                pop_i = np.diag(dm_lowdin)[start_idx:end_idx].sum()
                loewdin_charges.append(float(mol.atom_charge(i) - pop_i))
        except Exception as e:
            sys.stderr.write(f"Failed to calculate Loewdin charges: {str(e)}\n")
            loewdin_charges = [0.0] * mol.natm

    # Cube files
    if "homo_lumo_cubes" in params.get("properties", []):
        try:
            from pyscf.tools import cubegen
            if isinstance(nocc, list):
                homo_idx = nocc[0] - 1
                lumo_idx = nocc[0]
                mo_coeff = mf.mo_coeff[0]
            else:
                homo_idx = nocc - 1
                lumo_idx = nocc
                mo_coeff = mf.mo_coeff
            cubegen.orbital(mol, "homo.cube", mo_coeff[:, homo_idx])
            cubegen.orbital(mol, "lumo.cube", mo_coeff[:, lumo_idx])
        except Exception as e:
            sys.stderr.write(f"Failed to generate HOMO/LUMO cubes: {str(e)}\n")

    # ESP
    if "esp" in params.get("properties", []):
        try:
            from pyscf.tools import cubegen
            dm = mf.make_rdm1()
            if isinstance(dm, tuple) or (isinstance(dm, np.ndarray) and dm.ndim == 3):
                dm_tot = dm[0] + dm[1]
            else:
                dm_tot = dm
            cubegen.potential(mol, "esp.cube", dm_tot)
        except Exception as e:
            sys.stderr.write(f"Failed to generate ESP cube: {str(e)}\n")

    # Save to results.json
    results = {
        "ok": converged,
        "energy_hartree": float(energy_hartree),
        "energy_ev": float(energy_hartree * HARTREE_TO_EV),
        "dipole_moment_debye": dipole_list,
        "mo_energies_ev": (np.array(mo_energy_list) * HARTREE_TO_EV).tolist(),
        "nocc": nocc,
        "mulliken_charges": charges_list,
        "loewdin_charges": loewdin_charges,
        "forces_ev_angstrom": forces_list
    }
    
    with open("results.json", "w", encoding="utf-8") as out:
        json.dump(results, out, indent=2)

if __name__ == "__main__":
    main()
