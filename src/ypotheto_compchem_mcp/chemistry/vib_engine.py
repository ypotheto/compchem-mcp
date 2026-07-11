import io
import shutil
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo

from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace, save_molecule_coords
from ypotheto_compchem_mcp.chemistry.qm_engine import PySCFCalculator, RDKitCalculator, PYSCF_AVAILABLE, HARTREE_TO_EV
from ypotheto_compchem_mcp.workspace import workspace_manager

def run_vibrations_engine(
    workspace_id: str,
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Perform vibrational harmonic analysis, yielding frequencies and ZPE.
    """
    method_upper = method.upper()
    
    # 1. Load starting coordinate structure from workspace
    mol_rdkit = load_molecule_from_workspace(workspace_id, molecule_id)
    
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
    
    # 4. Run ASE Vibrations inside workspace scratch folder
    vib_dir = workspace_manager.get_workspace_dir(workspace_id) / "vibrations" / f"vib_{molecule_id}"
    if vib_dir.exists():
        shutil.rmtree(vib_dir)
    vib_dir.mkdir(parents=True, exist_ok=True)
    
    if progress_callback:
        progress_callback("Running numerical displacements for Hessian calculation...")
        
    # Use standard Vibrations class
    # For O(N) displacements, this evaluates energy/forces at each atomic displacement
    vib = Vibrations(atoms, name=str(vib_dir / "vib"))
    vib.run()
    
    frequencies = vib.get_frequencies()
    zpe = vib.get_zero_point_energy()
    
    # Clean frequencies list (extract real parts, convert complex to float/imag representation)
    freqs_clean = []
    imag_modes_count = 0
    for f in frequencies:
        if isinstance(f, complex):
            val = f.real if f.imag == 0 else -f.imag
        else:
            val = float(f)
            
        freqs_clean.append(val)
        if val < 0:
            imag_modes_count += 1
            
    # Calculate thermochemistry corrections using IdealGasThermo
    # Exclude translation/rotation modes (first 3 for linear, 5 for nonlinear)
    # We filter vibrational energies > 0.005 eV
    vib_energies = vib.get_energies()
    clean_energies = [e for e in vib_energies if e.real > 0.005 and e.imag == 0]
    
    # Check linear geometry via moments of inertia
    moments = atoms.get_moments_of_inertia()
    is_linear = moments[0] < 1e-4
    geom = "linear" if is_linear else "nonlinear"
    
    # rotational symmetry number
    symmetrynumber = 1
    
    thermo = IdealGasThermo(
        vib_energies=clean_energies,
        geometry=geom,
        atoms=atoms,
        symmetrynumber=symmetrynumber,
        spin=spin / 2.0
    )
    
    T = 298.15
    P = 101325.0
    
    enthalpy_corr = float(thermo.get_enthalpy(temperature=T))
    entropy_corr = float(thermo.get_entropy(temperature=T, pressure=P))
    gibbs_corr = float(thermo.get_gibbs_energy(temperature=T, pressure=P))
    
    # Clean up scratch files
    shutil.rmtree(vib_dir, ignore_errors=True)
    
    return {
        "ok": True,
        "results": {
            "frequencies_cm1": freqs_clean,
            "zero_point_energy_ev": float(zpe),
            "zero_point_energy_kcal": float(zpe * 23.0609), # 1 eV = 23.0609 kcal/mol
            "imaginary_modes_count": imag_modes_count,
            "thermochemistry": {
                "temperature_k": T,
                "pressure_pa": P,
                "enthalpy_ev": enthalpy_corr,
                "entropy_ev_k": entropy_corr,
                "gibbs_free_energy_ev": gibbs_corr
            }
        },
        "warnings": [{"type": "IMAGINARY_MODES", "message": f"Found {imag_modes_count} imaginary vibrational modes. Structure may not be a true minimum."}] if imag_modes_count > 0 else []
    }


def simulate_ir_spectrum_engine(
    workspace_id: str,
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Simulate IR intensities and generate a Lorentzian IR spectrum plot.
    """
    method_upper = method.upper()
    mol_rdkit = load_molecule_from_workspace(workspace_id, molecule_id)
    
    symbols = []
    positions = []
    conf = mol_rdkit.GetConformer()
    for i in range(mol_rdkit.GetNumAtoms()):
        atom = mol_rdkit.GetAtomWithIdx(i)
        symbols.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
        
    atoms = Atoms(symbols=symbols, positions=positions)
    
    if method_upper in ("MMFF94", "UFF"):
        calc = RDKitCalculator(mol_rdkit, forcefield=method_upper)
    else:
        if not PYSCF_AVAILABLE:
            raise RuntimeError("PySCF is not available on this Windows host. Please run inside Linux/Docker.")
        calc = PySCFCalculator(method=method, functional=functional, basis=basis, charge=charge, spin=spin)
        
    atoms.calc = calc
    
    vib_dir = workspace_manager.get_workspace_dir(workspace_id) / "vibrations" / f"ir_{molecule_id}"
    if vib_dir.exists():
        shutil.rmtree(vib_dir)
    vib_dir.mkdir(parents=True, exist_ok=True)
    
    if progress_callback:
        progress_callback("Running displacement steps for Infrared analysis...")
        
    # Execute IR / dipole derivatives
    # For force fields (which lack dipole moment), we run Vibrations and mock intensities as 1.0
    if method_upper in ("MMFF94", "UFF"):
        from ase.vibrations import Vibrations
        ir = Vibrations(atoms, name=str(vib_dir / "ir"))
        ir.run()
        frequencies = ir.get_frequencies()
        intensities = np.array([1.0] * len(frequencies))
        warnings = [{"type": "FORCEFIELD_IR", "message": "Forcefield methods do not yield charge dipole derivatives. All IR intensities are mocked as 1.0."}]
    else:
        from ase.vibrations import Infrared
        ir = Infrared(atoms, name=str(vib_dir / "ir"))
        ir.run()
        frequencies = ir.get_frequencies()
        intensities = ir.get_intensities()
        warnings = []
        
    # Clean lists
    freqs_clean = []
    ints_clean = []
    for f, intens in zip(frequencies, intensities):
        if isinstance(f, complex):
            freq_val = f.real if f.imag == 0 else -f.imag
        else:
            freq_val = float(f)
        freqs_clean.append(freq_val)
        ints_clean.append(float(intens))
        
    # Plotting: Generate Lorentzian lines
    x = np.linspace(400, 4000, 1000)
    y = np.zeros_like(x)
    gamma = 15.0  # Lorentzian broadening factor HWHM
    
    for f_val, intens in zip(freqs_clean, ints_clean):
        # Only plot real positive frequencies (exclude translation/rotation modes)
        if f_val > 10.0:
            y += intens * (gamma**2) / ((x - f_val)**2 + gamma**2)
            
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color="#2c3e50", lw=1.5)
    ax.fill_between(x, y, color="#34495e", alpha=0.1)
    
    # Reverse X axis for standard chemical representation
    ax.set_xlim(4000, 400)
    ax.set_ylim(0, max(y) * 1.1 if len(y) > 0 and max(y) > 0 else 1.0)
    
    ax.set_title(f"Simulated IR Spectrum ({method_upper})")
    ax.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax.set_ylabel("Intensity (Arbitrary Units)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    plot_bytes = buf.getvalue()
    
    # Clean up scratch files
    shutil.rmtree(vib_dir, ignore_errors=True)
    
    return {
        "ok": True,
        "results": {
            "frequencies_cm1": freqs_clean,
            "intensities": ints_clean
        },
        "plot_bytes": plot_bytes,
        "warnings": warnings
    }
