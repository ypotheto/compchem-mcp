import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin

from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace, save_molecule_coords
from ypotheto_compchem_mcp.chemistry.qm_engine import PySCFCalculator, RDKitCalculator, PYSCF_AVAILABLE
from ypotheto_compchem_mcp.workspace import workspace_manager

def run_molecular_dynamics_engine(
    workspace_id: str,
    molecule_id: str,
    steps: int = 200,
    time_step_fs: float = 0.5,
    temperature_k: float = 300.0,
    ensemble: str = "NVT",
    calculator_type: str = "MMFF94",
    functional: Optional[str] = "B3LYP",
    basis: Optional[str] = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Run molecular dynamics using Langevin (NVT) or VelocityVerlet (NVE) ensembles.
    Returns energies, temperature profile, trajectory XYZ frames, and plot bytes.
    """
    calc_type_upper = calculator_type.upper()
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
    
    # Attach Calculator
    if calc_type_upper in ("MMFF94", "UFF"):
        calc = RDKitCalculator(mol_rdkit, forcefield=calc_type_upper)
    else:
        if not PYSCF_AVAILABLE:
            raise RuntimeError("PySCF is not available on this Windows host. Please run inside Linux/Docker.")
        calc = PySCFCalculator(method=calc_type_upper, functional=functional, basis=basis, charge=charge, spin=spin)
        
    atoms.calc = calc
    
    # Initialize velocities using Maxwell-Boltzmann Distribution at target T
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k)
    
    # Configure Dynamics Propagator
    dt = time_step_fs * units.fs
    if ensemble.upper() == "NVT":
        # Langevin thermostat with friction drag
        dyn = Langevin(atoms, timestep=dt, temperature_K=temperature_k, friction=0.002)
    else:
        # Microcanonical NVE VelocityVerlet
        dyn = VelocityVerlet(atoms, timestep=dt)
        
    energy_log = []
    trajectory_frames = []
    log_interval = max(1, steps // 50)  # Log up to 50 data points
    
    def log_md_step(current_step: int):
        pot = atoms.get_potential_energy()
        kin = atoms.get_kinetic_energy()
        tot = pot + kin
        temp = atoms.get_temperature()
        
        energy_log.append({
            "step": current_step,
            "time_fs": current_step * time_step_fs,
            "potential_energy_ev": float(pot),
            "kinetic_energy_ev": float(kin),
            "total_energy_ev": float(tot),
            "temperature_k": float(temp)
        })
        
        # Build multi-frame XYZ coordinate block
        frame_lines = [
            f"{len(atoms)}",
            f"Frame step={current_step} time={current_step*time_step_fs:.1f}fs E_tot={tot:.4f}eV T={temp:.1f}K"
        ]
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            frame_lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        trajectory_frames.append("\n".join(frame_lines))

    # Run dynamics loop
    for step in range(steps):
        dyn.step()
        if step % log_interval == 0 or step == steps - 1:
            log_md_step(step)
            if progress_callback:
                progress_callback(f"MD Progress: Step {step + 1}/{steps} (T = {atoms.get_temperature():.1f} K)")
                
    # Plotting: Generate figures
    steps_arr = [x["step"] for x in energy_log]
    times_arr = [x["time_fs"] for x in energy_log]
    pot_arr = [x["potential_energy_ev"] for x in energy_log]
    kin_arr = [x["kinetic_energy_ev"] for x in energy_log]
    tot_arr = [x["total_energy_ev"] for x in energy_log]
    temp_arr = [x["temperature_k"] for x in energy_log]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    
    # Panel 1: Energy profile
    ax1.plot(times_arr, pot_arr, label="Potential", color="#e74c3c", lw=1.2)
    ax1.plot(times_arr, kin_arr, label="Kinetic", color="#2ecc71", lw=1.2)
    ax1.plot(times_arr, tot_arr, label="Total", color="#2c3e50", lw=1.5)
    ax1.set_ylabel("Energy (eV)")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title(f"MD Simulation Profile ({ensemble})")
    
    # Panel 2: Temperature profile
    ax2.plot(times_arr, temp_arr, color="#3498db", lw=1.2)
    ax2.axhline(temperature_k, color="#7f8c8d", linestyle=":", label=f"Target T ({temperature_k} K)")
    ax2.set_ylabel("Temperature (K)")
    ax2.set_xlabel("Time (fs)")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    plot_bytes = buf.getvalue()
    
    trajectory_xyz = "\n".join(trajectory_frames)
    
    return {
        "ok": True,
        "results": {
            "ensemble": ensemble,
            "steps_run": steps,
            "time_step_fs": time_step_fs,
            "final_temperature_k": float(atoms.get_temperature()),
            "energy_history": energy_log
        },
        "trajectory_xyz": trajectory_xyz,
        "plot_bytes": plot_bytes,
        "warnings": []
    }
