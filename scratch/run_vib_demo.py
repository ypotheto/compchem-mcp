import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import packages
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ypotheto_compchem_mcp.workspace import workspace_manager, current_workspace_id
from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.chemistry.vib_engine import run_vibrations_engine, simulate_ir_spectrum_engine

def main():
    # Setup the workspace ID using the ContextVar (exactly like the HTTP middleware does)
    workspace_id = "demo_workspace"
    current_workspace_id.set(workspace_id)
    
    workspace_dir = workspace_manager.get_workspace_dir(workspace_id)
    print(f"Using workspace directory: {workspace_dir}")
    
    # 1. Build Water (H2O) structure
    print("\n1. Building Water (H2O) structure...")
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    print(f"Successfully built H2O molecule. Molecule ID = {molecule_id}")
    print(f"Formula: {mol_res['formula']}, Atoms: {mol_res['num_atoms']}")
    
    # 2. Run Vibrations calculation using MMFF94 force field
    print("\n2. Running vibrational frequency analysis...")
    vib_res = run_vibrations_engine(workspace_id, molecule_id, method="MMFF94")
    
    results = vib_res["results"]
    print(f"Zero-Point Energy (ZPE): {results['zero_point_energy_ev']:.4f} eV ({results['zero_point_energy_kcal']:.2f} kcal/mol)")
    print(f"Number of vibrational frequencies calculated: {len(results['frequencies_cm1'])}")
    print("Harmonic Frequencies (cm^-1):")
    for i, freq in enumerate(results["frequencies_cm1"]):
        print(f"  Mode {i+1:2d}: {freq:8.2f} cm^-1")
        
    print("\n3. Calculating thermochemistry corrections (T = 298.15 K, P = 1 atm)...")
    thermo = results["thermochemistry"]
    print(f"  Enthalpy (H):             {thermo['enthalpy_ev']:.4f} eV")
    print(f"  Entropy (S):              {thermo['entropy_ev_k']:.6f} eV/K")
    print(f"  Gibbs Free Energy (G):    {thermo['gibbs_free_energy_ev']:.4f} eV")
    
    # 3. Simulate IR Spectrum plot
    print("\n4. Simulating IR Spectrum...")
    ir_res = simulate_ir_spectrum_engine(workspace_id, molecule_id, method="MMFF94")
    plot_bytes = ir_res["plot_bytes"]
    
    output_png = project_root / "scratch" / "water_ir_spectrum.png"
    output_png.parent.mkdir(exist_ok=True)
    output_png.write_bytes(plot_bytes)
    print(f"Successfully simulated IR spectrum. Saved plot to: {output_png}")

if __name__ == "__main__":
    main()
