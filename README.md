# Ypotheto Computational Chemistry MCP Server (`ypotheto-compchem-mcp`)

An MCP (Model Context Protocol) server that provides natural language interfaces and AI assistants with access to computational chemistry, cheminformatics, molecular modeling, engineering thermodynamics, reaction kinetics, materials science, and machine learning force fields.

Powered by **RDKit**, **PySCF**, **Atomic Simulation Environment (ASE)**, **Sella**, **Clapeyron.jl**, **Cantera**, and **PyTorch-based MLFFs** (CHGNet, MACE).

---

## Features & Capabilities

*   **Molecule Builder & Cheminformatics**: Convert SMILES strings to 3D optimized structures (MMFF94 or UFF), render 2D layouts (SVG), compute molecular descriptors (MW, LogP, TPSA), and evaluate Lipinski filters.
*   **Ab Initio Electronic Structure Theory**: Run Hartree-Fock (HF) and Density Functional Theory (DFT) calculations using PySCF. Retrieve potential energies, dipole moments, HOMO/LUMO gaps, and Mulliken charges.
*   **Affordable Semi-Empirical & Conformer Search**: Perform GFN-xTB calculations and execute conformer ensemble searches using CREST. Evaluates ensemble-averaged free energies and thermochemistry.
*   **Vibrational Spectroscopy & Molecular Dynamics**: Compute normal modes, frequencies, ZPE, and thermochemical corrections (Enthalpy, Entropy, Gibbs free energy). Simulates IR intensities (Lorentzian broadening plots) and runs Langevin or Verlet molecular dynamics.
*   **Engineering Thermodynamics**: Calculate mixture properties, vapor-liquid equilibria (VLE/LLE), bubble/dew points, azeotropes, and flash calculations using equation-of-state methods (via Clapeyron).
*   **Reaction Kinetics & Reactor Modeling**: Model chemical reactor networks, calculate ignition delays, solve constant-pressure/volume kinetics, and retrieve transport properties using Cantera.
*   **Polymers & Soft Matter**: Pack molecules into periodic cells with target densities using Packmol. Run classical MD simulations via LAMMPS and post-process trajectories (Radius of Gyration, Mean Squared Displacement, Radial Distribution Functions) using MDAnalysis.
*   **Transition States & Reaction Pathways**: Find first-order saddle-points (transition states) using the Sella optimizer. Trace minimum energy pathways (MEP) and activation barriers ($\Delta E^{\ddagger}$) using Nudged Elastic Band (NEB).
*   **Periodic DFT & Adsorption**: Construct surface slabs with custom Miller indices and vacuum spaces. Add molecular adsorbates onto surface sites (ontop, bridge, hollow) and run periodic calculations (PBC DFT or xTB).
*   **Machine Learning Force Fields (MLFF)**: Run fast geometry optimizations and molecular dynamics simulations using pre-trained neural network potentials (CHGNet, MACE).
*   **Asynchronous Job Management**: Heavy computations run in background threads using a persistent job manager, avoiding client/LLM timeout issues.
*   **File-First Artifacts**: Visual plots, SVG diagrams, and coordinate files (SDF, XYZ, CIF, PDB) are written to a local workspace directory and returned as public URLs.

---

## Local Development & Setup

This project uses **`uv`** for virtualenv and dependency management.

### 1. Prerequisites
*   Python `>= 3.11`
*   Windows, macOS, or Linux.
*   *Note: Heavy dependencies (like PySCF, Packmol, LAMMPS) have robust fallback mechanisms in the code. If a binary is missing or incompatible on a specific platform, the server automatically routes calculations to classical/semi-empirical fallbacks, ensuring portability.*

### 2. Install Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd compchem-mcp

# Create virtual environment and install dependencies
uv venv
uv pip install -e .
```

### 3. Run the Tests
```bash
.\.venv\Scripts\python.exe -m pytest
```

### 4. Run the Server
The server supports two transport mechanisms: **STDIO** (default) for desktop client plugins and **HTTP** (SSE/Streamable HTTP) for web applications.

```bash
# Start STDIO mode
.\.venv\Scripts\ypotheto-compchem-mcp --transport stdio

# Start HTTP mode
.\.venv\Scripts\ypotheto-compchem-mcp --transport http --port 8348
```

---

## Running in Docker

To run inside a container environment containing pre-compiled binaries:

```bash
# Build the Docker image
docker build -t ypotheto-compchem-mcp .

# Run the container (binds workspace data to a local directory)
docker run -p 8348:8348 -v C:/data/compchem:/data ypotheto-compchem-mcp
```

---

## Client Integration Configuration

### 1. Claude Desktop (STDIO)
Add the server to your `claude_desktop_config.json` configuration file:

```json
{
  "mcpServers": {
    "ypotheto-compchem": {
      "command": "C:/Users/<your-user>/PycharmProjects/compchem-mcp/.venv/Scripts/ypotheto-compchem-mcp.exe",
      "args": [
        "--transport",
        "stdio"
      ],
      "env": {
        "COMPCHEM_DATA_DIR": "C:/Users/<your-user>/.compchem-mcp"
      }
    }
  }
}
```

### 2. SSE Web Client
To connect a remote client over streamable HTTP with Bearer token authentication:

1. Start the server:
   ```bash
   COMPCHEM_API_TOKEN="secret_api_token" ypotheto-compchem-mcp --transport http --port 8348
   ```
2. Configure your client to connect to `http://localhost:8348/mcp` with the header `Authorization: Bearer secret_api_token`.

---

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `COMPCHEM_API_TOKEN` | Bearer token required for authentication | `""` (Disabled) |
| `COMPCHEM_DATA_DIR` | Directory on disk to store molecule structures and artifacts | `~/.compchem-mcp` |
| `COMPCHEM_PORT` | Port to run the HTTP/SSE server | `8348` |
| `COMPCHEM_PUBLIC_BASE_URL` | Base URL used to prefix artifact download links | `http://localhost:8348` |

---

## Tool Catalog Overview

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `ping` | None | Verify server connectivity. |
| `build_molecule_from_smiles` | `smiles`, `name` | Create a 3D model and 2D diagram from SMILES. |
| `get_3d_coordinates` | `molecule_id`, `format` | Export XYZ or SDF molecular coordinates. |
| `calculate_descriptors` | `molecule_id` | Compute MW, LogP, TPSA, and Lipinski filter. |
| `estimate_calculation_time` | `molecule_id`, `method`, `basis` | Estimate duration of a calculation. |
| `run_single_point` | `molecule_id`, `method`, ... | Run DFT/HF energy, HOMO/LUMO, and dipole moments. |
| `optimize_geometry` | `molecule_id`, `method`, ... | Relax molecular coordinates. |
| `calculate_vibrations` | `molecule_id`, `method`, ... | Run frequency analysis and thermochemical corrections. |
| `simulate_ir_spectrum` | `molecule_id`, `method`, ... | Simulates vibrational IR spectrum plot. |
| `run_molecular_dynamics` | `molecule_id`, `steps`, ... | Run Langevin or Verlet molecular dynamics. |
| `run_xtb_calculation` | `molecule_id`, `method`, ... | Run GFN-xTB calculations. |
| `run_conformer_search` | `molecule_id`, `method`, ... | Run CREST conformer ensemble searches. |
| `run_ensemble_thermochemistry`| `molecule_id`, `method`, ... | Evaluates conformer ensemble-averaged free energies. |
| `run_mixture_flash` | `compounds`, `fractions`, ... | Run phase equilibrium and flash calculations (Clapeyron). |
| `run_reactor_kinetics` | `species`, `reactor_type`, ... | Model reactor kinetic networks (Cantera). |
| `pack_amorphous_cell` | `compounds`, `density`, ... | Pack molecules into periodic cells (Packmol). |
| `run_lammps_simulation` | `molecule_id`, `steps`, ... | Run classical periodic molecular dynamics (LAMMPS). |
| `analyze_md_trajectory` | `trajectory_url`, `analysis_type` | Post-process trajectory files (MDAnalysis). |
| `run_transition_state_search` | `molecule_id`, `method`, ... | Optimize reaction saddle-point transition states (Sella). |
| `run_neb_calculation` | `initial_id`, `final_id`, ... | Optimize reaction pathway and barrier (ASE NEB). |
| `build_surface_slab` | `bulk_molecule_id`, ... | Slice bulk crystals into surface slabs (ASE). |
| `add_adsorbate_to_surface` | `slab_id`, `adsorbate_id`, ... | Place adsorbates onto surfaces (ASE). |
| `run_periodic_dft` | `molecule_id`, `kpts`, ... | Perform periodic boundary DFT or xTB calculations. |
| `run_mlff_optimization` | `molecule_id`, `model_name` | Optimize structures using pre-trained MLFFs (CHGNet/MACE). |
| `run_mlff_molecular_dynamics` | `molecule_id`, `model_name` | Run MD simulations using MLFF force fields. |
| `get_job_status` | `job_id` | Poll background calculation status and get results. |
