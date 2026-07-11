# Ypotheto Computational Chemistry MCP Server (`ypotheto-compchem-mcp`)

An MCP (Model Context Protocol) server that provides natural language interfaces and AI assistants with access to computational chemistry, cheminformatics, and molecular modeling tools.

Powered by **RDKit**, **PySCF**, and the **Atomic Simulation Environment (ASE)**.

---

## Features

*   **Molecule Builder**: Convert SMILES strings to 3D optimized geometries (MMFF94 or UFF) and render 2D layout diagrams (SVG).
*   **Cheminformatics**: Calculate molecular descriptors (Molecular Weight, LogP, TPSA, Rotatable Bonds) and evaluate compliance with Lipinski's Rule of Five.
*   **Electronic Structure Theory**: Run Hartree-Fock (HF) and Density Functional Theory (DFT) single-point calculations using PySCF. Retrieve total energies, dipole moments, HOMO/LUMO band gaps, and Mulliken atomic charges.
*   **Geometry Optimization**: Relax molecular structures to energy minima using ASE's LBFGS optimizer coupled with PySCF or RDKit force fields.
*   **Vibrational Spectroscopy**: Run harmonic vibrational analysis to determine normal modes, frequencies, Zero-Point Energy (ZPE), and thermochemical corrections (Enthalpy, Entropy, Gibbs energy). Simulates IR intensities and generates Lorentzian line-broadened IR spectra plots.
*   **Molecular Dynamics (MD)**: Propagate NVE (VelocityVerlet) or NVT (Langevin) trajectories. Saves multi-frame XYZ files and renders energy/temperature conservation plots.
*   **Asynchronous Job Manager**: Long-running calculations (like DFT optimizations or dynamics) run in background threads with progress updates, avoiding client timeouts.
*   **File-First Artifacts**: Visual outputs (plots, diagrams) and heavy coordinate files (SDF, XYZ, PDB) are written to a local data directory and returned as public URLs instead of bloating the LLM's chat context.

---

## Local Development & Setup

This project uses **`uv`** for lightning-fast virtualenv and dependency management.

### 1. Prerequisites
*   Python `>= 3.11`
*   Windows, macOS, or Linux.
*   *Note: PySCF requires C compilation libraries. If running natively on Windows without WSL/Conda, PySCF tools will automatically report as unavailable, but the RDKit builder, descriptors, and forcefield-based optimizations/dynamics/vibrations will still run perfectly.*

### 2. Install Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd compchem-mcp

# Create a virtual environment and install in editable mode
uv venv
uv pip install -e .

# (Optional) Install pytest to run tests
uv pip install pytest
```

### 3. Run the Tests
```bash
.\.venv\Scripts\python.exe -m pytest
```

### 4. Run the Server
The server supports two transport mechanisms: **STDIO** (default) for desktop client plugins and **HTTP** (SSE/Streamable HTTP) for hosted web apps.

```bash
# Start STDIO mode
.\.venv\Scripts\python.exe -m ypotheto_compchem_mcp --transport stdio

# Start HTTP mode
.\.venv\Scripts\python.exe -m ypotheto_compchem_mcp --transport http --port 8348
```

---

## Running in Docker

To run full PySCF quantum calculations natively on Windows or deploy to DigitalOcean, compile and run inside Docker:

```bash
# Build the Docker image
docker build -t ypotheto-compchem-mcp .

# Run the container (binds workspace data to a local directory)
docker run -p 8348:8348 -v C:/data/compchem:/data ypotheto-compchem-mcp
```

---

## Client Integration Configuration

### 1. Claude Desktop (STDIO)
To register the server directly with Claude Desktop, add it to your `claude_desktop_config.json` configuration file:

```json
{
  "mcpServers": {
    "ypotheto-compchem": {
      "command": "python",
      "args": [
        "-m",
        "ypotheto_compchem_mcp",
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

### 2. ypotheto-core / SSE Web Client
To connect a remote client over streamable HTTP with Bearer token authentication:

1. Start the server:
   ```bash
   COMPCHEM_API_TOKEN="secret_api_token" python -m ypotheto_compchem_mcp --transport http --port 8348
   ```
2. Configure your client (e.g. `StreamableHttpMcpClient`) to connect to:
   *   URL: `http://localhost:8348/mcp`
   *   Headers: `Authorization: Bearer secret_api_token`

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
| `run_single_point` | `molecule_id`, `method`, `functional`, `basis`, `charge`, `spin`, `run_async` | Run DFT/HF energy, HOMO/LUMO, dipole moments, and atomic charges. |
| `optimize_geometry` | `molecule_id`, `method`, `functional`, `basis`, `charge`, `spin`, `max_steps`, `run_async` | Relax molecular coordinates. |
| `calculate_vibrations` | `molecule_id`, `method`, `functional`, `basis`, `charge`, `spin`, `run_async` | Run frequency analysis and thermochemical corrections. |
| `simulate_ir_spectrum` | `molecule_id`, `method`, `functional`, `basis`, `charge`, `spin`, `run_async` | Simulates vibrational IR spectrum plot. |
| `run_molecular_dynamics` | `molecule_id`, `steps`, `time_step_fs`, `temperature_k`, `ensemble`, `calculator_type`, `run_async` | Run Langevin or Verlet molecular dynamics. |
| `get_job_status` | `job_id` | Poll background calculation status and get results. |
