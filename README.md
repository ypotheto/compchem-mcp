# Ypotheto Computational Chemistry MCP Server (`ypotheto-compchem-mcp`)

An MCP (Model Context Protocol) server that provides natural language interfaces and AI assistants with access to computational chemistry, cheminformatics, molecular modeling, engineering thermodynamics, reaction kinetics, materials science, and machine learning force fields.

Powered by **RDKit**, **PySCF**, **Atomic Simulation Environment (ASE)**, **Sella**, **Clapeyron.jl**, **Cantera**, and **PyTorch-based MLFFs** (CHGNet, MACE).

---

## Features & Capabilities

*   **Molecule Builder & Cheminformatics**: Convert SMILES strings to 3D optimized structures (MMFF94 or UFF), render 2D layouts (SVG), compute molecular descriptors (MW, LogP, TPSA), and evaluate Lipinski filters. `list_molecules`/`describe_molecule`/`delete_molecule` manage the workspace's stored molecule archive directly.
*   **Ab Initio Electronic Structure Theory**: Run Hartree-Fock (HF) and Density Functional Theory (DFT) calculations using PySCF. Retrieve potential energies, dipole moments, HOMO/LUMO gaps, and Mulliken charges.
*   **Affordable Semi-Empirical & Conformer Search**: Perform GFN-xTB calculations and execute conformer ensemble searches using CREST. Evaluates ensemble-averaged free energies and thermochemistry.
*   **Vibrational Spectroscopy & Molecular Dynamics**: Compute normal modes, frequencies, ZPE, and thermochemical corrections (Enthalpy, Entropy, Gibbs free energy). Simulates IR intensities (Lorentzian broadening plots) and runs Langevin or Verlet molecular dynamics.
*   **Engineering Thermodynamics**: Calculate mixture properties, vapor-liquid equilibria (VLE/LLE), bubble/dew points, azeotropes, and flash calculations using equation-of-state methods (via Clapeyron).
*   **Reaction Kinetics & Reactor Modeling**: Model chemical reactor networks, calculate ignition delays, solve constant-pressure/volume kinetics, and retrieve transport properties using Cantera.
*   **Polymers & Soft Matter**: Pack molecules into periodic cells with target densities using Packmol. Run classical MD simulations via LAMMPS and post-process trajectories (Radius of Gyration, Mean Squared Displacement, Radial Distribution Functions) using MDAnalysis.
*   **Transition States & Reaction Pathways**: Find first-order saddle-points (transition states) using the Sella optimizer. Trace minimum energy pathways (MEP) and activation barriers ($\Delta E^{\ddagger}$) using Nudged Elastic Band (NEB).
*   **Periodic DFT & Adsorption**: Construct surface slabs with custom Miller indices and vacuum spaces. Add molecular adsorbates onto surface sites (ontop, bridge, hollow) and run periodic calculations (PBC DFT or xTB).
*   **Machine Learning Force Fields (MLFF)**: Run fast geometry optimizations and molecular dynamics simulations using pre-trained neural network potentials (CHGNet, MACE).
*   **Advisor & Guidance Layer**: `recommend_workflow` chains the right tools together for a plain-language goal (e.g. "find the activation barrier"), tailored to a molecule's size; `explain_concept` looks up 30+ plain-language explanations of core concepts (basis sets, GFN2 vs. DFT, HSP, imaginary frequencies, etc.); guided MCP prompts (`compute_reaction_barrier`, `characterize_a_molecule`, `screen_solvent_compatibility`, `simulate_polymer_properties`) walk a client LLM through common multi-step workflows end-to-end.
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

Heavy backends are optional extras, not hard dependencies - a core-only install
starts the server and serves the RDKit/ASE tool subset (molecule building,
cheminformatics, descriptors). Any tool whose backend isn't installed returns a
`BACKEND_UNAVAILABLE` error naming the exact extra to add.

```bash
# Clone the repository
git clone <repo-url>
cd compchem-mcp

# Create virtual environment
uv venv

# Core install only (RDKit/ASE tools)
uv pip install -e .

# ...or pick the extras you need:
uv pip install -e ".[qm]"       # PySCF ab initio DFT/HF (+ cclib log parsing)
uv pip install -e ".[mlff]"     # CHGNet / MACE machine-learned force fields
uv pip install -e ".[thermo]"   # Cantera + Clapeyron.jl (via juliacall)
uv pip install -e ".[md]"       # MDAnalysis trajectory post-processing
uv pip install -e ".[ts]"       # Sella transition-state optimization
uv pip install -e ".[db]"       # PostgreSQL-backed durable job queue
uv pip install -e ".[s3]"       # DigitalOcean Spaces (S3-compatible) storage
uv pip install -e ".[all]"      # everything above

# Development tooling (ruff, mypy, pytest, moto)
uv pip install -e ".[dev]"
```

`xtb`, `CREST`, `Packmol`, and `LAMMPS` are external binaries, not pip
packages - the server auto-detects them via `shutil.which` and falls back
gracefully (or raises `BACKEND_UNAVAILABLE`) when they're missing. The
[Dockerfile](Dockerfile) installs all of the above, plus these binaries and a
Julia + Clapeyron.jl environment, and is the canonical way to get every
backend available at once.

### 3. Run the Tests
```bash
.\.venv\Scripts\python.exe -m pytest
```

Every heavy backend is mocked in the default test run above, so it never
verifies a single *real* calculation. A separate `@pytest.mark.integration`
tier (`tests/test_integration.py`) runs actual xtb/PySCF/CREST/Packmol/LAMMPS
calculations and checks results against known reference values (e.g. GFN2-xTB
water ≈ -5.070 Ha, HF/STO-3G water ≈ -74.96 Ha) - each test auto-skips if its
backend isn't installed, so it's safe to run anywhere:

```bash
pytest -m integration
```

Since most dev machines won't have every binary installed, the project Docker
image is the canonical venue for the full integration suite (it installs
everything - see below):

```bash
docker build --target test -t ypotheto-compchem-mcp:test .
docker run --rm ypotheto-compchem-mcp:test
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

## Authentication

Controlled by `COMPCHEM_AUTH_MODE`, checked live on every request:

* **`token`** (default): a single shared secret (`COMPCHEM_API_TOKEN`). Every caller that presents it lands in the same workspace derived from that token; unset entirely, auth is effectively open.
* **`none`**: no credential required at all, regardless of `COMPCHEM_API_TOKEN`. Whatever Bearer token a caller does supply (if any) still selects their own isolated workspace.
* **`keys`**: a per-tenant API-key table (`ypotheto_compchem_mcp.apikeys`), backed by SQLite (`{COMPCHEM_DATA_DIR}/keys.db`) by default or Postgres when `COMPCHEM_DATABASE_URL` is set. Manage keys with `python scripts/issue_key.py {issue,disable,list}`. Each key maps to its own workspace, hashed at rest (the raw key is only ever shown once, at issuance).
* **`oauth`**: OIDC resource-server mode (Kinde or any RS256-signing provider). Requires `COMPCHEM_OAUTH_ISSUER` and `COMPCHEM_OAUTH_AUDIENCE`; a valid token's `sub` claim resolves to a stable per-user workspace, and the required `COMPCHEM_OAUTH_REQUIRED_PERMISSION` must appear in its `permissions` claim. A missing/invalid token gets a `401` with a `WWW-Authenticate` header pointing at `/.well-known/oauth-protected-resource` (RFC 9728) so a client can discover where to authenticate.

---

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `COMPCHEM_API_TOKEN` | Shared-secret Bearer token required for authentication in `auth_mode=token` (the default) | `""` (Disabled) |
| `COMPCHEM_AUTH_MODE` | Authentication mode: `token` (single shared secret, `COMPCHEM_API_TOKEN`), `none` (no auth at all), `keys` (per-tenant API-key table, see below), or `oauth` (OIDC/Kinde-style resource server, see below) | `"token"` |
| `COMPCHEM_OAUTH_ISSUER` / `COMPCHEM_OAUTH_AUDIENCE` | OIDC provider base URL and this API's registered audience (required together when `auth_mode=oauth`) | unset |
| `COMPCHEM_OAUTH_REQUIRED_PERMISSION` | The `permissions` claim a valid token must carry | `"access:ypotheto-compchem-mcp"` |
| `COMPCHEM_DATA_DIR` | Directory on disk to store molecule structures and artifacts | `~/.compchem-mcp` |
| `COMPCHEM_PORT` | Port to run the HTTP/SSE server | `8348` |
| `COMPCHEM_PUBLIC_BASE_URL` | Base URL used to prefix artifact download links; also used to derive the allowed `Host` for DNS-rebinding protection | `http://localhost:8348` |
| `COMPCHEM_ALLOWED_ORIGINS` | Comma-separated CORS allowlist. Empty means no CORS headers at all (same-origin only) | `""` (none) |
| `COMPCHEM_REQUEST_TIMEOUT_SECONDS` | Hard timeout for a single POST tool-call request before returning `504`; never applies to GET (health check, artifact download, streamable-HTTP push) | `120` |
| `COMPCHEM_ARTIFACT_URL_EXPIRY_SECONDS` | Lifetime of a signed artifact download URL (`?exp=&sig=`) before it expires | `604800` (7 days) |
| `COMPCHEM_DATABASE_URL` | PostgreSQL connection string for the durable job queue, molecule archive, and (in `auth_mode=keys`) the API-key table (requires the `[db]` extra). Unset falls back to a local thread pool + on-disk job state, and a local SQLite file for API keys | `""` (disabled) |
| `COMPCHEM_SPACES_BUCKET` / `_ENDPOINT` / `_KEY` / `_SECRET` / `_REGION` / `_PREFIX` | DigitalOcean Spaces (S3-compatible) storage backend for artifacts (requires the `[s3]` extra). Unset falls back to local disk storage | unset (local disk) |

---

## Tool Catalog Overview

Generated from the actual tool registrations - regenerate with
`python scripts/gen_tool_catalog.py` (pipe into this section) whenever tools
are added or removed, so this table can't silently drift out of sync again.

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `add_adsorbate_to_surface` | `slab_molecule_id`, `adsorbate_molecule_id`, `height`, ... | Place a non-periodic adsorbate molecule onto a periodic surface slab. |
| `analyze_crystal_symmetry` | `molecule_id` | Perform deep crystallographic symmetry and space group analysis for a stored structure. |
| `analyze_md_trajectory` | `trajectory_file_id` | Analyze MD trajectory XYZ file to compute Radius of Gyration, RDF, and MSD. |
| `build_molecule_from_smiles` | `smiles`, `name` | Generate optimized 3D coordinates from a SMILES representation. |
| `build_polymer_chain` | `monomer_id`, `dp`, `tacticity`, ... | Assemble repeat units head-to-tail to form a 3D-relaxed polymer chain of specified length. |
| `build_surface_slab` | `bulk_molecule_id`, `miller_indices`, `layers`, ... | Generate a surface slab from bulk periodic crystal structure. |
| `calculate_descriptors` | `molecule_id` | Calculate molecular properties (descriptors) and Lipinski's Rule of Five compliance. |
| `calculate_hsp` | `molecule_id` | Calculate the Hansen Solubility Parameters (HSP) and Cohesive Energy Density (CED) for a stored molecule using the Hoftyzer-Van Krevelen (HVK) group contribution method. |
| `calculate_hsp_distance` | `molecule_id_1`, `molecule_id_2` | Calculate the Hansen Solubility Parameter (HSP) distance (Ra) between two stored molecules. |
| `calculate_transport_properties` | `components`, `mole_fractions`, `temperature_k`, ... | Calculate viscosity, thermal conductivity, and binary diffusion coefficients. |
| `calculate_vibrations` | `molecule_id`, `method`, `functional`, ... | Run vibrational frequency analysis and calculate thermochemistry corrections. |
| `delete_molecule` | `molecule_id` | Permanently delete a stored molecule's coordinates and metadata from this workspace. |
| `describe_molecule` | `molecule_id` | Retrieve stored metadata (name, formula, SMILES, atom count, method) for a molecule without loading its full 3D coordinates. |
| `enumerate_tautomers` | `molecule_id` | Enumerate all tautomeric forms for a stored molecule. |
| `estimate_calculation_time` | `molecule_id`, `method`, `basis` | Estimate the execution time for a quantum chemistry calculation before running it. |
| `explain_concept` | `concept` | Look up a short, plain-language explanation of a core computational chemistry concept (basis sets, DFT functionals, transition states, HSP, etc.); call with an empty string to list all available concepts. |
| `generate_supercell` | `molecule_id`, `sc_matrix`, `name` | Expand a unit cell periodic structure into a supercell. |
| `get_3d_coordinates` | `molecule_id`, `format` | Retrieve coordinate contents (SDF, XYZ, or PDB) of a stored molecule. |
| `get_job_status` | `job_id` | Check progress or fetch results of a background calculation job. |
| `import_periodic_structure` | `cif_content`, `name` | Import a periodic crystal structure from a CIF file. |
| `list_molecules` | None | List all molecules stored in the current workspace. |
| `optimize_geometry` | `molecule_id`, `method`, `functional`, ... | Relax molecule coordinates using ASE LBFGS optimizer coupled with PySCF energy/gradients. |
| `pack_amorphous_cell` | `molecule_ids`, `counts`, `density_g_cm3`, ... | Pack polymer chains and solvent molecules into a periodic box using Packmol. |
| `ping` | None | Check if the Ypotheto Computational Chemistry MCP Server is responsive. |
| `recommend_workflow` | `goal`, `molecule_id` | Recommend a chain of tool calls for a described computational-chemistry goal, with rationale for each step (deterministic keyword rules, not an LLM call); tailors to a molecule's size when `molecule_id` is given. |
| `register_monomer` | `smiles`, `name`, `head_idx`, ... | Register a monomer repeat unit, defining attachment connection points for polymer building. |
| `run_conformer_search` | `molecule_id`, `method`, `solvent`, ... | Generate conformer ensembles using CREST (Conformer-Rotamer Ensemble Sampling Tool). |
| `run_ensemble_thermochemistry` | `molecule_id`, `method`, `solvent`, ... | Run the Ensemble Thermochemistry Pipeline (enumerate -> optimize -> frequency-check -> Boltzmann rank). |
| `run_lammps_simulation` | `packed_molecule_id`, `steps`, `timestep_fs`, ... | Run classical MD simulation in LAMMPS (or ASE fallback). |
| `run_mixture_flash` | `components`, `mole_fractions`, `temperature_k`, ... | Perform flash equilibrium calculations for a mixture using Clapeyron.jl. |
| `run_mlff_molecular_dynamics` | `molecule_id`, `model_name`, `steps`, ... | Run classical MD simulations driven by MLFF forces. |
| `run_mlff_optimization` | `molecule_id`, `model_name`, `fmax`, ... | Optimize molecular or periodic structures using pre-trained Machine Learning Force Fields (MLFFs). |
| `run_molecular_dynamics` | `molecule_id`, `steps`, `time_step_fs`, ... | Run molecular dynamics (MD) simulations to study motion and thermal relaxation. |
| `run_neb_calculation` | `reactant_molecule_id`, `product_molecule_id`, `num_images`, ... | Optimize reaction pathway and energy barrier using Nudged Elastic Band (NEB). |
| `run_periodic_dft` | `molecule_id`, `kpts`, `method`, ... | Perform periodic DFT or semi-empirical GFN-xTB PBC energy calculations. |
| `run_pyscf_properties` | `molecule_id`, `method`, `functional`, ... | Perform advanced electronic structure calculations to compute properties like Mulliken and Loewdin populations, Electrostatic Potential (ESP) cubes, and HOMO/LUMO orbital cubes. |
| `run_reactor_kinetics` | `mechanism`, `initial_state`, `reactor_type`, ... | Simulate chemical kinetics and species concentrations over time using Cantera. |
| `run_scientific_preflight` | `molecule_id`, `method`, `basis`, ... | Validate molecule consistency and estimate calculation resources before submission. |
| `run_single_point` | `molecule_id`, `method`, `functional`, ... | Compute single-point energy, dipole moments, HOMO/LUMO energies, and Mulliken charges. |
| `run_transition_state_search` | `molecule_id`, `method`, `functional`, ... | Perform a transition state search (first-order saddle point) using the Sella optimizer. |
| `run_xtb_calculation` | `molecule_id`, `task`, `method`, ... | Run fast semi-empirical GFN-xTB calculations. |
| `save_conformer_as_molecule` | `parent_molecule_id`, `rdkit_conformer_id`, `name` | Extract a single conformer from a search result and save it as a new molecule in the workspace. |
| `search_conformers` | `molecule_id`, `num_conformers`, `rmsd_threshold` | Generate multiple conformers for a molecule, relax them, prune duplicates, and rank them by forcefield energy and Boltzmann populations. |
| `simulate_ir_spectrum` | `molecule_id`, `method`, `functional`, ... | Simulate IR intensities and generate a Lorentzian IR spectrum plot. |
| `standardize_molecule` | `smiles_or_sdf`, `strip_salts`, `neutralize`, ... | Standardize a molecule: parses structure, strips salts, neutralizes formal charge, canonicalizes tautomers, and sanitizes/minimizes the output. |
