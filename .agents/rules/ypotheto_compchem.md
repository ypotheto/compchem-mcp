# Ypotheto Computational Chemistry MCP Server Rules

These custom instructions must guide all development and styling decisions for the `ypotheto-compchem-mcp` project.

## 1. Branding & Context
*   Brand: Always use the **Ypotheto** brand name (e.g. `ypotheto-compchem-mcp` or `ypotheto_compchem_mcp` for python naming).
*   Domain: Natural language access to computational chemistry, cheminformatics, and atomistic/molecular modeling tools.

## 2. Technology Stack & Design Decisions
*   **MCP Server Framework**: `FastMCP` (python-mcp SDK) for tool definition and registration.
*   **HTTP Transport & Artifact Serving**: `Starlette` web framework for hosting the HTTP server, handling SSE connections, and serving the static files stored in the Artifact Store.
*   **Cheminformatics**: RDKit (`rdkit`) for molecular structure generation from SMILES, molecular drawings (2D representations), and descriptor calculations.
*   **Quantum Chemistry**: PySCF (`pyscf`) for electronic structure calculations (Hartree-Fock, Density Functional Theory, electronic property analysis).
*   **Simulation & Optimization**: ASE (`ase`) to handle atomic configuration structures, coordinate optimization (BFGS, LBFGS), molecular dynamics (NVE/NVT ensembles), and vibrational/harmonic analysis.
*   **Plotting/Graphing**: Matplotlib (`matplotlib`) for plotting spectra (IR), optimization convergence trajectories, or MD thermodynamic profiles.

## 3. Planning & Documentation
*   **ALL PLAN DOCUMENTS** we generate must be stored in the `planning/` folder at the root of the repository (e.g., `planning/my_plan_name.md`).
*   This folder is gitignored to avoid cluttering version control.

## 4. Coding & Execution Constraints
*   **No stdout logging in STDIO mode**: In standard input/output execution, do not print or log to standard output (`sys.stdout`) as it corrupts JSON-RPC frames. Use python logging to `stderr`.
*   **File-first responses**: Return file paths or URLs for large files (XYZ, plots, reports) using the Artifact Store rather than outputting massive text strings.
*   **Modular architecture**: Keep `server.py` as a lightweight register of tools, delegating all complex logic to modules under `src/ypotheto_compchem_mcp/workflows/`.

## 5. Versioning & Git Commits
*   **Version Bump & Tagging**: Whenever committing changes to the codebase, we must increment the package version (major, minor, or patch according to semver) in `pyproject.toml` and `src/ypotheto_compchem_mcp/__init__.py`.
*   **Git Tag**: Create an annotated git tag in the format `vx.x.x` matching the new version and push the tag along with the commit.
*   **Automation Script**: Use `scripts/bump_version.py` to automate version incrementing, file updates, staging, committing, and tagging.
