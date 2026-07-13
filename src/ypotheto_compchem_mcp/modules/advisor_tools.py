import importlib.resources
from functools import lru_cache
from typing import Any

import yaml
from mcp.server.fastmcp import FastMCP

from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
from ypotheto_compchem_mcp.chemistry.qm_engine import estimate_time_seconds
from ypotheto_compchem_mcp.envelope import make_success_response, mcp_tool_decorator
from ypotheto_compchem_mcp.errors import ValidationError
from ypotheto_compchem_mcp.workspace import get_workspace_id

# Slow-DFT threshold (seconds) above which recommend_workflow suggests a
# cheaper xTB/MLFF pre-screen instead of jumping straight to DFT, when a
# molecule_id is supplied. Matches the sync/async cutoff philosophy of
# estimate_calculation_time but set higher since this is advisory, not a
# hard routing decision.
_SLOW_DFT_THRESHOLD_SECONDS = 60


@lru_cache(maxsize=1)
def _load_concepts() -> dict[str, dict[str, str]]:
    resource = importlib.resources.files("ypotheto_compchem_mcp.content").joinpath("concepts.yaml")
    return yaml.safe_load(resource.read_text(encoding="utf-8"))


# Each rule: (keywords, category, chain-of-steps, caveats). Checked in this
# fixed priority order against the lowercased goal string; every rule whose
# keywords match is included, in this order, so a goal matching an earlier
# rule ranks that chain first. Tool names below are verified against the
# actual registered @mcp.tool() names (see scripts/gen_tool_catalog.py).
_WORKFLOW_RULES: list[tuple[list[str], str, list[dict[str, Any]], list[str]]] = [
    (
        ["barrier", "transition state", "reaction rate", "activation energy", "ts search"],
        "activation_barrier_workflow",
        [
            {
                "step": 1,
                "tool": "build_molecule_from_smiles",
                "why": "Build 3D structures for the reactant and product endpoints (call once per endpoint).",
                "typical_args": {"smiles": "<endpoint SMILES>", "name": "<endpoint name>"},
            },
            {
                "step": 2,
                "tool": "run_scientific_preflight",
                "why": "Catch valence/charge/multiplicity problems on each built structure before spending compute.",
                "typical_args": {"molecule_id": "<from step 1>"},
            },
            {
                "step": 3,
                "tool": "optimize_geometry",
                "why": "Relax both the reactant and product endpoints to their nearest energy minimum.",
                "typical_args": {"molecule_id": "<reactant_id or product_id>"},
            },
            {
                "step": 4,
                "tool": "run_neb_calculation",
                "why": (
                    "Locate the connecting transition state from the two optimized endpoints. Use "
                    "run_transition_state_search instead if you already have a good guess structure for the TS."
                ),
                "typical_args": {"reactant_molecule_id": "<from step 3>", "product_molecule_id": "<from step 3>"},
            },
            {
                "step": 5,
                "tool": "calculate_vibrations",
                "why": "Confirm the located structure is a genuine TS (exactly one imaginary frequency) before trusting the barrier.",
                "typical_args": {"molecule_id": "<ts molecule_id from step 4>"},
            },
        ],
        [
            "Barriers should be compared using Gibbs free energy (ZPE + thermal corrections), not raw "
            "electronic energy - see explain_concept('gibbs_thermochemistry').",
            "See explain_concept('imaginary_frequencies') for how to interpret step 5's result.",
        ],
    ),
    (
        ["conformer", "flexible", "ensemble", "rotamer"],
        "conformer_ensemble_workflow",
        [
            {
                "step": 1,
                "tool": "build_molecule_from_smiles",
                "why": "Build the initial 3D structure.",
                "typical_args": {"smiles": "<SMILES>", "name": "<molecule name>"},
            },
            {
                "step": 2,
                "tool": "run_conformer_search",
                "why": "Generate a representative set of low-energy 3D conformers (search_conformers is a lighter RDKit-only alternative).",
                "typical_args": {"molecule_id": "<from step 1>"},
            },
            {
                "step": 3,
                "tool": "run_ensemble_thermochemistry",
                "why": "Boltzmann-weight the ensemble and get population-averaged thermochemistry rather than a single-conformer answer.",
                "typical_args": {"molecule_id": "<from step 1>"},
            },
        ],
        [
            "See explain_concept('boltzmann_weighting') - a single lowest-energy conformer can be a "
            "biased answer for flexible molecules."
        ],
    ),
    (
        ["ir spectrum", "infrared", "spectrum", "vibrational frequencies", "frequencies"],
        "vibrational_spectrum_workflow",
        [
            {
                "step": 1,
                "tool": "optimize_geometry",
                "why": "A vibrational analysis is only valid at a converged energy minimum.",
                "typical_args": {"molecule_id": "<molecule_id>"},
            },
            {
                "step": 2,
                "tool": "calculate_vibrations",
                "why": "Compute the harmonic vibrational frequencies and confirm zero imaginary frequencies (a genuine minimum).",
                "typical_args": {"molecule_id": "<from step 1>"},
            },
            {
                "step": 3,
                "tool": "simulate_ir_spectrum",
                "why": "Convert the vibrational data into a simulated IR spectrum.",
                "typical_args": {"molecule_id": "<from step 1>"},
            },
        ],
        [
            "Harmonic frequencies are systematically off from experiment by an amount that depends on "
            "the method/basis - a scaling factor is often applied before comparing to measured spectra."
        ],
    ),
    (
        ["solubility", "miscib", "polymer compatibility", "solvent"],
        "solubility_screening_workflow",
        [
            {
                "step": 1,
                "tool": "build_molecule_from_smiles",
                "why": "Build 3D structures for the solute and each candidate solvent.",
                "typical_args": {"smiles": "<SMILES>", "name": "<name>"},
            },
            {
                "step": 2,
                "tool": "calculate_hsp",
                "why": "Compute Hansen Solubility Parameters for the solute and every candidate.",
                "typical_args": {"molecule_id": "<from step 1>"},
            },
            {
                "step": 3,
                "tool": "calculate_hsp_distance",
                "why": "Rank candidates by Ra distance to the solute - smaller Ra means better predicted compatibility.",
                "typical_args": {"molecule_id_1": "<solute molecule_id>", "molecule_id_2": "<candidate molecule_id>"},
            },
        ],
        [
            "See explain_concept('hansen_solubility') and explain_concept('ra_distance') for how to "
            "interpret the numbers."
        ],
    ),
    (
        ["adsorption", "surface", "catalyst", "catalysis"],
        "surface_adsorption_workflow",
        [
            {
                "step": 1,
                "tool": "import_periodic_structure",
                "why": "Import the bulk crystal structure (e.g. a CIF file).",
                "typical_args": {"cif_content": "<CIF text>"},
            },
            {
                "step": 2,
                "tool": "build_surface_slab",
                "why": "Cut a finite-thickness slab of the desired facet (Miller index) with a vacuum gap.",
                "typical_args": {"bulk_molecule_id": "<from step 1>", "miller_indices": "<e.g. [1, 1, 1]>"},
            },
            {
                "step": 3,
                "tool": "add_adsorbate_to_surface",
                "why": "Place the adsorbate molecule at a candidate site on the slab.",
                "typical_args": {"slab_molecule_id": "<from step 2>", "adsorbate_molecule_id": "<adsorbate>"},
            },
            {
                "step": 4,
                "tool": "run_periodic_dft",
                "why": "Compute the adsorption energy with periodic DFT.",
                "typical_args": {"molecule_id": "<from step 3>"},
            },
        ],
        [
            "Check k-point and slab-thickness convergence before trusting an absolute binding energy - "
            "see explain_concept('k_points') and explain_concept('surface_slabs').",
            "Consider sampling more than one adsorption site - see explain_concept('adsorption_sites').",
        ],
    ),
    (
        ["phase equilibrium", "boiling", "azeotrope", "flash", "vle"],
        "phase_equilibrium_workflow",
        [
            {
                "step": 1,
                "tool": "run_mixture_flash",
                "why": "Compute the vapor-liquid split for the mixture at the given temperature/pressure/composition.",
                "typical_args": {"components": "<list of species>", "mole_fractions": "<list of fractions>"},
            },
        ],
        [
            "See explain_concept('vle_flash') - result quality depends entirely on the underlying "
            "equation-of-state/activity model."
        ],
    ),
    (
        ["combustion", "ignition", "reactor", "kinetics", "mechanism"],
        "reactor_kinetics_workflow",
        [
            {
                "step": 1,
                "tool": "run_reactor_kinetics",
                "why": "Simulate the reactor (batch/PFR/etc.) with the chosen kinetic mechanism.",
                "typical_args": {"mechanism": "<mechanism file/name>", "reactor_type": "<batch|pfr|...>"},
            },
        ],
        [],
    ),
    (
        ["polymer", "amorphous", "glass", "chain packing"],
        "polymer_simulation_workflow",
        [
            {
                "step": 1,
                "tool": "register_monomer",
                "why": "Register the repeat-unit monomer structure.",
                "typical_args": {"smiles": "<monomer SMILES>"},
            },
            {
                "step": 2,
                "tool": "build_polymer_chain",
                "why": "Build a single chain of the requested length from the registered monomer.",
                "typical_args": {"monomer_id": "<from step 1>", "count": "<repeat units>"},
            },
            {
                "step": 3,
                "tool": "pack_amorphous_cell",
                "why": "Pack multiple chains into a periodic amorphous cell at a reasonable density.",
                "typical_args": {"molecule_id": "<from step 2>"},
            },
            {
                "step": 4,
                "tool": "run_lammps_simulation",
                "why": "Equilibrate the packed cell (NVT is usually the right ensemble - see explain_concept('md_ensembles')).",
                "typical_args": {"packed_molecule_id": "<from step 3>"},
            },
            {
                "step": 5,
                "tool": "analyze_md_trajectory",
                "why": "Check radius_of_gyration/RDF/MSD for equilibration before trusting any measured property.",
                "typical_args": {"trajectory_file_id": "<from step 4>"},
            },
        ],
        [
            "See explain_concept('radius_of_gyration'), explain_concept('rdf'), and explain_concept('msd') "
            "for how to judge whether step 5's trajectory has actually equilibrated."
        ],
    ),
    (
        ["drug", "screening", "descriptor", "lipinski", "bioavailab"],
        "druglike_screening_workflow",
        [
            {
                "step": 1,
                "tool": "build_molecule_from_smiles",
                "why": "Build the 3D structure.",
                "typical_args": {"smiles": "<SMILES>", "name": "<name>"},
            },
            {
                "step": 2,
                "tool": "standardize_molecule",
                "why": "Normalize tautomer/charge/salt state before computing descriptors.",
                "typical_args": {"molecule_id": "<from step 1>"},
            },
            {
                "step": 3,
                "tool": "calculate_descriptors",
                "why": "Compute physicochemical descriptors, including a Lipinski's Rule of Five check.",
                "typical_args": {"molecule_id": "<from step 2>"},
            },
        ],
        [
            "See explain_concept('lipinski_rule_of_five') - this is a cheap heuristic filter, not a "
            "rigorous ADMET prediction."
        ],
    ),
]

_DEFAULT_RECOMMENDATION: dict[str, Any] = {
    "goal_category": "general_starting_point",
    "why": "No specific keyword pattern matched this goal - it may be unclear or span multiple tool families.",
    "steps": [
        {
            "step": 1,
            "tool": "run_scientific_preflight",
            "why": "Sanity-check any molecule you already have before running anything expensive on it.",
            "typical_args": {"molecule_id": "<molecule_id>"},
        },
        {
            "step": 2,
            "tool": "explain_concept",
            "why": "List available concepts (call with an empty string) to see what guidance is available.",
            "typical_args": {"concept": ""},
        },
    ],
    "caveats": [
        "Tool families available: molecule building/standardization, QM (xTB/PySCF), vibrations/IR, "
        "conformers/ensembles, transition-state search, periodic DFT/surfaces, solubility (HSP), "
        "reactor kinetics, phase equilibrium, polymers/MD."
    ],
}


def _recommend(goal: str, workspace_id: str, molecule_id: str | None) -> list[dict[str, Any]]:
    goal_lower = goal.lower()
    recommendations = []
    for keywords, category, steps, caveats in _WORKFLOW_RULES:
        if any(kw in goal_lower for kw in keywords):
            recommendations.append({"goal_category": category, "steps": steps, "caveats": list(caveats)})

    if not recommendations:
        recommendations = [dict(_DEFAULT_RECOMMENDATION, caveats=list(_DEFAULT_RECOMMENDATION["caveats"]))]

    if molecule_id is not None:
        _tailor_to_molecule(recommendations[0], workspace_id, molecule_id)

    return recommendations


def _tailor_to_molecule(top_recommendation: dict[str, Any], workspace_id: str, molecule_id: str) -> None:
    """Add a molecule-size caveat to the top recommendation when a real molecule_id is
    given and it looks too large for a routine DFT step. Advisory only - a bad/unknown
    molecule_id should not make recommend_workflow itself fail, so lookup errors are
    swallowed rather than raised."""
    try:
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
        natoms = mol.GetNumAtoms()
        est_seconds = estimate_time_seconds(workspace_id, molecule_id, "DFT", "sto-3g")
    except Exception:
        return

    if est_seconds >= _SLOW_DFT_THRESHOLD_SECONDS:
        top_recommendation["caveats"].append(
            f"{molecule_id} has {natoms} atoms; a DFT/STO-3G single point is estimated to take "
            f"~{est_seconds}s. Consider run_xtb_calculation or an MLFF tool (run_mlff_optimization) "
            "for a fast pre-screen before committing to DFT - see explain_concept('gfn2_vs_dft') and "
            "explain_concept('when_to_trust_mlff')."
        )


@mcp_tool_decorator
def recommend_workflow(goal: str, molecule_id: str | None = None) -> dict:
    """
    Recommend a chain of tool calls for a described computational-chemistry goal, with
    rationale for each step. Deterministic keyword rules, not an LLM call - pass
    molecule_id so the recommendation can be tailored to that molecule's size (e.g.
    suggesting a fast semi-empirical/MLFF pre-screen over DFT for a large molecule).

    Parameters:
    - goal: A plain-language description of what you're trying to accomplish (e.g.
      "find the activation barrier for my reaction").
    - molecule_id: Optional stored molecule handle to tailor the top recommendation to.
    """
    workspace_id = get_workspace_id()
    recommendations = _recommend(goal, workspace_id, molecule_id)
    top = recommendations[0]

    interpretation = (
        f"Top recommendation: {top['goal_category']} ({len(top['steps'])} step"
        f"{'s' if len(top['steps']) != 1 else ''}), starting with {top['steps'][0]['tool']}."
    )

    return make_success_response(
        results={"recommendations": recommendations},
        interpretation=interpretation,
        meta={"goal": goal, "molecule_id": molecule_id},
    )


@mcp_tool_decorator
def explain_concept(concept: str) -> dict:
    """
    Look up a short, plain-language explanation of a core computational chemistry
    concept (e.g. 'gfn2_vs_dft', 'imaginary_frequencies', 'hansen_solubility').
    Call with an empty string to list all available concepts.

    Parameters:
    - concept: The concept key (case/space/dash-insensitive), or "" to list all keys.
    """
    concepts = _load_concepts()
    if not concept:
        return make_success_response(
            results={"available_concepts": sorted(concepts.keys())},
            interpretation=f"{len(concepts)} concepts available. Call again with one of these keys.",
        )

    key = concept.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in concepts:
        raise ValidationError(
            f"unknown concept '{concept}'",
            hint=f"available concepts: {', '.join(sorted(concepts.keys()))}",
        )
    entry = concepts[key]
    return make_success_response(
        results={"concept": key, "title": entry["title"], "explanation": entry["explanation"].strip()},
        interpretation=entry["title"],
    )


def compute_reaction_barrier(reactant_smiles: str, product_smiles: str, reaction_name: str = "this reaction") -> str:
    """Walk through computing the activation barrier for a reaction from reactant and product SMILES."""
    return (
        f"I want to compute the activation barrier for {reaction_name}.\n"
        f"Reactant SMILES: {reactant_smiles}\n"
        f"Product SMILES: {product_smiles}\n\n"
        "Please help me by: (1) calling build_molecule_from_smiles for both the reactant and "
        "product, then run_scientific_preflight on each to catch valence/charge problems before "
        "spending compute; (2) calling optimize_geometry on both endpoints; (3) calling "
        "run_neb_calculation with the two optimized endpoints to locate the transition state (or "
        "run_transition_state_search if I already have a good guess structure for it); (4) calling "
        "calculate_vibrations on the resulting TS structure and confirming it has exactly one "
        "imaginary frequency (call explain_concept('imaginary_frequencies') if that's unclear) "
        "before reporting the barrier; (5) reporting the barrier as a Gibbs free energy difference "
        "(not a bare electronic energy difference) in kcal/mol, noting the method/functional/basis "
        "used."
    )


def characterize_a_molecule(smiles: str, molecule_name: str | None = None) -> str:
    """Walk through a full structural/electronic/thermochemical characterization of a molecule."""
    name = molecule_name or "this molecule"
    return (
        f"I want to characterize {name} (SMILES: {smiles}).\n\n"
        "Please help me by: (1) calling build_molecule_from_smiles, then standardize_molecule to "
        "normalize tautomer/charge state, then calculate_descriptors for basic physicochemical "
        "properties (including a Lipinski's Rule of Five check); (2) calling "
        "run_scientific_preflight before any expensive calculation; (3) if the molecule has "
        "rotatable bonds, calling run_conformer_search and run_ensemble_thermochemistry for a "
        "Boltzmann-weighted picture rather than a single conformer; (4) calling "
        "calculate_vibrations on the lowest-energy conformer and reporting the HOMO-LUMO gap and "
        "any imaginary frequencies; (5) summarizing the results with units and flagging anything "
        "that looks physically off (e.g. unexpected imaginary frequencies, or preflight warnings "
        "that were never resolved)."
    )


def screen_solvent_compatibility(
    solute_smiles: str, candidate_solvent_smiles: list[str], solute_name: str = "the solute"
) -> str:
    """Walk through screening candidate solvents for compatibility with a solute via Hansen solubility parameters."""
    solvents = ", ".join(candidate_solvent_smiles)
    return (
        f"I want to screen solvent compatibility for {solute_name} (SMILES: {solute_smiles}) "
        f"against these candidate solvents: {solvents}.\n\n"
        "Please help me by: (1) calling build_molecule_from_smiles for the solute and each "
        "candidate solvent; (2) calling calculate_hsp for the solute and every candidate; "
        "(3) calling calculate_hsp_distance between the solute and each candidate, and ranking "
        "candidates by the resulting Ra distance (smaller Ra = more compatible - call "
        "explain_concept('ra_distance') if that needs unpacking); (4) reporting the ranked list "
        "with each candidate's Ra value and a plain-language compatibility verdict, and flagging "
        "that this is a fast group-contribution estimate, not a substitute for measuring actual "
        "solubility when the decision matters."
    )


def simulate_polymer_properties(monomer_smiles: str, repeat_units: int = 20, polymer_name: str = "the polymer") -> str:
    """Walk through building and simulating an amorphous polymer cell to get bulk properties."""
    return (
        f"I want to simulate {polymer_name} built from the monomer SMILES {monomer_smiles}, "
        f"as a chain of {repeat_units} repeat units.\n\n"
        "Please help me by: (1) calling register_monomer with the monomer SMILES, then "
        f"build_polymer_chain with count={repeat_units}; (2) calling pack_amorphous_cell to build "
        "a periodic amorphous cell containing multiple chains at a reasonable density; "
        "(3) calling run_lammps_simulation to equilibrate the cell (NVT first, per "
        "explain_concept('md_ensembles')); (4) calling analyze_md_trajectory and checking that "
        "radius_of_gyration has stabilized (not still drifting) and inspecting the RDF/MSD before "
        "trusting any reported property; (5) reporting the equilibrated bulk properties with units "
        "and noting how long the simulation was run and whether it looked equilibrated."
    )


def register_advisor_tools(mcp: FastMCP) -> None:
    mcp.tool()(recommend_workflow)
    mcp.tool()(explain_concept)
    mcp.prompt()(compute_reaction_barrier)
    mcp.prompt()(characterize_a_molecule)
    mcp.prompt()(screen_solvent_compatibility)
    mcp.prompt()(simulate_polymer_properties)
