import asyncio

from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.modules.advisor_tools import (
    explain_concept,
    recommend_workflow,
)
from ypotheto_compchem_mcp.server import mcp


def test_explain_concept_lists_at_least_25_concepts():
    listing = explain_concept("")
    assert listing["ok"] is True
    assert len(listing["results"]["available_concepts"]) >= 25


def test_explain_concept_lookup_is_case_and_separator_insensitive():
    canonical = explain_concept("gfn2_vs_dft")
    alias_form = explain_concept("GFN2 Vs DFT")

    assert canonical["ok"] is True
    assert canonical["results"]["concept"] == "gfn2_vs_dft"
    assert alias_form["ok"] is True
    assert alias_form["results"]["concept"] == "gfn2_vs_dft"
    assert alias_form["results"]["title"] == canonical["results"]["title"]


def test_explain_concept_unknown_returns_error_with_hint():
    result = explain_concept("not_a_real_concept")
    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_ARGUMENT"
    assert "gfn2_vs_dft" in result["error"]["hint"]


def test_recommend_workflow_ranks_transition_state_chain_first():
    result = recommend_workflow("find the activation barrier for my reaction")
    assert result["ok"] is True
    top = result["results"]["recommendations"][0]
    assert top["goal_category"] == "activation_barrier_workflow"
    assert top["steps"][0]["tool"] == "build_molecule_from_smiles"
    assert any(s["tool"] == "run_neb_calculation" for s in top["steps"])


def test_recommend_workflow_matches_other_rules():
    solvent = recommend_workflow("is this a good solvent for my polymer, miscibility check")
    polymer = recommend_workflow("simulate an amorphous polymer glass")
    drug = recommend_workflow("screen this drug-like compound with Lipinski descriptors")

    assert solvent["results"]["recommendations"][0]["goal_category"] == "solubility_screening_workflow"
    assert polymer["results"]["recommendations"][0]["goal_category"] == "polymer_simulation_workflow"
    assert drug["results"]["recommendations"][0]["goal_category"] == "druglike_screening_workflow"


def test_recommend_workflow_falls_back_to_default_when_no_keyword_matches():
    result = recommend_workflow("xyzzy plugh unrelated nonsense")
    top = result["results"]["recommendations"][0]
    assert top["goal_category"] == "general_starting_point"
    assert top["steps"][0]["tool"] == "run_scientific_preflight"


def test_recommend_workflow_tolerates_unknown_molecule_id():
    # molecule_id is advisory-only tailoring input; a bad handle must not make
    # the whole recommendation fail.
    result = recommend_workflow("barrier for my reaction", molecule_id="not_a_real_molecule_id")
    assert result["ok"] is True


def test_recommend_workflow_tailors_caveat_for_a_slow_molecule():
    # Decane (C10H22, 32 atoms with explicit Hs) is comfortably above the
    # ~60s DFT/STO-3G estimate threshold but still embeds without issue -
    # unlike a pathologically long chain, which can fail RDKit's ETKDG/
    # random-coordinate embedding fallback entirely (a separate, pre-existing
    # bug in build_molecule_from_smiles_engine, out of scope here).
    big = build_molecule_from_smiles_engine("CCCCCCCCCC", "decane")
    result = recommend_workflow("find the activation barrier", molecule_id=big["molecule_id"])
    caveats = " ".join(result["results"]["recommendations"][0]["caveats"])
    assert "run_xtb_calculation" in caveats


def test_advisor_prompts_are_registered():
    async def _list_prompt_names():
        prompts = await mcp.list_prompts()
        return {p.name for p in prompts}

    names = asyncio.run(_list_prompt_names())
    assert names == {
        "compute_reaction_barrier",
        "characterize_a_molecule",
        "screen_solvent_compatibility",
        "simulate_polymer_properties",
    }


def test_compute_reaction_barrier_prompt_content():
    result = asyncio.run(
        mcp.get_prompt(
            "compute_reaction_barrier",
            {"reactant_smiles": "CCO", "product_smiles": "CC=O", "reaction_name": "ethanol oxidation"},
        )
    )
    text = result.messages[0].content.text
    assert "ethanol oxidation" in text
    assert "run_neb_calculation" in text
