import asyncio

from mediscribe.mcp_client import discover_tools, is_read_only
from mediscribe.tools.drugs import check_allergy


def test_sulfa_allergy_blocks_bactrim():
    result = check_allergy("Bactrim DS", ["sulfonamide antibiotics"])
    assert not result["safe"] and result["conflicts"][0]["severity"] == "high"


def test_nsaid_class_allergy():
    assert not check_allergy("naproxen 500 mg", ["ibuprofen (NSAIDs)"])["safe"]


def test_penicillin_cephalosporin_cross_reactivity_is_moderate():
    conflicts = check_allergy("cephalexin", ["penicillin"])["conflicts"]
    assert conflicts and conflicts[0]["severity"] == "moderate"


def test_unrelated_drug_is_safe():
    assert check_allergy("lisinopril", ["penicillin"])["safe"]


def test_mcp_servers_advertise_tools_with_read_write_annotations():
    servers = asyncio.run(discover_tools())
    names = {s: {t.name for t in tools} for s, tools in servers.items()}
    assert "get_patient_summary" in names["ehr"]
    assert "lookup_icd10_code" in names["terminology"]
    assert "check_drug_allergy" in names["drugs"]
    writes = {t.name for tools in servers.values() for t in tools if not is_read_only(t)}
    assert writes == {"write_clinical_note", "add_diagnosis", "create_prescription"}


# --- combination drugs and class-aware interaction search (regression tests) ---
from mediscribe.tools import drugs  # noqa: E402


def test_normalises_combination_and_brand_names():
    assert drugs.components("sulfamethoxazole-trimethoprim") == ["sulfamethoxazole", "trimethoprim"]
    assert drugs.components("Bactrim DS") == ["sulfamethoxazole", "trimethoprim"]
    assert drugs.components("lisinopril 10 mg oral once daily") == ["lisinopril"]


def test_hyphenated_combination_drug_is_checked_per_ingredient():
    result = check_allergy("sulfamethoxazole-trimethoprim", ["sulfonamide antibiotics"])
    assert not result["safe"]
    assert result["conflicts"][0]["ingredient"] == "sulfamethoxazole"


def test_interaction_found_by_drug_class(monkeypatch):
    fake_labels = {
        "sulfamethoxazole and trimethoprim": {"warnings": [
            "Hyperkalemia: high dosage of trimethoprim may cause hyperkalemia, especially with "
            "angiotensin converting enzyme inhibitors."]},
        "lisinopril": {"drug_interactions": ["Diuretics may cause hypotension."]},
    }
    monkeypatch.setattr(drugs, "_fetch_label", lambda d: fake_labels.get(drugs._norm(d)))
    result = drugs.check_interactions("sulfamethoxazole-trimethoprim", ["lisinopril 10 mg oral once daily"])
    assert result["interactions_found"] == 1
    assert result["findings"][0]["matched"] == "angiotensin converting enzyme"
