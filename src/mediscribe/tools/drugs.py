"""Drug safety helpers.

- Label information and interactions come from openFDA drug labels (free, no key).
  https://open.fda.gov/apis/drug/label/
- Allergy cross-reactivity uses a small built-in drug-class table: deterministic and
  auditable, which matters more than coverage for a safety check.

These are decision SUPPORT for the physician, not a complete interaction database.
"""

from __future__ import annotations

import re
from functools import lru_cache

import requests

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
LABEL_SECTIONS = ("boxed_warning", "contraindications", "drug_interactions",
                  "warnings_and_cautions", "warnings")

# Allergy class -> drugs in that class (and closely cross-reactive ones)
ALLERGY_CLASSES: dict[str, set[str]] = {
    "penicillin": {"penicillin", "amoxicillin", "ampicillin", "augmentin", "piperacillin",
                   "dicloxacillin", "nafcillin"},
    "cephalosporin": {"cephalexin", "cefazolin", "ceftriaxone", "cefuroxime", "cefdinir"},
    "sulfonamide": {"sulfamethoxazole", "bactrim", "septra", "sulfasalazine", "sulfadiazine"},
    "nsaid": {"ibuprofen", "naproxen", "aspirin", "diclofenac", "ketorolac", "celecoxib",
              "meloxicam", "indomethacin"},
    "opioid": {"codeine", "morphine", "hydrocodone", "oxycodone", "tramadol", "hydromorphone"},
}
# Words in an allergy record that map to a class
ALLERGY_ALIASES = {
    "penicillin": "penicillin", "amoxicillin": "penicillin",
    "cephalosporin": "cephalosporin",
    "sulfa": "sulfonamide", "sulfonamide": "sulfonamide",
    "nsaid": "nsaid", "ibuprofen": "nsaid", "aspirin": "nsaid", "naproxen": "nsaid",
    "codeine": "opioid", "opioid": "opioid", "morphine": "opioid",
}
CROSS_REACTIVE = {"penicillin": {"cephalosporin"}}  # small, well-known cross-reactivity

# Labels often warn about a drug CLASS rather than naming each drug ("ACE inhibitors"), so when
# searching another drug's label for one of these medications we also look for its class terms.
CLASS_TERMS: dict[str, list[str]] = {
    "lisinopril": ["ACE inhibitor", "angiotensin converting enzyme", "angiotensin-converting enzyme"],
    "enalapril": ["ACE inhibitor", "angiotensin converting enzyme", "angiotensin-converting enzyme"],
    "losartan": ["angiotensin receptor blocker", "angiotensin II receptor"],
    "furosemide": ["loop diuretic", "diuretic"],
    "spironolactone": ["potassium-sparing diuretic", "diuretic"],
    "warfarin": ["anticoagulant", "coumarin"],
    "apixaban": ["anticoagulant"],
    "sertraline": ["SSRI", "selective serotonin reuptake inhibitor"],
    "metformin": ["antidiabetic", "hypoglycemic"],
    "atorvastatin": ["statin", "HMG-CoA reductase inhibitor"],
    "levothyroxine": ["thyroid hormone"],
    "tacrolimus": ["calcineurin inhibitor", "immunosuppressant"],
}
BRAND_TO_GENERIC = {"bactrim": "sulfamethoxazole and trimethoprim", "septra": "sulfamethoxazole and trimethoprim",
                    "augmentin": "amoxicillin and clavulanate potassium", "lasix": "furosemide",
                    "zestril": "lisinopril", "coumadin": "warfarin", "zoloft": "sertraline"}


def _norm(drug: str) -> str:
    """'Lisinopril 10 mg daily' -> 'lisinopril'. Combination names keep their parts joined
    with ' and ' ('sulfamethoxazole-trimethoprim' -> 'sulfamethoxazole and trimethoprim')."""
    text = drug.strip().lower()
    text = re.split(r"\s+\d|\(|,", text)[0]                  # drop dose, brackets, extras
    text = re.sub(r"\s*(?:/|-|\+|\band\b)\s*", " and ", text)  # unify combination separators
    text = re.sub(r"\s+(tablets?|capsules?|ds|er|xr|oral|inhaler)\b.*$", "", text).strip()
    return BRAND_TO_GENERIC.get(text, text)


def components(drug: str) -> list[str]:
    """Single ingredients of a (possibly combination) drug."""
    return [c.strip() for c in _norm(drug).split(" and ") if c.strip()]


def search_terms(drug: str) -> list[str]:
    """Words to look for in another drug's label: each ingredient plus its class terms."""
    terms = []
    for comp in components(drug):
        terms += [comp] + CLASS_TERMS.get(comp, [])
    return terms


def _query_label(search: str) -> dict | None:
    r = requests.get(OPENFDA_LABEL_URL, params={"search": search, "limit": 1}, timeout=20)
    if r.status_code == 404:  # openFDA returns 404 for "no matches"
        return None
    r.raise_for_status()
    results = r.json().get("results", [])
    return results[0] if results else None


@lru_cache(maxsize=256)
def _fetch_label(drug: str) -> dict | None:
    name = _norm(drug)
    attempts = [f'openfda.generic_name:"{name}"', f'openfda.brand_name:"{name}"']
    if " and " in name:  # combination: fall back to a label containing all ingredients
        attempts.append(" AND ".join(f"openfda.generic_name:{c}" for c in components(name)))
    for search in attempts:
        if label := _query_label(search):
            return label
    return None


def drug_label_info(drug: str, max_chars: int = 1200) -> dict:
    """Key safety sections of the FDA label, truncated so they fit an LLM context."""
    label = _fetch_label(drug)
    if label is None:
        return {"drug": drug, "found": False}
    info = {"drug": drug, "found": True,
            "generic_name": label.get("openfda", {}).get("generic_name", [None])[0]}
    for section in LABEL_SECTIONS:
        if label.get(section):
            text = " ".join(label[section])
            info[section] = text[:max_chars] + ("…" if len(text) > max_chars else "")
    return info


def _find_in_label(label: dict, terms: list[str]) -> dict | None:
    for section in ("drug_interactions", "contraindications", "warnings_and_cautions", "warnings"):
        text = " ".join(label.get(section, []))
        for term in terms:
            m = re.search(rf"\b{re.escape(term)}s?\b", text, re.I)
            if m:
                snippet = text[max(0, m.start() - 200): m.end() + 250]
                return {"section": section, "matched": term, "excerpt": "…" + snippet.strip() + "…"}
    return None


def check_interactions(new_drug: str, current_medications: list[str]) -> dict:
    """Search the new drug's FDA label for each current medication (by name or drug class),
    and each current medication's label for the new drug's ingredients."""
    findings, labels_missing = [], []
    new_label = _fetch_label(new_drug)
    if new_label is None:
        labels_missing.append(_norm(new_drug))
    for med in current_medications:
        med_label = _fetch_label(med)
        if med_label is None:
            labels_missing.append(_norm(med))
        for label_of, label, terms in ((new_drug, new_label, search_terms(med)),
                                       (med, med_label, search_terms(new_drug))):
            if label and (hit := _find_in_label(label, terms)):
                findings.append({"between": [_norm(new_drug), _norm(med)], "label_of": _norm(label_of), **hit})
                break  # one piece of evidence per pair is enough
    return {"new_drug": _norm(new_drug), "checked_against": [_norm(m) for m in current_medications],
            "interactions_found": len(findings), "findings": findings,
            "labels_not_found": labels_missing,
            "note": "Label text search, not a complete interaction database. "
                    "Absence of a finding is not proof of safety."}


def check_allergy(drug: str, allergies: list[str]) -> dict:
    """Checks every ingredient of the drug (combination products included)."""
    conflicts = []
    for name in components(drug):
        drug_classes = {cls for cls, members in ALLERGY_CLASSES.items() if name in members}
        for allergy in allergies:
            text = allergy.lower()
            allergy_classes = {cls for word, cls in ALLERGY_ALIASES.items() if word in text}
            if name in text:
                conflicts.append({"allergy": allergy, "ingredient": name, "reason": "same drug", "severity": "high"})
                continue
            for cls in allergy_classes:
                if cls in drug_classes:
                    conflicts.append({"allergy": allergy, "ingredient": name,
                                      "reason": f"{name} is a {cls}", "severity": "high"})
                elif drug_classes & CROSS_REACTIVE.get(cls, set()):
                    conflicts.append({"allergy": allergy, "ingredient": name,
                                      "reason": f"possible {cls} cross-reactivity", "severity": "moderate"})
    return {"drug": drug, "ingredients": components(drug), "allergies": allergies,
            "safe": not conflicts, "conflicts": conflicts}
