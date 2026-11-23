"""Seed the HAPI FHIR server with one synthetic patient per ACI-Bench encounter.

Name, age, gender and existing conditions come from ACI-Bench metadata.
Current medications and allergies are SYNTHETIC additions (deterministic per patient)
so the Pharmacist agent has something real to check prescriptions against.

The FHIR server uses in-memory storage, so re-run this after restarting Docker.
Safe to run twice (conditional create).

Run:  python scripts/seed_fhir.py            (all splits)
      python scripts/seed_fhir.py --split valid
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import zlib

import pandas as pd

from mediscribe import fhir_client as fhir
from mediscribe.data_sources import ACI_SPLITS, load_aci

# Typical long-term medication for common conditions (synthetic, for demo safety checks).
CONDITION_MEDS = {
    "hypertension": "lisinopril 10 mg oral once daily",
    "diabetes": "metformin 500 mg oral twice daily",
    "congestive heart failure": "furosemide 20 mg oral once daily",
    "chf": "furosemide 20 mg oral once daily",
    "atrial fibrillation": "warfarin 5 mg oral once daily",
    "depression": "sertraline 50 mg oral once daily",
    "anxiety": "sertraline 50 mg oral once daily",
    "reflux": "omeprazole 20 mg oral once daily",
    "hypothyroidism": "levothyroxine 50 mcg oral once daily",
    "asthma": "albuterol inhaler 90 mcg as needed",
    "coronary artery disease": "atorvastatin 40 mg oral once daily",
    "gout": "allopurinol 100 mg oral once daily",
    "epilepsy": "levetiracetam 500 mg oral twice daily",
    "kidney transplant": "tacrolimus 2 mg oral twice daily",
}
# Weighted so most patients have no allergy, some have the classic ones.
ALLERGIES = ["penicillin", "sulfonamide antibiotics", "ibuprofen (NSAIDs)", "codeine",
             None, None, None, None]
FIRST_NAMES = {"male": ["Arjun", "David", "James", "Rahul"], "female": ["Priya", "Sarah", "Emma", "Anita"]}


def birth_date(age) -> str:
    today = dt.date(2026, 11, 1)  # fixed reference date so ages are reproducible
    s = str(age)
    if "month" in s:
        months = int(s.split("-")[0])
        return (today - dt.timedelta(days=30 * months)).isoformat()
    years = int(float(s)) if s not in ("nan", "") else 45
    return today.replace(year=today.year - years).isoformat()


def conditions_of(row) -> list[str]:
    raw = row.get("2nd_complaints")
    if pd.isna(raw):
        return []
    return [c.strip() for c in str(raw).split(";") if c.strip() and c.strip().lower() != "none"]


def med_for(condition: str) -> str | None:
    c = condition.lower()
    return next((med for key, med in CONDITION_MEDS.items() if key in c), None)


def seed_row(row) -> str:
    enc = row["encounter_id"]
    rng = random.Random(zlib.crc32(enc.encode()))  # same "random" data every run
    gender = row["patient_gender"] if row["patient_gender"] in ("male", "female") else "unknown"
    first = row["patient_firstname"]
    if pd.isna(first):
        first = rng.choice(FIRST_NAMES.get(gender, ["Alex"]))
    family = row["patient_familyname"] if not pd.isna(row["patient_familyname"]) else "Doe"

    patient, created = fhir.create(
        {
            "resourceType": "Patient",
            "identifier": [{"system": fhir.ID_SYSTEM, "value": enc}],
            "name": [{"given": [str(first).title()], "family": str(family).title()}],
            "gender": gender,
            "birthDate": birth_date(row["patient_age"]),
            "meta": {"tag": [{"system": "urn:mediscribe", "code": "synthetic"}]},
        },
        if_none_exist=f"identifier={fhir.ID_SYSTEM}|{enc}",
    )
    if not created:
        return "exists"

    ref = {"reference": f"Patient/{patient['id']}"}
    meds = set()
    for cond in conditions_of(row):
        fhir.create({
            "resourceType": "Condition",
            "subject": ref,
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "code": {"text": cond},
        })
        if med := med_for(cond):
            meds.add(med)
    for med in sorted(meds):
        fhir.create({
            "resourceType": "MedicationStatement",
            "status": "active",
            "subject": ref,
            "medicationCodeableConcept": {"text": med},
        })
    if allergy := rng.choice(ALLERGIES):
        fhir.create({
            "resourceType": "AllergyIntolerance",
            "patient": ref,
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
            "code": {"text": allergy},
        })
    return "created"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(ACI_SPLITS), default=None)
    args = ap.parse_args()
    splits = [args.split] if args.split else list(ACI_SPLITS)

    fhir.wait_until_ready()

    counts = {"created": 0, "exists": 0}
    for split in splits:
        for _, row in load_aci(split).iterrows():
            counts[seed_row(row)] += 1
    print(f"Patients created: {counts['created']}, already present: {counts['exists']}")

    # Show one example so you can see what the agents will get
    sample = fhir.find_patient_by_encounter(load_aci(splits[0]).encounter_id.iloc[0])
    print("Example patient summary:", fhir.patient_summary(sample["id"]))


if __name__ == "__main__":
    main()
