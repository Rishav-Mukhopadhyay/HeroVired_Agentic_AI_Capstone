"""Minimal FHIR R4 client for the HAPI server.

Kept deliberately small and readable. These functions are exposed as MCP tools that
the EHR agent can discover and call.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from mediscribe.config import settings

HEADERS = {"Content-Type": "application/fhir+json", "Accept": "application/fhir+json"}
ID_SYSTEM = "urn:mediscribe:aci-encounter"  # links a FHIR Patient to its ACI-Bench encounter


def _url(path: str) -> str:
    return f"{settings.fhir_base_url.rstrip('/')}/{path.lstrip('/')}"


def wait_until_ready(timeout_s: int = 240) -> None:
    """HAPI FHIR is a Java app that needs 1-2 minutes to start after `docker compose up`.
    Until then it accepts connections and drops them. Poll /metadata until it answers."""
    deadline = time.time() + timeout_s
    first = True
    while True:
        try:
            if requests.get(_url("metadata"), timeout=10).ok:
                if not first:
                    print(" ready.")
                return
        except requests.RequestException:
            pass
        if time.time() > deadline:
            raise TimeoutError(f"FHIR server not ready after {timeout_s}s - check `docker compose logs fhir`")
        if first:
            print("Waiting for the FHIR server to start", end="", flush=True)
            first = False
        print(".", end="", flush=True)
        time.sleep(5)


def create(resource: dict, if_none_exist: str | None = None) -> tuple[dict, bool]:
    """POST a resource. With if_none_exist (a search query) the server only creates it
    when nothing matches - this makes our seed script safe to run twice.
    Returns (resource, created?)."""
    headers = dict(HEADERS)
    if if_none_exist:
        headers["If-None-Exist"] = if_none_exist
    r = requests.post(_url(resource["resourceType"]), json=resource, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json(), r.status_code == 201


def search(resource_type: str, **params: Any) -> list[dict]:
    r = requests.get(_url(resource_type), params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return [e["resource"] for e in r.json().get("entry", [])]


def read(resource_type: str, resource_id: str) -> dict:
    r = requests.get(_url(f"{resource_type}/{resource_id}"), headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def find_patient_by_encounter(encounter_id: str) -> dict | None:
    hits = search("Patient", identifier=f"{ID_SYSTEM}|{encounter_id}")
    return hits[0] if hits else None


def patient_summary(patient_id: str) -> dict:
    """Everything the agents need about a patient in one call."""
    patient = read("Patient", patient_id)
    ref = f"Patient/{patient_id}"

    def texts(rtype: str, field: str) -> list[str]:
        return [res.get(field, {}).get("text", "?") for res in search(rtype, patient=ref)]

    name = patient.get("name", [{}])[0]
    return {
        "id": patient_id,
        "name": " ".join(name.get("given", []) + [name.get("family", "")]).strip(),
        "gender": patient.get("gender"),
        "birthDate": patient.get("birthDate"),
        "conditions": texts("Condition", "code"),
        "medications": texts("MedicationStatement", "medicationCodeableConcept"),
        "allergies": texts("AllergyIntolerance", "code"),
    }


def find_patients_by_name(name: str) -> list[dict]:
    return [{"id": p["id"],
             "name": " ".join(p["name"][0].get("given", []) + [p["name"][0].get("family", "")]),
             "birthDate": p.get("birthDate"), "gender": p.get("gender")}
            for p in search("Patient", name=name, _count=10)]


# ---------- Write operations (only after physician approval) ----------

def write_clinical_note(patient_id: str, note_text: str, title: str = "Consultation note") -> dict:
    """Store an approved note as a FHIR DocumentReference."""
    import base64
    import datetime as dt

    resource, _ = create({
        "resourceType": "DocumentReference",
        "status": "current",
        "docStatus": "final",
        "type": {"coding": [{"system": "http://loinc.org", "code": "11488-4", "display": "Consult note"}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "description": title,
        "content": [{"attachment": {"contentType": "text/plain", "title": title,
                                    "data": base64.b64encode(note_text.encode()).decode()}}],
    })
    return {"resourceType": "DocumentReference", "id": resource["id"]}


def add_condition(patient_id: str, diagnosis: str, icd10_code: str | None = None) -> dict:
    code = {"text": diagnosis}
    if icd10_code:
        code["coding"] = [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": icd10_code, "display": diagnosis}]
    resource, _ = create({
        "resourceType": "Condition",
        "subject": {"reference": f"Patient/{patient_id}"},
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
        "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "provisional"}]},
        "code": code,
    })
    return {"resourceType": "Condition", "id": resource["id"]}


def create_medication_request(patient_id: str, medication: str, dosage: str,
                              note: str | None = None) -> dict:
    """A prescription. status=active, intent=order: only call after the physician signs."""
    resource, _ = create({
        "resourceType": "MedicationRequest",
        "status": "active",
        "intent": "order",
        "subject": {"reference": f"Patient/{patient_id}"},
        "medicationCodeableConcept": {"text": medication},
        "dosageInstruction": [{"text": dosage}],
        **({"note": [{"text": note}]} if note else {}),
    })
    return {"resourceType": "MedicationRequest", "id": resource["id"]}
