"""Minimal FHIR R4 client for the HAPI server.

Kept deliberately small and readable. These functions are exposed as MCP tools that
the EHR agent can discover and call.
"""

from __future__ import annotations

from typing import Any

import requests

from mediscribe.config import settings

HEADERS = {"Content-Type": "application/fhir+json", "Accept": "application/fhir+json"}
ID_SYSTEM = "urn:mediscribe:aci-encounter"  # links a FHIR Patient to its ACI-Bench encounter


def _url(path: str) -> str:
    return f"{settings.fhir_base_url.rstrip('/')}/{path.lstrip('/')}"


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
