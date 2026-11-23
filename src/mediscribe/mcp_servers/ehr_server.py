"""MCP server: the hospital EHR (HAPI FHIR).

Read tools are marked readOnlyHint; write tools are marked destructiveHint so a client
(and the human-in-the-loop step) can treat them differently.

Run standalone (stdio):  python -m mediscribe.mcp_servers.ehr_server
Inspect in a browser:    mcp dev src/mediscribe/mcp_servers/ehr_server.py
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from mediscribe import fhir_client as fhir

mcp = FastMCP("ehr", instructions="Hospital EHR (FHIR R4). Read patient context; write only "
                                   "physician-approved notes, diagnoses and prescriptions.")
READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)


@mcp.tool(annotations=READ)
def find_patient_by_encounter(encounter_id: str) -> dict:
    """Find the patient for a consultation by its encounter id (e.g. 'D2N068').
    Returns the FHIR patient id, or an error if not found."""
    p = fhir.find_patient_by_encounter(encounter_id)
    return {"patient_id": p["id"]} if p else {"error": f"no patient for encounter {encounter_id}"}


@mcp.tool(annotations=READ)
def search_patients(name: str) -> list[dict]:
    """Search patients by (part of) their name. Returns id, name, birth date, gender."""
    return fhir.find_patients_by_name(name)


@mcp.tool(annotations=READ)
def get_patient_summary(patient_id: str) -> dict:
    """Get a patient's demographics, active conditions, current medications and allergies.
    Always call this before suggesting a diagnosis or prescription."""
    return fhir.patient_summary(patient_id)


@mcp.tool(annotations=WRITE)
def write_clinical_note(patient_id: str, note_text: str, title: str = "Consultation note") -> dict:
    """Save a PHYSICIAN-APPROVED clinical note to the patient's record."""
    return fhir.write_clinical_note(patient_id, note_text, title)


@mcp.tool(annotations=WRITE)
def add_diagnosis(patient_id: str, diagnosis: str, icd10_code: str | None = None) -> dict:
    """Record a PHYSICIAN-APPROVED diagnosis (as a provisional Condition) with its ICD-10-CM code."""
    return fhir.add_condition(patient_id, diagnosis, icd10_code)


@mcp.tool(annotations=WRITE)
def create_prescription(patient_id: str, medication: str, dosage: str, note: str | None = None) -> dict:
    """Create a PHYSICIAN-APPROVED prescription. dosage example: '40 mg orally once daily for 5 days'."""
    return fhir.create_medication_request(patient_id, medication, dosage, note)


if __name__ == "__main__":
    mcp.run()  # stdio transport
