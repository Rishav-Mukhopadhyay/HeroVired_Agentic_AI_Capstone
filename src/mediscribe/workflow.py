"""The consultation workflow: a LangGraph multi-agent graph with a human-in-the-loop step.

    load_patient -> planner -> scribe -+-> diagnostician -+-> physician_review -> ehr_writer
                                       +-> pharmacist ----+        (pause)

Design choices
- Planner decides WHICH agents run (conditional edges). Diagnostician and pharmacist then run
  in parallel; the review waits for both.
- Safety checks are fixed steps, not LLM choices: the pharmacist node ALWAYS runs the allergy
  and interaction tools for every drafted prescription. The LLM only extracts the drafts.
- physician_review calls interrupt(): the graph stops and its state is saved by the
  checkpointer (Postgres). The physician can answer minutes or days later, from another process.
- Only ehr_writer uses the EHR write tools, and only after an approval. Blocked prescriptions
  are not written unless the physician gives an override reason.
- State holds plain dicts (model_dump()) so checkpoints are plain JSON-like data.
"""

from __future__ import annotations

import datetime as dt
import operator
from contextlib import contextmanager
from typing import Annotated, Any, Iterator, TypedDict

from langgraph.graph import END, START, StateGraph
from google.genai import errors as genai_errors
from langgraph.types import RetryPolicy, interrupt

from mediscribe.agents.scribe import generate_note
from mediscribe.config import settings
from mediscribe.llm import generate_structured, last_model_used
from mediscribe.mcp_client import ToolCallError, call_tool
from mediscribe.prompts import load_prompt
from mediscribe.schemas import (DiagnosisSuggestion, DiagnosisSuggestions, PhysicianDecision,
                                PrescriptionDraft, PrescriptionDrafts, SOAPNote, WorkflowPlan)


class ConsultationState(TypedDict, total=False):
    encounter_id: str
    transcript: str
    patient_id: str
    patient: dict
    plan: dict
    note: dict
    diagnoses: list[dict]
    differentials: list[str]
    prescriptions: list[dict]
    decision: dict
    excluded_prescriptions: list[dict]  # blocked items left out at review (no override)
    ehr_writes: list[dict]
    status: str
    audit: Annotated[list[dict], operator.add]  # every node appends; nothing is overwritten


def log(node: str, event: str, **data: Any) -> list[dict]:
    return [{"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "node": node, "event": event, **data}]


# ---------------------------------------------------------------- nodes

def load_patient(state: ConsultationState) -> dict:
    """Fetch the patient's record through the EHR MCP server (read-only tools)."""
    found = call_tool("ehr", "find_patient_by_encounter", {"encounter_id": state["encounter_id"]})
    if "patient_id" not in found:
        raise ValueError(found.get("error", "patient not found"))
    patient = call_tool("ehr", "get_patient_summary", {"patient_id": found["patient_id"]})
    return {"patient_id": found["patient_id"], "patient": patient, "status": "in_progress",
            "audit": log("load_patient", "patient loaded", patient_id=found["patient_id"])}


def planner(state: ConsultationState) -> dict:
    """Planning & decisioning: which specialist agents does this visit need?"""
    plan = generate_structured(f"### TRANSCRIPT\n{state['transcript']}", WorkflowPlan,
                               system=load_prompt("planner", "v1"))
    return {"plan": plan.model_dump(),
            "audit": log("planner", "plan made", model=last_model_used(), needs_diagnosis=plan.needs_diagnosis,
                         needs_prescription=plan.needs_prescription, red_flags=plan.red_flags)}


def scribe(state: ConsultationState) -> dict:
    note = generate_note(state["transcript"])
    return {"note": note.model_dump(), "audit": log("scribe", "SOAP note drafted", model=last_model_used())}


def patient_context(patient: dict) -> str:
    return (f"Patient: {patient.get('name')}, {patient.get('gender')}, born {patient.get('birthDate')}\n"
            f"Known conditions: {', '.join(patient.get('conditions', [])) or 'none recorded'}\n"
            f"Current medications: {', '.join(patient.get('medications', [])) or 'none recorded'}\n"
            f"Allergies: {', '.join(patient.get('allergies', [])) or 'none recorded'}")


def diagnostician(state: ConsultationState) -> dict:
    """Suggest diagnoses (LLM), then attach official ICD-10-CM codes (terminology MCP tool)."""
    note = SOAPNote.model_validate(state["note"])
    result = generate_structured(
        f"### PATIENT RECORD\n{patient_context(state['patient'])}\n\n### CLINICAL NOTE\n{note.to_text()}",
        DiagnosisSuggestions, system=load_prompt("diagnostician", "v1"))

    diagnoses = [DiagnosisSuggestion(**d.model_dump()) for d in result.diagnoses]
    for dx in diagnoses:
        for term in (dx.icd10_search_term, dx.diagnosis):  # retry with the full name if needed
            try:
                candidates = call_tool("terminology", "lookup_icd10_code", {"diagnosis": term})
            except ToolCallError:
                candidates = []
            if candidates:
                dx.icd10_candidates = candidates[:5]
                dx.icd10_code, dx.icd10_name = candidates[0]["code"], candidates[0]["name"]
                break
    return {"diagnoses": [d.model_dump() for d in diagnoses],
            "differentials": result.differentials_to_consider,
            "audit": log("diagnostician", "diagnoses suggested", model=last_model_used(),
                         codes=[d.icd10_code for d in diagnoses])}


def pharmacist(state: ConsultationState) -> dict:
    """Extract prescriptions (LLM), then ALWAYS run both safety checks (MCP drug tools)."""
    note = SOAPNote.model_validate(state["note"])
    items = generate_structured(f"### CLINICAL NOTE\n{note.to_text()}", PrescriptionDrafts,
                                system=load_prompt("pharmacist", "v1")).prescriptions
    drafts = [PrescriptionDraft(**item.model_dump()) for item in items]
    patient = state["patient"]
    for rx in drafts:
        rx_checks(rx, patient.get("allergies", []), patient.get("medications", []))
    return {"prescriptions": [rx.model_dump() for rx in drafts],
            "audit": log("pharmacist", "prescriptions drafted and checked", model=last_model_used(),
                         statuses={rx.medication: rx.safety_status for rx in drafts})}


def rx_checks(rx: PrescriptionDraft, allergies: list[str], medications: list[str]) -> None:
    notes, status = [], "ok"
    try:
        rx.allergy_check = call_tool("drugs", "check_drug_allergy",
                                     {"drug": rx.medication, "allergies": allergies})
        for c in rx.allergy_check["conflicts"]:
            notes.append(f"ALLERGY ({c['severity']}): {c['reason']} - patient allergy: {c['allergy']}")
            status = "blocked" if c["severity"] == "high" else max(status, "caution", key=_rank)
    except ToolCallError as exc:
        notes.append(f"Allergy check could not run: {exc}")
        status = max(status, "caution", key=_rank)
    try:
        rx.interaction_check = call_tool("drugs", "check_drug_interactions",
                                         {"new_drug": rx.medication, "current_medications": medications})
        for f in rx.interaction_check["findings"]:
            notes.append(f"INTERACTION with {f['between'][1]} (FDA label of {f['label_of']}, "
                         f"{f['section']}): {f['excerpt'][:240]}")
            status = max(status, "caution", key=_rank)
        if rx.interaction_check.get("labels_not_found"):
            notes.append(f"No FDA label found for: {', '.join(rx.interaction_check['labels_not_found'])}")
    except ToolCallError as exc:
        notes.append(f"Interaction check could not run: {exc}")
        status = max(status, "caution", key=_rank)
    rx.safety_status, rx.safety_notes = status, notes


def _rank(status: str) -> int:
    return {"unchecked": 0, "ok": 1, "caution": 2, "blocked": 3}[status]


def review_packet(state: ConsultationState) -> dict:
    """Everything the physician needs to decide, in one object."""
    return {
        "encounter_id": state["encounter_id"],
        "patient": state["patient"],
        "red_flags": state.get("plan", {}).get("red_flags", []),
        "note_text": SOAPNote.model_validate(state["note"]).to_text(),
        "diagnoses": state.get("diagnoses", []),
        "differentials": state.get("differentials", []),
        "prescriptions": state.get("prescriptions", []),
    }


def physician_review(state: ConsultationState) -> dict:
    """HUMAN IN THE LOOP. interrupt() pauses the graph here until the physician responds."""
    answer = interrupt(review_packet(state))
    decision = PhysicianDecision.model_validate(answer)

    events, excluded = [], []
    if decision.decision == "approve":
        rx = state.get("prescriptions", [])
        chosen = decision.approved_prescriptions if decision.approved_prescriptions is not None else list(range(len(rx)))
        blocked = [i for i in chosen if rx[i]["safety_status"] == "blocked"]
        if blocked and not decision.override_reason:
            # Guardrail: a blocked prescription needs an explicit, recorded justification
            chosen = [i for i in chosen if i not in blocked]
            excluded = [rx[i] for i in blocked]
            events += log("physician_review", "blocked prescriptions excluded: no override reason",
                          excluded=[rx[i]["medication"] for i in blocked])
        elif blocked:
            events += log("physician_review", "safety block overridden by physician",
                          medications=[rx[i]["medication"] for i in blocked], reason=decision.override_reason)
        decision.approved_prescriptions = chosen

    return {"decision": decision.model_dump(), "excluded_prescriptions": excluded,
            "status": "approved" if decision.decision == "approve" else "rejected",
            "audit": events + log("physician_review", f"physician {decision.decision}d",
                                  reviewer=decision.reviewer, comments=decision.comments)}


def ehr_writer(state: ConsultationState) -> dict:
    """The ONLY node that uses EHR write tools, and only for approved items."""
    decision = PhysicianDecision.model_validate(state["decision"])
    pid = state["patient_id"]
    writes = []

    note_text = decision.note_text or SOAPNote.model_validate(state["note"]).to_text()
    if excluded := state.get("excluded_prescriptions"):
        # The note's plan may still mention a drug that was NOT prescribed: say so explicitly,
        # so nobody reading the record later assumes the patient received it.
        lines = [f"- {rx['medication']} {PrescriptionDraft.model_validate(rx).dosage_text()}: NOT prescribed. "
                 f"Blocked by safety check: {'; '.join(rx['safety_notes'][:1])}" for rx in excluded]
        note_text += ("\n\nADDENDUM (MediScribe safety check)\n\n" + "\n".join(lines)
                      + "\n- Physician to select an alternative if still indicated.")
    writes.append(call_tool("ehr", "write_clinical_note",
                            {"patient_id": pid, "note_text": note_text,
                             "title": f"Consultation note ({state['encounter_id']})"}))

    diagnoses = state.get("diagnoses", [])
    on_record = {c.strip().lower() for c in state.get("patient", {}).get("conditions", [])}
    skipped = []
    for i in (decision.approved_diagnoses if decision.approved_diagnoses is not None else range(len(diagnoses))):
        dx = diagnoses[i]
        names = {dx["diagnosis"].strip().lower(), (dx.get("icd10_name") or "").strip().lower()}
        if names & on_record:  # already an active condition: do not create a duplicate
            skipped.append(dx["diagnosis"])
            continue
        writes.append(call_tool("ehr", "add_diagnosis", {"patient_id": pid, "diagnosis": dx["diagnosis"],
                                                         "icd10_code": dx.get("icd10_code")}))

    prescriptions = state.get("prescriptions", [])
    for i in decision.approved_prescriptions or []:
        rx = PrescriptionDraft.model_validate(prescriptions[i])
        note = f"Override: {decision.override_reason}" if rx.safety_status == "blocked" else None
        writes.append(call_tool("ehr", "create_prescription",
                                {"patient_id": pid, "medication": rx.medication,
                                 "dosage": rx.dosage_text(), "note": note}))

    return {"ehr_writes": writes, "status": "written_to_ehr",
            "audit": log("ehr_writer", "written to EHR", resources=[f"{w['resourceType']}/{w['id']}" for w in writes],
                         already_on_record=skipped)}


# ---------------------------------------------------------------- routing

def route_after_scribe(state: ConsultationState) -> list[str]:
    plan = state.get("plan", {})
    nxt = [name for name, needed in (("diagnostician", plan.get("needs_diagnosis", True)),
                                     ("pharmacist", plan.get("needs_prescription", True))) if needed]
    return nxt or ["physician_review"]


def route_after_review(state: ConsultationState) -> str:
    return "ehr_writer" if state.get("status") == "approved" else END


# If every model fails, wait and run the node once more before giving up. A node that still
# fails stops the graph, but the checkpoint keeps the finished nodes: --resume continues from it.
LLM_RETRY = RetryPolicy(max_attempts=2, initial_interval=20.0,
                        retry_on=lambda exc: isinstance(exc, genai_errors.APIError))
LLM_NODES = {"planner", "scribe", "diagnostician", "pharmacist"}


def build_graph(checkpointer=None):
    g = StateGraph(ConsultationState)
    for node in (load_patient, planner, scribe, diagnostician, pharmacist, physician_review, ehr_writer):
        g.add_node(node.__name__, node,
                   retry_policy=LLM_RETRY if node.__name__ in LLM_NODES else None)
    g.add_edge(START, "load_patient")
    g.add_edge("load_patient", "planner")
    g.add_edge("planner", "scribe")
    g.add_conditional_edges("scribe", route_after_scribe,
                            ["diagnostician", "pharmacist", "physician_review"])
    g.add_edge("diagnostician", "physician_review")
    g.add_edge("pharmacist", "physician_review")
    g.add_conditional_edges("physician_review", route_after_review, ["ehr_writer", END])
    g.add_edge("ehr_writer", END)
    return g.compile(checkpointer=checkpointer)


@contextmanager
def postgres_checkpointer() -> Iterator[Any]:
    """Workflow state saved in Postgres after every step (survives restarts)."""
    from langgraph.checkpoint.postgres import PostgresSaver

    url = settings.postgres_url.replace("postgresql+psycopg://", "postgresql://")
    with PostgresSaver.from_conn_string(url) as saver:
        saver.setup()  # creates the checkpoint tables on first use
        yield saver
