"""Workflow tests with a fake LLM and fake MCP tools: no network, no API quota.

They check the orchestration itself: routing, parallel agents, the human-in-the-loop pause,
resuming from a checkpoint, and the blocked-prescription guardrail.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from mediscribe import workflow
from mediscribe.schemas import (DiagnosisDraft, DiagnosisSuggestions, PrescriptionDrafts,
                                PrescriptionItem, ProblemAssessment, SOAPNote, WorkflowPlan)
from mediscribe.tools.drugs import check_allergy

PATIENT = {"id": "1000", "name": "Martha Collins", "gender": "female", "birthDate": "1976-11-01",
           "conditions": ["hypertension"], "medications": ["lisinopril 10 mg oral once daily"],
           "allergies": ["sulfonamide antibiotics"]}


def fake_structured(plan: WorkflowPlan):
    def _fake(prompt, schema, system=None, **_):
        if schema is WorkflowPlan:
            return plan
        if schema is DiagnosisSuggestions:
            return DiagnosisSuggestions(diagnoses=[DiagnosisDraft(
                diagnosis="Acute cystitis", icd10_search_term="acute cystitis", confidence="high",
                reasoning="Dysuria and frequency", supporting_evidence=["burning on urination"])])
        if schema is PrescriptionDrafts:
            return PrescriptionDrafts(prescriptions=[
                PrescriptionItem(medication="Bactrim DS", dose="800/160 mg", frequency="twice daily",
                                  duration="3 days", indication="UTI"),
                PrescriptionItem(medication="phenazopyridine", dose="200 mg", frequency="three times daily",
                                  duration="2 days", indication="dysuria")])
        raise AssertionError(schema)
    return _fake


def fake_tool(writes: list):
    def _fake(server, tool, args):
        if tool == "find_patient_by_encounter":
            return {"patient_id": "1000"}
        if tool == "get_patient_summary":
            return PATIENT
        if tool == "lookup_icd10_code":
            return [{"code": "N30.00", "name": "Acute cystitis without hematuria"}]
        if tool == "check_drug_allergy":
            return check_allergy(args["drug"], args["allergies"])  # real deterministic logic
        if tool == "check_drug_interactions":
            return {"findings": [], "labels_not_found": []}
        writes.append((tool, args))
        return {"resourceType": tool, "id": str(len(writes))}
    return _fake


@pytest.fixture
def graph(monkeypatch):
    writes: list = []
    plan = WorkflowPlan(visit_type="acute illness", needs_diagnosis=True, needs_prescription=True,
                        red_flags=[], rationale="UTI treated with antibiotics")
    monkeypatch.setattr(workflow, "generate_structured", fake_structured(plan))
    monkeypatch.setattr(workflow, "generate_note", lambda transcript: SOAPNote(
        chief_complaint="Burning on urination", history_of_present_illness="2 days of dysuria.",
        assessment_and_plan=[ProblemAssessment(problem="UTI", assessment="Likely cystitis",
                                               plan=["Bactrim DS twice daily for 3 days"])]))
    monkeypatch.setattr(workflow, "call_tool", fake_tool(writes))
    g = workflow.build_graph(checkpointer=InMemorySaver())
    return g, writes, plan


def start(g, thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    result = g.invoke({"encounter_id": "D2N001", "transcript": "[doctor] ...", "audit": []}, config)
    return config, result


def test_pauses_for_physician_with_safety_results(graph):
    g, writes, _ = graph
    config, result = start(g)
    packet = result["__interrupt__"][0].value
    statuses = {rx["medication"]: rx["safety_status"] for rx in packet["prescriptions"]}
    assert statuses == {"Bactrim DS": "blocked", "phenazopyridine": "ok"}
    assert packet["diagnoses"][0]["icd10_code"] == "N30.00"
    assert g.get_state(config).next == ("physician_review",)
    assert writes == []  # nothing written before approval


def test_approve_without_override_excludes_blocked_prescription(graph):
    g, writes, _ = graph
    config, _ = start(g)
    final = g.invoke(Command(resume={"decision": "approve"}), config)
    assert final["status"] == "written_to_ehr"
    prescribed = [a["medication"] for t, a in writes if t == "create_prescription"]
    assert prescribed == ["phenazopyridine"]
    assert any("no override reason" in e["event"] for e in final["audit"])
    note = next(a["note_text"] for t, a in writes if t == "write_clinical_note")
    assert "ADDENDUM" in note and "Bactrim DS" in note and "NOT prescribed" in note


def test_override_with_reason_is_written_and_audited(graph):
    g, writes, _ = graph
    config, _ = start(g)
    final = g.invoke(Command(resume={"decision": "approve", "override_reason": "allergy was rash only, tolerated before"}),
                     config)
    prescribed = [a for t, a in writes if t == "create_prescription"]
    assert [p["medication"] for p in prescribed] == ["Bactrim DS", "phenazopyridine"]
    assert prescribed[0]["note"].startswith("Override:")
    assert any("overridden" in e["event"] for e in final["audit"])


def test_reject_writes_nothing(graph):
    g, writes, _ = graph
    config, _ = start(g)
    final = g.invoke(Command(resume={"decision": "reject", "comments": "wrong patient"}), config)
    assert final["status"] == "rejected" and writes == []


def test_planner_skips_pharmacist_when_no_medication_change(graph, monkeypatch):
    g, writes, plan = graph
    plan.needs_prescription = False
    config, result = start(g, "t2")
    assert result["__interrupt__"][0].value["prescriptions"] == []
    nodes = [e["node"] for e in g.get_state(config).values["audit"]]
    assert "pharmacist" not in nodes and "diagnostician" in nodes


def test_resume_after_failure_continues_from_checkpoint(graph, monkeypatch):
    """A crash mid-workflow loses nothing: resuming re-runs only the failed step."""
    g, writes, _ = graph
    calls = {"scribe": 0}
    real_scribe = workflow.generate_note

    def flaky_note(transcript):
        calls["scribe"] += 1
        if calls["scribe"] == 1:
            raise RuntimeError("all models overloaded")
        return real_scribe(transcript)

    monkeypatch.setattr(workflow, "generate_note", flaky_note)
    g = workflow.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "crash"}}

    with pytest.raises(RuntimeError):
        g.invoke({"encounter_id": "D2N001", "transcript": "[doctor] ...", "audit": []}, config)
    assert g.get_state(config).next == ("scribe",)  # stopped exactly at the failed node

    result = g.invoke(None, config)  # resume from the checkpoint
    assert "__interrupt__" in result
    nodes = [e["node"] for e in g.get_state(config).values["audit"]]
    assert nodes.count("load_patient") == 1 and nodes.count("planner") == 1  # not repeated


def test_diagnosis_already_on_record_is_not_duplicated(graph, monkeypatch):
    g, writes, plan = graph
    base = workflow.generate_structured

    def with_hypertension(prompt, schema, system=None, **kw):
        result = base(prompt, schema, system, **kw)
        if schema is DiagnosisSuggestions:  # PATIENT already has "hypertension" on record
            result.diagnoses.append(DiagnosisDraft(diagnosis="Hypertension", icd10_search_term="hypertension",
                                                   confidence="high", reasoning="BP discussed",
                                                   supporting_evidence=["BP 132/82"]))
        return result

    monkeypatch.setattr(workflow, "generate_structured", with_hypertension)
    g = workflow.build_graph(checkpointer=InMemorySaver())
    config, _ = start(g, "dup")
    final = g.invoke(Command(resume={"decision": "approve"}), config)
    written = [a["diagnosis"] for t, a in writes if t == "add_diagnosis"]
    assert written == ["Acute cystitis"]
    assert final["audit"][-1]["already_on_record"] == ["Hypertension"]
