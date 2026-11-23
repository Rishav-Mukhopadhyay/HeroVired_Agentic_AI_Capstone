"""Structured objects shared by the agents.

Every agent returns one of these Pydantic models instead of free text. The LLM is forced to
fill the schema (structured output), so downstream code - the physician review screen, FHIR
writer, PDF generator - can rely on exact fields instead of parsing prose.
Field descriptions are sent to the model, so they double as instructions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProblemAssessment(BaseModel):
    problem: str = Field(description="Diagnosis or problem name, e.g. 'Type 2 diabetes mellitus'")
    assessment: str = Field(description="Clinician's assessment of this problem as discussed in the visit")
    plan: list[str] = Field(description="Concrete plan items for this problem: tests, medications "
                                        "with dose and frequency, referrals, lifestyle advice")


class SOAPNote(BaseModel):
    """Clinical note in SOAP format (Subjective, Objective, Assessment, Plan)."""

    # --- Subjective ---
    chief_complaint: str = Field(description="Main reason for the visit, a short phrase")
    history_of_present_illness: str = Field(
        description="Narrative paragraph: onset, duration, character, severity, aggravating and "
                    "relieving factors of the current complaint, in third person")
    review_of_systems: list[str] = Field(
        default_factory=list, description="Symptoms the patient confirmed or denied, e.g. 'Denies chest pain'")
    past_medical_history: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(
        default_factory=list, description="Medications the patient is taking before this visit")
    allergies: list[str] = Field(default_factory=list)
    social_history: list[str] = Field(default_factory=list)

    # --- Objective ---
    vitals: list[str] = Field(default_factory=list, description="Only values stated in the conversation")
    physical_exam: list[str] = Field(default_factory=list, description="Exam findings stated by the clinician")
    results: list[str] = Field(default_factory=list, description="Lab, imaging or test results discussed")

    # --- Assessment & Plan ---
    assessment_and_plan: list[ProblemAssessment]
    follow_up: str = Field(default="", description="When and why the patient should return")

    # --- Safety ---
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Anything unclear, inaudible or contradictory in the conversation that the "
                    "physician should verify. Never guess; list it here instead.")

    def to_text(self) -> str:
        """Plain-text rendering with conventional headings (also used for scoring)."""

        def section(title: str, body) -> str:
            if not body:
                return ""
            if isinstance(body, list):
                body = "\n".join(f"- {b}" for b in body)
            return f"{title}\n\n{body}\n\n"

        ap = "\n\n".join(
            f"{i}. {p.problem}\n{p.assessment}\n" + "\n".join(f"- {x}" for x in p.plan)
            for i, p in enumerate(self.assessment_and_plan, 1)
        )
        return "".join([
            section("CHIEF COMPLAINT", self.chief_complaint),
            section("HISTORY OF PRESENT ILLNESS", self.history_of_present_illness),
            section("REVIEW OF SYSTEMS", self.review_of_systems),
            section("PAST MEDICAL HISTORY", self.past_medical_history),
            section("CURRENT MEDICATIONS", self.current_medications),
            section("ALLERGIES", self.allergies),
            section("SOCIAL HISTORY", self.social_history),
            section("VITALS", self.vitals),
            section("PHYSICAL EXAM", self.physical_exam),
            section("RESULTS", self.results),
            section("ASSESSMENT AND PLAN", ap),
            section("FOLLOW-UP", self.follow_up),
            section("ITEMS TO VERIFY", self.uncertainties),
        ]).strip()


# =====================================================================================
# Objects passed between the agents of the consultation workflow
# =====================================================================================

class WorkflowPlan(BaseModel):
    """Planner output: decides which specialist agents this consultation needs."""

    visit_type: str = Field(description="e.g. 'acute illness', 'follow-up', 'annual exam'")
    needs_diagnosis: bool = Field(description="True if a new or changed diagnosis was discussed")
    needs_prescription: bool = Field(
        description="True if any medication was started, changed, stopped or refilled")
    red_flags: list[str] = Field(
        default_factory=list,
        description="Findings needing urgent physician attention (e.g. chest pain with "
                    "shortness of breath, suicidal ideation, signs of sepsis). Empty if none.")
    rationale: str = Field(description="One or two sentences explaining the plan")


# LLM-facing schemas contain only simple, strictly typed fields: the Gemini Developer API rejects
# free-form dict fields (JSON-schema "additionalProperties") in a response schema. Fields filled
# in later by tools live in subclasses that are never sent to the model.

class DiagnosisDraft(BaseModel):
    """What the LLM fills in."""

    diagnosis: str = Field(description="Clinical diagnosis name")
    icd10_search_term: str = Field(
        description="Short plain term for an ICD-10-CM lookup, e.g. 'acute cystitis'")
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="Why this diagnosis fits, in one or two sentences")
    supporting_evidence: list[str] = Field(description="Facts from the note that support it")


class DiagnosisSuggestion(DiagnosisDraft):
    """DiagnosisDraft plus fields filled in by the terminology tool."""

    icd10_code: str | None = None
    icd10_name: str | None = None
    icd10_candidates: list[dict] = Field(default_factory=list)


class DiagnosisSuggestions(BaseModel):
    diagnoses: list[DiagnosisDraft]
    differentials_to_consider: list[str] = Field(
        default_factory=list, description="Other diagnoses the physician may want to rule out")


class PrescriptionItem(BaseModel):
    """What the LLM fills in."""

    medication: str = Field(description="Drug name as prescribed, e.g. 'Bactrim DS'")
    dose: str = Field(description="e.g. '800/160 mg'")
    route: str = Field(default="oral")
    frequency: str = Field(description="e.g. 'twice daily'")
    duration: str = Field(default="", description="e.g. '3 days'")
    indication: str = Field(description="What it is for")

    def dosage_text(self) -> str:
        return " ".join(x for x in (self.dose, self.route, self.frequency,
                                    f"for {self.duration}" if self.duration else "") if x)


class PrescriptionDraft(PrescriptionItem):
    """PrescriptionItem plus the results of the safety tools."""

    safety_status: Literal["unchecked", "ok", "caution", "blocked"] = "unchecked"
    safety_notes: list[str] = Field(default_factory=list)
    allergy_check: dict | None = None
    interaction_check: dict | None = None


class PrescriptionDrafts(BaseModel):
    prescriptions: list[PrescriptionItem] = Field(
        description="Only medications started, changed or refilled in THIS visit. "
                    "Do not include medications the patient simply continues unchanged.")


class PhysicianDecision(BaseModel):
    """What the physician returns at the review step (human in the loop)."""

    decision: Literal["approve", "reject"]
    reviewer: str = "physician"
    note_text: str | None = Field(default=None, description="Edited note; None keeps the draft")
    approved_diagnoses: list[int] | None = Field(default=None, description="Indexes; None = all")
    approved_prescriptions: list[int] | None = Field(default=None, description="Indexes; None = all")
    override_reason: str | None = Field(
        default=None, description="Required to approve a prescription the safety check blocked")
    comments: str = ""
