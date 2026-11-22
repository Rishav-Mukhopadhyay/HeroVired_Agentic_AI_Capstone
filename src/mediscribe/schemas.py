"""Structured objects shared by the agents.

Every agent returns one of these Pydantic models instead of free text. The LLM is forced to
fill the schema (structured output), so downstream code - the physician review screen, FHIR
writer, PDF generator - can rely on exact fields instead of parsing prose.
Field descriptions are sent to the model, so they double as instructions.
"""

from __future__ import annotations

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
