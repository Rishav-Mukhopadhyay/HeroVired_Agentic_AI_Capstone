from mediscribe.prompts import list_versions, load_prompt
from mediscribe.schemas import ProblemAssessment, SOAPNote


def test_all_scribe_prompt_versions_present():
    assert {"v1_baseline", "v2_role_rules", "v3_few_shot"} <= set(list_versions("scribe"))


def test_few_shot_prompt_has_placeholders():
    text = load_prompt("scribe", "v3_few_shot")
    assert "{example_dialogue}" in text and "{example_note}" in text


def test_note_renders_sections_and_skips_empty_ones():
    note = SOAPNote(
        chief_complaint="Knee pain",
        history_of_present_illness="Two weeks of right knee pain.",
        assessment_and_plan=[ProblemAssessment(problem="Knee sprain", assessment="Likely sprain.",
                                               plan=["Ibuprofen 400 mg as needed"])],
    )
    text = note.to_text()
    assert "CHIEF COMPLAINT" in text and "1. Knee sprain" in text
    assert "VITALS" not in text  # empty sections are omitted
