"""Every schema sent to the LLM must be accepted by the Gemini Developer API.

Fake-LLM tests never send schemas to Google, so this checks them directly: no free-form
dict fields (JSON-schema "additionalProperties"), which the Developer API rejects.
"""

import pytest

from mediscribe.schemas import (AlternativeSuggestions, DiagnosisSuggestions, PrescriptionDrafts,
                                SOAPNote, WorkflowPlan)
from mediscribe.transcription import SpeakerLabels

LLM_SCHEMAS = [SOAPNote, WorkflowPlan, DiagnosisSuggestions, PrescriptionDrafts, AlternativeSuggestions,
               SpeakerLabels]


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


@pytest.mark.parametrize("schema", LLM_SCHEMAS, ids=lambda s: s.__name__)
def test_llm_schema_has_no_free_form_objects(schema):
    offending = [n for n in _walk(schema.model_json_schema()) if "additionalProperties" in n]
    assert not offending, f"{schema.__name__} has free-form dict fields: {offending}"
