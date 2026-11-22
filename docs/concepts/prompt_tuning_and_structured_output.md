# Prompt tuning and structured output: the Scribe agent

The Scribe agent turns a transcript into a SOAP note (Subjective, Objective, Assessment, Plan),
the standard format physicians use.

## Structured output

The note is a Pydantic model (`mediscribe/schemas.py`), not free text. The schema, including each
field's `description`, is sent to Gemini, which must return JSON matching it. Benefits:

- Downstream code (review screen, FHIR writer, PDF generator) uses exact fields.
- Field descriptions act as extra instructions ("Only values stated in the conversation").
- An `uncertainties` field gives the model a legitimate place to put doubts instead of guessing,
  a simple but effective hallucination guard.

## Prompt versions

Prompts live in `src/mediscribe/prompts/scribe/` as text files, versioned and reviewed like code.
Each version changes one thing, so the comparison isolates its effect:

| Version | Technique | What changes |
|---|---|---|
| `v1_baseline` | Zero-shot, one sentence | Nothing but the task |
| `v2_role_rules` | Role prompting + explicit rules | Scribe persona, "only stated facts", ASR-noise handling, style and medication rules |
| `v3_few_shot` | v2 + one worked example | A short train-split conversation and its expert note, showing the expected detail and tone |

The few-shot example comes from the **train** split; evaluation uses the **valid** split, so the
model never sees the answers it is scored against.

## How versions are compared

`scripts/compare_prompts.py` runs every version on the same validation encounters and scores the
notes against the ACI-Bench expert notes:

- **ROUGE-L F1**: overlap of word sequences with the expert note.
- **Section coverage**: share of the expert note's sections our note also filled in.
- **Unsupported medications**: drug names in our note that never occur in the conversation, a cheap
  hallucination detector (lower is better).
- **Flags**: how many items the model asked the physician to verify.
- **Latency** per note.

Results are written to `docs/results/prompt_comparison.md`.
