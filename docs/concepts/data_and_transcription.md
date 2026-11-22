# Data and the Transcriber agent

## The data, and why it is synthetic

| Data | Source | Used for |
|---|---|---|
| 127 doctor–patient dialogues + expert-written notes | ACI-Bench (CC BY 4.0) | Input for the agents; reference notes for evaluation |
| 127 FHIR patients (name, age, gender, conditions) | ACI-Bench metadata, seeded by `scripts/seed_fhir.py` | The "hospital EHR" the agents read and write |
| Current medications, allergies | **Synthetic**, added by the seed script | Gives the Pharmacist agent something to check against |
| Consultation audio | Generated from dialogues with edge-tts | Input for the Transcriber |

Splits: **train** (67) for few-shot examples and prompt tuning, **valid** (20) for development,
**test** (40) held out for final evaluation. Never tune prompts on the test split.

Real patient data would need consent, de-identification and legal approval (HIPAA in the US,
DPDP Act in India). Using synthetic data avoids that entirely.

## How the Transcriber works

```
audio.mp3 ──► faster-whisper (local) ──► timed segments ──► LLM labels speakers ──► turns
                    "what text?"                               "who said it?"
```

- **faster-whisper** is OpenAI's Whisper model, optimised to run on a laptop CPU (`int8`).
- Whisper's `initial_prompt` is a small example of **prompt tuning** even for speech models:
  telling it this is a medical consultation improves drug and symptom spelling.
- **Speaker labelling with structured output:** we pass numbered segments to Gemini and require
  the answer to match a Pydantic schema (`SpeakerLabels`). The model cannot return free text; we
  get a validated Python object back. This "structured objects" pattern is used by every agent.
- **Retries:** `mediscribe/llm.py` retries automatically on 429 (rate limit) errors from the free tier.

## Measuring accuracy

`scripts/eval_transcription.py` compares against the ground truth:
- **WER** (word error rate): share of words wrong.
- **Speaker accuracy**: share of words attributed to the right speaker.

Result on encounter D2N081 (6.4 min audio, Whisper `base` on a laptop CPU):

| Metric | Value |
|---|---|
| Word error rate | 9.7% |
| Speaker accuracy | 96.7% |
| Speech-to-text time | 65 s (~6× faster than real time) |

Known limitation: Whisper sometimes puts two speakers in one segment, and labels are assigned
per segment, so a few short replies get merged into the other speaker's turn. Some word errors come
from the dataset itself (its dialogues were produced by speech recognition, e.g. "non-smile cell"
for "non-small cell").
