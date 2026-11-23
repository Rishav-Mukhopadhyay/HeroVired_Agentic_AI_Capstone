# RAG, vector databases and long-term memory

## Why RAG

An LLM's medical knowledge is broad but unverifiable, can be out of date, and cannot be cited.
**Retrieval-augmented generation (RAG)** gives the model a small set of trusted documents at question
time and asks it to answer from them and cite them. The physician can then check every claim.

## The knowledge base

`knowledge/guidelines/` holds 14 short summaries of widely used guidelines (IDSA cystitis, ACC/AHA
hypertension, ADA diabetes, GINA asthma, GOLD COPD, ACR gout, and others), written for this project in
our own words and naming their sources. They cover the conditions that appear most in ACI-Bench.

```
knowledge/guidelines/*.md --chunk by "## " section--> ~45 chunks --embed--> ChromaDB "clinical-guidelines"
                                                                                    |
question / clinical note --embed--> nearest chunks (cosine distance <= 0.8) --> prompt with [chunk-id] excerpts
```

- **Chunking:** one chunk per section, each prefixed with the document title so it stands alone.
  Ids are readable and citable, e.g. `uti_cystitis#first-line-treatment`.
- **Embeddings:** Chroma's built-in `all-MiniLM-L6-v2` model, run locally through ONNX: free,
  offline, and no API quota (important on the Gemini free tier).
- **Distance threshold:** chunks that are not similar enough are dropped, so unrelated text never
  reaches the model.

## Where RAG is used

| Where | Query | What the LLM must do |
|---|---|---|
| Diagnostician (`prompts/diagnostician/v2.txt`) | chief complaint + assessed problems | cite supporting excerpt ids in `guideline_refs`; name red-flag differentials |
| Pharmacist, when a prescription is **blocked** (`prompts/pharmacist_alternatives/v1.txt`) | indication + "first-line alternatives to <drug>" | suggest substitutes **only** from the excerpts, with the excerpt's dose |
| `scripts/ask_guidelines.py` | the user's question | answer only from excerpts or say it cannot |

### Grounding checks (hallucination guards)

- A citation is kept only if its id was among the retrieved chunks. Invented ids are removed.
- An alternative drug is kept only if its `guideline_ref` was retrieved **and** it passes the same
  deterministic allergy and interaction checks as the original prescription. In the demo, Bactrim is
  blocked (sulfa allergy) and the guideline's nitrofurantoin is suggested and checked; a sulfonamide
  alternative would be removed by the allergy check.

## Long-term patient memory

A second Chroma collection, `patient-memory`, stores one summary per **approved** visit: complaint,
coded diagnoses, what was prescribed, what was blocked and why, and physician comments. Each entry is
tagged with the patient id.

- At the start of a consultation, `load_patient` recalls that patient's past visits (filtered by patient
  id, ranked by similarity to the new transcript).
- The diagnostician sees them ("PREVIOUS VISITS"), and the physician sees them on the review screen.
- `ehr_writer` stores the new summary after approval.

Memory types in MediScribe:

| Memory | Lifetime | Store |
|---|---|---|
| Workflow state of one consultation | until the visit is finished (survives restarts) | Postgres checkpoints |
| Patient history across visits | permanent | ChromaDB `patient-memory` |
| Official record | permanent | FHIR EHR |

## Commands

```
python scripts/build_knowledge_base.py           # index the guidelines, show sample retrievals
python scripts/ask_guidelines.py "question"      # grounded Q&A with citations
```
