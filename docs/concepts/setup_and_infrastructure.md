# Setup and infrastructure

## 1. Why an agentic project needs "boring" infrastructure first

An agent is an LLM that can **decide** what to do next and **act** through tools.
Acting means it needs real systems to act on. For MediScribe those are:

| Service | Role for the agents | Runs in |
|---|---|---|
| HAPI FHIR | The "hospital EHR" the EHR agent writes approved notes into | Docker |
| PostgreSQL | App database + saved agent state (checkpoints) + analytics data | Docker |
| Neo4j | Knowledge graph for GraphRAG | Docker |
| Langfuse Cloud | Watches every agent step (monitoring) | Cloud, free |
| Gemini / Claude | The "brains" | API |

Docker lets us start all of these with one command and throw them away safely.

## 2. What is FHIR?

FHIR (Fast Healthcare Interoperability Resources) is the standard way hospital systems exchange data.
Everything is a **resource** with a fixed JSON shape: `Patient`, `Encounter`, `Condition` (diagnosis),
`MedicationRequest` (prescription), `DocumentReference` (clinical note). Real EHRs like Epic expose FHIR APIs,
so an agent that can write FHIR can, in principle, update a real EHR.

## 3. Secrets hygiene (first AI-security lesson)

- API keys live only in `.env`, which is in `.gitignore`, so it can never be committed.
- `.env.example` documents which keys exist, with empty values, and **is** committed.
- Code never reads `os.environ` directly; it goes through `mediscribe/config.py` (one place to audit).
- The check script prints only the first 6 characters of each key.

## 4. SDLC with AI

This project was built with an AI pair-programmer (Claude) generating code while the developer
reviewed every file, ran and debugged it, decided what to keep, and made every commit personally.
That is the "human in the loop" applied to the software development lifecycle itself.
