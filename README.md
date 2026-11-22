# MediScribe: Agentic Post-Consultation Assistant

> ⚠️ Educational capstone project. Uses **synthetic data only**. Not a medical device; every output requires physician review.

A multi-agent system that takes a doctor–patient consultation and:

1. Transcribes it with speaker labels
2. Writes a structured clinical note (SOAP)
3. Suggests diagnoses with ICD-10 codes, backed by guideline RAG and a knowledge graph
4. Drafts a prescription and checks allergies, interactions and dose limits
5. **Pauses for the physician to approve, edit or reject** (human in the loop)
6. Writes the approved record to a FHIR EHR and produces PDF documents

An analytics agent lets the doctor ask questions about the data in plain English and get charts back.

## Quick start (Windows, PowerShell)

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# 2. Secrets
copy .env.example .env      # then fill in the keys

# 3. Local services (FHIR EHR, Postgres, Neo4j)
docker compose up -d

# 4. Check everything is connected
python scripts/check_setup.py
```

## Project layout

```
src/mediscribe/     application code (agents, tools, guardrails, ...)
scripts/            one-off scripts: setup check, data loading
tests/              pytest tests and evals
docs/concepts/      notes explaining each agentic AI concept used
docs/screenshots/   screenshots used in this README
data/               downloaded/generated data (git-ignored)
```

## Tech stack

LangGraph · CrewAI (comparison) · Gemini + Claude · faster-whisper · MCP (FastMCP) · HAPI FHIR · PostgreSQL · ChromaDB · Neo4j · Langfuse · Presidio · Streamlit · FastAPI · Docker · GCP Cloud Run
