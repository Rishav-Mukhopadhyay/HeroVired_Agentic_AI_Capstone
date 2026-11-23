# Clinical knowledge base

`guidelines/` holds short, educational summaries of widely used clinical practice guidelines,
written for this project in our own words. Each file names the guideline it summarises.

**These summaries are not a substitute for the source guidelines** and may omit details,
exceptions and updates. They exist so the agents can retrieve and cite relevant guidance (RAG).
Every agent output remains decision support for a physician.

Add more `.md` or `.txt` files here and rebuild the index:

```
python scripts/build_knowledge_base.py
```
