"""Ask a clinical question; get an answer grounded in the guideline knowledge base, with citations.

This is RAG on its own: retrieve relevant chunks -> give them to the LLM -> the LLM must answer
only from them and cite chunk ids -> citations are checked against what was retrieved.

Run:  python scripts/ask_guidelines.py "First-line antibiotic for cystitis if allergic to sulfa?"
"""

import sys

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from mediscribe.knowledge import format_excerpts, retrieve_guidelines
from mediscribe.llm import generate_structured, last_model_used

SYSTEM = """Answer the clinician's question using ONLY the guideline excerpts provided.
Cite the ids of the excerpts you used. If the excerpts do not answer the question, say so
and do not answer from general knowledge."""


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(description="Ids of the excerpts used, exactly as given in brackets")
    answerable_from_excerpts: bool


console = Console()
question = " ".join(sys.argv[1:]) or "First-line antibiotic for uncomplicated cystitis if the patient is allergic to sulfa?"
hits = retrieve_guidelines(question, k=4)
console.print(f"[bold]Q:[/bold] {question}\n[dim]Retrieved: {', '.join(h['id'] for h in hits) or 'nothing'}[/dim]\n")
result = generate_structured(f"### GUIDELINE EXCERPTS\n{format_excerpts(hits)}\n\n### QUESTION\n{question}",
                             GroundedAnswer, system=SYSTEM)
valid = [c for c in result.citations if c in {h["id"] for h in hits}]
invented = [c for c in result.citations if c not in valid]
console.print(Panel(result.answer, title=f"Answer ({last_model_used()})"))
console.print(f"Citations: {', '.join(valid) or 'none'}")
if invented:
    console.print(f"[red]Removed citations not in the retrieved excerpts: {', '.join(invented)}[/red]")
if not result.answerable_from_excerpts:
    console.print("[yellow]The knowledge base does not cover this question.[/yellow]")
