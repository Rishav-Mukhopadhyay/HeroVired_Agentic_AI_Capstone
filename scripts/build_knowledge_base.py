"""Build (or rebuild) the guideline vector index in ChromaDB from knowledge/guidelines/.

Run:  python scripts/build_knowledge_base.py
The first run downloads the local embedding model (~80 MB, once).
"""

from rich.console import Console
from rich.table import Table

from mediscribe.knowledge import CHROMA_DIR, GUIDELINE_DIR, build_guideline_index, retrieve_guidelines

console = Console()
n = build_guideline_index()
console.print(f"Indexed {n} chunks from {GUIDELINE_DIR} into {CHROMA_DIR}\n")

table = Table(title="Retrieval check: top 3 chunks per query (lower distance = more similar)")
for col in ("Query", "Distance", "Chunk id"):
    table.add_column(col, overflow="fold")
for query in ("burning urination positive nitrites, patient allergic to sulfa",
              "blood pressure 150/95 already on lisinopril",
              "gout flare big toe, kidney disease",
              "patient on warfarin needs an antibiotic"):
    for i, hit in enumerate(retrieve_guidelines(query, k=3, max_distance=1.0)):
        table.add_row(query if i == 0 else "", f"{hit['distance']:.3f}", hit["id"])
console.print(table)
