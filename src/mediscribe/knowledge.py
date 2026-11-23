"""Vector memory: guideline retrieval (RAG) and long-term patient memory, in ChromaDB.

Two collections in one persistent store (data/chroma):
- "clinical-guidelines": chunks of knowledge/guidelines/*.md, one chunk per "## " section.
  Agents retrieve the most relevant chunks and cite them by id, e.g. "uti_cystitis#first-line-treatment".
- "patient-memory": one short summary per approved visit, tagged with the patient id. The next
  visit for the same patient recalls them (filtered by patient, ranked by similarity).

Embeddings come from Chroma's built-in local model (all-MiniLM-L6-v2 via ONNX): free, offline,
no API quota. The first use downloads the model (~80 MB).
"""

from __future__ import annotations

import datetime as dt
import re
import threading
from pathlib import Path
from typing import Any

from mediscribe.config import DATA_DIR, ROOT_DIR

GUIDELINE_DIR = ROOT_DIR / "knowledge" / "guidelines"
CHROMA_DIR = DATA_DIR / "chroma"
GUIDELINES = "clinical-guidelines"
MEMORY = "patient-memory"

_lock = threading.Lock()
_client = None
_embedding_function = None


def configure(client=None, embedding_function=None) -> None:
    """Swap the store or embedding model (tests use an in-memory client and a fake embedder)."""
    global _client, _embedding_function
    _client, _embedding_function = client, embedding_function


def _collection(name: str):
    global _client, _embedding_function
    with _lock:  # agents run in parallel threads: create the client once
        if _client is None:
            import chromadb

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        if _embedding_function is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            _embedding_function = DefaultEmbeddingFunction()
        return _client.get_or_create_collection(name, embedding_function=_embedding_function,
                                                metadata={"hnsw:space": "cosine"})


# ---------------------------------------------------------------- guidelines (RAG)

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chunk_markdown(path: Path) -> list[dict]:
    """One chunk per '## ' section; each chunk repeats the document title and source so it
    still makes sense on its own when retrieved."""
    text = path.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", text, re.M)
    title = title.group(1).strip() if title else path.stem
    source = re.search(r"^Source: (.+)$", text, re.M)
    source = source.group(1).strip() if source else "unspecified"
    chunks = []
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        heading, _, body = section.partition("\n")
        heading = heading.strip()
        chunks.append({
            "id": f"{path.stem}#{slug(heading)}",
            "text": f"{title} - {heading}\n{body.strip()}",
            "metadata": {"title": title, "section": heading, "source": source, "file": path.name},
        })
    return chunks


def build_guideline_index(folder: Path = GUIDELINE_DIR) -> int:
    """(Re)build the guideline collection from every .md/.txt file in the folder."""
    files = sorted(list(folder.glob("*.md")) + list(folder.glob("*.txt")))
    chunks = [c for f in files for c in chunk_markdown(f)]
    col = _collection(GUIDELINES)
    existing = col.get()["ids"]
    if existing:
        col.delete(ids=existing)
    if chunks:
        col.upsert(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                   metadatas=[c["metadata"] for c in chunks])
    return len(chunks)


def retrieve_guidelines(query: str, k: int = 4, max_distance: float = 0.8) -> list[dict]:
    """Top-k guideline chunks for a query, most relevant first. Chunks farther than
    max_distance (cosine distance) are dropped so unrelated text is not passed to the LLM."""
    col = _collection(GUIDELINES)
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    hits = []
    for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= max_distance:
            hits.append({"id": id_, "text": doc, "distance": round(dist, 3), **meta})
    return hits


def format_excerpts(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['id']}] ({h['source']})\n{h['text']}" for h in hits) or "(none found)"


# ---------------------------------------------------------------- patient memory

def remember_visit(patient_id: str, encounter_id: str, summary: str, **extra: Any) -> str:
    memory_id = f"{patient_id}:{encounter_id}:{dt.datetime.now(dt.timezone.utc):%Y%m%d%H%M%S}"
    _collection(MEMORY).upsert(
        ids=[memory_id], documents=[summary],
        metadatas=[{"patient_id": str(patient_id), "encounter_id": encounter_id,
                    "date": dt.date.today().isoformat(),
                    **{k: str(v) for k, v in extra.items()}}])
    return memory_id


def recall_visits(patient_id: str, query: str, k: int = 3) -> list[dict]:
    """This patient's past visit summaries, most relevant to the current visit first."""
    col = _collection(MEMORY)
    mine = col.get(where={"patient_id": str(patient_id)})["ids"]
    if not mine:
        return []
    res = col.query(query_texts=[query], n_results=min(k, len(mine)), where={"patient_id": str(patient_id)})
    return [{"id": i, "summary": doc, **meta}
            for i, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]
