"""Guideline RAG and patient memory, using an in-memory Chroma and a keyword-hashing embedder
(no model download, deterministic)."""

import hashlib
import re

import chromadb
import numpy as np
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from mediscribe import knowledge as kb

STOP = {"the", "a", "an", "and", "or", "of", "for", "to", "in", "with", "is", "are", "on", "by", "be"}


class KeywordEmbedding(EmbeddingFunction[Documents]):
    """Bag of words hashed into 256 dimensions: texts sharing words end up close."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            v = np.zeros(256)
            for w in re.findall(r"[a-z]+", text.lower()):
                if w not in STOP:
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            vectors.append(v / (np.linalg.norm(v) or 1))
        return vectors

    @staticmethod
    def name():
        return "keyword-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return KeywordEmbedding()


@pytest.fixture
def store():
    client = chromadb.EphemeralClient()
    for name in (kb.GUIDELINES, kb.MEMORY):  # EphemeralClient is shared per process: start clean
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001
            pass
    kb.configure(client=client, embedding_function=KeywordEmbedding())
    yield
    kb.configure()


def test_chunks_one_per_section_with_title_and_source():
    chunks = kb.chunk_markdown(kb.GUIDELINE_DIR / "uti_cystitis.md")
    ids = [c["id"] for c in chunks]
    assert "uti_cystitis#first-line-treatment" in ids
    first_line = next(c for c in chunks if c["id"] == "uti_cystitis#first-line-treatment")
    assert first_line["text"].startswith("Acute uncomplicated cystitis")
    assert "IDSA" in first_line["metadata"]["source"]


def test_retrieves_relevant_guideline_section(store):
    assert kb.build_guideline_index() > 30
    hits = kb.retrieve_guidelines("nitrofurantoin fosfomycin cystitis treatment", k=3, max_distance=1.0)
    assert hits[0]["file"] == "uti_cystitis.md"


def test_unrelated_chunks_are_filtered_by_distance(store):
    kb.build_guideline_index()
    assert kb.retrieve_guidelines("quantum chromodynamics lattice", max_distance=0.5) == []


def test_patient_memory_is_scoped_to_the_patient(store):
    kb.remember_visit("1000", "D2N001", "Acute cystitis. Bactrim blocked: sulfonamide allergy.")
    kb.remember_visit("2000", "D2N002", "Knee osteoarthritis. Started topical diclofenac.")
    recalled = kb.recall_visits("1000", "urinary symptoms again")
    assert [r["encounter_id"] for r in recalled] == ["D2N001"]
    assert kb.recall_visits("3000", "anything") == []
