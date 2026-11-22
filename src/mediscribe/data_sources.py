"""Loading the ACI-Bench dataset (doctor-patient dialogues + expert clinical notes).

ACI-Bench: Yim et al., "ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for
Benchmarking Automatic Visit Note Generation", Nature Scientific Data 2023. CC BY 4.0.
https://github.com/wyim/aci-bench
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from mediscribe.config import DATA_DIR

ACI_DIR = DATA_DIR / "raw" / "aci_bench"
ACI_BASE_URL = "https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data"
# train = few-shot examples / prompt tuning, valid = development, test1 = held-out evaluation
ACI_SPLITS = {"train": "train", "valid": "valid", "test": "clinicalnlp_taskB_test1"}

_TURN = re.compile(r"^\[(doctor|patient|patient_guest)\]\s*(.*)$")


def clean_aci_text(text: str) -> str:
    """ACI-Bench text is tokenized ("i , i do n't know ."). Make it read naturally."""
    text = re.sub(r"\s+n't\b", "n't", text)
    text = re.sub(r"\s+([,.?!;:%)])", r"\1", text)
    text = re.sub(r"(\w)\s+-(\w)", r"\1-\2", text)
    text = re.sub(r"([($])\s+", r"\1", text)
    text = re.sub(r"\s+'(s|re|ve|ll|d|m)\b", r"'\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_dialogue(dialogue: str) -> list[dict]:
    """'[doctor] hi .\\n[patient] hello .' -> [{'speaker': 'doctor', 'text': 'hi.'}, ...]"""
    turns = []
    for line in dialogue.splitlines():
        m = _TURN.match(line.strip())
        if m:
            speaker = "patient" if m.group(1) == "patient" else m.group(1)
            if speaker == "patient_guest":
                speaker = "other"
            turns.append({"speaker": speaker, "text": clean_aci_text(m.group(2))})
        elif turns and line.strip():  # continuation line
            turns[-1]["text"] += " " + clean_aci_text(line)
    return turns


def load_aci(split: str = "valid") -> pd.DataFrame:
    """Dialogue + reference note + metadata for one split. Run scripts/download_aci_bench.py first."""
    name = ACI_SPLITS[split]
    dialogues = ACI_DIR / f"{name}.csv"
    if not dialogues.exists():
        raise FileNotFoundError(f"{dialogues} missing - run: python scripts/download_aci_bench.py")
    df = pd.read_csv(dialogues)
    meta = pd.read_csv(ACI_DIR / f"{name}_metadata.csv")
    meta["patient_gender"] = meta["patient_gender"].str.strip()
    df = df.merge(meta.drop(columns=["dataset"]), on="encounter_id", how="left")
    df["split"] = split
    return df


def get_encounter(encounter_id: str) -> pd.Series:
    for split in ACI_SPLITS:
        try:
            df = load_aci(split)
        except FileNotFoundError:
            continue
        hit = df[df.encounter_id == encounter_id]
        if not hit.empty:
            return hit.iloc[0]
    raise KeyError(f"encounter {encounter_id} not found in ACI-Bench")


def raw_dir() -> Path:
    ACI_DIR.mkdir(parents=True, exist_ok=True)
    return ACI_DIR
