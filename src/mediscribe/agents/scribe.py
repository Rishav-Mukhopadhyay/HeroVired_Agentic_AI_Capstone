"""Scribe agent: transcript -> structured SOAP note.

The prompt version is a parameter, so the same agent can be run with each version and the
results compared (see scripts/compare_prompts.py).

Run:  python -m mediscribe.agents.scribe data/processed/transcripts/D2N081.json
      python -m mediscribe.agents.scribe --encounter D2N070 --version v2_role_rules
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from mediscribe.config import DATA_DIR
from mediscribe.data_sources import get_encounter, parse_dialogue
from mediscribe.llm import generate_structured
from mediscribe.prompts import load_prompt
from mediscribe.schemas import SOAPNote

DEFAULT_VERSION = "v3_few_shot"
FEW_SHOT_ENCOUNTER = "D2N005"  # short example from the TRAIN split (never valid/test)


def dialogue_to_text(dialogue: str) -> str:
    return "\n".join(f"[{t['speaker']}] {t['text']}" for t in parse_dialogue(dialogue))


@lru_cache
def _few_shot_example() -> dict[str, str]:
    row = get_encounter(FEW_SHOT_ENCOUNTER)
    return {"example_dialogue": dialogue_to_text(row["dialogue"]), "example_note": row["note"]}


def system_prompt(version: str) -> str:
    variables = _few_shot_example() if "few_shot" in version else {}
    return load_prompt("scribe", version, **variables)


def generate_note(transcript: str, version: str = DEFAULT_VERSION, model: str | None = None) -> SOAPNote:
    """transcript: '[doctor] ...\\n[patient] ...' lines."""
    return generate_structured(
        prompt=f"### CONVERSATION TO DOCUMENT\n{transcript}",
        schema=SOAPNote,
        system=system_prompt(version),
        model=model,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", nargs="?", help="transcript JSON from mediscribe.transcription")
    ap.add_argument("--encounter", help="use an ACI-Bench dialogue directly instead")
    ap.add_argument("--version", default=DEFAULT_VERSION)
    args = ap.parse_args()

    if args.encounter:
        text, name = dialogue_to_text(get_encounter(args.encounter)["dialogue"]), args.encounter
    elif args.transcript:
        from mediscribe.transcription import Transcript

        tr = Transcript.model_validate_json(Path(args.transcript).read_text(encoding="utf-8"))
        text, name = tr.to_text(), Path(args.transcript).stem
    else:
        ap.error("give a transcript JSON path or --encounter")

    note = generate_note(text, args.version)
    out_dir = DATA_DIR / "processed" / "notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_{args.version}.json"
    out.write_text(note.model_dump_json(indent=2), encoding="utf-8")
    print(note.to_text())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
