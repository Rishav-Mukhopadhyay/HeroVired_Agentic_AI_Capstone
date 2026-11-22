"""Turn an ACI-Bench dialogue into a consultation audio file, with a different
voice for doctor and patient, so we can test transcription end to end.

Uses edge-tts (free Microsoft neural voices, needs internet, no API key).
Also saves the ground-truth turns, so we can measure transcription accuracy.

Run:  python scripts/make_demo_audio.py                      (shortest valid encounter)
      python scripts/make_demo_audio.py --encounter D2N070 --max-turns 40

Tip for the demo video: you can also record a real role-play on your phone and
transcribe that instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import edge_tts

from mediscribe.config import DATA_DIR
from mediscribe.data_sources import get_encounter, load_aci, parse_dialogue

VOICES = {
    "doctor": "en-IN-PrabhatNeural",
    "patient_male": "en-US-GuyNeural",
    "patient_female": "en-US-JennyNeural",
    "other": "en-GB-SoniaNeural",
}


async def synth(text: str, voice: str) -> bytes:
    audio = b""
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encounter", help="ACI-Bench encounter id, e.g. D2N070")
    ap.add_argument("--max-turns", type=int, default=40,
                    help="cut long dialogues to keep audio around 2-4 minutes")
    args = ap.parse_args()

    if args.encounter:
        row = get_encounter(args.encounter)
    else:  # shortest dialogue with at least 30 turns in the validation split
        df = load_aci("valid")
        df["turns"] = df.dialogue.str.count(r"\n")
        row = df[df.turns >= 30].sort_values("turns").iloc[0]

    enc = row["encounter_id"]
    turns = parse_dialogue(row["dialogue"])[: args.max_turns]
    patient_voice = VOICES["patient_female" if row["patient_gender"] == "female" else "patient_male"]

    out_dir = DATA_DIR / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = b""
    for i, t in enumerate(turns, 1):
        voice = VOICES["doctor"] if t["speaker"] == "doctor" else (
            patient_voice if t["speaker"] == "patient" else VOICES["other"])
        audio += await synth(t["text"], voice)
        print(f"\r  synthesising turn {i}/{len(turns)}", end="")
    print()

    (out_dir / f"{enc}.mp3").write_bytes(audio)
    (out_dir / f"{enc}_ground_truth.json").write_text(
        json.dumps({"encounter_id": enc, "turns": turns}, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / f'{enc}.mp3'} ({len(audio) / 1024:.0f} KB) and ground truth JSON")


if __name__ == "__main__":
    asyncio.run(main())
