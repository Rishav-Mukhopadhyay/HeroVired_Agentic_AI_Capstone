"""Measure how good the Transcriber is against the ground-truth dialogue.

- WER (word error rate): % of words wrong. Lower is better; under ~10% is good.
- Speaker accuracy: % of ground-truth words attributed to the right speaker
  (aligned word by word, so it does not depend on how segments were split).

Run:  python scripts/eval_transcription.py [ENCOUNTER_ID]
"""

from __future__ import annotations

import difflib
import json
import re
import sys

import jiwer

from mediscribe.config import DATA_DIR
from mediscribe.transcription import Transcript


def words_with_speakers(turns) -> tuple[list[str], list[str]]:
    words, speakers = [], []
    for t in turns:
        for w in re.findall(r"[a-z0-9']+", t["text"].lower()):
            words.append(w)
            speakers.append(t["speaker"])
    return words, speakers


if len(sys.argv) > 1:
    enc = sys.argv[1]
else:  # default: the most recently generated demo audio
    latest = max((DATA_DIR / "audio").glob("*_ground_truth.json"), key=lambda p: p.stat().st_mtime)
    enc = latest.name.removesuffix("_ground_truth.json")
truth = json.loads((DATA_DIR / "audio" / f"{enc}_ground_truth.json").read_text(encoding="utf-8"))
hyp = Transcript.model_validate_json(
    (DATA_DIR / "processed" / "transcripts" / f"{enc}.json").read_text(encoding="utf-8"))

ref_w, ref_s = words_with_speakers(truth["turns"])
hyp_w, hyp_s = words_with_speakers([t.model_dump() for t in hyp.turns])

wer = jiwer.wer(" ".join(ref_w), " ".join(hyp_w))

matched = correct = 0
for block in difflib.SequenceMatcher(a=ref_w, b=hyp_w, autojunk=False).get_matching_blocks():
    for k in range(block.size):
        matched += 1
        correct += ref_s[block.a + k] == hyp_s[block.b + k]

print(f"Encounter {enc}")
print(f"  Word error rate (WER):  {wer:.1%}")
print(f"  Speaker accuracy:       {correct / max(matched, 1):.1%}  (over {matched} aligned words)")
print(f"  Turns: ground truth {len(truth['turns'])}, transcript {len(hyp.turns)}")
