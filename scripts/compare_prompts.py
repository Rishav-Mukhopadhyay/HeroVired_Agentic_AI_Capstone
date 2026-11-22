"""Prompt tuning experiment: run every Scribe prompt version on the same consultations and
score each note against the ACI-Bench expert note.

Metrics (higher is better unless noted):
- ROUGE-L F1        word-sequence overlap with the expert note (style + content)
- Section coverage  share of the expert note's sections that our note also filled in
- Unsupported meds  medication names in our note that never appear in the conversation
                    (a hallucination signal, LOWER is better)
- Flags             items the model listed in `uncertainties` for the physician to verify
- Seconds           latency per note

Uses the VALID split only. Each note is one LLM call, so --n 3 = 9 calls on the free tier.

Run:  python scripts/compare_prompts.py --n 3
"""

from __future__ import annotations

import argparse
import re
import time

import pandas as pd
from rouge_score import rouge_scorer

from mediscribe.agents.scribe import dialogue_to_text, generate_note
from mediscribe.config import DATA_DIR, ROOT_DIR
from mediscribe.data_sources import load_aci
from mediscribe.prompts import list_versions
from mediscribe.schemas import SOAPNote

# Reference-note headings -> the SOAPNote field that covers them
SECTION_MAP = {
    r"CHIEF COMPLAINT|^CC": "chief_complaint",
    r"HISTORY OF PRESENT ILLNESS|^HPI": "history_of_present_illness",
    r"REVIEW OF SYSTEMS": "review_of_systems",
    r"MEDICATIONS": "current_medications",
    r"(PAST|MEDICAL) HISTORY": "past_medical_history",
    r"SOCIAL HISTORY": "social_history",
    r"VITALS": "vitals",
    r"EXAM": "physical_exam",
    r"RESULTS": "results",
    r"ASSESSMENT|PLAN|IMPRESSION": "assessment_and_plan",
}
DOSE = re.compile(r"\b([a-zA-Z][a-zA-Z-]{3,})\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|ml)\b", re.I)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def reference_fields(note: str) -> set[str]:
    headings = re.findall(r"^([A-Z][A-Z /&]+):?\s*$", note, re.M)
    return {field for h in headings for pat, field in SECTION_MAP.items() if re.search(pat, h.strip())}


def section_coverage(note: SOAPNote, reference: str) -> float:
    wanted = reference_fields(reference)
    if not wanted:
        return float("nan")
    return sum(bool(getattr(note, f)) for f in wanted) / len(wanted)


def unsupported_meds(note: SOAPNote, transcript: str) -> list[str]:
    names = {m.split()[0].strip(",.").lower() for m in note.current_medications if m.split()}
    plan_text = " ".join(x for p in note.assessment_and_plan for x in p.plan)
    names |= {m.group(1).lower() for m in DOSE.finditer(plan_text)}
    said = transcript.lower()
    return sorted(n for n in names if len(n) > 3 and n not in said)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of valid encounters")
    ap.add_argument("--versions", nargs="*", default=list_versions("scribe"))
    args = ap.parse_args()

    df = load_aci("valid").head(args.n)
    rows = []
    for _, enc in df.iterrows():
        transcript = dialogue_to_text(enc["dialogue"])
        for version in args.versions:
            t0 = time.time()
            note = generate_note(transcript, version)
            secs = time.time() - t0
            bad = unsupported_meds(note, transcript)
            rows.append({
                "encounter": enc["encounter_id"],
                "version": version,
                "rougeL_f1": scorer.score(enc["note"], note.to_text())["rougeL"].fmeasure,
                "section_coverage": section_coverage(note, enc["note"]),
                "unsupported_meds": len(bad),
                "unsupported_list": ", ".join(bad),
                "flags": len(note.uncertainties),
                "seconds": secs,
            })
            r = rows[-1]
            print(f"{r['encounter']} {version:15s} ROUGE-L {r['rougeL_f1']:.3f}  "
                  f"coverage {r['section_coverage']:.0%}  unsupported {r['unsupported_meds']}  "
                  f"flags {r['flags']}  {secs:.1f}s")

    res = pd.DataFrame(rows)
    out_csv = DATA_DIR / "processed" / "prompt_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)

    summary = (res.groupby("version")
               .agg(rougeL_f1=("rougeL_f1", "mean"), section_coverage=("section_coverage", "mean"),
                    unsupported_meds=("unsupported_meds", "sum"), flags=("flags", "mean"),
                    seconds=("seconds", "mean"))
               .round(3))
    print("\n", summary.to_string())

    md = ROOT_DIR / "docs" / "results" / "prompt_comparison.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "# Scribe prompt comparison\n\n"
        f"{len(df)} ACI-Bench validation encounters ({', '.join(df.encounter_id)}), "
        "scored against the expert-written reference notes.\n\n"
        + summary.to_markdown() + "\n\n"
        "- `rougeL_f1`, `section_coverage`, `flags`, `seconds`: mean per note\n"
        "- `unsupported_meds`: total medication names not found in the conversation (lower is better)\n\n"
        "## Per-encounter results\n\n" + res.round(3).to_markdown(index=False) + "\n",
        encoding="utf-8")
    print(f"\nSaved {out_csv} and {md}")


if __name__ == "__main__":
    main()
