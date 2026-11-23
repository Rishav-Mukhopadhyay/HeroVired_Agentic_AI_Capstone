"""Run a consultation through the multi-agent workflow, with physician review in the terminal.

Start:   python scripts/run_consultation.py --encounter D2N001 --transcript demo/consultations/D2N001_uti_followup.txt
         python scripts/run_consultation.py --encounter D2N081 --transcript data/processed/transcripts/D2N081.json
         python scripts/run_consultation.py --encounter D2N068          (uses the ACI-Bench dialogue)
Resume:  python scripts/run_consultation.py --resume <thread-id>         (review later, or continue after
                                                                          an error, even after a restart)
Diagram: python scripts/run_consultation.py --draw
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mediscribe.config import DATA_DIR, ROOT_DIR
from mediscribe.workflow import build_graph, postgres_checkpointer

console = Console()
STATUS_STYLE = {"ok": "green", "caution": "yellow", "blocked": "bold red", "unchecked": "dim"}


def load_transcript(args) -> str:
    if args.transcript:
        path = Path(args.transcript)
        if path.suffix == ".json":
            from mediscribe.transcription import Transcript

            return Transcript.model_validate_json(path.read_text(encoding="utf-8")).to_text()
        return path.read_text(encoding="utf-8")
    from mediscribe.agents.scribe import dialogue_to_text
    from mediscribe.data_sources import get_encounter

    return dialogue_to_text(get_encounter(args.encounter)["dialogue"])


def run(graph, payload, config) -> dict | None:
    """Stream the graph; print each finished node; return the review packet if it pauses."""
    for update in graph.stream(payload, config, stream_mode="updates"):
        for node, data in update.items():
            if node == "__interrupt__":
                return data[0].value
            for entry in (data or {}).get("audit", []):
                extra = {k: v for k, v in entry.items() if k not in ("ts", "node", "event")}
                console.print(f"[green]✓ {node:15s}[/green] {entry['event']}  [dim]{extra or ''}[/dim]")
    return None


def safe_run(graph, payload, config, thread_id: str) -> dict | None:
    """Like run(), but an error (e.g. every model overloaded) ends with a resume hint."""
    try:
        return run(graph, payload, config)
    except Exception as exc:  # noqa: BLE001
        nxt = ", ".join(graph.get_state(config).next) or "?"
        console.print(f"\n[red]Stopped at '{nxt}': {type(exc).__name__}: {str(exc)[:200]}[/red]\n"
                      f"Finished steps are saved. When the model is available again, continue with:\n"
                      f"  python scripts/run_consultation.py --resume {thread_id}")
        return None


def dosage(*parts: str) -> str:
    """Join dose parts, skipping blanks, placeholders and repeats ('single dose single dose')."""
    out: list[str] = []
    for part in parts:
        part = (part or "").strip()
        if part and part.lower() not in ("not specified", "n/a", "unspecified") and part.lower() not in \
                [o.lower() for o in out] and not any(part.lower() in o.lower() for o in out):
            out.append(part)
    return " ".join(out) or "dose per prescriber"


def show_packet(p: dict) -> None:
    pt = p["patient"]
    console.print(Panel(f"{pt['name']} · {pt['gender']} · born {pt['birthDate']}\n"
                        f"Conditions: {', '.join(pt['conditions']) or '-'}\n"
                        f"Medications: {', '.join(pt['medications']) or '-'}\n"
                        f"[bold]Allergies: {', '.join(pt['allergies']) or 'none'}[/bold]",
                        title=f"Physician review · encounter {p['encounter_id']}"))
    if p.get("memory"):
        console.print(Panel("\n".join(f"{m['date']}: {m['summary']}" for m in p["memory"]),
                            title="Previous visits (patient memory)", style="cyan"))
    if p["red_flags"]:
        console.print(Panel("\n".join(p["red_flags"]), title="RED FLAGS", style="bold red"))
    console.print(Panel(p["note_text"], title="Draft clinical note (SOAP)"))

    dx_table = Table(title="Suggested diagnoses")
    for col in ("#", "Diagnosis", "ICD-10-CM", "Confidence", "Reasoning", "Guideline"):
        dx_table.add_column(col, overflow="fold")
    for i, d in enumerate(p["diagnoses"]):
        dx_table.add_row(str(i), d["diagnosis"], f"{d.get('icd10_code') or '?'} {d.get('icd10_name') or ''}",
                         d["confidence"], d["reasoning"], "\n".join(d.get("guideline_refs", [])) or "-")
    console.print(dx_table)
    if p["differentials"]:
        console.print(f"[dim]Differentials to consider: {', '.join(p['differentials'])}[/dim]")

    rx_table = Table(title="Draft prescriptions (safety-checked)")
    for col in ("#", "Medication", "Dosage", "Safety", "Findings"):
        rx_table.add_column(col, overflow="fold")
    for i, rx in enumerate(p["prescriptions"]):
        style = STATUS_STYLE[rx["safety_status"]]
        rx_table.add_row(str(i), rx["medication"], dosage(rx["dose"], rx["route"], rx["frequency"], rx["duration"]), f"[{style}]{rx['safety_status'].upper()}[/{style}]",
                         "\n".join(rx["safety_notes"]) or "-")
    console.print(rx_table if p["prescriptions"] else "[dim]No new prescriptions.[/dim]")

    for rx in p["prescriptions"]:
        if rx.get("alternatives"):
            alt = Table(title=f"Guideline alternatives to {rx['medication']} (safety-checked)")
            for col in ("Medication", "Dosage", "Safety", "Why", "Guideline"):
                alt.add_column(col, overflow="fold")
            for a in rx["alternatives"]:
                style = STATUS_STYLE[a["safety_status"]]
                alt.add_row(a["medication"], dosage(a["dose"], a["frequency"], a["duration"]),
                            f"[{style}]{a['safety_status'].upper()}[/{style}]", a["rationale"], a["guideline_ref"])
            console.print(alt)

    cited = {r for d in p["diagnoses"] for r in d.get("guideline_refs", [])} | \
            {a["guideline_ref"] for rx in p["prescriptions"] for a in rx.get("alternatives", [])}
    if cited:
        console.print("[dim]Sources: " + "; ".join(
            f"{r} = {p['guidelines'][r]['source']}" for r in sorted(cited) if r in p.get("guidelines", {})) + "[/dim]")


def ask_decision(p: dict, thread_id: str) -> dict | None:
    console.print("\n[bold]Decision:[/bold] \\[a]pprove all  \\[s]elect items  "
                  "\\[e]dit note & approve  \\[r]eject  \\[l]ater")
    choice = input("> ").strip().lower()
    if choice == "l":
        console.print(f"Paused. Resume any time with:\n  python scripts/run_consultation.py --resume {thread_id}")
        return None
    if choice == "r":
        return {"decision": "reject", "comments": input("Reason for rejecting: ")}

    decision = {"decision": "approve", "reviewer": os.environ.get("USERNAME", "physician")}
    if choice == "s":
        decision["approved_diagnoses"] = _indexes(input("Diagnoses to approve (e.g. 0,1; blank = none): "))
        decision["approved_prescriptions"] = _indexes(input("Prescriptions to approve (e.g. 0; blank = none): "))
    if choice == "e":
        path = DATA_DIR / "processed" / "review" / f"{thread_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(p["note_text"], encoding="utf-8")
        if sys.platform == "win32":
            os.startfile(path)  # opens in Notepad
        input(f"Edit and save {path}, then press Enter… ")
        decision["note_text"] = path.read_text(encoding="utf-8")

    chosen = decision.get("approved_prescriptions", range(len(p["prescriptions"])))
    if any(p["prescriptions"][i]["safety_status"] == "blocked" for i in chosen):
        reason = input("A selected prescription is BLOCKED by a safety check. Override reason "
                       "(leave blank to exclude it): ").strip()
        if reason:
            decision["override_reason"] = reason
    decision["comments"] = input("Comments (optional): ")
    return decision


def _indexes(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def finish(graph, decision: dict, config) -> None:
    run(graph, Command(resume=decision), config)
    state = graph.get_state(config).values
    console.print(f"\n[bold]Status:[/bold] {state['status']}")
    for w in state.get("ehr_writes", []):
        console.print(f"  written: {w['resourceType']}/{w['id']}")
    audit = Table(title="Audit trail")
    for col in ("Time (UTC)", "Node", "Event"):
        audit.add_column(col, overflow="fold")
    for e in state["audit"]:
        audit.add_row(e["ts"][11:19], e["node"], e["event"])
    console.print(audit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encounter", help="encounter id; links to the patient in the EHR")
    ap.add_argument("--transcript", help=".txt with [doctor]/[patient] lines, or a transcript .json")
    ap.add_argument("--resume", metavar="THREAD_ID")
    ap.add_argument("--draw", action="store_true", help="save the workflow diagram (Mermaid)")
    args = ap.parse_args()

    if args.draw:
        out = ROOT_DIR / "docs" / "diagrams" / "consultation_workflow.mmd"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_graph().get_graph().draw_mermaid(), encoding="utf-8")
        console.print(f"Saved {out}. Paste it into https://mermaid.live to view or export as PNG.")
        return

    with postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)

        if args.resume:
            thread_id = args.resume
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = graph.get_state(config)
            if not snapshot.next:
                console.print(f"Thread {thread_id} has finished (status: "
                              f"{snapshot.values.get('status', 'unknown')}).")
                return
            done = [e["node"] for e in snapshot.values.get("audit", [])]
            console.print(f"[dim]Loaded thread {thread_id} from the Postgres checkpoint. "
                          f"Already done: {', '.join(done) or 'nothing'}. Next: {', '.join(snapshot.next)}[/dim]")
            if snapshot.next == ("physician_review",) and snapshot.tasks[0].interrupts:
                packet = snapshot.tasks[0].interrupts[0].value
            else:  # stopped by an error earlier: continue from the last saved step
                packet = safe_run(graph, None, config, thread_id)
        else:
            if not args.encounter:
                ap.error("--encounter is required (or use --resume / --draw)")
            thread_id = f"{args.encounter}-{dt.datetime.now():%Y%m%d-%H%M%S}"
            config = {"configurable": {"thread_id": thread_id}}
            console.print(f"[bold]Consultation {thread_id}[/bold]\n")
            packet = safe_run(graph, {"encounter_id": args.encounter, "transcript": load_transcript(args),
                                      "audit": []}, config, thread_id)
        if packet is None:
            return

        show_packet(packet)
        decision = ask_decision(packet, thread_id)
        if decision:
            finish(graph, decision, config)


if __name__ == "__main__":
    main()
