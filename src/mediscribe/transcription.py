"""Transcriber: audio -> speaker-labelled transcript.

Two steps, and why:
1. faster-whisper (runs locally, free) turns speech into timed text segments.
   Whisper does not know WHO is speaking.
2. An LLM labels each segment as doctor / patient / other from what is said
   ("what brings you in today?" is the doctor). This is cheaper and lighter than an
   audio diarization model, and works well for two-person consultations.

Run:  python -m mediscribe.transcription data/audio/D2N086.mp3
"""

from __future__ import annotations

import argparse
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from mediscribe.config import DATA_DIR, settings
from mediscribe.llm import generate_structured

# Windows can't create the symlinks Hugging Face's cache prefers; it still works, just warns.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono audio

Speaker = Literal["doctor", "patient", "other", "unknown"]


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Speaker = "unknown"


class Turn(BaseModel):
    speaker: Speaker
    text: str
    start: float
    end: float


class Transcript(BaseModel):
    source: str
    language: str
    duration_s: float
    asr_seconds: float = Field(description="time spent on speech-to-text")
    segments: list[Segment]
    turns: list[Turn] = []

    def to_text(self) -> str:
        """Same '[speaker] text' format as ACI-Bench, which the Scribe agent reads."""
        return "\n".join(f"[{t.speaker}] {t.text}" for t in self.turns)


# ---------- 1. Speech to text ----------

@lru_cache
def _whisper():
    from faster_whisper import WhisperModel  # imported lazily: slow to load

    # int8 on CPU = small memory, decent speed on a laptop
    return WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")


def load_audio(audio_path: str | Path):
    """Decode any audio file (mp3, m4a, wav...) to 16 kHz mono float32 with PyAV.

    We do this ourselves instead of letting faster-whisper do it, because some
    faster-whisper / PyAV version combinations are incompatible.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(str(audio_path)) as container:
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray() for f in resampler.resample(None))  # flush
    return np.concatenate(chunks, axis=1).flatten().astype(np.float32) / 32768.0


def speech_to_text(audio_path: str | Path) -> Transcript:
    t0 = time.time()
    audio = load_audio(audio_path)
    segments, info = _whisper().transcribe(
        audio,
        beam_size=5,
        vad_filter=True,  # skip silences
        # A short context prompt nudges Whisper toward medical vocabulary
        initial_prompt="A medical consultation between a doctor and a patient, "
                       "discussing symptoms, medications, blood pressure and lab results.",
    )
    segs = [Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
            for s in segments if s.text.strip()]
    return Transcript(source=str(audio_path), language=info.language,
                      duration_s=round(len(audio) / SAMPLE_RATE, 1), asr_seconds=round(time.time() - t0, 1),
                      segments=segs)


# ---------- 2. Speaker labelling with an LLM (structured output) ----------

class SegmentLabel(BaseModel):
    index: int
    speaker: Literal["doctor", "patient", "other"]


class SpeakerLabels(BaseModel):
    labels: list[SegmentLabel]


LABEL_SYSTEM = """You label who is speaking in a doctor-patient consultation transcript.
- doctor: asks clinical questions, examines, explains diagnoses, orders tests, prescribes.
- patient: describes their own symptoms, history and feelings, answers questions.
- other: anyone else (family member, nurse).
A segment boundary does not always mean the speaker changed. Use the conversation flow.
Return one label for every segment index."""


def label_speakers(transcript: Transcript) -> Transcript:
    numbered = "\n".join(f"{i}: {s.text}" for i, s in enumerate(transcript.segments))
    result = generate_structured(f"Segments:\n{numbered}", SpeakerLabels, system=LABEL_SYSTEM)
    by_index = {lab.index: lab.speaker for lab in result.labels}
    for i, seg in enumerate(transcript.segments):
        seg.speaker = by_index.get(i, "unknown")

    # merge consecutive segments from the same speaker into turns
    turns: list[Turn] = []
    for seg in transcript.segments:
        if turns and turns[-1].speaker == seg.speaker:
            turns[-1].text += " " + seg.text
            turns[-1].end = seg.end
        else:
            turns.append(Turn(speaker=seg.speaker, text=seg.text, start=seg.start, end=seg.end))
    transcript.turns = turns
    return transcript


def transcribe(audio_path: str | Path) -> Transcript:
    return label_speakers(speech_to_text(audio_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    args = ap.parse_args()

    tr = transcribe(args.audio)
    out_dir = DATA_DIR / "processed" / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(args.audio).stem}.json"
    out.write_text(tr.model_dump_json(indent=2), encoding="utf-8")

    print(f"Audio {tr.duration_s}s, speech-to-text took {tr.asr_seconds}s, "
          f"{len(tr.segments)} segments -> {len(tr.turns)} turns\n")
    print(tr.to_text()[:1500])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
