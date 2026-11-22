"""Versioned prompts, stored as plain text files so they can be reviewed and diffed like code.

prompts/<agent>/<version>.txt  ->  load_prompt("scribe", "v2_role_rules")
"""

from pathlib import Path

PROMPT_DIR = Path(__file__).parent


def list_versions(agent: str) -> list[str]:
    return sorted(p.stem for p in (PROMPT_DIR / agent).glob("*.txt"))


def load_prompt(agent: str, version: str, **variables: str) -> str:
    text = (PROMPT_DIR / agent / f"{version}.txt").read_text(encoding="utf-8")
    return text.format(**variables) if variables else text
