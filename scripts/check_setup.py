"""Health check: are the keys set, and can we reach every service?

Run:  python scripts/check_setup.py

Costs nothing: the Gemini test is one tiny free-tier call, and the Claude
check only lists models (no tokens used).
"""

from __future__ import annotations

import traceback

import requests
from rich.console import Console
from rich.table import Table

from mediscribe.config import settings

console = Console()
results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: run a check, record PASS/FAIL and a short detail message."""

    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            results.append((name, False, f"{type(exc).__name__}: {exc}"[:200]))
            if console.is_terminal:
                console.print(f"[dim]{traceback.format_exc(limit=1)}[/dim]")
        return fn

    return wrap


def mask(value: str) -> str:
    return f"{value[:6]}…({len(value)} chars)" if value else "MISSING"


@check(".env keys present")
def _keys():
    missing = [k for k in ("gemini_api_key", "anthropic_api_key",
                           "langfuse_public_key", "langfuse_secret_key")
               if not getattr(settings, k)]
    if missing:
        raise ValueError(f"empty in .env: {', '.join(missing)}")
    return f"Gemini {mask(settings.gemini_api_key)}, Langfuse {mask(settings.langfuse_public_key)}"


@check("Gemini API (free tier)")
def _gemini():
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    names = [m.name.removeprefix("models/") for m in client.models.list()]
    flash = sorted(n for n in names if "flash" in n)
    embed = sorted(n for n in names if "embed" in n)
    console.print(f"[cyan]Flash models your key can use:[/cyan] {', '.join(flash)}")
    console.print(f"[cyan]Embedding models:[/cyan] {', '.join(embed)}")
    reply = client.models.generate_content(
        model=settings.gemini_model, contents="Reply with exactly: MediScribe ready"
    )
    return f"model={settings.gemini_model} → {reply.text.strip()!r}"


@check("Claude API key (no tokens used)")
def _claude():
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    ids = [m.id for m in client.models.list(limit=50).data]
    haiku = [i for i in ids if "haiku" in i]
    if settings.claude_model not in ids:
        return f"key OK, but {settings.claude_model} not listed. Haiku models: {haiku}"
    return f"key OK, {settings.claude_model} available"


@check("Langfuse Cloud")
def _langfuse():
    from langfuse import Langfuse

    lf = Langfuse(public_key=settings.langfuse_public_key,
                  secret_key=settings.langfuse_secret_key,
                  host=settings.langfuse_host)
    if not lf.auth_check():
        raise RuntimeError("auth_check returned False - check keys and host region")
    return settings.langfuse_host


@check("FHIR server (HAPI)")
def _fhir():
    r = requests.get(f"{settings.fhir_base_url}/metadata", timeout=10)
    r.raise_for_status()
    return f"FHIR {r.json().get('fhirVersion')} at {settings.fhir_base_url}"


@check("PostgreSQL")
def _postgres():
    from sqlalchemy import create_engine, text

    with create_engine(settings.postgres_url).connect() as conn:
        return conn.execute(text("select version()")).scalar().split(",")[0]


@check("Neo4j")
def _neo4j():
    from neo4j import GraphDatabase

    with GraphDatabase.driver(settings.neo4j_uri,
                              auth=(settings.neo4j_user, settings.neo4j_password)) as d:
        d.verify_connectivity()
    return settings.neo4j_uri


table = Table(title="MediScribe setup check")
table.add_column("Check")
table.add_column("Status")
table.add_column("Detail", overflow="fold")
for name, ok, detail in results:
    table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
console.print(table)
