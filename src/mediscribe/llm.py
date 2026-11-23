"""Thin wrapper around the Gemini API.

Two jobs:
1. Structured output: ask the model to fill a Pydantic schema, get a validated object back.
2. Retries: the free tier returns 429 when we exceed rate limits; we wait and try again.

Agent workflows use LangChain/LangGraph; this module serves simple utility calls.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from mediscribe.config import settings

T = TypeVar("T", bound=BaseModel)

# "-lite" models use fixed sampling and warn that temperature is ignored; harmless.
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")
# google-genai's async client leaves an un-awaited coroutine when it parses API error responses.
# The error itself is still raised and handled; only the warning is noise.
warnings.filterwarnings("ignore", message="coroutine 'ClientResponse.json' was never awaited")


@lru_cache
def client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in (429, 500, 503)


@retry(retry=retry_if_exception(_retryable), wait=wait_exponential(min=4, max=60),
       stop=stop_after_attempt(6), reraise=True)
def generate_structured(prompt: str, schema: type[T], system: str | None = None,
                        model: str | None = None, temperature: float = 0.0) -> T:
    resp = client().models.generate_content(
        model=model or settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    if resp.parsed is None:  # model returned JSON that didn't fit the schema
        return schema.model_validate_json(resp.text)
    return resp.parsed


@retry(retry=retry_if_exception(_retryable), wait=wait_exponential(min=4, max=60),
       stop=stop_after_attempt(6), reraise=True)
def generate_text(prompt: str, system: str | None = None, model: str | None = None,
                  temperature: float = 0.2) -> str:
    resp = client().models.generate_content(
        model=model or settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
    )
    return resp.text


# ---------- LangChain chat models (used by agents) ----------

def chat_model(provider: str = "gemini", temperature: float = 0.0, model: str | None = None):
    """A LangChain chat model. Agents take this so the LLM can be swapped (Gemini vs Claude)."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Few client-side retries: on overload we prefer to switch model (see agent_middleware)
        # timeout: a call that hangs counts as a failure, so the fallback model takes over
        return ChatGoogleGenerativeAI(model=model or settings.gemini_model, api_key=settings.gemini_api_key,
                                      temperature=temperature, max_retries=1, timeout=60)
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.claude_model, api_key=settings.anthropic_api_key,
                             temperature=temperature, max_retries=3, max_tokens=4096)
    raise ValueError(f"unknown provider {provider!r}")


def agent_middleware(provider: str = "gemini") -> list:
    """Resilience for agent model calls.

    - ModelRetryMiddleware: retry a failing call with exponential backoff.
    - ModelFallbackMiddleware: if the model still fails (e.g. 503 "high demand" on the free
      tier), switch to the next model in GEMINI_FALLBACK_MODELS.
    Order matters: the first middleware wraps the others, so every fallback model is retried too.
    """
    from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware

    # 5s, 10s, 20s: long enough for a per-minute rate limit (429) to clear
    retry = ModelRetryMiddleware(max_retries=3, initial_delay=5.0, max_delay=30.0, on_failure="error")
    if provider != "gemini":
        return [retry]
    fallbacks = [chat_model("gemini", model=m.strip())
                 for m in settings.gemini_fallback_models.split(",") if m.strip()]
    return ([ModelFallbackMiddleware(*fallbacks)] if fallbacks else []) + [retry]
