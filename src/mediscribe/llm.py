"""Thin wrapper around the Gemini API.

Two jobs:
1. Structured output: ask the model to fill a Pydantic schema, get a validated object back.
2. Retries: the free tier returns 429 when we exceed rate limits; we wait and try again.

Agent workflows use LangChain/LangGraph; this module serves simple utility calls.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from mediscribe.config import settings

T = TypeVar("T", bound=BaseModel)


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
