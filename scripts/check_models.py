"""Which Gemini models are answering right now, and how fast?

For the main model and each fallback, sends:
  1. a plain prompt through the Google SDK directly
  2. a plain prompt through LangChain (what the agents use)
  3. a tool-calling prompt through LangChain
Each call is limited to 30 seconds. Prints before every call so you can see where it waits.

Run:  python scripts/check_models.py
"""

import time

from google.genai import types
from langchain_core.tools import tool

from mediscribe.config import settings
from mediscribe.llm import chat_model, client

TIMEOUT_S = 30


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


def sdk_call(model: str) -> str:
    resp = client().models.generate_content(
        model=model, contents="Reply with: ok",
        config=types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000,
                                           retry_options=types.HttpRetryOptions(attempts=1))))
    return resp.text


models = [settings.gemini_model] + [m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()]
for name in models:
    llm = chat_model("gemini", model=name).model_copy(update={"timeout": TIMEOUT_S})
    checks = (
        ("SDK plain", lambda: sdk_call(name)),
        ("LangChain plain", lambda: llm.invoke("Reply with: ok").content),
        ("LangChain tools", lambda: llm.bind_tools([get_weather]).invoke("What's the weather in Pune?").tool_calls),
    )
    for label, run in checks:
        print(f"{name:24s} {label:16s} calling…", end=" ", flush=True)
        t0 = time.time()
        try:
            out = run()
            print(f"OK    {time.time() - t0:5.1f}s  {str(out)[:60]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {time.time() - t0:5.1f}s  {type(exc).__name__}: {str(exc)[:100]}", flush=True)
