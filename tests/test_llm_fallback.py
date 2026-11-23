"""Resilience: when every Gemini model fails, the call falls back through the chain to Claude."""

from google.genai import errors

from mediscribe import llm
from mediscribe.schemas import WorkflowPlan

PLAN = WorkflowPlan(visit_type="acute", needs_diagnosis=True, needs_prescription=False, rationale="x")


def test_falls_back_through_gemini_models_then_claude(monkeypatch):
    tried = []

    def overloaded(prompt, schema, system, model, temperature):
        tried.append(model)
        raise errors.ServerError(503, {"error": {"message": "high demand"}})

    monkeypatch.setattr(llm, "_generate_structured_once", overloaded)
    monkeypatch.setattr(llm, "_claude_structured", lambda prompt, schema, system: PLAN)
    monkeypatch.setattr(llm.settings, "claude_fallback", True)
    monkeypatch.setattr(llm.settings, "anthropic_api_key", "test-key")

    assert llm.generate_structured("transcript", WorkflowPlan) == PLAN
    assert tried == llm.model_chain()  # every Gemini model was tried, in order
    assert llm.last_model_used() == llm.settings.claude_model


def test_second_model_answers_when_first_is_out_of_quota(monkeypatch):
    def first_fails(prompt, schema, system, model, temperature):
        if model == llm.model_chain()[0]:
            raise errors.ClientError(429, {"error": {"message": "quota"}})
        return PLAN

    monkeypatch.setattr(llm, "_generate_structured_once", first_fails)
    assert llm.generate_structured("transcript", WorkflowPlan) == PLAN
    assert llm.last_model_used() == llm.model_chain()[1]


def test_parallel_agents_share_one_gemini_client(monkeypatch):
    """Regression: parallel nodes used to race and create two clients; the discarded one
    closed the shared HTTP connection mid-request."""
    import threading

    monkeypatch.setattr(llm, "_client", None)
    barrier, seen = threading.Barrier(8), []

    def grab():
        barrier.wait()  # all threads ask at the same instant
        seen.append(llm.client())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(c) for c in seen}) == 1


def test_claude_request_does_not_send_temperature(monkeypatch):
    """Regression: claude-haiku-5-5 returns 400 'temperature is deprecated' if it is sent."""
    monkeypatch.setattr(llm.settings, "anthropic_api_key", "test-key")
    payload = llm.chat_model("claude")._get_request_payload([("user", "hi")])
    assert "temperature" not in payload
