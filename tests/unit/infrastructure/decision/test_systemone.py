"""Offline tests for the System One decision model (TypeSafe and Ollama) — HTTP is
stubbed with httpx2.MockTransport, handed to the SDK, so no server (or network) is
involved."""

import json

import httpx2
import pytest

from qhld_ai.domain.ports.decision_model import (
    Choice, ChoiceAnswer, DecisionUnavailable, Predicate, PredicateAnswer)
from qhld_ai.infrastructure.decision import tracing
from qhld_ai.infrastructure.decision.systemone import SystemOneDecisionModel

pytestmark = pytest.mark.unit

CHOICE = Choice(instructions="¿Qué es?", options={"a": "la primera", "b": "la segunda"})
PREDICATE = Predicate(instructions="¿Es una búsqueda?")


def _model(handler, api_key="ts_secret", base_url="https://api.typesafe.ai",
           provider="typesafe"):
    return SystemOneDecisionModel("jev-1.13.0", base_url, api_key, provider=provider,
                                  transport=httpx2.MockTransport(handler))


def _answers(usage=None, **answers):
    return lambda request: httpx2.Response(200, json={
        "model": "jev-1.13.0", "answers": answers,
        "usage": usage or {"input_tokens": 455, "output_tokens": 40}})


def _choice(choice, **probabilities):
    return {"type": "choice", "choice": choice, "confidence": 0.9,
            "probabilities": probabilities}


def test_sends_the_state_and_named_questions_in_the_system_one_shape():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _answers(kind=_choice("a", a=0.9, b=0.1),
                        yes={"type": "noul", "noul": 0.3})(request)

    _model(handler).decide("texto", {"kind": CHOICE, "yes": PREDICATE})

    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer ts_secret"
    assert seen["body"]["model"] == "jev-1.13.0"
    assert seen["body"]["state"] == "texto"
    assert seen["body"]["questions"]["kind"] == {
        "type": "choice", "instructions": "¿Qué es?",
        "criteria": {"a": "la primera", "b": "la segunda"}}
    assert seen["body"]["questions"]["yes"] == {
        "type": "noul", "instructions": "¿Es una búsqueda?"}


def test_decodes_a_choice_and_a_predicate_by_the_callers_names():
    answers = _model(_answers(
        kind={"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8},
              "confidence": 0.6},
        yes={"type": "noul", "noul": 0.25},
    )).decide("texto", {"kind": CHOICE, "yes": PREDICATE})

    assert answers["kind"] == ChoiceAnswer(
        choice="b", probabilities={"a": 0.2, "b": 0.8}, confidence=0.6)
    assert answers["kind"].probability("a") == 0.2
    assert answers["yes"] == PredicateAnswer(probability=0.25)


def test_a_local_server_is_called_on_the_same_route_with_a_placeholder_key():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return _answers(kind=_choice("a", a=1.0))(request)

    _model(handler, api_key="", base_url="http://localhost:11434/",
           provider="ollama").decide("texto", {"kind": CHOICE})

    assert seen["url"] == "http://localhost:11434/v1/systemone"
    assert seen["auth"] == "Bearer ollama"


@pytest.mark.parametrize("handler", [
    lambda request: httpx2.Response(500, json={"error": "boom"}),
    lambda request: httpx2.Response(401, json={"error": "bad key"}),
    lambda request: httpx2.Response(200, json={"model": "m", "answers": {},
                                               "usage": {"input_tokens": 1,
                                                         "output_tokens": 0}}),
    lambda request: httpx2.Response(200, text="not json"),
    _answers(kind={"type": "refusal"}),
    _answers(kind={"type": "noul", "noul": 0.5}),
])
def test_any_unusable_response_is_unavailable(handler):
    with pytest.raises(DecisionUnavailable):
        _model(handler).decide("texto", {"kind": CHOICE})


def test_a_timeout_is_unavailable():
    def handler(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(DecisionUnavailable):
        _model(handler).decide("texto", {"kind": CHOICE})


def test_a_rate_limit_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(429, json={"error": "slow down"})

    with pytest.raises(DecisionUnavailable):
        _model(handler).decide("texto", {"kind": CHOICE})
    assert len(calls) == 1


class _Run:
    def __init__(self):
        self.metadata, self.usage = None, None

    def set(self, metadata=None, usage_metadata=None):
        self.metadata, self.usage = metadata, usage_metadata


def test_the_reported_tokens_go_on_the_trace(monkeypatch):
    run = _Run()
    monkeypatch.setattr(tracing, "get_current_run_tree", lambda: run)

    _model(_answers(usage={"input_tokens": 455, "output_tokens": 40},
                    kind=_choice("a", a=1.0))).decide("texto", {"kind": CHOICE})

    assert run.metadata == {"ls_provider": "typesafe", "ls_model_name": "jev-1.13.0"}
    assert run.usage == {"input_tokens": 455, "output_tokens": 40, "total_tokens": 495}
