"""Offline tests for the System One decision model (TypeSafe and Ollama) — HTTP is
stubbed with httpx.MockTransport so no server (or network) is involved."""

import json

import httpx
import pytest

from qhld_ai.domain.ports.decision_model import (
    Choice, ChoiceAnswer, DecisionUnavailable, Predicate, PredicateAnswer)
from qhld_ai.infrastructure.decision.systemone import SystemOneDecisionModel

pytestmark = pytest.mark.unit

CHOICE = Choice(instructions="¿Qué es?", options={"a": "la primera", "b": "la segunda"})
PREDICATE = Predicate(instructions="¿Es una búsqueda?")


def _model(handler, api_key="ts_secret", base_url="https://api.typesafe.ai"):
    model = SystemOneDecisionModel("jev-1.13.0", base_url, api_key)
    model._client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        transport=httpx.MockTransport(handler))
    return model


def _answers(**answers):
    return lambda request: httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers})


def test_sends_the_state_and_named_questions_in_the_system_one_shape():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _answers(
            kind={"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}},
            yes={"type": "noul", "noul": 0.3})(request)

    _model(handler).decide("texto", {"kind": CHOICE, "yes": PREDICATE})

    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer ts_secret"
    assert seen["body"] == {
        "model": "jev-1.13.0",
        "state": "texto",
        "questions": {
            "kind": {"type": "choice", "instructions": "¿Qué es?",
                     "criteria": {"a": "la primera", "b": "la segunda"}},
            "yes": {"type": "noul", "instructions": "¿Es una búsqueda?"},
        },
    }


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


def test_a_local_server_is_called_on_the_same_route_without_auth():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return _answers(kind={"type": "choice", "choice": "a", "probabilities": {}})(request)

    _model(handler, api_key="", base_url="http://localhost:11434/").decide(
        "texto", {"kind": CHOICE})

    assert seen["url"] == "http://localhost:11434/v1/systemone"
    assert seen["auth"] is None


@pytest.mark.parametrize("handler", [
    lambda request: httpx.Response(500, json={"error": "boom"}),
    lambda request: httpx.Response(200, json={"answers": {}}),
    lambda request: httpx.Response(200, text="not json"),
    _answers(kind={"type": "refusal"}),
])
def test_any_unusable_response_is_unavailable(handler):
    with pytest.raises(DecisionUnavailable):
        _model(handler).decide("texto", {"kind": CHOICE})


def test_a_timeout_is_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(DecisionUnavailable):
        _model(handler).decide("texto", {"kind": CHOICE})
