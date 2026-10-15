"""Offline tests for the OpenAI Decisions model — HTTP is stubbed with
httpx2.MockTransport so no server (or network) is involved."""

import json

import httpx2
import pytest

from qhld_ai.domain.ports.decision_model import (
    Choice, ChoiceAnswer, DecisionUnavailable, Predicate, PredicateAnswer)
from qhld_ai.infrastructure.decision.openai import OpenAIDecisionModel

pytestmark = pytest.mark.unit

CHOICE = Choice(instructions="¿Qué es?", options={"a": "la primera", "b": "la segunda"})
PREDICATE = Predicate(instructions="¿Es una búsqueda?")


def _model(handler, api_key="sk_secret"):
    model = OpenAIDecisionModel("gpt-6-luna", api_key)
    model._client = httpx2.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        transport=httpx2.MockTransport(handler))
    return model


def _answers(*answers):
    return lambda request: httpx2.Response(200, json={"model": "gpt-6-luna",
                                                     "answers": list(answers)})


def test_sends_the_input_and_a_list_of_named_questions():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _answers(
            {"type": "choice", "name": "kind", "choice": "a",
             "probabilities": [{"value": "a", "probability": 0.9}]},
            {"type": "predicate", "name": "yes", "probability": 0.3})(request)

    _model(handler).decide("texto", {"kind": CHOICE, "yes": PREDICATE})

    assert seen["url"] == "https://api.openai.com/v1/decisions"
    assert seen["auth"] == "Bearer sk_secret"
    assert seen["body"] == {
        "model": "gpt-6-luna",
        "input": "texto",
        "questions": [
            {"type": "choice", "name": "kind", "instructions": "¿Qué es?",
             "choices": [{"value": "a", "description": "la primera"},
                         {"value": "b", "description": "la segunda"}]},
            {"type": "predicate", "name": "yes", "instructions": "¿Es una búsqueda?"},
        ],
    }


def test_decodes_answers_by_name_whatever_their_order():
    answers = _model(_answers(
        {"type": "predicate", "name": "yes", "probability": 0.25},
        {"type": "choice", "name": "kind", "choice": "b", "confidence": 0.6,
         "probabilities": [{"value": "a", "probability": 0.2},
                           {"value": "b", "probability": 0.8}]},
    )).decide("texto", {"kind": CHOICE, "yes": PREDICATE})

    assert answers["kind"] == ChoiceAnswer(
        choice="b", probabilities={"a": 0.2, "b": 0.8}, confidence=0.6)
    assert answers["yes"] == PredicateAnswer(probability=0.25)


@pytest.mark.parametrize("handler", [
    lambda request: httpx2.Response(429, json={"error": "rate limited"}),
    lambda request: httpx2.Response(200, json={"answers": []}),
    _answers({"type": "refusal", "name": "kind"}),
])
def test_any_unusable_response_is_unavailable(handler):
    with pytest.raises(DecisionUnavailable):
        _model(handler).decide("texto", {"kind": CHOICE})


class _Run:
    def __init__(self):
        self.metadata, self.usage = None, None

    def set(self, metadata=None, usage_metadata=None):
        self.metadata, self.usage = metadata, usage_metadata


def test_the_reported_tokens_go_on_the_trace_cached_ones_included(monkeypatch):
    from qhld_ai.infrastructure.decision import tracing

    run = _Run()
    monkeypatch.setattr(tracing, "get_current_run_tree", lambda: run)

    def handler(request):
        return httpx2.Response(200, json={
            "model": "gpt-6-luna",
            "answers": [{"type": "choice", "name": "kind", "choice": "a",
                         "probabilities": [{"value": "a", "probability": 1.0}]}],
            "usage": {"input_tokens": 236, "output_tokens": 0,
                      "input_tokens_details": {"cached_tokens": 128}}})

    _model(handler).decide("texto", {"kind": CHOICE})

    assert run.metadata == {"ls_provider": "openai", "ls_model_name": "gpt-6-luna"}
    assert run.usage == {"input_tokens": 236, "output_tokens": 0, "total_tokens": 236,
                         "input_token_details": {"cache_read": 128}}
