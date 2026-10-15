"""Offline tests for the decision model factory registry."""

import pytest

from qhld_ai.infrastructure.config.settings import Settings
from qhld_ai.infrastructure.decision.factory import create_decision_model_from_env
from qhld_ai.infrastructure.decision.openai import OpenAIDecisionModel
from qhld_ai.infrastructure.decision.systemone import SystemOneDecisionModel

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def test_typesafe_provider_calls_the_hosted_api_with_its_key():
    model = create_decision_model_from_env(_settings(
        decision_provider="typesafe", decision_model="jev-1.13.0",
        typesafe_api_key="ts_secret", decision_timeout=1.5))
    assert isinstance(model, SystemOneDecisionModel)
    assert model.model == "jev-1.13.0"
    assert model._url == "https://api.typesafe.ai/v1/systemone"
    assert model._api_key == "ts_secret"
    assert model._timeout == 1.5


def test_ollama_provider_calls_the_local_server_without_a_key():
    model = create_decision_model_from_env(_settings(
        decision_provider="ollama", decision_model="clef:27b",
        ollama_base_url="http://localhost:11434"))
    assert isinstance(model, SystemOneDecisionModel)
    assert model._url == "http://localhost:11434/v1/systemone"
    assert model._api_key == ""


def test_openai_provider_uses_the_openai_key():
    model = create_decision_model_from_env(_settings(
        decision_provider="OpenAI", decision_model="gpt-6-luna", openai_api_key="sk"))
    assert isinstance(model, OpenAIDecisionModel)
    assert model.model == "gpt-6-luna"
    assert model._api_key == "sk"


def test_unknown_provider_names_the_valid_ones():
    with pytest.raises(ValueError, match="typesafe"):
        create_decision_model_from_env(_settings(decision_provider="nope"))
