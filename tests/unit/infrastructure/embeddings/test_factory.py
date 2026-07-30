"""Unit tests for the embeddings provider registry + factory — no network.

Ported from vinculante; same shape as the LLM factory tests.
"""

import pytest
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

import qhld_ai.infrastructure.embeddings.factory as factory_module
from qhld_ai.infrastructure.config.settings import Settings
from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class _RecordingEmbedder:
    received: Settings | None = None

    def __init__(self, settings: Settings) -> None:
        _RecordingEmbedder.received = settings


def test_create_embedder_uses_embedding_provider(monkeypatch):
    monkeypatch.setitem(factory_module._PROVIDERS, "google", _RecordingEmbedder)
    s = _settings(embedding_provider="google", embedding_model="embedding-001")
    create_embedder_from_env(s)
    assert _RecordingEmbedder.received.embedding_provider == "google"
    assert _RecordingEmbedder.received.embedding_model == "embedding-001"


def test_create_embedder_unknown_provider_raises():
    s = _settings(embedding_provider="unknown_xyz")
    with pytest.raises(ValueError, match="unknown_xyz"):
        create_embedder_from_env(s)


def test_all_embedding_providers_registered():
    assert {"openai", "google", "ollama", "vmlx", "novita",
            "digitalocean"} <= set(factory_module._PROVIDERS)


def test_novita_uses_its_own_key_and_the_hosted_endpoint():
    s = _settings(
        embedding_provider="novita",
        embedding_model="baai/bge-m3",
        openai_api_key="openai_secret",
        novita_api_key="novita_secret",
    )
    embedder = create_embedder_from_env(s)
    assert embedder.openai_api_key.get_secret_value() == "novita_secret"
    assert embedder.openai_api_base == "https://api.novita.ai/openai/v1"
    # Raw strings, not tiktoken token arrays — Novita is not OpenAI's backend.
    assert embedder.check_embedding_ctx_length is False


def test_digitalocean_uses_its_own_key_and_the_hosted_endpoint():
    s = _settings(
        embedding_provider="digitalocean",
        embedding_model="bge-m3",
        openai_api_key="openai_secret",
        digitalocean_api_key="do_secret",
    )
    embedder = create_embedder_from_env(s)
    assert embedder.openai_api_key.get_secret_value() == "do_secret"
    assert embedder.openai_api_base == "https://inference.do-ai.run/v1"
    # Raw strings, not tiktoken token arrays — this is not OpenAI's backend.
    assert embedder.check_embedding_ctx_length is False


def test_vmlx_uses_its_own_base_url():
    s = _settings(
        embedding_provider="vmlx",
        ollama_base_url="http://ollama:11434",
        vmlx_base_url="http://vmlx:8080",
    )
    embedder = create_embedder_from_env(s)
    assert embedder.base_url == "http://vmlx:8080"


@pytest.mark.parametrize(
    "provider, expected_cls",
    [
        ("openai", OpenAIEmbeddings),
        ("google", GoogleGenerativeAIEmbeddings),
        ("ollama", OllamaEmbeddings),
        ("vmlx", OllamaEmbeddings),
        ("novita", OpenAIEmbeddings),
        ("digitalocean", OpenAIEmbeddings),
    ],
)
def test_each_provider_builds_real_embedder(provider, expected_cls):
    s = _settings(
        embedding_provider=provider,
        openai_api_key="x",
        google_api_key="x",
        novita_api_key="x",
        digitalocean_api_key="x",
    )
    assert isinstance(create_embedder_from_env(s), expected_cls)
