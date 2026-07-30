"""Offline tests for the reranker factory registry."""

import pytest

from qhld_ai.infrastructure.config.settings import Settings
from qhld_ai.infrastructure.reranker.cohere import CohereReranker
from qhld_ai.infrastructure.reranker.cross_encoder import CrossEncoderReranker
from qhld_ai.infrastructure.reranker.digitalocean import DigitalOceanReranker
from qhld_ai.infrastructure.reranker.factory import create_reranker_from_env
from qhld_ai.infrastructure.reranker.jina import JinaReranker
from qhld_ai.infrastructure.reranker.noop import NoOpReranker
from qhld_ai.infrastructure.reranker.rerank_api import RerankAPIReranker
from qhld_ai.infrastructure.reranker.tei import TEIReranker
from qhld_ai.infrastructure.reranker.voyage import VoyageReranker

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def test_noop_provider_builds_noop_reranker():
    reranker = create_reranker_from_env(_settings(reranker_provider="noop"))
    assert isinstance(reranker, NoOpReranker)


def test_cross_encoder_provider_builds_cross_encoder():
    settings = _settings(
        reranker_provider="cross_encoder",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_top_n=25,
    )
    reranker = create_reranker_from_env(settings)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._model == "BAAI/bge-reranker-v2-m3"
    assert reranker._top_n == 25       # constructing it does not load the model


def test_tei_provider_builds_tei_reranker():
    settings = _settings(
        reranker_provider="tei",
        reranker_base_url="http://localhost:8087/",
    )
    reranker = create_reranker_from_env(settings)
    assert isinstance(reranker, TEIReranker)
    assert reranker._base_url == "http://localhost:8087"  # trailing slash stripped
    assert reranker._client is None    # constructing it does not open connections


def test_rerank_api_provider_builds_rerank_api_reranker():
    settings = _settings(
        reranker_provider="rerank_api",
        reranker_model="jinaai/jina-reranker-v3-mlx",
        reranker_base_url="http://localhost:11438/v1/rerank",
    )
    reranker = create_reranker_from_env(settings)
    assert isinstance(reranker, RerankAPIReranker)
    assert reranker._url == "http://localhost:11438/v1/rerank"
    assert reranker._model == "jinaai/jina-reranker-v3-mlx"
    assert reranker._client is None    # constructing it does not open connections


def test_jina_provider_builds_jina_reranker():
    settings = _settings(
        reranker_provider="jina",
        reranker_model="jina-reranker-v2-base-multilingual",
        jina_api_key="secret",
    )
    reranker = create_reranker_from_env(settings)
    assert isinstance(reranker, JinaReranker)
    assert reranker._model == "jina-reranker-v2-base-multilingual"
    assert reranker._api_key == "secret"
    assert reranker._client is None    # constructing it does not open connections


def test_cohere_provider_builds_cohere_reranker():
    settings = _settings(
        reranker_provider="cohere",
        reranker_model="rerank-v4.0-fast",
        cohere_api_key="secret",
    )
    reranker = create_reranker_from_env(settings)
    assert isinstance(reranker, CohereReranker)
    assert reranker._model == "rerank-v4.0-fast"
    assert reranker._client is None    # constructing it does not open connections


def test_voyage_provider_builds_voyage_reranker():
    settings = _settings(
        reranker_provider="voyage",
        reranker_model="rerank-2.5",
        voyage_api_key="secret",
    )
    reranker = create_reranker_from_env(settings)
    assert isinstance(reranker, VoyageReranker)
    assert reranker._model == "rerank-2.5"
    assert reranker._client is None    # constructing it does not open connections


def test_digitalocean_provider_builds_digitalocean_reranker():
    settings = _settings(
        reranker_provider="digitalocean",
        reranker_model="bge-reranker-v2-m3",
        digitalocean_api_key="secret",
    )
    reranker = create_reranker_from_env(settings)
    assert isinstance(reranker, DigitalOceanReranker)
    assert reranker._model == "bge-reranker-v2-m3"
    assert reranker._api_key == "secret"
    assert reranker._client is None    # constructing it does not open connections


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        create_reranker_from_env(_settings(reranker_provider="bogus"))
