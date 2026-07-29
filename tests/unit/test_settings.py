"""Unit tests for the Pydantic ``Settings`` — no DB.

``_env_file=None`` keeps these tests from reading the repo-root ``.env``;
``monkeypatch.delenv`` clears container-injected env vars so genuine defaults
are asserted.
"""

import pytest

from qhld_ai.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


def test_defaults(monkeypatch):
    # _env_file=None only ignores the .env *file*; pydantic-settings still reads
    # os.environ. Containers and dev shells inject these as real env vars, so
    # clear them to test the genuine defaults.
    for key in (
        "LOGLEVEL",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "QDRANT_HOST",
        "QDRANT_COLLECTION",
        "QUERY_PARSER_PROVIDER",
        "QUERY_PARSER_LLM_PROVIDER",
        "QUERY_PARSER_LLM_MODEL",
        "RERANKER_PROVIDER",
        "RERANKER_MODEL",
        "SPARSE_PROVIDER",
        "SPARSE_MODEL",
        "NER_PROVIDER",
        "NER_MODEL",
        "NER_GAZETTEER",
        "MENTION_MATCH_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.loglevel == "INFO"
    assert settings.llm_provider == "anthropic"
    assert settings.embedding_provider == "openai"
    assert settings.qdrant_host == "qdrant"
    assert settings.qdrant_collection == ""
    assert settings.query_parser_provider == "llm"
    assert settings.reranker_provider == "noop"
    assert settings.sparse_provider == "none"
    assert settings.ner_provider == "spacy"
    assert settings.ner_gazetteer is True
    assert settings.mention_match_threshold == 90


@pytest.mark.parametrize("raw", ["False", "false", "0", "no", "off"])
def test_ner_gazetteer_falsey_strings_are_false(raw):
    settings = Settings(_env_file=None, ner_gazetteer=raw)
    assert settings.ner_gazetteer is False


def test_ints_coerced_from_strings():
    settings = Settings(
        _env_file=None,
        qdrant_port="7333",
        reranker_top_n="25",
        mention_match_threshold="85",
    )
    assert settings.qdrant_port == 7333
    assert settings.reranker_top_n == 25
    assert settings.mention_match_threshold == 85


def test_llm_temperature_coerced_to_float():
    settings = Settings(_env_file=None, llm_temperature="0.2")
    assert settings.llm_temperature == 0.2


def test_retrieval_tuning_defaults_to_no_opinion(monkeypatch):
    for key in ("QDRANT_HNSW_EF", "QDRANT_QUANTIZATION", "QDRANT_QUANTIZATION_RESCORE"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.qdrant_hnsw_ef is None
    assert settings.qdrant_quantization == "none"
    assert settings.qdrant_quantization_rescore is None


def test_blank_retrieval_tuning_values_mean_unset():
    # "VAR=" is how an env file says "leave this to the server"; it must not be a
    # startup error.
    settings = Settings(
        _env_file=None, qdrant_hnsw_ef="", qdrant_quantization_rescore="")
    assert settings.qdrant_hnsw_ef is None
    assert settings.qdrant_quantization_rescore is None


def test_retrieval_tuning_coerced_from_strings():
    settings = Settings(
        _env_file=None, qdrant_hnsw_ef="1024", qdrant_quantization_rescore="false")
    assert settings.qdrant_hnsw_ef == 1024
    assert settings.qdrant_quantization_rescore is False
