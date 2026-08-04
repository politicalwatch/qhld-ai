"""Tests for the per-model collection naming (pure string logic)."""

import pytest

from qhld_ai.infrastructure.config.settings import Settings
from qhld_ai.infrastructure.vectorstore.naming import collection_name

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(
        _env_file=None,
        embedding_provider="ollama",
        embedding_model="bge-m3:567m",
        **overrides,
    )


def test_derives_per_model_name():
    assert collection_name(_settings(), 1024) == "speeches__ollama__bge_m3_567m__1024"


def test_sparse_provider_adds_suffix():
    assert (
        collection_name(_settings(sparse_provider="bm25"), 1024)
        == "speeches__ollama__bge_m3_567m__1024__bm25"
    )


def test_sparse_none_keeps_dense_name():
    assert (
        collection_name(_settings(sparse_provider="none"), 1024)
        == "speeches__ollama__bge_m3_567m__1024"
    )


def test_quantization_adds_suffix():
    assert (
        collection_name(_settings(qdrant_quantization="tq4"), 1024)
        == "speeches__ollama__bge_m3_567m__1024__tq4"
    )


@pytest.mark.parametrize("value", ["none", "NONE", ""])
def test_uncompressed_keeps_the_existing_name(value):
    """An uncompressed collection must keep the name it already has — otherwise
    enabling the setting would orphan every collection built before it."""
    assert (
        collection_name(_settings(qdrant_quantization=value), 1024)
        == "speeches__ollama__bge_m3_567m__1024"
    )


def test_sparse_and_quantization_order():
    """Lexical branch first, compression last: dense shape -> retrieval branch ->
    storage. Pinned because the name is a cross-service contract."""
    settings = _settings(sparse_provider="bm25", qdrant_quantization="tq4")
    assert collection_name(settings, 1024) == "speeches__ollama__bge_m3_567m__1024__bm25__tq4"


def test_quantization_is_not_tq4_specific():
    """Any rung of the compression ladder names itself; the dotted rungs keep a
    legible token rather than a literal '.'."""
    assert (
        collection_name(_settings(qdrant_quantization="sq8"), 768)
        == "speeches__ollama__bge_m3_567m__768__sq8"
    )
    assert (
        collection_name(_settings(qdrant_quantization="tq1_5"), 768)
        == "speeches__ollama__bge_m3_567m__768__tq1_5"
    )


def test_explicit_collection_overrides_everything():
    settings = _settings(
        qdrant_collection="fixed", sparse_provider="bm25", qdrant_quantization="tq4")
    assert collection_name(settings, 1024) == "fixed"
