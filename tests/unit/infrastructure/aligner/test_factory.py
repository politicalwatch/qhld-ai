"""Offline tests for the aligner factory registry."""

import pytest

from qhld_ai.infrastructure.aligner.factory import create_aligner_from_env
from qhld_ai.infrastructure.aligner.mms_onnx import MmsOnnxAligner
from qhld_ai.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def test_mms_onnx_provider_builds_the_aligner():
    settings = _settings(aligner_provider="mms_onnx", aligner_threads=5)
    aligner = create_aligner_from_env(settings)

    assert isinstance(aligner, MmsOnnxAligner)
    assert aligner._repo == "onnx-community/mms-300m-1130-forced-aligner-ONNX"
    assert aligner._revision == "2100fb247d8e"
    assert aligner._threads == 5
    assert aligner._session is None      # constructing it loads no model
    assert aligner._vocab is None


def test_local_model_path_overrides_the_download():
    aligner = create_aligner_from_env(
        _settings(aligner_provider="mms_onnx", aligner_model_path="/models/mms.onnx"))

    assert aligner.paths == ("/models/mms.onnx", "/models/vocab.json")


def test_default_provider_is_not_registered():
    # "none" means the feature is off, so callers must not build an aligner at all
    # rather than getting one that silently does nothing.
    with pytest.raises(ValueError, match="Unknown aligner provider"):
        create_aligner_from_env(_settings())


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown aligner provider"):
        create_aligner_from_env(_settings(aligner_provider="bogus"))
