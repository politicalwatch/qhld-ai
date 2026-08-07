"""Forced-aligner factory — mirrors the reranker/sparse registry pattern.

Adapters self-register a ``create(settings)`` callable under a provider name; the
factory dispatches on ``settings.aligner_provider``. "mms_onnx" runs the MMS-300m
CTC aligner in-process through ONNX Runtime.

There is no noop adapter, as in the sparse factory: "none" (the default) means
subtitle alignment is switched off and callers simply don't build an aligner, so a
misspelled provider fails loudly here instead of silently producing no subtitles.
"""

from qhld_ai.domain.ports.aligner import AlignerPort
from qhld_ai.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _PROVIDERS[name] = fn
        return fn
    return decorator


def create_aligner_from_env(settings: Settings | None = None) -> AlignerPort:
    from qhld_ai.infrastructure.config.settings import get_settings

    s = settings or get_settings()
    provider = s.aligner_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown aligner provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Trigger adapter self-registration. onnxruntime, numpy and huggingface_hub are all
# lazy-imported inside the adapter, so this stays cheap even where the align extra
# is not installed.
from qhld_ai.infrastructure.aligner import mms_onnx  # noqa: E402, F401
