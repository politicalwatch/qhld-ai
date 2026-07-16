"""Reranker factory — mirrors the embeddings/vectorstore registry pattern.

Adapters self-register a ``create(settings)`` callable under a provider name; the
factory dispatches on ``settings.reranker_provider``. "noop" leaves the bi-encoder
order untouched (the clean baseline); "cross_encoder" loads a sentence-transformers
cross-encoder in-process; "tei" calls the same model served over HTTP by a
text-embeddings-inference server; "rerank_api" calls a LOCAL Jina-schema rerank
server (a vMLX engine, ...); "jina" and "novita" call those vendors' hosted
rerank APIs; "cohere" and "voyage" call those vendors' hosted rerank models
through their own SDKs (via langchain-cohere / langchain-voyageai).
"""

from qhld_ai.domain.ports.reranker import RerankerPort
from qhld_ai.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _PROVIDERS[name] = fn
        return fn
    return decorator


def create_reranker_from_env(settings: Settings | None = None) -> RerankerPort:
    from qhld_ai.infrastructure.config.settings import get_settings

    s = settings or get_settings()
    provider = s.reranker_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown reranker provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Trigger adapter self-registration.
from qhld_ai.infrastructure.reranker import (  # noqa: E402, F401
    cohere,
    cross_encoder,
    jina,
    noop,
    novita,
    rerank_api,
    tei,
    voyage,
)
