"""Decision model factory — mirrors the reranker registry pattern.

Adapters self-register a ``create(settings)`` callable under a provider name; the
factory dispatches on ``settings.decision_provider``. "typesafe" calls TypeSafe's
hosted Jev models; "ollama" calls decision models served by a local Ollama, which
speaks the same request format; "openai" calls OpenAI's Decisions API.

A task that wants its own provider or model (the intent gate does) passes a
Settings copy with ``decision_*`` overridden, the way the query parser overrides
``llm_*``.
"""

from qhld_ai.domain.ports.decision_model import DecisionModelPort
from qhld_ai.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _PROVIDERS[name] = fn
        return fn
    return decorator


def create_decision_model_from_env(settings: Settings | None = None) -> DecisionModelPort:
    from qhld_ai.infrastructure.config.settings import get_settings

    s = settings or get_settings()
    provider = s.decision_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown decision model provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Trigger adapter self-registration.
from qhld_ai.infrastructure.decision import openai, systemone  # noqa: E402, F401
