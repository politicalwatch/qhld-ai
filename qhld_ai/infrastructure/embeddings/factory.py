from langchain_core.embeddings import Embeddings

from qhld_ai.infrastructure.config.settings import Settings

_PROVIDERS: dict[str, type] = {}


def _register(name: str):
    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls
    return decorator


def create_embedder_from_env(settings: Settings | None = None) -> Embeddings:
    from qhld_ai.infrastructure.config.settings import get_settings
    s = settings or get_settings()
    provider = s.embedding_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Valid: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](s)


# Import providers to trigger registration. Each adapter imports its
# langchain-* SDK at module top, so a provider whose SDK isn't installed (a
# slimmer image built with a subset of the optional extras) is skipped rather
# than crashing the factory import. Selecting a skipped provider still raises
# the clear "Unknown provider" ValueError above. A broken *internal* import
# (a qhld_ai module) is re-raised so real bugs surface.
import importlib  # noqa: E402

from qhld_ai.logger import get_logger  # noqa: E402

_logger = get_logger(__name__)


def _register_available(package: str, modules: tuple[str, ...]) -> None:
    for name in modules:
        try:
            importlib.import_module(f"{package}.{name}")
        except ImportError as exc:
            if exc.name and exc.name.startswith("qhld_ai"):
                raise
            _logger.debug("skip %s provider %r (SDK not installed): %s",
                          package, name, exc)


_register_available(
    "qhld_ai.infrastructure.embeddings",
    ("openai", "google", "ollama", "vmlx", "novita", "digitalocean"),
)
