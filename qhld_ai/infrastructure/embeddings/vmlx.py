from langchain_ollama import OllamaEmbeddings

from qhld_ai.infrastructure.config.settings import Settings
from .factory import _register


@_register("vmlx")
def create(settings: Settings) -> OllamaEmbeddings:
    # vMLX serves an Ollama-compatible API; model names are HuggingFace paths.
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.vmlx_base_url,
    )
