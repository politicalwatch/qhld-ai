from langchain_ollama import ChatOllama

from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register


@_register("vmlx")
def create(settings: Settings) -> ChatOllama:
    # vMLX serves an Ollama-compatible API; model names are HuggingFace paths.
    kwargs = {"model": settings.llm_model, "base_url": settings.vmlx_base_url}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatOllama(**kwargs)
