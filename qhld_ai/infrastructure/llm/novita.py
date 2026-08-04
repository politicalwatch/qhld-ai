from langchain_openai import ChatOpenAI

from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register


@_register("novita")
def create(settings: Settings) -> ChatOpenAI:
    # Novita serves an OpenAI-compatible chat API at a fixed endpoint,
    # authenticated with the same per-vendor key as its embeddings/reranker
    # providers. Model names are the vendor's slugs (e.g. "openai/gpt-oss-20b").
    kwargs = {
        "model": settings.llm_model,
        "api_key": settings.novita_api_key,
        "base_url": "https://api.novita.ai/openai/v1",
    }
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    # Novita's gpt-oss models are reasoning models, so the knob is meaningful
    # here too. Whether their endpoint honours the parameter is UNVERIFIED —
    # langchain's gpt-5-only temperature handling does not apply to vendor slugs
    # like "openai/gpt-oss-120b", so temperature keeps working regardless.
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    return ChatOpenAI(**kwargs)
