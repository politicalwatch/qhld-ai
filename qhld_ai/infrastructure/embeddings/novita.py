from langchain_openai import OpenAIEmbeddings

from qhld_ai.infrastructure.config.settings import Settings
from .factory import _register


@_register("novita")
def create(settings: Settings) -> OpenAIEmbeddings:
    # Novita serves an OpenAI-compatible embeddings API at a fixed endpoint,
    # authenticated with the same per-vendor key as its reranker provider.
    # check_embedding_ctx_length=False sends raw strings instead of the
    # default tiktoken token arrays, which only OpenAI's own backend accepts.
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.novita_api_key,
        base_url="https://api.novita.ai/openai/v1",
        check_embedding_ctx_length=False,
    )
