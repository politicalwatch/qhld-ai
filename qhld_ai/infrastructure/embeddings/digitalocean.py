from langchain_openai import OpenAIEmbeddings

from qhld_ai.infrastructure.config.settings import Settings
from .factory import _register


@_register("digitalocean")
def create(settings: Settings) -> OpenAIEmbeddings:
    # DigitalOcean serves an OpenAI-compatible embeddings API at a fixed
    # endpoint, authenticated with the same per-vendor key as its reranker
    # provider. check_embedding_ctx_length=False sends raw strings instead of
    # the default tiktoken token arrays, which only OpenAI's own backend
    # accepts. Note the model slug is unprefixed here (e.g. "bge-m3"), so the
    # derived collection name differs from another vendor's serving of the same
    # model.
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.digitalocean_api_key,
        base_url="https://inference.do-ai.run/v1",
        check_embedding_ctx_length=False,
    )
