"""Reranker behind a local Jina-schema rerank HTTP server.

Speaks the request/response shape of Jina's rerank API — ``POST {model,
query, documents}`` returning ``results`` rows of ``{index, relevance_score}``
— as emulated by local servers (a vMLX engine serving an MLX reranker, ...).
``reranker_base_url`` is the FULL endpoint URL because servers mount the route
at different paths (e.g. vMLX serves http://127.0.0.1:11438/v1/rerank). Local
servers are unauthenticated and unmetered, so there is no API key and no retry
layer; hosted vendors get dedicated providers ("jina", "cohere", "voyage")
instead. Score space is the served model's own (e.g. jina-reranker-v3 returns
cosine similarities, not sigmoid probabilities), so relevance floors
calibrated for another model do NOT carry over. The ``httpx.Client`` is
created lazily on first use, so importing this module stays cheap.
"""

from qhld_ai.domain.ports.reranker import RerankerPort
from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register


class RerankAPIReranker(RerankerPort):
    def __init__(self, url: str, model: str, timeout: float = 120.0):
        self._url = url
        self._model = model
        self._timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        if not hits:
            return []
        documents = [hit.payload.get("text") or "" for hit in hits]
        response = self.client.post(
            self._url,
            json={"model": self._model, "query": query, "documents": documents},
        )
        response.raise_for_status()
        scores = {
            row["index"]: row["relevance_score"]
            for row in response.json()["results"]
        }
        rescored = [
            SearchHit(id=hit.id, score=float(scores[i]), payload=hit.payload)
            for i, hit in enumerate(hits)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:k]


@_register("rerank_api")
def create(settings: Settings) -> RerankAPIReranker:
    return RerankAPIReranker(
        settings.reranker_base_url,
        settings.reranker_model,
    )
