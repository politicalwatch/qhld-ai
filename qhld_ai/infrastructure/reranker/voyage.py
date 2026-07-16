"""Reranker behind Voyage AI's hosted rerank API, via langchain-voyageai.

Mirrors the ``cohere`` adapter: the HTTP surface is delegated to
``langchain_voyageai.VoyageAIRerank`` (Voyage's own SDK client underneath), so
endpoint routing, request shape and truncation follow the vendor. The score
space is Voyage's own 0-1 relevance — floors calibrated for another model do
NOT carry over. The API key resolves from ``settings.voyage_api_key`` when
set, else the SDK's standard VOYAGE_API_KEY environment variable. No retry
layer: Voyage's rate limits (2,000 RPM / 2M TPM) sit far above our request
patterns, so a 429 here is exceptional and should surface, not be absorbed.
The LangChain client is created lazily on first use, so importing this module
stays cheap.
"""

from qhld_ai.domain.ports.reranker import RerankerPort
from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register


class VoyageReranker(RerankerPort):
    def __init__(self, model: str, api_key: str = ""):
        self._model = model
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from langchain_voyageai import VoyageAIRerank

            # top_k stays None so Voyage scores every document — downstream
            # floor/grouping needs all of them, not a truncated head.
            kwargs = {"model": self._model, "top_k": None}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = VoyageAIRerank(**kwargs)
        return self._client

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        if not hits:
            return []
        texts = [hit.payload.get("text") or "" for hit in hits]
        results = self.client._rerank(texts, query)
        scores = {row.index: row.relevance_score for row in results.results}
        rescored = [
            SearchHit(id=hit.id, score=float(scores[i]), payload=hit.payload)
            for i, hit in enumerate(hits)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:k]


@_register("voyage")
def create(settings: Settings) -> VoyageReranker:
    return VoyageReranker(settings.reranker_model, settings.voyage_api_key)
