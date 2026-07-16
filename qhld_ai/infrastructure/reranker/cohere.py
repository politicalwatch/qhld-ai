"""Reranker behind Cohere's hosted rerank API, via langchain-cohere.

The HTTP surface is delegated to ``langchain_cohere.CohereRerank`` (Cohere's
own ClientV2 under the hood), so endpoint routing, request shape and API
versioning follow the vendor SDK. The score space is Cohere's own 0-1
relevance — floors calibrated for another model do NOT carry over. The API
key resolves from ``settings.cohere_api_key`` when set, else the SDK's
standard COHERE_API_KEY environment variable. No retry layer: a rate-limited
call fails fast and surfaces, the same policy as the other hosted providers.
The LangChain client is created lazily on first use, so importing this module
stays cheap.
"""

from qhld_ai.domain.ports.reranker import RerankerPort
from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register


class CohereReranker(RerankerPort):
    def __init__(self, model: str, api_key: str = ""):
        self._model = model
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from langchain_cohere import CohereRerank

            kwargs = {"model": self._model}
            if self._api_key:
                kwargs["cohere_api_key"] = self._api_key
            self._client = CohereRerank(**kwargs)
        return self._client

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        if not hits:
            return []
        texts = [hit.payload.get("text") or "" for hit in hits]
        # top_n=None asks for every document's score, not the class default
        # of 3 — downstream floor/grouping needs all of them.
        results = self.client.rerank(documents=texts, query=query, top_n=None)
        scores = {row["index"]: row["relevance_score"] for row in results}
        rescored = [
            SearchHit(id=hit.id, score=float(scores[i]), payload=hit.payload)
            for i, hit in enumerate(hits)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:k]


@_register("cohere")
def create(settings: Settings) -> CohereReranker:
    return CohereReranker(settings.reranker_model, settings.cohere_api_key)
