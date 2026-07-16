"""Reranker behind Novita's hosted rerank API.

The endpoint is fixed (https://api.novita.ai/openai/v1/rerank) — that is what
makes this a dedicated provider rather than a ``rerank_api`` configuration;
local Jina-schema servers (vMLX) go through ``rerank_api`` instead. Same
request and response shape: ``POST {model, query, documents}`` returning
``results`` rows of ``{index, relevance_score}``. Authenticates with
``settings.novita_api_key`` (env NOVITA_API_KEY) as a bearer token. No retry
layer: a rate-limited call fails fast and surfaces, the same policy as the
other hosted providers. Novita serves the same bge-reranker-v2-m3 we run
in-process, but the score space is still the SERVED model's own — verify
parity before carrying a calibrated relevance floor over. The
``httpx.Client`` is created lazily on first use, so importing this module
stays cheap.
"""

from qhld_ai.domain.ports.reranker import RerankerPort
from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register

_RERANK_URL = "https://api.novita.ai/openai/v1/rerank"


class NovitaReranker(RerankerPort):
    def __init__(self, model: str, api_key: str, timeout: float = 120.0):
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        return self._client

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        if not hits:
            return []
        documents = [hit.payload.get("text") or "" for hit in hits]
        response = self.client.post(
            _RERANK_URL,
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


@_register("novita")
def create(settings: Settings) -> NovitaReranker:
    return NovitaReranker(settings.reranker_model, settings.novita_api_key)
