"""Reranker behind DigitalOcean's serverless inference rerank API.

The endpoint is fixed (https://inference.do-ai.run/v1/rerank), which is what
makes this a dedicated provider rather than a ``rerank_api`` configuration.
Request and response shape match the other hosted providers:
``POST {model, query, documents}`` returning ``results`` rows of
``{index, relevance_score}``, mapped back by index rather than by row order.
Authenticates with ``settings.digitalocean_api_key`` (env DIGITALOCEAN_API_KEY)
as a bearer token. No retry layer: a rate-limited call fails fast and surfaces,
the same policy as the other hosted rerankers. The ``httpx.Client`` is created
lazily on first use, so importing this module stays cheap.

Two traits of this API shape the code:

* **Scores arrive as raw cross-encoder logits**, roughly -11 to +8, where the
  in-process cross-encoder and the other hosted providers return 0-1
  probabilities. ``_sigmoid`` maps them into that shared space, so a relevance
  floor calibrated against those providers keeps its meaning here. If the
  vendor ever starts returning normalized scores this transform has to go —
  applying it twice would squash the range.
* **A request takes at most 150 documents** and is rejected outright above
  that, while a single-speech passage search reranks many more. Documents are
  therefore sent in slices and their scores stitched back together by position
  in the original list. Slicing cannot change a score, because the model scores
  each (query, document) pair independently.
"""

import math

from qhld_ai.domain.ports.reranker import RerankerPort
from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register

_RERANK_URL = "https://inference.do-ai.run/v1/rerank"
_MAX_DOCUMENTS = 150


def _sigmoid(score: float) -> float:
    """Logit to probability, evaluated on whichever side of zero keeps ``exp``
    from overflowing on large-magnitude scores."""
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    weight = math.exp(score)
    return weight / (1.0 + weight)


class DigitalOceanReranker(RerankerPort):
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
        scores: dict[int, float] = {}
        for offset in range(0, len(documents), _MAX_DOCUMENTS):
            response = self.client.post(
                _RERANK_URL,
                json={
                    "model": self._model,
                    "query": query,
                    "documents": documents[offset:offset + _MAX_DOCUMENTS],
                },
            )
            response.raise_for_status()
            for row in response.json()["results"]:
                scores[offset + row["index"]] = _sigmoid(row["relevance_score"])
        rescored = [
            SearchHit(id=hit.id, score=float(scores[i]), payload=hit.payload)
            for i, hit in enumerate(hits)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:k]


@_register("digitalocean")
def create(settings: Settings) -> DigitalOceanReranker:
    return DigitalOceanReranker(
        settings.reranker_model, settings.digitalocean_api_key)
