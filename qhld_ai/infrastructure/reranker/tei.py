"""Reranker served over HTTP by text-embeddings-inference (TEI).

Same cross-encoder scoring as the in-process ``cross_encoder`` adapter, but the
model lives in a dedicated TEI server process (GPU-resident, persistent across
callers), so search processes stay torch-free and repeated runs don't reload
the weights. TEI's ``/rerank`` applies the model's sigmoid by default, matching
``CrossEncoder.predict``'s score space, so score thresholds calibrated against
the in-process adapter carry over. The ``httpx.Client`` is created lazily on
first use, so importing this module stays cheap.
"""

from qhld_ai.domain.ports.reranker import RerankerPort
from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register


class TEIReranker(RerankerPort):
    def __init__(self, base_url: str, timeout: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                base_url=self._base_url, timeout=self._timeout
            )
        return self._client

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        if not hits:
            return []
        texts = [hit.payload.get("text") or "" for hit in hits]
        # truncate: passages beyond the model's max sequence length are cut
        # server-side instead of failing the whole batch.
        response = self.client.post(
            "/rerank", json={"query": query, "texts": texts, "truncate": True}
        )
        response.raise_for_status()
        scores = {row["index"]: row["score"] for row in response.json()}
        rescored = [
            SearchHit(id=hit.id, score=float(scores[i]), payload=hit.payload)
            for i, hit in enumerate(hits)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:k]


@_register("tei")
def create(settings: Settings) -> TEIReranker:
    return TEIReranker(settings.reranker_base_url)
