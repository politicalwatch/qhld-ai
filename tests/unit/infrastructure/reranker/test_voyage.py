"""Offline tests for the VoyageReranker — the langchain-voyageai client is
stubbed (the adapter delegates HTTP to the vendor SDK, so the seam is the
client's ``_rerank`` method, not a transport)."""

from types import SimpleNamespace

import pytest

from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.reranker.voyage import VoyageReranker

pytestmark = pytest.mark.unit


class _StubClient:
    """Mimics langchain_voyageai.VoyageAIRerank._rerank: scores each document
    by its trailing int and returns rows sorted by score (as the vendor does),
    proving the adapter maps by index, not order."""

    def __init__(self):
        self.calls = []

    def _rerank(self, documents, query):
        self.calls.append({"documents": documents, "query": query})
        rows = [
            SimpleNamespace(index=i, relevance_score=float(doc.split("t")[-1]))
            for i, doc in enumerate(documents)
        ]
        rows.sort(key=lambda row: row.relevance_score, reverse=True)
        return SimpleNamespace(results=rows)


def _reranker(client):
    reranker = VoyageReranker("rerank-test-model")
    reranker._client = client
    return reranker


def _hit(id_, score, text):
    return SearchHit(id=id_, score=score, payload={"text": text})


def test_rerank_reorders_and_rescores_by_served_scores():
    reranker = _reranker(_StubClient())
    hits = [_hit("a", 0.9, "t1"), _hit("b", 0.8, "t3"), _hit("c", 0.7, "t2")]

    out = reranker.rerank("q", hits, k=2)

    assert [h.id for h in out] == ["b", "c"]   # served scores 3 > 2 > 1
    assert out[0].score == 3.0                 # score replaced by served score
    assert out[0].payload["text"] == "t3"      # payload preserved


def test_rerank_sends_query_and_documents():
    client = _StubClient()
    reranker = _reranker(client)
    reranker.rerank("la consulta", [_hit("a", 0.9, "un pasaje t1")], k=1)

    assert client.calls[0]["query"] == "la consulta"
    assert client.calls[0]["documents"] == ["un pasaje t1"]


def test_rerank_empty_hits_returns_empty_without_calling_the_client():
    client = _StubClient()
    reranker = _reranker(client)
    assert reranker.rerank("q", [], k=5) == []
    assert client.calls == []


def test_lazy_client_requests_every_documents_score():
    # top_k=None asks Voyage for all scores — a truncated head would starve
    # the floor and grouping of the tail scores.
    reranker = VoyageReranker("rerank-test-model", api_key="voyage_secret")
    client = reranker.client
    assert client.top_k is None
    assert client.model == "rerank-test-model"
