"""Offline tests for the CohereReranker — the langchain-cohere client is
stubbed (the adapter delegates HTTP to the vendor SDK, so the seam is the
client's ``rerank`` method, not a transport)."""

import pytest
from cohere.core import ApiError

from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.reranker.cohere import CohereReranker

pytestmark = pytest.mark.unit


class _StubClient:
    """Mimics langchain_cohere.CohereRerank.rerank: scores each document by
    its trailing int and returns rows sorted by score (as the vendor does),
    proving the adapter maps by index, not order."""

    def __init__(self, errors=None):
        self.calls = []
        self.errors = list(errors or [])

    def rerank(self, documents, query, top_n="unset"):
        self.calls.append({"documents": documents, "query": query, "top_n": top_n})
        if self.errors:
            raise self.errors.pop(0)
        rows = [
            {"index": i, "relevance_score": float(doc.split("t")[-1])}
            for i, doc in enumerate(documents)
        ]
        rows.sort(key=lambda row: row["relevance_score"], reverse=True)
        return rows


def _reranker(client):
    reranker = CohereReranker("rerank-test-model")
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


def test_rerank_requests_every_documents_score():
    # top_n=None asks Cohere for all scores — the class default of 3 would
    # starve the floor and grouping of the tail scores.
    client = _StubClient()
    reranker = _reranker(client)
    reranker.rerank("la consulta", [_hit("a", 0.9, "un pasaje t1")], k=1)

    assert client.calls[0]["query"] == "la consulta"
    assert client.calls[0]["documents"] == ["un pasaje t1"]
    assert client.calls[0]["top_n"] is None


def test_rerank_empty_hits_returns_empty_without_calling_the_client():
    client = _StubClient()
    reranker = _reranker(client)
    assert reranker.rerank("q", [], k=5) == []
    assert client.calls == []


def test_rerank_fails_fast_on_rate_limits():
    client = _StubClient(errors=[ApiError(status_code=429, body="")])
    reranker = _reranker(client)

    with pytest.raises(ApiError):
        reranker.rerank("q", [_hit("a", 0.9, "t1")], k=1)
    assert len(client.calls) == 1              # no retry: surface, don't absorb
