"""Offline tests for the RerankAPIReranker — HTTP is stubbed with
httpx.MockTransport so no server (or network) is involved."""

import json

import httpx
import pytest

from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.reranker.rerank_api import RerankAPIReranker

pytestmark = pytest.mark.unit

URL = "http://rerank.test/v1/rerank"
MODEL = "jinaai/jina-reranker-v3-mlx"


def _reranker(handler, api_key=""):
    """A RerankAPIReranker whose lazy client is replaced by a mock transport,
    keeping the real client's headers (built the same way)."""
    reranker = RerankAPIReranker(URL, MODEL, api_key)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    reranker._client = httpx.Client(
        headers=headers, transport=httpx.MockTransport(handler)
    )
    return reranker


def _hit(id_, score, text):
    return SearchHit(id=id_, score=score, payload={"text": text})


def _scores_by_trailing_int(request):
    """Scores each document by its trailing int; rows come back sorted by score
    (as vendors return them) to prove the adapter maps by index, not order."""
    body = json.loads(request.content)
    rows = [
        {"index": i, "relevance_score": float(doc.split("t")[-1])}
        for i, doc in enumerate(body["documents"])
    ]
    rows.sort(key=lambda row: row["relevance_score"], reverse=True)
    return httpx.Response(200, json={"results": rows})


def test_rerank_reorders_and_rescores_by_served_scores():
    reranker = _reranker(_scores_by_trailing_int)
    hits = [_hit("a", 0.9, "t1"), _hit("b", 0.8, "t3"), _hit("c", 0.7, "t2")]

    out = reranker.rerank("q", hits, k=2)

    assert [h.id for h in out] == ["b", "c"]   # served scores 3 > 2 > 1
    assert out[0].score == 3.0                 # score replaced by served score
    assert out[0].payload["text"] == "t3"      # payload preserved


def test_rerank_sends_model_query_and_documents_to_the_endpoint():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.5}]}
        )

    reranker = _reranker(handler)
    reranker.rerank("la consulta", [_hit("a", 0.9, "un pasaje")], k=1)

    assert seen["url"] == URL
    assert seen["auth"] is None                # no api key -> no auth header
    assert seen["body"]["model"] == MODEL
    assert seen["body"]["query"] == "la consulta"
    assert seen["body"]["documents"] == ["un pasaje"]


def test_rerank_sends_bearer_header_when_api_key_is_set():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.5}]}
        )

    reranker = _reranker(handler, api_key="jina_secret")
    reranker.rerank("q", [_hit("a", 0.9, "t1")], k=1)

    assert seen["auth"] == "Bearer jina_secret"


def test_rerank_empty_hits_returns_empty_without_calling_the_server():
    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("no HTTP request expected for empty hits")

    reranker = _reranker(handler)
    assert reranker.rerank("q", [], k=5) == []


def test_rerank_raises_on_server_error():
    reranker = _reranker(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError):
        reranker.rerank("q", [_hit("a", 0.9, "t1")], k=1)
