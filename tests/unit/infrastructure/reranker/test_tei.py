"""Offline tests for the TEIReranker — HTTP is stubbed with httpx.MockTransport
so no server (or network) is involved."""

import json

import httpx
import pytest

from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.reranker.tei import TEIReranker

pytestmark = pytest.mark.unit

BASE_URL = "http://tei.test"


def _reranker(handler):
    """A TEIReranker whose lazy client is replaced by a mock-transport client."""
    reranker = TEIReranker(BASE_URL)
    reranker._client = httpx.Client(
        base_url=BASE_URL, transport=httpx.MockTransport(handler)
    )
    return reranker


def _hit(id_, score, text):
    return SearchHit(id=id_, score=score, payload={"text": text})


def _scores_by_trailing_int(request):
    """Scores each text by its trailing int — mirrors the cross_encoder test's
    fake encoder, and returns rows out of order to prove index mapping is used."""
    body = json.loads(request.content)
    rows = [
        {"index": i, "score": float(text.split("t")[-1])}
        for i, text in enumerate(body["texts"])
    ]
    rows.sort(key=lambda row: row["score"], reverse=True)  # TEI returns sorted
    return httpx.Response(200, json=rows)


def test_rerank_reorders_and_rescores_by_served_scores():
    reranker = _reranker(_scores_by_trailing_int)
    hits = [_hit("a", 0.9, "t1"), _hit("b", 0.8, "t3"), _hit("c", 0.7, "t2")]

    out = reranker.rerank("q", hits, k=2)

    assert [h.id for h in out] == ["b", "c"]   # served scores 3 > 2 > 1
    assert out[0].score == 3.0                 # score replaced by served score
    assert out[0].payload["text"] == "t3"      # payload preserved


def test_rerank_sends_query_and_texts_to_the_rerank_route():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=[{"index": 0, "score": 0.5}])

    reranker = _reranker(handler)
    reranker.rerank("la consulta", [_hit("a", 0.9, "un pasaje")], k=1)

    assert seen["path"] == "/rerank"
    assert seen["body"]["query"] == "la consulta"
    assert seen["body"]["texts"] == ["un pasaje"]
    assert seen["body"]["truncate"] is True


def test_rerank_empty_hits_returns_empty_without_calling_the_server():
    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("no HTTP request expected for empty hits")

    reranker = _reranker(handler)
    assert reranker.rerank("q", [], k=5) == []


def test_rerank_raises_on_server_error():
    reranker = _reranker(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError):
        reranker.rerank("q", [_hit("a", 0.9, "t1")], k=1)
