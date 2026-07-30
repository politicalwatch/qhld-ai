"""Offline tests for the DigitalOceanReranker — HTTP is stubbed with
httpx.MockTransport so no server (or network) is involved."""

import json
import math

import httpx
import pytest

from qhld_ai.domain.ports.vector_store import SearchHit
from qhld_ai.infrastructure.reranker.digitalocean import (
    _MAX_DOCUMENTS,
    DigitalOceanReranker,
)

pytestmark = pytest.mark.unit

MODEL = "bge-reranker-v2-m3"


def _reranker(handler, api_key="do_secret"):
    """A DigitalOceanReranker whose lazy client is replaced by a mock transport,
    keeping the real client's headers (built the same way)."""
    reranker = DigitalOceanReranker(MODEL, api_key)
    reranker._client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        transport=httpx.MockTransport(handler),
    )
    return reranker


def _hit(id_, score, text):
    return SearchHit(id=id_, score=score, payload={"text": text})


def _logits_by_trailing_int(request):
    """Scores each document by its trailing int, as a LOGIT; rows come back
    sorted (as the vendor returns them) to prove mapping is by index."""
    body = json.loads(request.content)
    rows = [
        {"index": i, "relevance_score": float(doc.split("t")[-1])}
        for i, doc in enumerate(body["documents"])
    ]
    rows.sort(key=lambda row: row["relevance_score"], reverse=True)
    return httpx.Response(200, json={"results": rows})


def test_rerank_reorders_and_rescores_by_served_scores():
    reranker = _reranker(_logits_by_trailing_int)
    hits = [_hit("a", 0.9, "t1"), _hit("b", 0.8, "t3"), _hit("c", 0.7, "t2")]

    out = reranker.rerank("q", hits, k=2)

    assert [h.id for h in out] == ["b", "c"]   # served logits 3 > 2 > 1
    assert out[0].payload["text"] == "t3"      # payload preserved


def test_rerank_maps_vendor_logits_into_probability_space():
    """The vendor returns raw cross-encoder logits; the adapter converts them so
    a floor calibrated on the other providers' 0-1 scores still applies."""
    reranker = _reranker(_logits_by_trailing_int)

    out = reranker.rerank("q", [_hit("a", 0.5, "t3")], k=1)

    assert out[0].score == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))
    assert 0.0 < out[0].score < 1.0


def test_rerank_maps_a_strongly_negative_logit_near_zero():
    """Irrelevance has to land near 0, which is what makes the floor work."""

    def handler(request):
        return httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": -11.03}]}
        )

    out = _reranker(handler).rerank("q", [_hit("a", 0.5, "irrelevante")], k=1)

    assert out[0].score < 0.001


def test_rerank_splits_batches_over_the_document_cap_and_stitches_scores():
    """The API rejects more than _MAX_DOCUMENTS per call, so documents are sent
    in slices; each score must come back to its own hit."""
    seen = []

    def handler(request):
        # Logits stay in the model's real range (roughly -11..+8); larger ones
        # would all saturate to 1.0 under the sigmoid and tie.
        body = json.loads(request.content)
        seen.append(len(body["documents"]))
        rows = [
            {"index": i, "relevance_score": float(doc.split("t")[-1]) / 100.0}
            for i, doc in enumerate(body["documents"])
        ]
        return httpx.Response(200, json={"results": rows})

    total = _MAX_DOCUMENTS + 10
    hits = [_hit(f"h{i}", 0.5, f"t{i}") for i in range(total)]

    out = _reranker(handler).rerank("q", hits, k=total)

    assert seen == [_MAX_DOCUMENTS, 10]        # sliced, not one oversized call
    assert len(out) == total
    # highest trailing int wins, and every hit keeps its own document's score
    assert out[0].id == f"h{total - 1}"
    by_id = {h.id: h.score for h in out}
    assert by_id["h0"] == pytest.approx(0.5)   # sigmoid(0)
    # a hit from the SECOND slice keeps its own score, not the first slice's
    assert by_id[f"h{_MAX_DOCUMENTS}"] == pytest.approx(
        1.0 / (1.0 + math.exp(-_MAX_DOCUMENTS / 100.0)))


def test_rerank_calls_the_hosted_endpoint_with_bearer_auth():
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

    assert seen["url"] == "https://inference.do-ai.run/v1/rerank"  # fixed, not configured
    assert seen["auth"] == "Bearer do_secret"
    assert seen["body"]["model"] == MODEL
    assert seen["body"]["query"] == "la consulta"
    assert seen["body"]["documents"] == ["un pasaje"]


def test_rerank_empty_hits_returns_empty_without_calling_the_server():
    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("no HTTP request expected for empty hits")

    reranker = _reranker(handler)
    assert reranker.rerank("q", [], k=5) == []


def test_rerank_fails_fast_on_rate_limits():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, text="slow down")

    reranker = _reranker(handler)
    with pytest.raises(httpx.HTTPStatusError):
        reranker.rerank("q", [_hit("a", 0.9, "t1")], k=1)
    assert len(calls) == 1                     # no retry: surface, don't absorb
