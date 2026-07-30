"""Unit test for the SearchSpeeches service — embedder and vector store stubbed."""

import pytest

from qhld_ai.application.search.search_speeches import SearchSpeeches
from qhld_ai.domain.ports.vector_store import SearchHit, SparseVector, SpeechGroup
from qhld_ai.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


def _settings(**overrides):
    return Settings(
        _env_file=None,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:0.6b",
        **overrides,
    )


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]  # dim 3


class _FakeStore:
    def __init__(self):
        self.searched = None

    def search(self, name, vector, k, filters=None):
        self.searched = (name, vector, k, filters)
        return [SearchHit(id="p1", score=0.9, payload={"speaker": "X"})]


def test_search_embeds_query_and_uses_per_model_collection():
    store = _FakeStore()
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)

    hits = service.search("financiación autonómica", k=5)

    name, vector, k, filters = store.searched
    assert name == "speeches__ollama__qwen3_embedding_0_6b__3"  # dim from query vector
    assert vector == [0.1, 0.2, 0.3]
    assert k == 5
    assert filters is None
    assert hits[0].id == "p1"


def test_none_filters_are_dropped():
    store = _FakeStore()
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)

    service.search("hola", filters={"group": "GMx", "lang": None, "speaker": None})

    assert store.searched[3] == {"group": "GMx"}


class _GroupStore:
    def __init__(self):
        self.called = None

    def search_grouped(self, name, vector, group_by, limit, group_size,
                       filters=None, exclude=None):
        self.called = dict(
            name=name, group_by=group_by, limit=limit, group_size=group_size,
            filters=filters, exclude=exclude)
        return [SpeechGroup(
            speech_id="A", score=0.9,
            highlights=[SearchHit(id="p1", score=0.9, payload={"speech_id": "A"})])]


def test_search_grouped_passes_params_and_drops_none_filters():
    store = _GroupStore()
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)

    groups = service.search_grouped(
        "q", page_size=5, highlights=2,
        filters={"lang": "gl", "group": None}, exclude={"X"})

    call = store.called
    assert call["name"] == "speeches__ollama__qwen3_embedding_0_6b__3"
    assert call["group_by"] == "speech_id"
    assert call["limit"] == 5          # page_size → number of speeches
    assert call["group_size"] == 2     # highlights per speech
    assert call["filters"] == {"lang": "gl"}  # None dropped
    assert call["exclude"] == {"X"}
    assert groups[0].speech_id == "A"
    assert groups[0].highlights[0].id == "p1"


def test_search_grouped_no_filters_or_cursor_pass_none():
    store = _GroupStore()
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=store)

    service.search_grouped("q")

    assert store.called["filters"] is None
    assert store.called["exclude"] is None


class _WideStore:
    """Returns a fixed pool regardless of k, recording the k it was asked for."""

    def __init__(self, pool):
        self.pool = pool
        self.k = None

    def search(self, name, vector, k, filters=None):
        self.k = k
        return self.pool


class _FakeReranker:
    def __init__(self):
        self.call = None

    def rerank(self, query, hits, k):
        self.call = (query, list(hits), k)
        # reverse the pool and rescore, then trim — a visible reordering
        rescored = [SearchHit(id=h.id, score=float(i), payload=h.payload)
                    for i, h in enumerate(reversed(hits))]
        return rescored[:k]


def test_search_overfetches_and_reranks_when_reranker_set():
    pool = [SearchHit(id=f"p{i}", score=1.0 - i / 10, payload={"text": f"t{i}"}) for i in range(6)]
    store = _WideStore(pool)
    reranker = _FakeReranker()
    service = SearchSpeeches(
        settings=_settings(), embedder=_FakeEmbedder(), store=store, reranker=reranker)

    hits = service.search("q", k=3)

    assert store.k == 50                      # over-fetched to reranker_top_n, not k
    assert reranker.call[0] == "q" and reranker.call[2] == 3
    assert [h.id for h in hits] == ["p5", "p4", "p3"]  # reranker's reversed top-3


def test_reranker_none_by_default_keeps_baseline():
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=_FakeStore())
    assert service.reranker is None           # noop provider => no reranking


class _ScoringReranker:
    """Returns hits with fixed reranked scores (already sorted desc)."""

    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, hits, k):
        out = [SearchHit(id=f"r{i}", score=s, payload={"text": "t"})
               for i, s in enumerate(self.scores)]
        return out[:k]


def test_reranker_score_floor_drops_low_scoring_hits():
    # An off-domain query's reranked scores fall below the floor → those hits are
    # dropped rather than returned as top-k least-irrelevant passages.
    store = _WideStore([SearchHit(id=f"p{i}", score=0.5, payload={"text": "t"})
                        for i in range(6)])
    reranker = _ScoringReranker([0.9, 0.2, 0.05])
    service = SearchSpeeches(
        settings=_settings(reranker_score_floor=0.15), embedder=_FakeEmbedder(),
        store=store, reranker=reranker)

    hits = service.search("q", k=3)

    assert [round(h.score, 2) for h in hits] == [0.9, 0.2]  # 0.05 dropped


def test_apply_floor_false_exempts_low_scoring_hits():
    # Entity-anchored queries pass apply_floor=False: a brief-mention hit reranks
    # as low as junk but is a valid result, so the floor must not drop it.
    store = _WideStore([SearchHit(id=f"p{i}", score=0.5, payload={"text": "t"})
                        for i in range(6)])
    reranker = _ScoringReranker([0.9, 0.2, 0.05])
    service = SearchSpeeches(
        settings=_settings(reranker_score_floor=0.15), embedder=_FakeEmbedder(),
        store=store, reranker=reranker)

    hits = service.search("q", k=3, apply_floor=False)

    assert [round(h.score, 2) for h in hits] == [0.9, 0.2, 0.05]  # nothing dropped


def test_reranker_score_floor_zero_keeps_every_hit():
    store = _WideStore([SearchHit(id=f"p{i}", score=0.5, payload={"text": "t"})
                        for i in range(6)])
    reranker = _ScoringReranker([0.9, 0.2, 0.05])
    service = SearchSpeeches(  # default floor 0.0 => no-op
        settings=_settings(), embedder=_FakeEmbedder(), store=store, reranker=reranker)

    assert len(service.search("q", k=3)) == 3


class _TwoGroupStore:
    def search_grouped(self, name, vector, group_by, limit, group_size,
                       filters=None, exclude=None):
        return [
            SpeechGroup(speech_id="A", score=0.5,
                        highlights=[SearchHit(id="a1", score=0.5, payload={"text": "x"})]),
            SpeechGroup(speech_id="B", score=0.5,
                        highlights=[SearchHit(id="b1", score=0.5, payload={"text": "z"})]),
        ]


class _ScoresById:
    """Rescores each pooled hit by a fixed per-id score (already sorted desc)."""

    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, hits, k):
        rescored = [SearchHit(id=h.id, score=self.scores[h.id], payload=h.payload)
                    for h in hits]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:k]


def test_reranker_score_floor_drops_low_scoring_groups():
    service = SearchSpeeches(
        settings=_settings(reranker_score_floor=0.15), embedder=_FakeEmbedder(),
        store=_TwoGroupStore(), reranker=_ScoresById({"a1": 0.9, "b1": 0.05}))

    groups = service.search_grouped("q", page_size=5, highlights=1)

    assert [g.speech_id for g in groups] == ["A"]  # B (0.05) dropped by the floor


def test_apply_floor_false_exempts_low_scoring_groups():
    service = SearchSpeeches(
        settings=_settings(reranker_score_floor=0.15), embedder=_FakeEmbedder(),
        store=_TwoGroupStore(), reranker=_ScoresById({"a1": 0.9, "b1": 0.05}))

    groups = service.search_grouped("q", page_size=5, highlights=1, apply_floor=False)

    assert [g.speech_id for g in groups] == ["A", "B"]  # B kept despite 0.05


# --- Hybrid (dense + sparse) search -----------------------------------------

_SPARSE = SparseVector(indices=[7], values=[1.0])


class _FakeSparseEmbedder:
    def embed_query(self, text):
        return _SPARSE


class _HybridStore:
    """Records the sparse_vector keyword the service passes along."""

    def __init__(self):
        self.searched = None
        self.grouped = None

    def search(self, name, vector, k, filters=None, sparse_vector=None):
        self.searched = dict(name=name, k=k, filters=filters, sparse_vector=sparse_vector)
        return [SearchHit(id="p1", score=0.9, payload={"text": "t"})]

    def search_grouped(self, name, vector, group_by, limit, group_size,
                       filters=None, exclude=None, sparse_vector=None):
        self.grouped = dict(name=name, sparse_vector=sparse_vector)
        return [SpeechGroup(
            speech_id="A", score=0.9,
            highlights=[SearchHit(id="p1", score=0.9, payload={})])]


def test_sparse_embedder_none_by_default_keeps_baseline():
    service = SearchSpeeches(settings=_settings(), embedder=_FakeEmbedder(), store=_FakeStore())
    assert service.sparse_embedder is None    # "none" provider => dense-only search


def test_hybrid_search_passes_sparse_vector_and_suffixed_collection():
    store = _HybridStore()
    service = SearchSpeeches(
        settings=_settings(sparse_provider="bm25"), embedder=_FakeEmbedder(),
        store=store, sparse_embedder=_FakeSparseEmbedder())

    service.search("AP-9", k=5)

    assert store.searched["name"] == "speeches__ollama__qwen3_embedding_0_6b__3__bm25"
    assert store.searched["k"] == 5
    assert store.searched["sparse_vector"] == _SPARSE


def test_hybrid_search_composes_with_reranker_overfetch():
    store = _HybridStore()
    reranker = _FakeReranker()
    service = SearchSpeeches(
        settings=_settings(sparse_provider="bm25"), embedder=_FakeEmbedder(),
        store=store, reranker=reranker, sparse_embedder=_FakeSparseEmbedder())

    hits = service.search("q", k=3)

    assert store.searched["k"] == 50          # over-fetched to reranker_top_n
    assert store.searched["sparse_vector"] == _SPARSE
    assert reranker.call[2] == 3
    assert len(hits) == 1


def test_hybrid_search_grouped_passes_sparse_vector():
    store = _HybridStore()
    service = SearchSpeeches(
        settings=_settings(sparse_provider="bm25"), embedder=_FakeEmbedder(),
        store=store, sparse_embedder=_FakeSparseEmbedder())

    groups = service.search_grouped("q", page_size=2, highlights=1)

    assert store.grouped["name"] == "speeches__ollama__qwen3_embedding_0_6b__3__bm25"
    assert store.grouped["sparse_vector"] == _SPARSE
    assert groups[0].speech_id == "A"


def test_search_grouped_reranks_highlights_and_resorts():
    hi_a = [SearchHit(id="a1", score=0.5, payload={"text": "x"}),
            SearchHit(id="a2", score=0.4, payload={"text": "y"})]
    hi_b = [SearchHit(id="b1", score=0.9, payload={"text": "z"})]

    class _Store:
        def search_grouped(self, name, vector, group_by, limit, group_size,
                           filters=None, exclude=None):
            # A ranked first by the bi-encoder, B second
            return [SpeechGroup(speech_id="A", score=0.5, highlights=hi_a),
                    SpeechGroup(speech_id="B", score=0.9, highlights=hi_b)]

    # Gives group B's highlight a higher reranked score so B should overtake A
    reranker = _ScoresById({"a1": 1.0, "a2": 0.5, "b1": 9.0})
    service = SearchSpeeches(
        settings=_settings(), embedder=_FakeEmbedder(), store=_Store(), reranker=reranker)

    groups = service.search_grouped("q", page_size=2, highlights=1)

    assert [g.speech_id for g in groups] == ["B", "A"]  # rerank promoted B
    assert groups[0].score == 9.0                        # group score = best highlight
    assert [h.id for h in groups[1].highlights] == ["a1"]  # trimmed to best highlight


def test_search_grouped_reranks_all_groups_in_one_call():
    # One rerank call over the pooled highlights of every group — the shape a
    # reranker served over HTTP depends on (one round-trip per search, not one
    # per group). Scores come back per hit and are redistributed to their groups.
    hi_a = [SearchHit(id="a1", score=0.5, payload={"text": "x"}),
            SearchHit(id="a2", score=0.4, payload={"text": "y"})]
    hi_b = [SearchHit(id="b1", score=0.9, payload={"text": "z"})]

    class _Store:
        def search_grouped(self, name, vector, group_by, limit, group_size,
                           filters=None, exclude=None):
            return [SpeechGroup(speech_id="A", score=0.5, highlights=hi_a),
                    SpeechGroup(speech_id="B", score=0.9, highlights=hi_b)]

    calls = []

    class _Recording(_ScoresById):
        def rerank(self, query, hits, k):
            calls.append(([h.id for h in hits], k))
            return super().rerank(query, hits, k)

    reranker = _Recording({"a1": 0.7, "a2": 0.8, "b1": 0.6})
    service = SearchSpeeches(
        settings=_settings(), embedder=_FakeEmbedder(), store=_Store(), reranker=reranker)

    groups = service.search_grouped("q", page_size=2, highlights=2)

    assert calls == [(["a1", "a2", "b1"], 3)]        # ONE call, pooled, k = all
    assert [g.speech_id for g in groups] == ["A", "B"]
    assert [h.id for h in groups[0].highlights] == ["a2", "a1"]  # re-sorted in-group
    assert groups[0].score == 0.8 and groups[1].score == 0.6


def test_search_grouped_floors_individual_highlights():
    # The floor gates each PASSAGE, not just the speech: a speech whose best
    # passage clears the floor is kept, but its below-floor passages are dropped —
    # so a card never shows a snippet the detail page (which floors per passage)
    # would then omit.
    hi = [SearchHit(id="a1", score=0.5, payload={"text": "x", "lang": "es"}),
          SearchHit(id="a2", score=0.5, payload={"text": "y", "lang": "es"})]

    class _Store:
        def search_grouped(self, name, vector, group_by, limit, group_size,
                           filters=None, exclude=None):
            return [SpeechGroup(speech_id="A", score=0.5, highlights=hi)]

    service = SearchSpeeches(
        settings=_settings(reranker_score_floor=0.15), embedder=_FakeEmbedder(),
        store=_Store(), reranker=_ScoresById({"a1": 0.9, "a2": 0.05}))

    groups = service.search_grouped("q", page_size=5, highlights=3)

    assert [g.speech_id for g in groups] == ["A"]            # kept: a1 clears the floor
    assert [h.id for h in groups[0].highlights] == ["a1"]    # a2 (0.05) dropped by the floor


def test_search_grouped_keeps_one_language_per_card():
    # A passage indexed in two languages (original + translation) must not appear
    # twice on a card: keep the matched language (the top-scoring survivor's lang),
    # dropping the lower-scoring twin — even though both clear the floor.
    hi = [SearchHit(id="es1", score=0.5, payload={"text": "es", "lang": "es"}),
          SearchHit(id="ca1", score=0.5, payload={"text": "ca", "lang": "ca"})]

    class _Store:
        def search_grouped(self, name, vector, group_by, limit, group_size,
                           filters=None, exclude=None):
            return [SpeechGroup(speech_id="A", score=0.5, highlights=hi)]

    service = SearchSpeeches(
        settings=_settings(reranker_score_floor=0.15), embedder=_FakeEmbedder(),
        store=_Store(), reranker=_ScoresById({"es1": 0.9, "ca1": 0.4}))

    groups = service.search_grouped("q", page_size=5, highlights=3)

    assert [h.id for h in groups[0].highlights] == ["es1"]   # matched lang wins; ca twin dropped


# --- Grouped top-up: cards must show what the detail page would --------------

def _passage(id_, speech_id, score=0.5):
    return SearchHit(id=id_, score=score,
                     payload={"text": id_, "lang": "es", "speech_id": speech_id})


class _SaturatedStore:
    """Grouped retrieval returns a FULL pool (group_size passages) for speech A,
    so more of its passages may exist; the scoped search reveals two more."""

    def __init__(self, pool_size=5):
        self.pool_size = pool_size
        self.searches = []

    def search_grouped(self, name, vector, group_by, limit, group_size,
                       filters=None, exclude=None):
        pooled = [_passage(f"a{i}", "A") for i in range(1, group_size + 1)]
        return [SpeechGroup(speech_id="A", score=0.5, highlights=pooled)]

    def search(self, name, vector, k, filters=None):
        self.searches.append((k, filters))
        return ([_passage(f"a{i}", "A") for i in range(1, self.pool_size + 1)]
                + [_passage("a90", "A"), _passage("a91", "A")])


class _CountingReranker:
    """Scores by id, recording every batch it was asked to score."""

    def __init__(self, scores):
        self.scores = scores
        self.batches = []

    def rerank(self, query, hits, k):
        self.batches.append([h.id for h in hits])
        rescored = [SearchHit(id=h.id, score=self.scores.get(h.id, 0.0), payload=h.payload)
                    for h in hits]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:k]


def test_deficient_card_is_refilled_from_the_rest_of_its_speech():
    # Only a1 clears the floor in the pooled passages, so the card shows 1 of 3 —
    # but the speech has two more above-floor passages the pool never returned.
    store = _SaturatedStore()
    reranker = _CountingReranker(
        {"a1": 0.9, "a2": 0.05, "a3": 0.05, "a4": 0.05, "a5": 0.05,
         "a90": 0.8, "a91": 0.7})
    service = SearchSpeeches(settings=_settings(reranker_score_floor=0.15),
                             embedder=_FakeEmbedder(), store=store, reranker=reranker)

    groups = service.search_grouped("q", page_size=5, highlights=3)

    assert [h.id for h in groups[0].highlights] == ["a1", "a90", "a91"]
    assert store.searches[0][1] == {"speech_id": ["A"]}   # one scoped retrieval
    # the already-scored passages are never sent to the reranker a second time
    assert reranker.batches[1] == ["a90", "a91"]


def test_unsaturated_pool_is_never_topped_up():
    # The pool came back SHORT of group_size, so every passage of the speech was
    # already scored: a card with one survivor is genuinely all there is.
    class _Store:
        def __init__(self):
            self.searches = []

        def search_grouped(self, name, vector, group_by, limit, group_size,
                           filters=None, exclude=None):
            return [SpeechGroup(speech_id="A", score=0.5,
                                highlights=[_passage("a1", "A"), _passage("a2", "A")])]

        def search(self, name, vector, k, filters=None):  # pragma: no cover
            raise AssertionError("no top-up retrieval expected")

    service = SearchSpeeches(
        settings=_settings(reranker_score_floor=0.15), embedder=_FakeEmbedder(),
        store=_Store(), reranker=_ScoresById({"a1": 0.9, "a2": 0.05}))

    groups = service.search_grouped("q", page_size=5, highlights=3)

    assert [h.id for h in groups[0].highlights] == ["a1"]


def test_full_card_costs_no_extra_call():
    # Every slot already filled: the page is returned untouched, no second
    # retrieval and no second rerank.
    store = _SaturatedStore()
    reranker = _CountingReranker({f"a{i}": 0.9 - i / 100 for i in range(1, 6)})
    service = SearchSpeeches(settings=_settings(reranker_score_floor=0.15),
                             embedder=_FakeEmbedder(), store=store, reranker=reranker)

    groups = service.search_grouped("q", page_size=5, highlights=3)

    assert [h.id for h in groups[0].highlights] == ["a1", "a2", "a3"]
    assert store.searches == []          # no scoped retrieval
    assert len(reranker.batches) == 1    # no second rerank


def test_top_up_keeps_the_query_filters():
    # The scoped retrieval carries the query's own filters alongside speech_id;
    # the speeches already satisfy them, and dropping them could widen the match.
    store = _SaturatedStore()
    reranker = _CountingReranker(
        {"a1": 0.9, "a2": 0.05, "a3": 0.05, "a4": 0.05, "a5": 0.05,
         "a90": 0.8, "a91": 0.7})
    service = SearchSpeeches(settings=_settings(reranker_score_floor=0.15),
                             embedder=_FakeEmbedder(), store=store, reranker=reranker)

    service.search_grouped("q", page_size=5, highlights=3, filters={"group": "GS"})

    assert store.searches[0][1] == {"group": "GS", "speech_id": ["A"]}


# --- Browse: no query, no vector, no rerank ----------------------------------
# What a filters-only query gets. Two store calls: the grouped browse names the
# newest speeches, then one filtered fetch turns each into its opening passages.


def _chunk(id_, speech_id, block=0, chunk=0, lang="es", original=True):
    return SearchHit(id=id_, score=0.0, payload={
        "text": id_, "speech_id": speech_id, "lang": lang, "original": original,
        "block_index": block, "chunk_index": chunk})


class _BrowseStore:
    def __init__(self, groups, chunks):
        self._groups = groups
        self._chunks = chunks
        self.grouped_calls = []
        self.browse_calls = []

    def browse_grouped(self, name, group_by, limit, filters=None, exclude=None,
                       order_key="date", descending=True):
        self.grouped_calls.append(dict(
            name=name, group_by=group_by, limit=limit, filters=filters,
            exclude=exclude, order_key=order_key, descending=descending))
        return [SpeechGroup(speech_id=speech_id, score=0.0, highlights=[])
                for speech_id in self._groups[:limit]]

    def browse(self, name, filters=None, limit=10, order_key="date", descending=True):
        self.browse_calls.append(dict(
            name=name, filters=filters, limit=limit, order_key=order_key))
        return list(self._chunks)


def _browse_service(store, **overrides):
    return SearchSpeeches(settings=_settings(**overrides), embedder=_FakeEmbedder(),
                          store=store, reranker=_FakeReranker())


def test_browse_returns_the_filtered_set_without_embedding_a_query():
    class _Store(_FakeStore):
        def browse(self, name, filters=None, limit=10, order_key="date",
                   descending=True):
            self.browsed = dict(name=name, filters=filters, limit=limit,
                                order_key=order_key)
            return [SearchHit(id="p1", score=0.0, payload={})]

    store = _Store()
    hits = _browse_service(store).browse(k=7, filters={"group": "GS", "lang": None})

    assert store.browsed == dict(
        name="speeches__ollama__qwen3_embedding_0_6b__3",  # dim from a probe embedding
        filters={"group": "GS"},                           # None dropped as elsewhere
        limit=7, order_key="date")
    assert hits[0].id == "p1"
    assert store.searched is None                           # never the searching path


def test_browse_grouped_previews_each_speech_with_its_opening_passage():
    store = _BrowseStore(
        groups=["new", "old"],
        chunks=[_chunk("new-2", "new", chunk=2), _chunk("old-0", "old"),
                _chunk("new-0", "new", chunk=0), _chunk("new-1", "new", chunk=1)])

    groups = _browse_service(store).browse_grouped(page_size=5)

    assert [g.speech_id for g in groups] == ["new", "old"]   # store's date order kept
    assert [h.id for h in groups[0].highlights] == ["new-0"]  # the speech's start
    assert [h.id for h in groups[1].highlights] == ["old-0"]
    assert all(group.score == 0.0 for group in groups)


def test_browse_grouped_can_show_several_opening_passages_in_reading_order():
    store = _BrowseStore(
        groups=["A"],
        chunks=[_chunk("a2", "A", block=1, chunk=0), _chunk("a0", "A", chunk=0),
                _chunk("a1", "A", chunk=1)])

    groups = _browse_service(store).browse_grouped(page_size=5, excerpts=3)

    assert [h.id for h in groups[0].highlights] == ["a0", "a1", "a2"]


def test_browse_grouped_previews_the_language_as_delivered():
    # No query means no matched language, so the card shows the speech as spoken
    # rather than its Spanish translation.
    store = _BrowseStore(
        groups=["A"],
        chunks=[_chunk("es0", "A", lang="es", original=False),
                _chunk("gl0", "A", lang="gl", original=True)])

    groups = _browse_service(store).browse_grouped(page_size=5)

    assert [h.id for h in groups[0].highlights] == ["gl0"]


def test_browse_grouped_scopes_the_excerpt_fetch_to_the_page_and_its_filters():
    store = _BrowseStore(groups=["A", "B"], chunks=[_chunk("a0", "A"), _chunk("b0", "B")])

    _browse_service(store).browse_grouped(
        page_size=2, filters={"speaker": "X", "role": None}, exclude={"seen"})

    grouped, fetch = store.grouped_calls[0], store.browse_calls[0]
    assert grouped["filters"] == {"speaker": "X"}       # None dropped
    assert (grouped["limit"], grouped["exclude"]) == (2, {"seen"})
    assert grouped["order_key"] == "date" and grouped["descending"] is True
    # The second call carries the query's filters plus the page's speeches, and
    # asks for no ordering: every passage of a speech shares its date.
    assert fetch["filters"] == {"speaker": "X", "speech_id": ["A", "B"]}
    assert fetch["order_key"] is None


def test_browse_grouped_of_an_empty_page_costs_no_second_call():
    store = _BrowseStore(groups=[], chunks=[])

    assert _browse_service(store).browse_grouped(page_size=5) == []
    assert store.browse_calls == []
