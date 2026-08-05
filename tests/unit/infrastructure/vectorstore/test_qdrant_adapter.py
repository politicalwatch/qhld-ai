"""Unit tests for the Qdrant adapter against an in-process store — no Docker."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from qhld_ai.domain.ports.vector_store import SparseVector, VectorPoint
from qhld_ai.infrastructure.config.settings import Settings
from qhld_ai.infrastructure.vectorstore import qdrant as qdrant_mod
from qhld_ai.infrastructure.vectorstore.qdrant import QdrantAdapter

pytestmark = pytest.mark.unit


@pytest.fixture
def adapter():
    return QdrantAdapter(Settings(_env_file=None, qdrant_host=":memory:"))


def _point(payload):
    return VectorPoint(id=str(uuid4()), vector=[0.1, 0.2, 0.3], payload=payload)


def test_ensure_collection_is_idempotent(adapter):
    adapter.ensure_collection("c", 3)
    adapter.ensure_collection("c", 3)  # no error on second call
    assert adapter.client.collection_exists("c")


def test_upsert_and_search_round_trip(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "lang": "es"}),
        _point({"speech_id": "b", "lang": "gl"}),
    ])
    hits = adapter.search("c", [0.1, 0.2, 0.3], k=5)
    assert {h.payload["speech_id"] for h in hits} == {"a", "b"}
    assert all(isinstance(h.score, float) for h in hits)


def test_search_applies_payload_filter(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "lang": "es"}),
        _point({"speech_id": "b", "lang": "gl"}),
    ])
    hits = adapter.search("c", [0.1, 0.2, 0.3], k=5, filters={"lang": "es"})
    assert [h.payload["speech_id"] for h in hits] == ["a"]


def test_search_applies_numeric_range_filter(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "old", "date": 20240101}),
        _point({"speech_id": "mid", "date": 20250501}),
        _point({"speech_id": "new", "date": 20250901}),
    ])
    hits = adapter.search(
        "c", [0.1, 0.2, 0.3], k=5, filters={"date": {"gte": 20250101, "lte": 20250701}})
    assert [h.payload["speech_id"] for h in hits] == ["mid"]


def test_search_list_filter_matches_any_value(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "group": "GS"}),
        _point({"speech_id": "b", "group": "GP"}),
        _point({"speech_id": "c", "group": "GVOX"}),
    ])
    hits = adapter.search("c", [0.1, 0.2, 0.3], k=5, filters={"group": ["GS", "GP"]})
    assert {h.payload["speech_id"] for h in hits} == {"a", "b"}


def test_search_all_filter_requires_every_value_in_list_payload(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "both", "mentions": ["ayuso", "putin"]}),
        _point({"speech_id": "one", "mentions": ["ayuso"]}),
        _point({"speech_id": "other", "mentions": ["putin", "sanchez"]}),
    ])
    hits = adapter.search(
        "c", [0.1, 0.2, 0.3], k=5, filters={"mentions": {"all": ["ayuso", "putin"]}})
    assert [h.payload["speech_id"] for h in hits] == ["both"]


def test_search_combines_range_and_exact_filters(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "date": 20250501, "lang": "es"}),
        _point({"speech_id": "b", "date": 20250501, "lang": "gl"}),
        _point({"speech_id": "c", "date": 20240101, "lang": "es"}),
    ])
    hits = adapter.search(
        "c", [0.1, 0.2, 0.3], k=5,
        filters={"date": {"gte": 20250101}, "lang": "es"})
    assert [h.payload["speech_id"] for h in hits] == ["a"]


def test_search_grouped_applies_range_filter(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "A", "date": 20250901}),
        _point({"speech_id": "B", "date": 20240101}),
    ])
    groups = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=10, group_size=3,
        filters={"date": {"gte": 20250101}})
    assert [g.speech_id for g in groups] == ["A"]


def test_delete_by_removes_matching_points(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "lang": "es"}),
        _point({"speech_id": "b", "lang": "gl"}),
    ])
    adapter.delete_by("c", "speech_id", "a")
    hits = adapter.search("c", [0.1, 0.2, 0.3], k=5)
    assert [h.payload["speech_id"] for h in hits] == ["b"]


def test_upsert_empty_is_a_noop(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [])  # must not raise
    assert adapter.search("c", [0.1, 0.2, 0.3], k=5) == []


def test_distinct_values_returns_unique_payload_values(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "lang": "es"}),
        _point({"speech_id": "a", "lang": "gl"}),  # same speech, second block
        _point({"speech_id": "b", "lang": "es"}),
    ])
    assert adapter.distinct_values("c", "speech_id") == {"a", "b"}


def test_distinct_values_empty_collection(adapter):
    adapter.ensure_collection("c", 3)
    assert adapter.distinct_values("c", "speech_id") == set()


def test_distinct_values_can_be_restricted_to_matching_points(adapter):
    # "Which speakers have spoken for this group" — what narrows a tied surname to the
    # people the rest of the query leaves possible.
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speaker": "Sánchez Díaz, María Carmen", "group": "GS", "date": 20240110}),
        _point({"speaker": "Sánchez Serna, Javier", "group": "GMx", "date": 20240110}),
        _point({"speaker": "Vaquero Montero, Maribel", "group": "GV", "date": 20250315}),
    ])

    assert adapter.distinct_values("c", "speaker", {"group": "GS"}) == {
        "Sánchez Díaz, María Carmen"}
    # The same filter shapes a search takes — a range here.
    assert adapter.distinct_values("c", "speaker", {"date": {"gte": 20250101}}) == {
        "Vaquero Montero, Maribel"}
    # A filter nobody matches is an empty answer, not everybody.
    assert adapter.distinct_values("c", "speaker", {"group": "GSUMAR"}) == set()
    # And no filter still means the whole collection.
    assert len(adapter.distinct_values("c", "speaker")) == 3


def test_distinct_values_flattens_list_payloads(adapter):
    # A list-valued key (entities, mentions) yields the member vocabulary.
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "a", "entities": ["eurovision", "guerra de gaza"]}),
        _point({"speech_id": "b", "entities": ["eurovision"]}),
        _point({"speech_id": "c", "entities": []}),
    ])
    assert adapter.distinct_values("c", "entities") == {"eurovision", "guerra de gaza"}


def test_search_scalar_filter_matches_list_payload_membership(adapter):
    # The entities filter in its most common shape: one key against the
    # speech-level list payload.
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "euro", "entities": ["eurovision", "rtve"]}),
        _point({"speech_id": "gaza", "entities": ["guerra de gaza"]}),
        _point({"speech_id": "none", "entities": []}),
    ])
    hits = adapter.search("c", [0.1, 0.2, 0.3], k=5, filters={"entities": "eurovision"})
    assert [h.payload["speech_id"] for h in hits] == ["euro"]


def test_search_grouped_returns_speeches_with_capped_highlights(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "A", "lang": "es"}),
        _point({"speech_id": "A", "lang": "es"}),
        _point({"speech_id": "A", "lang": "es"}),
        _point({"speech_id": "B", "lang": "es"}),
    ])
    groups = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=10, group_size=2)

    by_id = {g.speech_id: g for g in groups}
    assert set(by_id) == {"A", "B"}
    assert len(by_id["A"].highlights) == 2      # 3 passages, capped at group_size
    assert len(by_id["B"].highlights) == 1
    assert by_id["A"].score == by_id["A"].highlights[0].score


def test_search_grouped_limit_gives_stable_speech_count(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_point({"speech_id": s}) for s in ["A", "B", "C"]])
    groups = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=2, group_size=3)
    assert len(groups) == 2  # number of speeches == limit, regardless of passages


def test_search_grouped_exclude_paginates(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_point({"speech_id": s}) for s in ["A", "B", "C"]])
    first = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=2, group_size=1)
    seen = {g.speech_id for g in first}
    nxt = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=2, group_size=1, exclude=seen)
    assert {g.speech_id for g in nxt}.isdisjoint(seen)  # load-more returns new speeches


def test_search_grouped_applies_exact_filter(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _point({"speech_id": "A", "lang": "es"}),
        _point({"speech_id": "B", "lang": "gl"}),
    ])
    groups = adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=10, group_size=3,
        filters={"lang": "gl"})
    assert [g.speech_id for g in groups] == ["B"]


# --- Browse: the vector-free path ------------------------------------------
# What a filters-only query uses. No vector goes in, so no score comes out and
# the order is the payload's, not relevance.


def _dated(speech_id, date, **payload):
    return _point({"speech_id": speech_id, "date": date, **payload})


def test_browse_returns_filtered_points_newest_first(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _dated("old", 20240101, speaker="A"),
        _dated("new", 20260601, speaker="A"),
        _dated("mid", 20250301, speaker="A"),
        _dated("other", 20260701, speaker="B"),
    ])
    hits = adapter.browse("c", filters={"speaker": "A"}, limit=10)
    assert [h.payload["speech_id"] for h in hits] == ["new", "mid", "old"]
    assert all(hit.score == 0.0 for hit in hits)


def test_browse_can_order_oldest_first(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_dated("old", 20240101), _dated("new", 20260601)])
    hits = adapter.browse("c", limit=10, descending=False)
    assert [h.payload["speech_id"] for h in hits] == ["old", "new"]


def test_browse_honours_the_limit(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_dated(f"s{i}", 20260101 + i) for i in range(5)])
    assert len(adapter.browse("c", limit=2)) == 2


def test_browse_without_an_order_key_still_filters(adapter):
    # How a speech's own passages are fetched: they all share one date, so
    # ordering them by it is meaningless and the caller sorts by chunk instead.
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _dated("A", 20260101, chunk_index=1),
        _dated("A", 20260101, chunk_index=0),
        _dated("B", 20260101, chunk_index=0),
    ])
    hits = adapter.browse("c", filters={"speech_id": ["A"]}, limit=100, order_key=None)
    assert {h.payload["chunk_index"] for h in hits} == {0, 1}
    assert {h.payload["speech_id"] for h in hits} == {"A"}


def test_browse_grouped_returns_distinct_speeches_newest_first(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _dated("old", 20240101), _dated("old", 20240101),
        _dated("new", 20260601), _dated("new", 20260601), _dated("new", 20260601),
        _dated("mid", 20250301),
    ])
    groups = adapter.browse_grouped("c", group_by="speech_id", limit=10)
    assert [g.speech_id for g in groups] == ["new", "mid", "old"]
    # The store names the speeches; which passages preview them is the caller's
    # call, since every passage of a speech shares its date.
    assert all(group.highlights == [] and group.score == 0.0 for group in groups)


def test_browse_grouped_honours_the_limit_and_excludes_seen_speeches(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_dated(s, d) for s, d in
                         [("A", 20260301), ("B", 20260201), ("C", 20260101)]])
    first = adapter.browse_grouped("c", group_by="speech_id", limit=2)
    assert [g.speech_id for g in first] == ["A", "B"]
    nxt = adapter.browse_grouped("c", group_by="speech_id", limit=2, exclude={"A", "B"})
    assert [g.speech_id for g in nxt] == ["C"]


def test_browse_grouped_applies_filters(adapter):
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [
        _dated("A", 20260301, group="GS"),
        _dated("B", 20260201, group="GP"),
    ])
    groups = adapter.browse_grouped(
        "c", group_by="speech_id", limit=10, filters={"group": "GP"})
    assert [g.speech_id for g in groups] == ["B"]


# --- Payload indexes -------------------------------------------------------
# The in-process store ignores payload indexes, so these assert on the requests
# the adapter makes. A real server refuses to order_by an unindexed key at all,
# which is what makes the date index load-bearing rather than a nicety.


def _index_spy(adapter):
    requested = {}
    original = adapter.client.create_payload_index

    def recording(**kwargs):
        requested[kwargs["field_name"]] = kwargs["field_schema"]
        return original(**kwargs)

    adapter.client.create_payload_index = recording
    return requested


def test_ensure_collection_indexes_every_filterable_payload_key(adapter):
    requested = _index_spy(adapter)
    adapter.ensure_collection("c", 3)
    assert requested == qdrant_mod._PAYLOAD_INDEXES
    # date carries a range index because that is what order_by demands.
    assert requested["date"] == qdrant_mod.models.PayloadSchemaType.INTEGER


def test_an_existing_collection_gains_missing_payload_indexes(adapter):
    # The migration path: a collection built before these existed picks them up on
    # the next index run, without re-embedding anything.
    adapter.ensure_collection("c", 3)
    requested = _index_spy(adapter)
    adapter.ensure_collection("c", 3)
    assert set(requested) == set(qdrant_mod._PAYLOAD_INDEXES)


def test_payload_indexes_already_present_are_left_alone(adapter):
    adapter.ensure_collection("c", 3)
    requested = _index_spy(adapter)
    adapter.client.get_collection = lambda name: SimpleNamespace(
        payload_schema=dict.fromkeys(qdrant_mod._PAYLOAD_INDEXES, "indexed"))
    adapter.ensure_collection("c", 3)
    assert requested == {}


# --- Hybrid (dense + sparse) collections -----------------------------------

def _hybrid_point(payload, dense, terms):
    """A point with a dense vector and a sparse vector given as {term_id: weight}."""
    return VectorPoint(
        id=str(uuid4()),
        vector=dense,
        payload=payload,
        sparse=SparseVector(indices=list(terms), values=list(terms.values())),
    )


def _query(adapter, terms, k=5, filters=None):
    return adapter.search(
        "h", [1.0, 0.0, 0.0], k=k, filters=filters,
        sparse_vector=SparseVector(indices=list(terms), values=list(terms.values())))


def test_ensure_sparse_collection_is_idempotent(adapter):
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.ensure_collection("h", 3, sparse=True)  # no error on second call
    assert adapter.client.collection_exists("h")


def test_hybrid_search_surfaces_lexical_only_match(adapter):
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [
        # Semantically close to the query vector, no shared terms.
        _hybrid_point({"speech_id": "sem"}, [1.0, 0.0, 0.0], {11: 1.0}),
        # Semantically orthogonal, but shares the query's term.
        _hybrid_point({"speech_id": "lex"}, [0.0, 1.0, 0.0], {7: 1.0}),
    ])
    hits = _query(adapter, {7: 1.0})
    assert {h.payload["speech_id"] for h in hits} == {"sem", "lex"}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_hybrid_search_applies_filters_to_both_branches(adapter):
    # Regression guard: under a fusion query a top-level filter is not applied,
    # so the filter must ride on each prefetch branch.
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [
        _hybrid_point({"speech_id": "a", "lang": "es", "date": 20250501}, [1.0, 0.0, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "b", "lang": "gl", "date": 20250501}, [1.0, 0.0, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "c", "lang": "es", "date": 20240101}, [1.0, 0.0, 0.0], {7: 1.0}),
    ])
    hits = _query(adapter, {7: 1.0}, filters={"lang": "es", "date": {"gte": 20250101}})
    assert [h.payload["speech_id"] for h in hits] == ["a"]


def test_hybrid_search_applies_list_filter_to_both_branches(adapter):
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [
        _hybrid_point({"speech_id": "a", "group": "GS"}, [1.0, 0.0, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "b", "group": "GP"}, [0.0, 1.0, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "c", "group": "GVOX"}, [1.0, 0.0, 0.0], {7: 1.0}),
    ])
    hits = _query(adapter, {7: 1.0}, filters={"group": ["GS", "GP"]})
    assert {h.payload["speech_id"] for h in hits} == {"a", "b"}


def test_hybrid_upsert_accepts_empty_sparse_vector(adapter):
    # A stopword-only passage encodes to an empty sparse vector; the point must
    # still be stored and reachable through the dense branch.
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [
        _hybrid_point({"speech_id": "empty"}, [1.0, 0.0, 0.0], {}),
    ])
    hits = _query(adapter, {7: 1.0})
    assert [h.payload["speech_id"] for h in hits] == ["empty"]


def test_hybrid_search_grouped_fuses_and_caps_highlights(adapter):
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [
        _hybrid_point({"speech_id": "A"}, [1.0, 0.0, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "A"}, [0.9, 0.1, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "A"}, [0.8, 0.2, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "B"}, [0.0, 1.0, 0.0], {7: 1.0}),
    ])
    groups = adapter.search_grouped(
        "h", [1.0, 0.0, 0.0], group_by="speech_id", limit=10, group_size=2,
        sparse_vector=SparseVector(indices=[7], values=[1.0]))
    by_id = {g.speech_id: g for g in groups}
    assert set(by_id) == {"A", "B"}
    assert len(by_id["A"].highlights) == 2      # 3 passages, capped at group_size
    assert by_id["A"].score == by_id["A"].highlights[0].score


def test_hybrid_search_grouped_applies_filters_and_exclude(adapter):
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [
        _hybrid_point({"speech_id": "A", "lang": "es"}, [1.0, 0.0, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "B", "lang": "es"}, [0.9, 0.1, 0.0], {7: 1.0}),
        _hybrid_point({"speech_id": "C", "lang": "gl"}, [0.8, 0.2, 0.0], {7: 1.0}),
    ])
    sparse = SparseVector(indices=[7], values=[1.0])
    groups = adapter.search_grouped(
        "h", [1.0, 0.0, 0.0], group_by="speech_id", limit=10, group_size=1,
        filters={"lang": "es"}, sparse_vector=sparse)
    assert {g.speech_id for g in groups} == {"A", "B"}
    nxt = adapter.search_grouped(
        "h", [1.0, 0.0, 0.0], group_by="speech_id", limit=10, group_size=1,
        filters={"lang": "es"}, exclude={"A"}, sparse_vector=sparse)
    assert {g.speech_id for g in nxt} == {"B"}


# --- Search tuning and vector compression ----------------------------------
#
# A beam width has no observable effect on the in-process store, so these tests
# assert on the request the adapter builds rather than on the results it gets.
# That is the only way to catch a parameter that is accepted and then ignored.

def _tuned(**overrides):
    return QdrantAdapter(Settings(_env_file=None, qdrant_host=":memory:", **overrides))


def _spy(adapter, method):
    """Capture the keyword arguments of a client call while still letting it run,
    so the request is inspectable and the results stay real."""
    captured = {}
    original = getattr(adapter.client, method)

    def recording(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    setattr(adapter.client, method, recording)
    return captured


def test_search_sends_the_configured_beam_width():
    adapter = _tuned(qdrant_hnsw_ef=512)
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_point({"speech_id": "A"})])
    captured = _spy(adapter, "query_points")
    assert len(adapter.search("c", [0.1, 0.2, 0.3], k=5)) == 1
    assert captured["search_params"].hnsw_ef == 512


def test_search_grouped_sends_the_configured_beam_width():
    adapter = _tuned(qdrant_hnsw_ef=512)
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_point({"speech_id": "A"})])
    captured = _spy(adapter, "query_points_groups")
    assert len(adapter.search_grouped(
        "c", [0.1, 0.2, 0.3], group_by="speech_id", limit=10, group_size=3)) == 1
    assert captured["search_params"].hnsw_ef == 512


def test_hybrid_search_sends_the_beam_width_on_the_dense_branch_only():
    # Regression guard, the same shape as the filter one above: under a fusion
    # query top-level search parameters are not applied to the prefetched
    # candidates, so they must ride on the branch that needs them — and only the
    # dense branch traverses a vector graph.
    adapter = _tuned(qdrant_hnsw_ef=512)
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [_hybrid_point({"speech_id": "a"}, [1.0, 0.0, 0.0], {7: 1.0})])
    captured = _spy(adapter, "query_points")
    hits = _query(adapter, {7: 1.0})
    assert [h.payload["speech_id"] for h in hits] == ["a"]
    dense, sparse = captured["prefetch"]
    assert dense.params.hnsw_ef == 512
    assert sparse.params is None
    assert captured.get("search_params") is None


def test_hybrid_search_grouped_sends_the_beam_width_on_the_dense_branch_only():
    adapter = _tuned(qdrant_hnsw_ef=512)
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [_hybrid_point({"speech_id": "A"}, [1.0, 0.0, 0.0], {7: 1.0})])
    captured = _spy(adapter, "query_points_groups")
    groups = adapter.search_grouped(
        "h", [1.0, 0.0, 0.0], group_by="speech_id", limit=10, group_size=1,
        sparse_vector=SparseVector(indices=[7], values=[1.0]))
    assert [g.speech_id for g in groups] == ["A"]
    dense, sparse = captured["prefetch"]
    assert dense.params.hnsw_ef == 512
    assert sparse.params is None


def test_search_sends_no_parameters_when_nothing_is_tuned(adapter):
    # Unset means the server's own defaults, exactly as before these were
    # configurable.
    adapter.ensure_collection("c", 3)
    adapter.upsert("c", [_point({"speech_id": "A"})])
    captured = _spy(adapter, "query_points")
    adapter.search("c", [0.1, 0.2, 0.3], k=5)
    assert captured["search_params"] is None
    assert adapter._search_params() is None


def test_hybrid_search_sends_no_branch_parameters_when_nothing_is_tuned(adapter):
    adapter.ensure_collection("h", 3, sparse=True)
    adapter.upsert("h", [_hybrid_point({"speech_id": "a"}, [1.0, 0.0, 0.0], {7: 1.0})])
    captured = _spy(adapter, "query_points")
    _query(adapter, {7: 1.0})
    assert all(branch.params is None for branch in captured["prefetch"])


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("name", sorted(qdrant_mod._QUANTIZATION))
def test_ensure_collection_applies_the_configured_compression(name, sparse):
    adapter = _tuned(qdrant_quantization=name)
    adapter.ensure_collection("c", 3, sparse=sparse)
    vectors = adapter.client.get_collection("c").config.params.vectors
    dense = vectors["dense"] if sparse else vectors
    assert dense.quantization_config == qdrant_mod._QUANTIZATION[name]
    # The compressed copy is the resident one, so the originals go to disk.
    assert dense.on_disk is True


def test_ensure_collection_stores_plain_vectors_by_default(adapter):
    adapter.ensure_collection("c", 3)
    dense = adapter.client.get_collection("c").config.params.vectors
    assert dense.quantization_config is None
    assert dense.on_disk is None


def test_unknown_compression_name_fails_fast():
    with pytest.raises(ValueError, match="tq3"):
        _tuned(qdrant_quantization="tq3")


def test_rescore_is_sent_only_against_a_compressed_collection():
    # Nothing to re-score without compression, whatever the setting says.
    assert _tuned(qdrant_quantization_rescore=False)._search_params() is None
    # Compression alone leaves the choice to Qdrant.
    assert _tuned(qdrant_quantization="tq4")._search_params() is None
    params = _tuned(
        qdrant_quantization="tq4", qdrant_quantization_rescore=False)._search_params()
    assert params.quantization.rescore is False
    assert params.hnsw_ef is None


def test_retry_recovers_after_transient_disconnect(adapter, monkeypatch):
    monkeypatch.setattr(qdrant_mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ResponseHandlingException(Exception("Server disconnected"))
        return "ok"

    assert adapter._retry(flaky) == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_retry_raises_after_max_attempts(adapter, monkeypatch):
    monkeypatch.setattr(qdrant_mod.time, "sleep", lambda *_: None)

    def always_down():
        raise ResponseHandlingException(Exception("down"))

    with pytest.raises(ResponseHandlingException):
        adapter._retry(always_down)
