"""Application service: natural-language semantic search over indexed speeches.

Embeds the query with the same configured embedder used for indexing, runs a vector
search in Qdrant (optionally filtered by exact payload matches — group, legislature,
lang, speaker…), and returns the ranked hits. Each hit's payload carries the speech
metadata and the passage snippet, so callers can render results without a Mongo
round-trip; ``Speeches.get`` is available for full-text hydration when needed.

``browse``/``browse_grouped`` are the same thing without a query: for a request
that carries only filters there is no text to embed and nothing to rank, so they
return the newest matching passages/speeches instead, unscored.
"""

from langsmith import traceable

from qhld_ai.domain.ports.vector_store import SearchHit, SpeechGroup
from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env
from qhld_ai.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name

# Retrieval ceiling for the grouped path's top-up: a speech_id filter narrows the
# candidates to a handful of speeches, so this only means "return all of them".
_TOPUP_K = 1000


def _drop_vectors(inputs: dict) -> dict:
    """Keep traced retrieval inputs readable: the query vectors are hundreds of
    floats and carry no diagnostic value next to collection/filters/k."""
    return {key: value for key, value in inputs.items()
            if key not in ("vector", "sparse_vector")}


class SearchSpeeches:
    def __init__(self, settings=None, embedder=None, store=None, reranker=None,
                 sparse_embedder=None):
        self.settings = settings or get_settings()
        self.embedder = embedder or create_embedder_from_env(self.settings)
        self.store = store or create_vector_store_from_env(self.settings)
        self.reranker = reranker if reranker is not None else self._reranker_from_settings()
        self.sparse_embedder = (
            sparse_embedder if sparse_embedder is not None
            else self._sparse_from_settings()
        )
        self._collection_name = None

    def _reranker_from_settings(self):
        """Build the configured reranker, or ``None`` for the "noop"/unset default
        so the bi-encoder baseline path stays byte-identical."""
        provider = (self.settings.reranker_provider or "").lower()
        if not provider or provider == "noop":
            return None
        from qhld_ai.infrastructure.reranker.factory import create_reranker_from_env

        return create_reranker_from_env(self.settings)

    def _sparse_from_settings(self):
        """Build the configured sparse embedder, or ``None`` for the "none"/unset
        default so the dense-only baseline path stays byte-identical."""
        provider = (self.settings.sparse_provider or "").lower()
        if not provider or provider == "none":
            return None
        from qhld_ai.infrastructure.sparse.factory import create_sparse_embedder_from_env

        return create_sparse_embedder_from_env(self.settings)

    def _collection(self):
        """The target collection name for a query that has no vector to read the
        model dimension off (the browse path). One probe embedding per process,
        memoized — the searching paths keep deriving it from the query vector
        they already computed."""
        if self._collection_name is None:
            dim = len(self.embedder.embed_query("probe"))
            self._collection_name = collection_name(self.settings, dim)
        return self._collection_name

    def _store_kwargs(self, query):
        """Hybrid searches pass the lexical query vector as an extra keyword; the
        dense-only call shape stays exactly as before."""
        if self.sparse_embedder is None:
            return {}
        return {"sparse_vector": self._sparse_embed_query(query)}

    # Traced pass-through helpers: each model/store call in a search becomes a
    # child span of the search's trace (inert unless LANGSMITH_TRACING is set),
    # so one trace shows the whole flow — embed, retrieve, rerank — with
    # timings. Vectors are elided from the logged payloads; passages are not
    # (seeing what the reranker scored is the point of the trace).

    @traceable(name="embed_query", run_type="embedding",
               process_outputs=lambda vector: {"dim": len(vector)})
    def _embed_query(self, query):
        return self.embedder.embed_query(query)

    @traceable(name="sparse_embed_query", run_type="embedding",
               process_outputs=lambda sv: {"nonzero": len(sv.indices)})
    def _sparse_embed_query(self, query):
        return self.sparse_embedder.embed_query(query)

    @traceable(name="vector_search", run_type="retriever",
               process_inputs=_drop_vectors)
    def _store_search(self, collection, vector, k, filters, **extra):
        return self.store.search(collection, vector, k, filters, **extra)

    @traceable(name="vector_search_grouped", run_type="retriever",
               process_inputs=_drop_vectors)
    def _store_search_grouped(self, collection, vector, **kwargs):
        return self.store.search_grouped(collection, vector, **kwargs)

    @traceable(name="browse", run_type="retriever")
    def _store_browse(self, collection, filters, limit, order_key="date"):
        return self.store.browse(
            collection, filters=filters, limit=limit, order_key=order_key)

    @traceable(name="browse_grouped", run_type="retriever")
    def _store_browse_grouped(self, collection, **kwargs):
        return self.store.browse_grouped(collection, **kwargs)

    @traceable(name="rerank", run_type="chain")
    def _rerank(self, query, hits, k):
        return self.reranker.rerank(query, hits, k)

    def _rerank_metadata(self):
        """Stamped on rerank spans so a trace names the serving provider/model."""
        return {"metadata": {"provider": self.settings.reranker_provider,
                             "model": self.settings.reranker_model}}

    @traceable(name="search_speeches", run_type="chain")
    def search(self, query, k=10, filters=None, apply_floor=True) -> list[SearchHit]:
        vector = self._embed_query(query)
        # The query vector's length is the model dimension, which is part of the
        # per-model collection name — no separate probe needed.
        collection = collection_name(self.settings, len(vector))
        clean = self._clean(filters)
        extra = self._store_kwargs(query)
        if self.reranker is None:
            return self._store_search(collection, vector, k, clean, **extra)
        # Over-fetch a wide candidate pool for the cross-encoder to reorder.
        fetch = max(k, self.settings.reranker_top_n)
        hits = self._store_search(collection, vector, fetch, clean, **extra)
        reranked = self._rerank(query, hits, k, langsmith_extra=self._rerank_metadata())
        return self._above_floor(reranked, apply_floor)

    @traceable(name="search_speeches_grouped", run_type="chain")
    def search_grouped(
        self, query, page_size=10, highlights=3, filters=None, exclude=None,
        apply_floor=True,
    ) -> list[SpeechGroup]:
        """Speech-level results: ``page_size`` distinct speeches, each with up to
        ``highlights`` matching passages. Pagination is stateless — the caller
        accumulates the returned ``speech_id``s and passes them back as ``exclude``
        to fetch the next page ("load more")."""
        vector = self._embed_query(query)
        collection = collection_name(self.settings, len(vector))
        clean = self._clean(filters)
        extra = self._store_kwargs(query)
        if self.reranker is None:
            return self._store_search_grouped(
                collection,
                vector,
                group_by="speech_id",
                limit=page_size,
                group_size=highlights,
                filters=clean,
                exclude=exclude,
                **extra,
            )
        # Over-fetch groups (and highlights per group) so the reranker can promote
        # a speech the bi-encoder ranked lower. All groups' highlights are pooled
        # into ONE rerank call: a pointwise reranker scores each (query, passage)
        # pair independently, so the scores are identical to per-group calls —
        # and a reranker served over HTTP pays one round-trip per search instead
        # of one per group. Each group is then rebuilt from its own reranked
        # highlights: drop below-floor passages, keep one language, group score =
        # best surviving highlight, re-sort, trim.
        pool_size = max(highlights, 5)
        groups = self._store_search_grouped(
            collection,
            vector,
            group_by="speech_id",
            limit=page_size * 2,
            group_size=pool_size,
            filters=clean,
            exclude=exclude,
            **extra,
        )
        pooled = [hit for group in groups for hit in group.highlights]
        rescored = self._rerank(query, pooled, len(pooled),
                                langsmith_extra=self._rerank_metadata())
        # Every score is kept, below-floor ones included: the floor is applied per
        # group in _build_group, and the top-up below reads this map to know which
        # passages were already scored so none is ever paid for twice.
        scored = {hit.id: hit.score for hit in rescored}
        reranked = [
            group for group in (
                self._build_group(source.speech_id, source.highlights, scored,
                                  highlights, apply_floor)
                for source in groups
            ) if group is not None
        ]
        reranked.sort(key=lambda group: group.score, reverse=True)
        page = reranked[:page_size]
        page = self._top_up(
            query, collection, vector, clean, extra, page,
            {source.speech_id: len(source.highlights) for source in groups},
            pool_size, highlights, apply_floor, scored)
        page.sort(key=lambda group: group.score, reverse=True)
        return page

    @traceable(name="browse_speeches", run_type="chain")
    def browse(self, k=10, filters=None) -> list[SearchHit]:
        """The newest ``k`` filtered passages — the vector-free counterpart of
        ``search``, for a query that named only filters and no topic. Nothing was
        searched for, so there is nothing to rank by: recency stands in."""
        return self._store_browse(self._collection(), self._clean(filters), k)

    @traceable(name="browse_speeches_grouped", run_type="chain")
    def browse_grouped(self, page_size=10, excerpts=1, filters=None,
                       exclude=None) -> list[SpeechGroup]:
        """The newest ``page_size`` filtered speeches, each with the first
        ``excerpts`` passages of the speech as its card text. Same stateless
        ``exclude`` cursor as ``search_grouped``.

        Two store calls, no embedding and no reranking: the grouped browse names
        the speeches (every passage of one shares its date, so the store cannot
        say which passage opens it), then one filtered fetch of just those
        speeches' passages puts the excerpt in reading order. The page keeps the
        store's date order — there are no scores to sort by."""
        collection = self._collection()
        clean = self._clean(filters)
        groups = self._store_browse_grouped(
            collection, group_by="speech_id", limit=page_size, filters=clean,
            exclude=exclude)
        if not groups:
            return []
        # k is a ceiling the store API requires, not a passage cap: the speech_id
        # filter narrows candidates to these speeches' own passages.
        hits = self._store_browse(
            collection, {**(clean or {}), "speech_id": [g.speech_id for g in groups]},
            _TOPUP_K, order_key=None)
        passages = {}
        for hit in hits:
            passages.setdefault(hit.payload.get("speech_id"), []).append(hit)
        return [
            self._opening_group(group.speech_id, passages.get(group.speech_id, []),
                                excerpts)
            for group in groups
        ]

    @staticmethod
    def _clean(filters):
        """Drop unset filter keys, as every retrieval path does before handing
        them to the store; ``None`` for "no filters at all"."""
        clean = {key: value for key, value in (filters or {}).items() if value is not None}
        return clean or None

    @staticmethod
    def _opening_group(speech_id, hits, excerpts) -> SpeechGroup:
        """One browsed speech's card: its opening passages, in reading order.

        The language kept is the as-delivered one (``original``), not the
        matched one ``_build_group`` picks — a browse matched no passage, so
        there is no query language to follow, and a Galician speech is more
        honestly previewed in Galician than in its Spanish translation. Score is
        0.0 throughout: recency ordered this page, not relevance."""
        original = [hit for hit in hits if hit.payload.get("original")] or hits
        ordered = sorted(
            original,
            key=lambda hit: (hit.payload.get("block_index") or 0,
                             hit.payload.get("chunk_index") or 0))
        return SpeechGroup(speech_id=speech_id, score=0.0, highlights=ordered[:excerpts])

    def _build_group(self, speech_id, hits, scored, highlights, apply_floor):
        """One speech's card, built from its scored passages — or ``None`` when the
        speech keeps nothing.

        Floors the passages themselves (the same gate the ungrouped path applies),
        so a card never shows a below-floor snippet the detail page then drops. A
        speech qualifies iff it keeps at least one passage, which is equivalent to
        the older "best passage ≥ floor" group gate.
        """
        ranked = sorted(
            (SearchHit(id=hit.id, score=scored[hit.id], payload=hit.payload)
             for hit in hits if hit.id in scored),
            key=lambda hit: hit.score, reverse=True,
        )
        ranked = self._above_floor(ranked, apply_floor)
        if not ranked:
            return None
        # One language per card (no original/translation twins of the same
        # passage): keep the matched language — the top-scoring survivor's lang,
        # which is the query's language for a same-language corpus hit — with
        # Spanish as the fallback, then whatever remains. Done BEFORE the top-N
        # trim so twins never consume card slots.
        matched = ranked[0].payload.get("lang")
        same = (
            [hit for hit in ranked if hit.payload.get("lang") == matched]
            or [hit for hit in ranked if hit.payload.get("lang") == "es"]
            or ranked
        )
        top = same[:highlights]
        return SpeechGroup(speech_id=speech_id, score=top[0].score, highlights=top)

    def _top_up(self, query, collection, vector, filters, extra, page, pool_sizes,
                pool_size, highlights, apply_floor, scored):
        """Refill cards showing fewer passages than their speech actually offers.

        Grouped retrieval returns at most ``pool_size`` passages per speech, so a
        passage the bi-encoder ranked below that but the reranker scores above the
        floor reaches the detail page — which scores ALL of a speech's passages —
        and never the card. Cards short of ``highlights`` are refilled here from
        the rest of their own speech, costing one extra retrieval and one extra
        rerank for the whole page, both reusing the query vectors already computed.

        A card whose pool came back short of ``pool_size`` is skipped: every
        passage of that speech was already scored, so nothing can be missing.
        Only the page's own cards are refilled — a speech that never made the page
        keeps the score its pooled passages earned. Refilling those too would mean
        reranking every candidate speech in full, which costs several times more
        and is still not exact for the longest speeches.
        """
        deficient = [group for group in page
                     if len(group.highlights) < highlights
                     and pool_sizes.get(group.speech_id) == pool_size]
        if not deficient:
            return page
        speech_ids = [group.speech_id for group in deficient]
        # k is a ceiling the store API requires, not a passage cap: the speech_id
        # filter narrows candidates to these speeches' own passages.
        hits = self._store_search(
            collection, vector, _TOPUP_K,
            {**(filters or {}), "speech_id": speech_ids}, **extra)
        fresh = [hit for hit in hits if hit.id not in scored]
        if not fresh:
            return page
        rescored = self._rerank(query, fresh, len(fresh),
                                langsmith_extra=self._rerank_metadata())
        scored = {**scored, **{hit.id: hit.score for hit in rescored}}
        passages = {}
        for hit in hits:
            passages.setdefault(hit.payload.get("speech_id"), []).append(hit)
        refilled = {}
        for group in deficient:
            rebuilt = self._build_group(
                group.speech_id, passages.get(group.speech_id, []), scored,
                highlights, apply_floor)
            if rebuilt is not None:
                refilled[group.speech_id] = rebuilt
        return [refilled.get(group.speech_id, group) for group in page]

    def _above_floor(self, items, apply_floor):
        """Drop reranked hits/groups scoring below the relevance floor. Only the
        reranked path calls this: cross-encoder scores separate in-domain from
        off-domain, so a floor here turns a nonsensical query into zero results
        instead of the top-k least-irrelevant passages. Callers pass
        ``apply_floor=False`` for pure-entity queries: a speech that merely
        mentions the entity in passing is a valid hit yet reranks as low as junk
        (the passage is about something else), so no floor value can separate the
        two — precision comes from the entity filter there instead. A floor of
        0.0 (the default) is a no-op, so the bi-encoder baseline stays
        byte-identical."""
        floor = self.settings.reranker_score_floor
        if not apply_floor or not floor:
            return items
        return [item for item in items if item.score >= floor]
