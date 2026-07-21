"""Application service: natural-language semantic search over indexed speeches.

Embeds the query with the same configured embedder used for indexing, runs a vector
search in Qdrant (optionally filtered by exact payload matches — group, legislature,
lang, speaker…), and returns the ranked hits. Each hit's payload carries the speech
metadata and the passage snippet, so callers can render results without a Mongo
round-trip; ``Speeches.get`` is available for full-text hydration when needed.
"""

from langsmith import traceable

from qhld_ai.domain.ports.vector_store import SearchHit, SpeechGroup
from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env
from qhld_ai.infrastructure.vectorstore.factory import create_vector_store_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name


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
        clean = {key: value for key, value in (filters or {}).items() if value is not None}
        extra = self._store_kwargs(query)
        if self.reranker is None:
            return self._store_search(collection, vector, k, clean or None, **extra)
        # Over-fetch a wide candidate pool for the cross-encoder to reorder.
        fetch = max(k, self.settings.reranker_top_n)
        hits = self._store_search(collection, vector, fetch, clean or None, **extra)
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
        clean = {key: value for key, value in (filters or {}).items() if value is not None}
        extra = self._store_kwargs(query)
        if self.reranker is None:
            return self._store_search_grouped(
                collection,
                vector,
                group_by="speech_id",
                limit=page_size,
                group_size=highlights,
                filters=clean or None,
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
        groups = self._store_search_grouped(
            collection,
            vector,
            group_by="speech_id",
            limit=page_size * 2,
            group_size=max(highlights, 5),
            filters=clean or None,
            exclude=exclude,
            **extra,
        )
        pooled = [hit for group in groups for hit in group.highlights]
        rescored = self._rerank(query, pooled, len(pooled),
                                langsmith_extra=self._rerank_metadata())
        # Floor the passages themselves (same gate the ungrouped path applies),
        # so a card never shows a below-floor snippet the detail page then drops.
        # A speech qualifies iff it keeps at least one passage — equivalent to the
        # old "best passage ≥ floor" group gate, so the result set is unchanged.
        surviving = self._above_floor(rescored, apply_floor)
        scores = {hit.id: hit.score for hit in surviving}
        reranked = []
        for group in groups:
            ranked = sorted(
                (SearchHit(id=hit.id, score=scores[hit.id], payload=hit.payload)
                 for hit in group.highlights if hit.id in scores),
                key=lambda hit: hit.score, reverse=True,
            )
            if not ranked:
                continue
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
            reranked.append(
                SpeechGroup(speech_id=group.speech_id, score=top[0].score, highlights=top))
        reranked.sort(key=lambda group: group.score, reverse=True)
        return reranked[:page_size]

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
