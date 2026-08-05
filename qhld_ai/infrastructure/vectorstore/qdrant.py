"""Qdrant implementation of ``VectorStorePort`` over the raw ``qdrant-client``.

We use the low-level client (not ``langchain-qdrant``) to keep control of
deterministic point ids and the payload shape. ``qdrant_host == ":memory:"``
selects qdrant-client's in-process mode, which lets the tests run with no Docker.

Every client call goes through ``_retry``: a slow embedder can leave the HTTP
connection idle past Qdrant's keep-alive timeout, so the server closes it and the
next call lands on a dead socket (``ResponseHandlingException``). httpx discards
the dead connection, so retrying dials a fresh one — this keeps long full-corpus
index runs (especially with slower/larger embedding models) from dying mid-way.
"""

import time

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from qhld_ai.domain.ports.vector_store import (
    SearchHit,
    SparseVector,
    SpeechGroup,
    VectorPoint,
    VectorStorePort,
)
from qhld_ai.infrastructure.config.settings import Settings
from qhld_ai.logger import get_logger
from .factory import _register

log = get_logger(__name__)

# Named vectors of a hybrid collection (dense-only collections keep the
# original unnamed vector, so they need no migration).
_DENSE = "dense"
_SPARSE = "sparse"


def _turbo(bits) -> models.TurboQuantization:
    return models.TurboQuantization(
        turbo=models.TurboQuantQuantizationConfig(bits=bits, always_ram=True))


def _binary(encoding) -> models.BinaryQuantization:
    return models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(encoding=encoding, always_ram=True))


_PAYLOAD_INDEXES = {
    # Range index: what a date-window filter narrows on, and what Qdrant demands
    # before it will ``order_by`` a key at all (the browse path orders by date).
    "date": models.PayloadSchemaType.INTEGER,
    # Keyword indexes for every payload key a search filters on — the resolved
    # NL-query filters plus ``speech_id``, which scopes the detail page's
    # passages and the grouped path's top-up. Unindexed keys make Qdrant scan
    # the whole collection per filtered query.
    "speech_id": models.PayloadSchemaType.KEYWORD,
    "speaker": models.PayloadSchemaType.KEYWORD,
    "role": models.PayloadSchemaType.KEYWORD,
    "group": models.PayloadSchemaType.KEYWORD,
    "constituency": models.PayloadSchemaType.KEYWORD,
    "mentions": models.PayloadSchemaType.KEYWORD,
    "entities": models.PayloadSchemaType.KEYWORD,
    "lang": models.PayloadSchemaType.KEYWORD,
    "legislature": models.PayloadSchemaType.KEYWORD,
}


# Vector compression presets, keyed by the ``qdrant_quantization`` setting: the
# scheme and the bits per dimension it leaves. Every one pins its compressed
# copy in RAM, which is what makes the copy worth keeping — the originals go to
# disk instead (see ``_dense_params``). Below 4 bits the shortlist is coarse
# enough that it needs re-scoring against the originals to be worth serving, so
# those presets are only usable with ``qdrant_quantization_rescore``.
_QUANTIZATION = {
    "sq8": models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, always_ram=True)),
    "tq4": _turbo(models.TurboQuantBitSize.BITS4),
    "tq2": _turbo(models.TurboQuantBitSize.BITS2),
    "tq1_5": _turbo(models.TurboQuantBitSize.BITS1_5),
    "tq1": _turbo(models.TurboQuantBitSize.BITS1),
    "bq2": _binary(models.BinaryQuantizationEncoding.TWO_BITS),
    "bq1_5": _binary(models.BinaryQuantizationEncoding.ONE_AND_HALF_BITS),
    "bq1": _binary(models.BinaryQuantizationEncoding.ONE_BIT),
}


class QdrantAdapter(VectorStorePort):
    _MAX_ATTEMPTS = 4
    _BACKOFF_SECONDS = 0.5

    def __init__(self, settings: Settings):
        if settings.qdrant_host == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=settings.qdrant_prefer_grpc,
            )
        self._prefetch_limit = settings.hybrid_prefetch_limit
        self._fusion = models.Fusion(settings.hybrid_fusion.lower())
        self._hnsw_ef = settings.qdrant_hnsw_ef
        self._quantization = self._quantization_config(settings.qdrant_quantization)
        self._rescore = settings.qdrant_quantization_rescore

    @staticmethod
    def _quantization_config(name: str):
        """Resolve the configured compression preset. An unknown name fails here
        rather than silently indexing uncompressed vectors."""
        name = (name or "none").lower()
        if name == "none":
            return None
        if name not in _QUANTIZATION:
            raise ValueError(
                f"Unknown qdrant_quantization {name!r}; "
                f"expected 'none' or one of {sorted(_QUANTIZATION)}")
        return _QUANTIZATION[name]

    def _retry(self, operation):
        """Run a Qdrant client call, retrying transient connection drops (stale
        keep-alive sockets closed by the server during long idle gaps)."""
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                return operation()
            except ResponseHandlingException as exc:
                if attempt == self._MAX_ATTEMPTS:
                    raise
                log.warning(
                    f"Qdrant connection error (attempt {attempt}/{self._MAX_ATTEMPTS}), "
                    f"retrying: {exc}")
                time.sleep(self._BACKOFF_SECONDS * attempt)

    def ensure_collection(self, name: str, dim: int, sparse: bool = False) -> None:
        def _ensure():
            if self.client.collection_exists(name):
                # Compression and layout are fixed at creation, so an existing
                # collection keeps whatever shape it was built with. Payload
                # indexes are not: they are added below on every run, which is
                # how a collection built before this existed acquires them.
                self._ensure_payload_indexes(name)
                return
            if sparse:
                # Hybrid collection: a named dense vector plus a named sparse
                # (lexical) vector. The IDF modifier makes Qdrant weight sparse
                # matches by term rarity server-side, so the client only sends
                # corpus-independent term weights.
                self.client.create_collection(
                    collection_name=name,
                    vectors_config={_DENSE: self._dense_params(dim)},
                    sparse_vectors_config={
                        _SPARSE: models.SparseVectorParams(
                            modifier=models.Modifier.IDF),
                    },
                )
            else:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=self._dense_params(dim),
                )
            self._ensure_payload_indexes(name)
        self._retry(_ensure)

    def _ensure_payload_indexes(self, name: str) -> None:
        """Index every payload key searches filter or order on. Re-creating an
        index that already exists is a no-op server-side, so this can run on
        every call; the in-process store ignores payload indexes altogether,
        which is why it is not an error there either."""
        existing = self.client.get_collection(name).payload_schema or {}
        for key, schema in _PAYLOAD_INDEXES.items():
            if key in existing:
                continue
            self.client.create_payload_index(
                collection_name=name, field_name=key, field_schema=schema)

    def _dense_params(self, dim: int) -> models.VectorParams:
        """Dense vector layout. Without compression this is Qdrant's default
        arrangement. With it, the compressed copy is the one held in RAM and the
        originals move to disk — keeping both resident would spend the memory the
        compression just saved."""
        if self._quantization is None:
            return models.VectorParams(size=dim, distance=models.Distance.COSINE)
        return models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
            quantization_config=self._quantization,
            on_disk=True,
        )

    def upsert(self, name: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        self._retry(lambda: self.client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(id=p.id, vector=self._vector(p), payload=p.payload)
                for p in points
            ],
        ))

    @staticmethod
    def _vector(point: VectorPoint):
        """A point with a sparse vector targets a hybrid collection's named
        vectors; without one, the original unnamed dense layout."""
        if point.sparse is None:
            return point.vector
        return {
            _DENSE: point.vector,
            _SPARSE: models.SparseVector(
                indices=point.sparse.indices, values=point.sparse.values),
        }

    def delete_by(self, name: str, key: str, value) -> None:
        self._retry(lambda: self.client.delete(
            collection_name=name,
            points_selector=models.Filter(must=self._conditions(key, value)),
        ))

    def distinct_values(self, name: str, key: str, where: dict | None = None) -> set:
        values = set()
        offset = None
        scroll_filter = self._query_filter(where)
        while True:
            records, offset = self._retry(lambda: self.client.scroll(
                collection_name=name,
                scroll_filter=scroll_filter,
                with_payload=[key],
                with_vectors=False,
                limit=1000,
                offset=offset,
            ))
            for record in records:
                if record.payload and key in record.payload:
                    value = record.payload[key]
                    # A list-valued key (entities, mentions) contributes its
                    # members: the distinct vocabulary, not distinct lists.
                    if isinstance(value, list):
                        values.update(value)
                    else:
                        values.add(value)
            if offset is None:
                break
        return values

    def search_grouped(
        self,
        name: str,
        vector: list[float],
        group_by: str,
        limit: int,
        group_size: int,
        filters: dict | None = None,
        exclude: set | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> list[SpeechGroup]:
        query_filter = self._query_filter(filters, exclude, group_by)
        if sparse_vector is None:
            response = self._retry(lambda: self.client.query_points_groups(
                collection_name=name,
                group_by=group_by,
                query=vector,
                limit=limit,
                group_size=group_size,
                query_filter=query_filter,
                search_params=self._search_params(),
                with_payload=True,
            ))
        else:
            fetch = max(limit * group_size, self._prefetch_limit)
            response = self._retry(lambda: self.client.query_points_groups(
                collection_name=name,
                group_by=group_by,
                prefetch=self._hybrid_prefetch(vector, sparse_vector, fetch, query_filter),
                query=models.FusionQuery(fusion=self._fusion),
                limit=limit,
                group_size=group_size,
                with_payload=True,
            ))
        groups = []
        for group in response.groups:
            highlights = [
                SearchHit(id=str(p.id), score=p.score, payload=p.payload or {})
                for p in group.hits
            ]
            top_score = highlights[0].score if highlights else 0.0
            groups.append(
                SpeechGroup(
                    speech_id=str(group.id), score=top_score, highlights=highlights))
        return groups

    def search(
        self,
        name: str,
        vector: list[float],
        k: int,
        filters: dict | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> list[SearchHit]:
        query_filter = self._query_filter(filters)
        if sparse_vector is None:
            response = self._retry(lambda: self.client.query_points(
                collection_name=name,
                query=vector,
                limit=k,
                query_filter=query_filter,
                search_params=self._search_params(),
                with_payload=True,
            ))
        else:
            response = self._retry(lambda: self.client.query_points(
                collection_name=name,
                prefetch=self._hybrid_prefetch(
                    vector, sparse_vector, max(k, self._prefetch_limit), query_filter),
                query=models.FusionQuery(fusion=self._fusion),
                limit=k,
                with_payload=True,
            ))
        return [
            SearchHit(id=str(point.id), score=point.score, payload=point.payload or {})
            for point in response.points
        ]

    def browse(
        self,
        name: str,
        filters: dict | None = None,
        limit: int = 10,
        order_key: str | None = "date",
        descending: bool = True,
    ) -> list[SearchHit]:
        query = (
            models.OrderByQuery(order_by=self._order_by(order_key, descending))
            if order_key
            else None
        )
        response = self._retry(lambda: self.client.query_points(
            collection_name=name,
            query=query,
            limit=limit,
            query_filter=self._query_filter(filters),
            with_payload=True,
        ))
        # No vector went in, so whatever the server puts in the score field is a
        # placeholder (the in-process store says 1.0). The port promises 0.0:
        # there is no relevance here to report or to sort by.
        return [
            SearchHit(id=str(point.id), score=0.0, payload=point.payload or {})
            for point in response.points
        ]

    def browse_grouped(
        self,
        name: str,
        group_by: str,
        limit: int,
        filters: dict | None = None,
        exclude: set | None = None,
        order_key: str = "date",
        descending: bool = True,
    ) -> list[SpeechGroup]:
        response = self._retry(lambda: self.client.query_points_groups(
            collection_name=name,
            group_by=group_by,
            query=models.OrderByQuery(order_by=self._order_by(order_key, descending)),
            limit=limit,
            # One hit per group is all it takes to name the group and read its
            # ordering value; the passages a card shows are the caller's to pick
            # (see the port docstring), so nothing more is fetched here.
            group_size=1,
            query_filter=self._query_filter(filters, exclude, group_by),
            with_payload=[group_by, order_key],
        ))
        return [
            SpeechGroup(speech_id=str(group.id), score=0.0, highlights=[])
            for group in response.groups
        ]

    @staticmethod
    def _order_by(key: str, descending: bool) -> models.OrderBy:
        direction = models.Direction.DESC if descending else models.Direction.ASC
        return models.OrderBy(key=key, direction=direction)

    def _hybrid_prefetch(
        self,
        vector: list[float],
        sparse_vector: SparseVector,
        fetch: int,
        query_filter: models.Filter | None,
    ) -> list[models.Prefetch]:
        """Dense and sparse candidate branches for a fusion query. The payload
        filter goes on each branch: a top-level filter is not applied to
        prefetched candidates under fusion, so it would be silently ignored.

        Search parameters ride along the same way, and only on the dense branch:
        they tune the vector-graph traversal, while a sparse branch matches an
        inverted index exactly and has nothing to tune."""
        return [
            models.Prefetch(
                query=vector, using=_DENSE, limit=fetch, filter=query_filter,
                params=self._search_params()),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices, values=sparse_vector.values),
                using=_SPARSE, limit=fetch, filter=query_filter),
        ]

    def _search_params(self) -> models.SearchParams | None:
        """Per-query search tuning, or None when there is nothing to tune — which
        leaves the server's own defaults in place, as before these were
        configurable. Re-scoring only means anything against a compressed
        collection, so it is sent only when one is configured."""
        quantization = (
            models.QuantizationSearchParams(rescore=self._rescore)
            if self._quantization is not None and self._rescore is not None
            else None
        )
        if self._hnsw_ef is None and quantization is None:
            return None
        return models.SearchParams(hnsw_ef=self._hnsw_ef, quantization=quantization)

    @classmethod
    def _query_filter(
        cls,
        filters: dict | None,
        exclude: set | None = None,
        group_by: str | None = None,
    ) -> models.Filter | None:
        """The payload filter of one query: the ``filters`` conditions, minus the
        ``exclude``d ``group_by`` values (the "load more" cursor). ``None`` when
        there is nothing to filter on, which is what Qdrant wants for "no
        filter"."""
        must = cls._build_conditions(filters)
        must_not = (
            [models.FieldCondition(key=group_by, match=models.MatchAny(any=list(exclude)))]
            if exclude and group_by
            else []
        )
        if not (must or must_not):
            return None
        return models.Filter(must=must or None, must_not=must_not or None)

    @classmethod
    def _build_conditions(cls, filters: dict | None) -> list[models.FieldCondition]:
        """Translate a ``{key: value}`` filter dict into Qdrant conditions. A scalar
        value is an exact ``MatchValue``; a list is a ``MatchAny`` (any-of); a dict
        is either ``{"all": [...]}`` — one condition per element, so a list payload
        must contain every one — or a numeric ``Range`` whose keys are
        ``gte``/``gt``/``lte``/``lt`` (used for the ``date`` YYYYMMDD int)."""
        return [
            condition
            for key, value in (filters or {}).items()
            for condition in cls._conditions(key, value)
        ]

    @staticmethod
    def _conditions(key: str, value) -> list[models.FieldCondition]:
        if isinstance(value, dict):
            if "all" in value:
                return [
                    models.FieldCondition(key=key, match=models.MatchValue(value=v))
                    for v in value["all"]
                ]
            allowed = {"gte", "gt", "lte", "lt"}
            bounds = {k: v for k, v in value.items() if k in allowed}
            return [models.FieldCondition(key=key, range=models.Range(**bounds))]
        if isinstance(value, (list, tuple, set)):
            return [models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))]
        return [models.FieldCondition(key=key, match=models.MatchValue(value=value))]


@_register("qdrant")
def create(settings: Settings) -> QdrantAdapter:
    return QdrantAdapter(settings)
