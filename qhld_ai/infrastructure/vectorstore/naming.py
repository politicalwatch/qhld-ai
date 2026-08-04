"""Collection naming for speech embeddings.

Pure string logic (kept beside the adapter for cohesion). Collections are named
per embedding model + dimension so that indexing different models — the TFM's
benchmark A/B — writes to separate collections instead of clobbering each other.
An explicit ``qdrant_collection`` setting overrides the derived name.

The name carries every property Qdrant bakes in when the collection is created
and that no later query can change: embedding provider and model, vector
dimension, lexical branch, compression scheme. Search-time knobs stay out of it —
``qdrant_hnsw_ef`` and ``qdrant_quantization_rescore`` travel on ``SearchParams``
and may differ between two queries against the same collection, so naming them
would promise a distinction the storage does not make.

Because the name is a contract between the indexer and every reader, each service
must derive it from the same settings: a reader that omits ``qdrant_quantization``
resolves the uncompressed name and searches a collection the indexer never writes.
"""

import re

from qhld_ai.infrastructure.config.settings import Settings


def _token(value: str) -> str:
    """Lowercase ``value`` and reduce anything a collection name cannot carry to
    underscores, so ``bge-m3:567m`` and ``tq1.5`` become ``bge_m3_567m`` and
    ``tq1_5``. Already-underscored values pass through unchanged."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def collection_name(settings: Settings, dim: int) -> str:
    if settings.qdrant_collection:
        return settings.qdrant_collection
    provider = settings.embedding_provider.lower()
    model = _token(settings.embedding_model)
    name = f"speeches__{provider}__{model}__{dim}"
    # Hybrid collections carry an extra sparse (lexical) vector per point, so
    # they get their own name — dense-only collections stay untouched.
    sparse = (settings.sparse_provider or "").lower()
    if sparse and sparse != "none":
        name += f"__{sparse}"
    # Compression is fixed when the collection is created, so it belongs in the
    # name: two compression schemes are two collections, and an uncompressed one
    # keeps the name it already has. Normalised exactly as the adapter's
    # ``_quantization_config`` does, so "" and "none" agree on "no suffix";
    # an unknown value raises there, at adapter construction.
    quantization = _token(settings.qdrant_quantization or "none")
    if quantization != "none":
        name += f"__{quantization}"
    return name
