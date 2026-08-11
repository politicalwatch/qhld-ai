"""How much one paragraph reads as a rendering of another, across languages.

The question is "is B a translation of A", and the instrument that answers it best of
those available is the cosine between multilingual sentence embeddings. Measured over a
hand-labelled bitext set, it ranks the true partner first 44 times in 45 where whole-token
overlap manages 25 in 42 — and unlike surface overlap it works for Basque, which shares
almost no vocabulary with Spanish and against which a lexical measure scores near zero
however complete the translation.

Two things it is NOT, both measured rather than assumed:

- **A cross-encoder reranker is worse here, not better.** ``bge-reranker-v2-m3`` is trained
  on relevance, and two paragraphs of one speech arguing the same point are maximally
  relevant to each other; its Basque separation is −0.963 against the bi-encoder's −0.132.
- **A bitext-mining model is not better either.** LaBSE, trained for exactly this task,
  matched bge-m3 rather than beating it, so there is no case for a serving lane we do not
  already have.

Within one speech every paragraph shares the subject, the names and the figures, so a high
score is necessary and not sufficient. For Basque no threshold separates true pairs from
near misses at all, and the caller is expected to send those on for adjudication rather
than decide on this number alone.
"""

from qhld_ai.domain.ports.embeddings import EmbedderProtocol


class ParagraphSimilarity:
    """Cosine between two paragraphs, over whatever embedder is configured.

    Caches by text because the aligner asks about the same paragraph many times — once
    per candidate on the other side — and a speech of sixteen paragraphs would otherwise
    embed each of them a dozen times over.
    """

    def __init__(self, embedder: EmbedderProtocol):
        self._embedder = embedder
        self._vectors: dict[str, list[float]] = {}

    def _vector(self, text: str) -> list[float]:
        cached = self._vectors.get(text)
        if cached is None:
            raw = self._embedder.embed_documents([text])[0]
            norm = sum(value * value for value in raw) ** 0.5 or 1.0
            cached = [value / norm for value in raw]
            self._vectors[text] = cached
        return cached

    def warm(self, texts) -> None:
        """Embed several paragraphs in one call.

        Worth doing before an alignment: the providers batch, so one request for a whole
        speech costs far less than one per paragraph.
        """
        missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if not missing:
            return
        for text, raw in zip(missing, self._embedder.embed_documents(missing)):
            norm = sum(value * value for value in raw) ** 0.5 or 1.0
            self._vectors[text] = [value / norm for value in raw]

    def __call__(self, source: str, candidate: str) -> float:
        a, b = self._vector(source), self._vector(candidate)
        return sum(x * y for x, y in zip(a, b))


def create_paragraph_similarity(embedder: EmbedderProtocol | None = None,
                                settings=None) -> ParagraphSimilarity:
    """The similarity the splitter injects, on the same embedder the indexer uses.

    Deliberately the configured lane rather than a model chosen for this job: a speech
    classified with one model and searched with another would drift apart silently.
    """
    if embedder is None:
        from qhld_ai.infrastructure.embeddings.factory import create_embedder_from_env
        embedder = create_embedder_from_env(settings)
    return ParagraphSimilarity(embedder)
