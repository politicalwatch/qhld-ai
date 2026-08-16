"""Pure pairing of a co-official passage with its Spanish interpretation — no
I/O, no deps.

A speech given in Catalan, Galician or Basque is published as two parallel
blocks: the words as delivered, and the Spanish interpretation of the same
speech. Search reranks a passage against its interpretation as well as itself,
because a cross-encoder scores a language-mismatched pair as junk however
relevant it is (see ``SearchSpeeches._rerank_against_siblings``). That needs to
know which Spanish passage renders which native one.

The pairing is cosine over the embeddings the indexer already computes, so it
costs no extra inference. Chunk position is NOT usable: the two blocks are
chunked independently and 16% of correct pairings are off-diagonal.

Ranking, not gating: which Spanish passage is closest is a question the
embeddings answer well (bge-m3 ranks the true partner first for 13/14 Basque,
16/16 Catalan and 12/12 Galician paragraphs), whereas "is this a rendering at
all" is one they answer badly — the refuted instruments in the bitext work were
all refuted as gates. Here the interpretation is known to exist, so only the
ranking question is ever asked.
"""

import math

# Two, because the blocks are chunked independently and their boundaries do not
# coincide: a native passage often straddles two Spanish ones, and then the true
# partner is split across the top two candidates. Every partial mispairing found
# in the Basque audit was a straddle whose other half was the runner-up.
SIBLINGS_PER_CHUNK = 2


def cosine(left, right):
    """Cosine similarity of two dense vectors. 0.0 if either has no magnitude."""
    dot = sum(x * y for x, y in zip(left, right))
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return dot / norm if norm else 0.0


def nearest_texts(vector, candidates, top_n=SIBLINGS_PER_CHUNK):
    """The ``top_n`` candidate texts closest to ``vector``, closest first.

    ``candidates`` are ``(text, vector)`` pairs — the Spanish passages of the
    same speech. Deliberately unthresholded: an audit of all 63 Basque passages
    measured pairing 93.7% correct but found NO margin or cosine cut that
    separates the right pairings from the wrong ones (the cleanest one rejects
    17.5% of correct pairings and still admits a defect). Search takes the best
    score across the native passage and these, so a wrong candidate scores low
    and loses instead of being gated out.
    """
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda pair: cosine(vector, pair[1]), reverse=True)
    return [text for text, _ in ranked[:top_n]]
