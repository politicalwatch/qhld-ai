"""Unit tests for pairing a co-official passage with its Spanish interpretation."""

import pytest

from qhld_ai.domain.sibling_pairing import SIBLINGS_PER_CHUNK, cosine, nearest_texts

pytestmark = pytest.mark.unit


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_ignores_magnitude():
    assert cosine([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_a_zero_vector_is_zero_not_an_error():
    """An empty passage embeds to nothing; pairing must not raise on it."""
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_nearest_returns_the_closest_texts_closest_first():
    candidates = [("far", [0.0, 1.0]), ("near", [1.0, 0.0]), ("mid", [1.0, 1.0])]

    assert nearest_texts([1.0, 0.0], candidates) == ["near", "mid"]


def test_nearest_returns_two_by_default():
    """Two, because a native passage often straddles two Spanish ones and the
    true partner is then split across the top two candidates."""
    candidates = [(f"c{i}", [1.0, float(i)]) for i in range(5)]

    assert len(nearest_texts([1.0, 0.0], candidates)) == SIBLINGS_PER_CHUNK == 2


def test_nearest_honours_an_explicit_top_n():
    candidates = [("a", [1.0, 0.0]), ("b", [1.0, 1.0]), ("c", [0.0, 1.0])]

    assert nearest_texts([1.0, 0.0], candidates, top_n=1) == ["a"]


def test_nearest_of_a_single_candidate_returns_it():
    assert nearest_texts([1.0, 0.0], [("only", [0.0, 1.0])]) == ["only"]


def test_nearest_without_candidates_is_empty():
    """A speech with no Spanish block — the query path must get an empty list,
    not an error."""
    assert nearest_texts([1.0, 0.0], []) == []


def test_nearest_does_not_threshold():
    """Deliberately unthresholded: the Basque audit found no margin or cosine cut
    that separates correct pairings from wrong ones, so search takes the best
    score across passage and siblings instead of gating here."""
    assert nearest_texts([1.0, 0.0], [("opposite", [-1.0, 0.0])]) == ["opposite"]
