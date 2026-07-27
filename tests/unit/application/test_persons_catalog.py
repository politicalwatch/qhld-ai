"""Unit tests for the person-catalog assembler's curated inputs — pure, no Mongo.

Also lints the shipped ``deputy_aliases.json``: a hand-edited data file is the one
part of this feature no code change can protect, so its invariants are asserted here.
"""

import pytest

from qhld_ai.application.persons_catalog import (
    alias_index,
    deputy_aliases_by_id,
    gazetteer_surfaces,
    load_deputy_aliases,
    load_person_index,
)
from qhld_ai.domain.mentions import normalize_span, resolve_person
from qhld_ai.infrastructure.config.settings import get_settings

pytestmark = pytest.mark.unit


RECORDS = [
    {"deputy_id": "andala-ubbi-teslem", "name": "Andala Ubbi, Teslem",
     "aliases": ["Tesh Sidi"]},
]


class FakeDeputy:
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_fullname(self):
        surname, given = (p.strip() for p in self.name.split(","))
        return f"{given} {surname}"


# --- the two shapes the records are consumed in ----------------------------

def test_aliases_by_id_is_keyed_for_the_index():
    assert deputy_aliases_by_id(RECORDS) == {"andala-ubbi-teslem": ("Tesh Sidi",)}


def test_aliases_by_id_tolerates_a_record_without_aliases():
    assert deputy_aliases_by_id([{"deputy_id": "x", "name": "X, Y"}]) == {"x": ()}


def test_alias_index_is_keyed_by_aliases_only():
    entry, = alias_index(RECORDS)
    assert (entry.person_id, entry.name, entry.keys) == (
        "andala-ubbi-teslem", "Andala Ubbi, Teslem", ("tesh sidi",))


def test_alias_index_resolves_a_subset_and_a_misspelling_like_mentions_do():
    # The point of scoring against an index instead of a dict: the speaker path now
    # accepts every surface the mentions path does.
    index = alias_index(RECORDS)
    for surface in ("Tesh Sidi", "Tesh", "la señora Tesh Sidi", "Tesh Sidí"):
        assert resolve_person(surface, index, 90).name == "Andala Ubbi, Teslem", surface


def test_alias_index_ignores_official_names_so_they_fall_through():
    # Keyed by aliases ALONE, so the official name and surname score ~0 here and the
    # caller falls back to the corpus fuzzy match.
    index = alias_index(RECORDS)
    for surface in ("Andala Ubbi, Teslem", "Andala Ubbi", "Teslem", "Sidi Ifni"):
        assert resolve_person(surface, index, 90) is None, surface


def test_alias_index_skips_records_without_a_canonical_name():
    # Without ``name`` there is nothing to filter the corpus on, so the record serves
    # the mentions path only.
    assert alias_index([{"deputy_id": "x", "aliases": ["Whoever"]}]) == []


def test_alias_index_skips_aliases_that_normalize_away():
    index = alias_index(
        [{"deputy_id": "x", "name": "X, Y", "aliases": ["el señor", "Real Name"]}])
    assert index[0].keys == ("real name",)


def test_aliases_reach_the_assembled_person_index():
    index = load_person_index(
        [FakeDeputy("andala-ubbi-teslem", "Andala Ubbi, Teslem")], 90,
        curated=[], nondeputy_speakers=[], deputy_aliases=RECORDS)
    assert resolve_person("Tesh Sidi", index, 90).person_id == "andala-ubbi-teslem"


# --- lint of the shipped data file -----------------------------------------

def test_shipped_alias_file_is_well_formed():
    records = load_deputy_aliases()
    assert records, "the file exists to carry at least one curated alias"
    for row in records:
        assert row.get("deputy_id"), row
        assert row.get("name"), row
        assert row.get("aliases"), row
        assert "," in row["name"], f"canonical name must be 'Apellido, Nombre': {row}"
        for alias in row["aliases"]:
            assert normalize_span(alias), f"alias normalizes away: {alias!r}"


def test_shipped_aliases_are_unique_across_records():
    seen = set()
    for row in load_deputy_aliases():
        for alias in row["aliases"]:
            key = normalize_span(alias)
            assert key not in seen, f"duplicate alias across records: {alias!r}"
            seen.add(key)


# --- tag surfaces (what the NER may tag) -----------------------------------

def test_gazetteer_surfaces_are_separate_from_aliases():
    # A surface can be resolvable without being taggable, which is the whole point of
    # the two lists: "Sidi" resolves as a subset of the alias but must never be tagged.
    records = [{"deputy_id": "andala-ubbi-teslem", "name": "Andala Ubbi, Teslem",
                "aliases": ["Tesh Sidi"], "tag_surfaces": ["Tesh"]}]
    assert gazetteer_surfaces(records) == ("Tesh",)
    assert alias_index(records)[0].keys == ("tesh sidi",)


def test_gazetteer_surfaces_default_to_nothing():
    assert gazetteer_surfaces(RECORDS) == ()
    assert gazetteer_surfaces([]) == ()
    assert gazetteer_surfaces(None) == ()


def test_shipped_tag_surfaces_are_out_of_vocabulary():
    # The adapter only turns OUT-OF-VOCABULARY terms into patterns, which is the second
    # line of defence behind curation: an in-vocabulary token ("Torres") would be
    # silently ignored, so curating one is a curation error worth catching here.
    spacy = pytest.importorskip("spacy")
    nlp = spacy.load(get_settings().ner_model)
    for row in load_deputy_aliases():
        for surface in row.get("tag_surfaces", ()):
            assert nlp.vocab[surface.lower()].is_oov, (
                f"{surface!r} is in-vocabulary, so the adapter will ignore it")
