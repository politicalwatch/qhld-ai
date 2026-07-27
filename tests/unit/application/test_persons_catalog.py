"""Unit tests for the person-catalog assembler's curated inputs — pure, no Mongo.

Also lints the shipped ``deputy_aliases.json``: a hand-edited data file is the one
part of this feature no code change can protect, so its invariants are asserted here.
"""

import pytest

from qhld_ai.application.persons_catalog import (
    deputy_aliases_by_id,
    load_deputy_aliases,
    load_person_index,
    speaker_alias_map,
)
from qhld_ai.domain.mentions import normalize_span, resolve_person

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


def test_speaker_alias_map_normalizes_the_query_side():
    assert speaker_alias_map(RECORDS) == {"tesh sidi": "Andala Ubbi, Teslem"}


def test_speaker_alias_map_skips_records_without_a_canonical_name():
    # Without ``name`` there is nothing to filter the corpus on, so the record serves
    # the mentions path only.
    assert speaker_alias_map([{"deputy_id": "x", "aliases": ["Whoever"]}]) == {}


def test_speaker_alias_map_skips_aliases_that_normalize_away():
    assert speaker_alias_map(
        [{"deputy_id": "x", "name": "X, Y", "aliases": ["el señor", "Real Name"]}]
    ) == {"real name": "X, Y"}


def test_speaker_alias_map_first_record_wins_a_duplicate():
    records = [{"deputy_id": "a", "name": "A, A", "aliases": ["Dup"]},
               {"deputy_id": "b", "name": "B, B", "aliases": ["Dup"]}]
    assert speaker_alias_map(records) == {"dup": "A, A"}


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


def test_shipped_aliases_are_whole_names_not_bare_tokens():
    # The invariant that actually prevents a wrong attribution: token_set_ratio scores
    # a subset at 100, so a whole public name already covers its parts, while a bare
    # token would swallow any surface containing it (a curated "Sidi" would match the
    # Moroccan town "Sidi Ifni").
    for row in load_deputy_aliases():
        for alias in row["aliases"]:
            assert len(normalize_span(alias).split()) > 1, (
                f"curate whole public names, not bare tokens: {alias!r}")


def test_shipped_aliases_are_unique_across_records():
    seen = set()
    for row in load_deputy_aliases():
        for alias in row["aliases"]:
            key = normalize_span(alias)
            assert key not in seen, f"duplicate alias across records: {alias!r}"
            seen.add(key)
