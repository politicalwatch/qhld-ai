"""Unit tests for the person-catalog assembler's curated inputs — pure, no Mongo.

Also lints the shipped ``deputy_aliases.json``: a hand-edited data file is the one
part of this feature no code change can protect, so its invariants are asserted here.
"""

import pytest

from qhld_ai.application.persons_catalog import (
    _gender_from_role,
    alias_index,
    deputy_aliases_by_id,
    gazetteer_surfaces,
    load_curated,
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


# --- bootstrapped non-deputy speakers --------------------------------------
# The source groups by (speaker, role), so a promoted or reworded office yields the
# same person twice. Two identical-keyed entries tie at 100 and the ambiguity guard
# drops the span, i.e. duplication makes a person UNMENTIONABLE.

PROMOTED = [
    {"speaker": "Cuerpo Caballero, Carlos",
     "role": "Ministro de Economía, Comercio y Empresa"},
    {"speaker": "Cuerpo Caballero, Carlos",
     "role": "Vicepresidente Primero del Gobierno y Ministro de Economía"},
]


def _index(speakers):
    return load_person_index([], 90, curated=[], nondeputy_speakers=speakers,
                             deputy_aliases=[])


def test_a_person_id_never_appears_twice_in_the_index():
    index = _index(PROMOTED)
    ids = [e.person_id for e in index]
    assert len(ids) == len(set(ids)), f"duplicated person_id: {ids}"


def test_a_promoted_minister_stays_resolvable():
    # The regression this guards: with one entry per role, "señor Cuerpo" tied 2-way
    # and resolved to None.
    index = _index(PROMOTED)
    entry = resolve_person("señor Cuerpo", index, 90)
    assert entry is not None
    assert entry.person_id == "cuerpo-caballero-carlos"
    assert entry.person_type == "minister"


def test_the_more_specific_role_wins_when_roles_disagree():
    # A government office is the informative label, whichever order the rows arrive in.
    witness_first = [
        {"speaker": "Pérez Pérez, Ana", "role": "Directora de la Agencia"},
        {"speaker": "Pérez Pérez, Ana", "role": "Ministra de Sanidad"},
    ]
    assert _index(witness_first)[0].person_type == "minister"
    assert _index(list(reversed(witness_first)))[0].person_type == "minister"


def test_a_single_role_speaker_is_unaffected():
    index = _index([{"speaker": "Pérez Pérez, Ana", "role": "Directora de la Agencia"}])
    assert [(e.person_id, e.person_type) for e in index] == [
        ("perez-perez-ana", "official")]


# --- gender, from the grammatical gender of the office ----------------------
# The corpus carries no gender field for non-deputy speakers, so their office title is
# the only source — and they are precisely the people who collide with a deputy surname.

@pytest.mark.parametrize("role, expected", [
    ("Ministra de Juventud e Infancia", "Mujer"),
    ("Ministro de Economía, Comercio y Empresa", "Hombre"),
    ("Vicepresidenta Tercera y Ministra para la Transición Ecológica", "Mujer"),
    ("Vicepresidente Primero del Gobierno y Ministro de Economía", "Hombre"),
    ("Presidenta del Congreso de los Diputados", "Mujer"),
    ("Presidente del Gobierno", "Hombre"),
    ("Directora de la Agencia", "Mujer"),
    ("Alcaldesa de Madrid", "Mujer"),
    # the corpus really does distinguish these two, so they carry gender as well
    ("Diputada", "Mujer"),
    ("Diputado", "Hombre"),
    # unknown or genderless in form => no signal at all
    ("Compareciente", None),
    ("", None),
    (None, None),
])
def test_gender_from_role(role, expected):
    assert _gender_from_role(role) == expected


def test_a_masculine_title_does_not_match_inside_its_feminine_form():
    # "alcalde" is a prefix of "alcaldesa"; the word boundary is what keeps them apart.
    assert _gender_from_role("Alcaldesa de Valencia") == "Mujer"
    assert _gender_from_role("Directora General") == "Mujer"


def test_a_trailing_office_does_not_override_the_leading_one():
    # Matched at the start, so the first title names the holder.
    assert _gender_from_role(
        "Vicepresidenta Primera del Gobierno y Ministro de Hacienda") == "Mujer"


def test_bootstrapped_speaker_carries_the_gender_of_its_office():
    index = _index(PROMOTED)
    assert index[0].gender == "Hombre"


def test_gender_is_taken_from_whichever_role_states_it():
    rows = [
        {"speaker": "Pérez Pérez, Ana", "role": "Compareciente"},
        {"speaker": "Pérez Pérez, Ana", "role": "Ministra de Sanidad"},
    ]
    assert _index(rows)[0].gender == "Mujer"
    assert _index(list(reversed(rows)))[0].gender == "Mujer"


def test_contradicting_roles_drop_the_gender_rather_than_guess():
    # Disagreement means the data is wrong; unknown merely disables the gender filter,
    # whereas picking a side would resolve spans to the wrong person.
    rows = [
        {"speaker": "Pérez Pérez, Ana", "role": "Ministra de Sanidad"},
        {"speaker": "Pérez Pérez, Ana", "role": "Ministro de Sanidad"},
    ]
    assert _index(rows)[0].gender is None


def test_curated_records_thread_their_gender():
    curated = [{"person_id": "x", "person_type": "former_pm", "name": "Aznar López, José",
                "aliases": ["Aznar"], "gender": "Hombre"}]
    index = load_person_index([], 90, curated=curated, nondeputy_speakers=[],
                              deputy_aliases=[])
    assert index[0].gender == "Hombre"


def test_shipped_curated_catalog_declares_a_gender_for_everyone():
    # Hand-edited file: gender is what lets a courtesy form rule these people out, and a
    # record added without it silently loses that. Only code can enforce it.
    for row in load_curated():
        assert row.get("gender") in ("Hombre", "Mujer"), row["person_id"]
