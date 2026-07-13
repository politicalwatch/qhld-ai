"""Unit tests for stenographer-annotation handling — no I/O, no spaCy.

Annotation texts are real shapes from the Diario de Sesiones corpus. NER over
interruption quotes is faked with a capitalized-token extractor, mirroring how
``resolve_interruptions`` receives it injected.
"""

import re

import pytest

from qhld_ai.domain.annotations import (
    Utterance,
    extract_annotations,
    parse_utterances,
    resolve_interruptions,
    strip_annotations,
)
from qhld_ai.domain.mentions import build_deputy_index

pytestmark = pytest.mark.unit


class FakeDeputy:
    """Duck-types the bits of tipi_data ``Deputy`` the index needs."""

    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_fullname(self):
        surname, given = (p.strip() for p in self.name.split(","))
        return f"{given} {surname}"


TELLADO = FakeDeputy("d1", "Tellado Filgueira, Miguel")
FEIJOO = FakeDeputy("d2", "Núñez Feijóo, Alberto")
ABALOS = FakeDeputy("d3", "Ábalos Meco, José Luis")
CERDAN = FakeDeputy("d4", "Cerdán León, Santos")
PEDRO = FakeDeputy("d5", "Sánchez Pérez-Castejón, Pedro")
MONTERO = FakeDeputy("d6", "Montero Cuadrado, María Jesús")
BRAVO = FakeDeputy("d7", "Bravo Baena, Juan")
HERNANDO = FakeDeputy("d8", "Hernando Fraile, Rafael")

INDEX = build_deputy_index(
    [TELLADO, FEIJOO, ABALOS, CERDAN, PEDRO, MONTERO, BRAVO, HERNANDO])
THRESHOLD = 90


def fake_quote_spans(text):
    """Capitalized tokens stand in for spaCy PER spans."""
    return re.findall(r"[A-ZÁÉÍÓÚÜÑ][\wáéíóúüñ’'-]*", text)


def resolve(utterances, **kwargs):
    return resolve_interruptions(
        utterances, INDEX, THRESHOLD, fake_quote_spans, **kwargs)


# --- stripping ---------------------------------------------------------------

def test_strip_removes_annotations_and_normalizes_spacing():
    text = ("Eso es mentira. (El señor Núñez Feijóo: ¡Qué disparate!) "
            "Y lo saben. (Aplausos)")
    assert strip_annotations(text) == "Eso es mentira. Y lo saben."


def test_strip_handles_empty_and_annotation_free_text():
    assert strip_annotations("") == ""
    assert strip_annotations(None) == ""
    assert strip_annotations("Sin acotaciones.") == "Sin acotaciones."


def test_extract_returns_inner_texts_in_order():
    text = "Uno (Risas) dos (Rumores.―Aplausos) tres"
    assert extract_annotations(text) == ["Risas", "Rumores.―Aplausos"]


# --- parsing: named verbal ----------------------------------------------------

def test_named_interjection_with_quote():
    utts = parse_utterances("El señor Núñez Feijóo: ¡Qué disparate!")
    assert utts == [Utterance(speaker="Núñez Feijóo", label="Núñez Feijóo",
                              quote="¡Qué disparate!")]


def test_named_interjection_after_stage_direction_segment():
    utts = parse_utterances(
        "Rumores.―El señor Tellado Filgueira: Ábalos. Cerdán en la cárcel…")
    assert len(utts) == 1
    assert utts[0].speaker == "Tellado Filgueira"
    assert utts[0].quote == "Ábalos. Cerdán en la cárcel…"


def test_ascii_hyphen_separates_only_after_punctuation():
    # ".-" chains segments; the hyphen in "Pérez-Castejón" must not split.
    utts = parse_utterances(
        "Aplausos.-El señor Sánchez Pérez-Castejón: Gracias")
    assert utts == [Utterance(speaker="Sánchez Pérez-Castejón",
                              label="Sánchez Pérez-Castejón", quote="Gracias")]


def test_lowercase_determiner_after_chained_segment_still_named():
    utts = parse_utterances("aplausos.―el señor Tellado Filgueira: ¡El Estado!")
    assert utts == [Utterance(speaker="Tellado Filgueira",
                              label="Tellado Filgueira", quote="¡El Estado!")]


def test_action_before_colon_still_yields_quote():
    utts = parse_utterances(
        "La señora Sánchez Torregrosa dándose palmadas en la mejilla: "
        "Vergüenza. ¡Mucha cara!")
    assert utts[0].speaker == "Sánchez Torregrosa"
    assert utts[0].quote == "Vergüenza. ¡Mucha cara!"


# --- parsing: named non-verbal (reactions) -------------------------------------

def test_inaudible_words_recorded_as_reaction():
    utts = parse_utterances(
        "Rumores.-El señor Tellado Filgueira pronuncia palabras que no se perciben")
    assert utts == [Utterance(
        speaker="Tellado Filgueira", label="Tellado Filgueira",
        reaction="pronuncia palabras que no se perciben")]


def test_gesture_recorded_as_reaction():
    utts = parse_utterances(
        "La señora Montesinos de Miguel hace signos negativos")
    assert utts[0].speaker == "Montesinos de Miguel"
    assert utts[0].reaction == "hace signos negativos"


def test_reaction_first_wording():
    utts = parse_utterances("Risas del señor Tellado Filgueira")
    assert utts == [Utterance(speaker="Tellado Filgueira",
                              label="Tellado Filgueira",
                              reaction="Risas", drop_unresolved=True)]


def test_reaction_first_with_office_wording_keeps_full_tail():
    utts = parse_utterances(
        "risas de la señora vicepresidenta primera y ministra de Hacienda, "
        "Montero Cuadrado")
    assert len(utts) == 1
    assert utts[0].reaction == "risas"
    assert "Montero Cuadrado" in utts[0].speaker


def test_plural_determiner_chains_two_interrupters():
    utts = parse_utterances(
        "Los señores Bravo Baena y Hernando Fraile pronuncian palabras "
        "que no se perciben")
    assert [u.speaker for u in utts] == ["Bravo Baena", "Hernando Fraile"]
    assert all(u.reaction for u in utts)


# --- parsing: collective / skipped ---------------------------------------------

def test_anonymous_interjection_keeps_transcript_label():
    utts = parse_utterances("Un señor diputado: ¡Sí, hombre!")
    assert utts == [Utterance(speaker="Un señor diputado",
                              label="Un señor diputado", quote="¡Sí, hombre!")]


def test_collective_label_trims_action_tail():
    utts = parse_utterances(
        "Una señora diputada canturrea, con la entonación utilizada en el "
        "sorteo de la Lotería de Navidad: ¡15 000 eeeuros!")
    assert len(utts) == 1
    assert utts[0].label == "Una señora diputada"
    assert utts[0].quote == "¡15 000 eeeuros!"


def test_group_chant_label_keeps_full_noun_phrase():
    utts = parse_utterances(
        "Las señoras y los señores diputados del Grupo Parlamentario Popular "
        "en el Congreso: ¡Dimisión! ¡Dimisión!")
    assert utts[0].label == "Las señoras y los señores diputados"
    assert utts[0].quote == "¡Dimisión! ¡Dimisión!"


def test_office_wording_label_falls_back_to_comma_piece():
    # "señor <office>" beats the noun whitelist; the label must not degenerate
    # to a bare "El señor" stub.
    utts = parse_utterances(
        "El señor ministro de la Presidencia, Justicia y Relaciones con las "
        "Cortes, Bolaños García: Tranquila, tranquila")
    assert utts[0].label == "El señor ministro de la Presidencia"


def test_non_person_colon_is_not_an_interjection():
    assert parse_utterances(
        "Desde el teléfono móvil se escucha: «El número marcado no se "
        "encuentra disponible en este momento»") == []


def test_pure_stage_directions_yield_nothing():
    for annotation in (
            "Risas", "Rumores.―Aplausos", "Prolongados aplausos",
            "muestra un recorte de prensa",
            "Aplausos de las señoras y los señores diputados del Grupo "
            "Parlamentario Socialista, puestos en pie",
            "El señor vicepresidente, Rodríguez Gómez de Celis, ocupa la "
            "Presidencia"):
        assert parse_utterances(annotation) == [], annotation


# --- resolution -----------------------------------------------------------------

def test_interruptions_group_by_person_with_quotes_reactions_and_mentions():
    utts = (parse_utterances("El señor Tellado Filgueira: ¡Mentira!")
            + parse_utterances(
                "Rumores.―El señor Tellado Filgueira: Ábalos. Cerdán en la cárcel…")
            + parse_utterances(
                "Rumores.-El señor Tellado Filgueira pronuncia palabras "
                "que no se perciben"))
    interruptions = resolve(utts)
    assert len(interruptions) == 1
    i = interruptions[0]
    assert i.person_id == "d1"
    assert i.person_type == "deputy"
    assert i.name == "Tellado Filgueira, Miguel"
    assert i.count == 3
    assert i.quotes == ["¡Mentira!", "Ábalos. Cerdán en la cárcel…"]
    assert i.reactions == ["pronuncia palabras que no se perciben"]
    assert {m.name for m in i.mentions} == {
        "Ábalos Meco, José Luis", "Cerdán León, Santos"}


def test_anonymous_interruption_has_null_person_and_resolved_quote_mentions():
    utts = parse_utterances("Un señor diputado: ¿Dónde estaba Ábalos?")
    interruptions = resolve(utts)
    assert len(interruptions) == 1
    i = interruptions[0]
    assert i.person_id is None
    assert i.name == "Un señor diputado"
    assert {m.person_id for m in i.mentions} == {"d3"}


def test_office_wording_resolves_via_comma_piece():
    utts = parse_utterances(
        "risas de la señora vicepresidenta primera y ministra de Hacienda, "
        "Montero Cuadrado")
    interruptions = resolve(utts)
    assert [i.person_id for i in interruptions] == ["d6"]
    assert interruptions[0].reactions == ["risas"]


def test_unresolved_reaction_first_tail_is_dropped():
    # Reaction-of wording whose tail is no catalog person: not an interrupter.
    utts = [Utterance(speaker="una persona del público", label="una persona",
                      reaction="Risas", drop_unresolved=True)]
    assert resolve(utts) == []


def test_speaker_own_closing_applause_is_not_an_interruption():
    utts = parse_utterances(
        "aplausos del señor presidente del Gobierno, Sánchez Pérez-Castejón, "
        "dirigidos a su grupo parlamentario")
    assert resolve(utts, speaker_name="Sánchez Pérez-Castejón, Pedro") == []
    # Without the speaker context the same annotation does resolve to him.
    assert [i.person_id for i in resolve(utts)] == ["d5"]


def test_sorted_by_count_then_name():
    utts = (parse_utterances("El señor Núñez Feijóo: ¡Qué disparate!")
            + parse_utterances("El señor Tellado Filgueira: ¡Mentira!")
            + parse_utterances("El señor Tellado Filgueira: ¡El Estado!"))
    names = [i.name for i in resolve(utts)]
    assert names == ["Tellado Filgueira, Miguel", "Núñez Feijóo, Alberto"]
