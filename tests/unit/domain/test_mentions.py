"""Unit tests for pure mention resolution — no I/O, no spaCy.

Feeds raw NER-style spans + a fake deputy catalog through the resolver and asserts
the canonicalization, dedupe/count, honorific stripping, ambiguity guard, and the
cues that let a shared surname resolve after all — the gendered courtesy form, the
office a role apposition names, and same-speech coreference.
"""

from dataclasses import replace

import pytest

from qhld_ai.domain.mentions import (
    COMMON_WORD_SURNAMES,
    build_deputy_index,
    build_office_surfaces,
    build_person_index,
    build_surname_gazetteer,
    context_excluded_surnames,
    make_person_entry,
    match_person,
    normalize_span,
    office_families,
    resolve_mentions,
    resolve_person,
    role_appositions,
    role_cues,
    span_gender,
)

pytestmark = pytest.mark.unit


class FakeDeputy:
    """Duck-types the bits of tipi_data ``Deputy`` the index needs."""

    def __init__(self, id, name, gender=None):
        self.id = id
        self.name = name
        self.gender = gender

    def get_fullname(self):
        surname, given = (p.strip() for p in self.name.split(","))
        return f"{given} {surname}"


PEDRO = FakeDeputy("d1", "Sánchez Pérez-Castejón, Pedro")
MONTERO = FakeDeputy("d2", "Montero Cuadrado, María Jesús")
GARCIA_A = FakeDeputy("d3", "García López, Ana")
GARCIA_J = FakeDeputy("d4", "García Ruiz, Juan")

INDEX = build_deputy_index([PEDRO, MONTERO, GARCIA_A, GARCIA_J])


def _names(mentions):
    return {m.name for m in mentions}


# --- normalization ---------------------------------------------------------

def test_normalize_strips_honorifics_and_articles():
    assert normalize_span("el señor Sánchez") == "sánchez"
    assert normalize_span("doña María Jesús Montero") == "maría jesús montero"
    # "al" is the contracted article (a + el), not part of the name.
    assert normalize_span("Al señor López") == "lópez"


def test_normalize_strips_role_words_too():
    # The model does fold a role word into a span, and the extra token dropped
    # token_set_ratio to 71 against "Torres Pérez, Ángel Víctor" — below any workable
    # threshold, so the mention was silently lost. Nothing is lost by stripping it: the
    # office is read off the surrounding text (see role_cues), not off the span.
    assert normalize_span("ministro Torres") == "torres"
    assert normalize_span("el señor ministro Torres") == "torres"
    assert normalize_span("la vicepresidenta Montero") == "montero"


def test_normalize_drops_pure_honorific_and_too_short():
    assert normalize_span("Su Señoría") == ""
    assert normalize_span("el") == ""
    assert normalize_span(",.") == ""
    # a role word with no name behind it names nobody
    assert normalize_span("Señora ministra") == ""
    assert normalize_span("Ministro") == ""


# --- resolution ------------------------------------------------------------

def test_surname_only_resolves_to_canonical_name():
    mentions = resolve_mentions(["Montero"], INDEX, 90)
    assert _names(mentions) == {"Montero Cuadrado, María Jesús"}
    assert mentions[0].person_id == "d2"
    assert mentions[0].person_type == "deputy"


def test_full_name_and_honorific_forms_merge_into_one_mention():
    spans = ["el señor Sánchez", "Pedro Sánchez", "Sánchez"]
    mentions = resolve_mentions(spans, INDEX, 90)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.name == "Sánchez Pérez-Castejón, Pedro"
    assert m.count == 3
    assert m.surface_forms == ["Pedro Sánchez", "Sánchez", "el señor Sánchez"]


def test_ambiguous_bare_surname_is_dropped():
    # "García" matches two deputies at the same top score → precision-safe drop.
    assert resolve_mentions(["García"], INDEX, 90) == []


def test_ambiguity_resolved_when_given_name_disambiguates():
    mentions = resolve_mentions(["Ana García"], INDEX, 90)
    assert _names(mentions) == {"García López, Ana"}


def test_unknown_person_below_threshold_is_dropped():
    assert resolve_mentions(["Winston Churchill"], INDEX, 90) == []


def test_result_sorted_by_count_desc():
    spans = ["Montero", "Sánchez", "Sánchez"]
    mentions = resolve_mentions(spans, INDEX, 90)
    assert [m.name for m in mentions] == [
        "Sánchez Pérez-Castejón, Pedro", "Montero Cuadrado, María Jesús"]


def test_empty_and_honorific_only_spans_yield_nothing():
    assert resolve_mentions(["Su Señoría", "", "  "], INDEX, 90) == []


# --- excluded (non-deputy) surnames ----------------------------------------

AZNAR_DEP = FakeDeputy("e1", "Aznar Teruel, Evarist")
GAMARRA = FakeDeputy("e2", "Gamarra Ruiz-Clavijo, Concepción")
ALBARES = FakeDeputy("e3", "Albares Bueno, José Manuel")
FEIJOO = FakeDeputy("e4", "Núñez Feijóo, Alberto")
INDEX_EXCL = build_deputy_index([AZNAR_DEP, GAMARRA, ALBARES, FEIJOO])


def test_referent_homonym_excluded_on_first_surname():
    # "Aznar" fuzzy-matches the deputy Aznar Teruel but denotes the ex-PM; the deputy's
    # OWN first surname is flagged, so it is dropped.
    assert resolve_mentions(["Aznar"], INDEX_EXCL, 90, frozenset({"aznar"})) == []


def test_mismatch_secondary_surname_excluded():
    # "Clavijo" (the Canarias president) resolves to Gamarra via her SECOND surname —
    # dropped because her first surname (Gamarra) is not in the span.
    assert resolve_mentions(["Clavijo"], INDEX_EXCL, 90, frozenset({"clavijo"})) == []


def test_full_name_containing_flagged_token_survives():
    # A genuine full-name mention that merely contains the flagged token is kept.
    mentions = resolve_mentions(
        ["Gamarra Ruiz-Clavijo"], INDEX_EXCL, 90, frozenset({"clavijo"}))
    assert _names(mentions) == {"Gamarra Ruiz-Clavijo, Concepción"}


def test_common_word_surname_excluded():
    assert resolve_mentions(["Bueno"], INDEX_EXCL, 90, COMMON_WORD_SURNAMES) == []


def test_deputy_known_by_second_surname_survives_denylist():
    # Regression guard: the exclusion set must not suppress a deputy universally named
    # by their SECOND surname (Feijóo of Núñez Feijóo) — "feijóo" is not flagged.
    mentions = resolve_mentions(
        ["Feijóo"], INDEX_EXCL, 90, frozenset({"aznar", "suárez", "clavijo"}))
    assert _names(mentions) == {"Núñez Feijóo, Alberto"}


def test_no_exclusion_by_default():
    assert _names(resolve_mentions(["Aznar"], INDEX_EXCL, 90)) == {
        "Aznar Teruel, Evarist"}


# --- context-cue exclusion (speech-scoped) ---------------------------------

def test_context_cue_magistrate():
    text = "Soldados del régimen como Macías, actual magistrado del Tribunal Constitucional."
    assert "macías" in context_excluded_surnames(text)


def test_context_cue_dictatorship_flags_franco():
    text = "La ley debe ser la orgánica de Franco, la manera fina de llamarla en la Dictadura."
    assert "franco" in context_excluded_surnames(text)


def test_no_context_cue_yields_empty():
    assert context_excluded_surnames("El señor Sánchez habló de vivienda.") == frozenset()


# --- tie-break (recall) ----------------------------------------------------

JUAN_BRAVO = FakeDeputy("t1", "Bravo Baena, Juan")
AITOR_ESTEBAN = FakeDeputy("t2", "Esteban Bravo, Aitor")
GONZALEZ_PONS = FakeDeputy("t3", "González Pons, Esteban")
RAMOS_ESTEBAN = FakeDeputy("t4", "Ramos Esteban, César Joaquín")
PEDRO_SANCHEZ = FakeDeputy("t5", "Sánchez Pérez-Castejón, Pedro")
CESAR_SANCHEZ = FakeDeputy("t6", "Sánchez Pérez, César")
MUNOZ_IGLESIA = FakeDeputy("t7", "Muñoz de la Iglesia, Ester")
MUNOZ_ABRINES = FakeDeputy("t8", "Muñoz Abrines, Pedro")
INDEX_TIE = build_deputy_index([
    JUAN_BRAVO, AITOR_ESTEBAN, GONZALEZ_PONS, RAMOS_ESTEBAN,
    PEDRO_SANCHEZ, CESAR_SANCHEZ, MUNOZ_IGLESIA, MUNOZ_ABRINES])


def test_tie_broken_toward_first_surname_holder():
    # "Bravo" is Juan Bravo's FIRST surname but Aitor Esteban Bravo's SECOND — resolves
    # to the former instead of dropping as an ambiguous tie.
    assert _names(resolve_mentions(["el señor Bravo"], INDEX_TIE, 90)) == {
        "Bravo Baena, Juan"}


def test_tie_first_surname_beats_given_name_and_second_surname():
    # "Esteban" is Aitor Esteban's FIRST surname, González Pons's GIVEN name and Ramos
    # Esteban's SECOND surname — resolves to the first-surname holder.
    assert _names(resolve_mentions(["Esteban"], INDEX_TIE, 90)) == {
        "Esteban Bravo, Aitor"}


def test_tie_broken_by_exact_token_order():
    # Both share first surname "Sánchez" and tie at token_set_ratio 100; the exact
    # surname order picks Pedro over the shorter "Sánchez Pérez".
    assert _names(resolve_mentions(["Sánchez Pérez-Castejón"], INDEX_TIE, 90)) == {
        "Sánchez Pérez-Castejón, Pedro"}


def test_ambiguous_shared_first_surname_still_drops():
    # Two deputies hold "Muñoz" as their first surname → genuinely ambiguous → dropped.
    assert resolve_mentions(["Muñoz"], INDEX_TIE, 90) == []


# --- resolve_person (query side) -------------------------------------------

def test_resolve_person_surname_resolves_to_deputy():
    entry = resolve_person("Montero", INDEX, 90)
    assert entry is not None
    assert (entry.person_id, entry.name) == ("d2", "Montero Cuadrado, María Jesús")


def test_resolve_person_full_name_resolves():
    entry = resolve_person("Pedro Sánchez", INDEX, 90)
    assert entry.person_id == "d1"


def test_resolve_person_ambiguous_surname_is_none():
    # "García" is borne by two deputies at the same score → ambiguous → not resolved.
    assert resolve_person("García", INDEX, 90) is None


def test_resolve_person_unknown_is_none():
    assert resolve_person("Winston Churchill", INDEX, 90) is None


def test_resolve_person_empty_after_normalize_is_none():
    assert resolve_person("Su Señoría", INDEX, 90) is None


def test_match_person_success_carries_score_and_no_candidates():
    match = match_person("Montero", INDEX, 90)
    assert match.entry.person_id == "d2"
    assert match.best_score >= 90
    assert match.candidates == []


def test_match_person_below_threshold_reports_near_miss():
    match = match_person("Monteros Cuadrados", INDEX, 101)  # force a miss
    assert match.entry is None
    assert 0 < match.best_score < 101
    assert "Montero Cuadrado, María Jesús" in match.candidate_names


def test_match_person_ambiguous_reports_tied_names():
    match = match_person("García", INDEX, 90)
    assert match.entry is None
    assert match.best_score >= 90
    assert set(match.candidate_names) == {"García López, Ana", "García Ruiz, Juan"}


def test_match_person_empty_span_has_no_diagnostics():
    match = match_person("Su Señoría", INDEX, 90)
    assert (match.entry, match.best_score, match.candidates) == (None, 0, [])


# --- surname gazetteer -----------------------------------------------------

def test_gazetteer_keeps_distinctive_surnames_and_compound_parts():
    deputies = [
        FakeDeputy("g1", "Vallugera Balañà, Pilar"),
        FakeDeputy("g2", "Grande-Marlaska Gómez, Fernando"),
        FakeDeputy("g3", "García López, Ana"),
        FakeDeputy("g4", "García Ruiz, Juan"),  # 'García' shared → excluded
    ]
    terms = build_surname_gazetteer(deputies)
    assert "Vallugera" in terms
    assert "Grande" in terms and "Marlaska" in terms  # hyphenated compound split
    assert "García" not in terms  # borne by two deputies → not distinctive
    assert all(t == t for t in terms) and terms == sorted(terms)


def test_gazetteer_includes_distinctive_second_surnames():
    # The chamber knows some deputies by the second surname: "Feijóo" must seed
    # the gazetteer even though the first surname ("Núñez") is shared.
    deputies = [
        FakeDeputy("g1", "Núñez Feijóo, Alberto"),
        FakeDeputy("g2", "Núñez González, Noelia"),
        FakeDeputy("g3", "Muñoz de la Iglesia, Marta"),
    ]
    terms = build_surname_gazetteer(deputies)
    assert "Feijóo" in terms
    assert "Núñez" not in terms  # shared first surname stays out
    assert "González" in terms and "Iglesia" in terms
    # connective particles are never name surfaces
    assert "de" not in terms and "la" not in terms and "del" not in terms


def test_gazetteer_keeps_unique_first_surname_shared_as_second_surname():
    # "Montero" is Montero Cuadrado's first surname and another deputy's second:
    # a bare "Montero" still resolves to its first-surname bearer (_break_tie),
    # so the token stays distinctive. A token shared as FIRST surname does not.
    deputies = [
        FakeDeputy("g1", "Montero Cuadrado, María Jesús"),
        FakeDeputy("g2", "Gómez Montero, Luis"),
        FakeDeputy("g3", "García López, Ana"),
        FakeDeputy("g4", "García Ruiz, Juan"),
    ]
    terms = build_surname_gazetteer(deputies)
    assert "Montero" in terms
    assert "García" not in terms


# --- non-deputy people (curated catalog + overrides) -----------------------

# A deputy who shares a surname with a famous non-deputy, plus one who doesn't.
GAMARRA = FakeDeputy("gamarra", "Gamarra Ruiz-Clavijo, Concepción")
AZNAR_DEP2 = FakeDeputy("aznar-teruel", "Aznar Teruel, Evarist")
FEIJOO2 = FakeDeputy("nunez-feijoo-alberto", "Núñez Feijóo, Alberto")

CURATED = [
    make_person_entry("fernando-clavijo", "regional_president", "Clavijo Batlle, Fernando",
                      aliases=["Clavijo", "Fernando Clavijo"], overrides_deputy=True),
    make_person_entry("jose-maria-aznar", "former_pm", "Aznar López, José María",
                      aliases=["Aznar", "José María Aznar"], overrides_deputy=True),
    make_person_entry("isabel-diaz-ayuso", "regional_president", "Díaz Ayuso, Isabel",
                      aliases=["Ayuso", "Díaz Ayuso"]),
    make_person_entry("donald-trump", "foreign_leader", "Trump, Donald", aliases=["Trump"]),
    make_person_entry("felipe-vi", "head_of_state", "Felipe VI",
                      aliases=["Felipe VI", "su majestad"]),
]
PERSON_INDEX = build_person_index([GAMARRA, AZNAR_DEP2, FEIJOO2], CURATED)


def _one(span):
    m = resolve_mentions([span], PERSON_INDEX, 90)
    return (m[0].name, m[0].person_type) if m else None


def test_non_deputy_resolves_to_catalog_person():
    assert _one("Ayuso") == ("Díaz Ayuso, Isabel", "regional_president")
    assert _one("Trump") == ("Trump, Donald", "foreign_leader")


def test_override_wins_over_colliding_deputy_on_bare_surname():
    # "Clavijo" is the Canarias president, not the deputy Gamarra Ruiz-Clavijo (2nd
    # surname); "Aznar" is the ex-PM, not the deputy Aznar Teruel.
    assert _one("Clavijo") == ("Clavijo Batlle, Fernando", "regional_president")
    assert _one("Aznar") == ("Aznar López, José María", "former_pm")


def test_deputy_full_name_beats_override():
    # The deputy's OWN full name outranks the surname-sharing override.
    assert _one("Gamarra Ruiz-Clavijo") == ("Gamarra Ruiz-Clavijo, Concepción", "deputy")


def test_deputy_second_surname_still_wins_when_no_override():
    # Feijóo has no override entry and only the deputy matches → deputy, unchanged.
    assert _one("Feijóo") == ("Núñez Feijóo, Alberto", "deputy")


def test_king_matched_by_explicit_alias_not_bare_common_noun():
    assert _one("su majestad") == ("Felipe VI", "head_of_state")
    assert _one("Felipe VI") == ("Felipe VI", "head_of_state")
    # bare "rey" is a common noun (and collides with the real deputy 'Rey de las
    # Heras'), deliberately not an alias → the King is not matched from it.
    assert _one("rey") is None


def test_non_deputy_never_excluded_by_deputy_denylist():
    # The exclusion set only guards deputy resolutions; a resolved non-deputy is kept
    # even if a homonymous surname would be flagged for deputies.
    mentions = resolve_mentions(["Aznar"], PERSON_INDEX, 90, frozenset({"aznar"}))
    assert _names(mentions) == {"Aznar López, José María"}


def test_resolve_person_resolves_non_deputy():
    entry = resolve_person("Ayuso", PERSON_INDEX, 90)
    assert (entry.person_id, entry.person_type) == ("isabel-diaz-ayuso", "regional_president")


def test_override_second_surname_does_not_hijack_ambiguous_tie():
    # 'Aznar López' (override ex-PM) shares his SECOND surname with a bare 'López', which
    # is ambiguous across several deputies. The override must not fire on a secondary
    # token: 'López' stays ambiguous (dropped), while the ex-PM still resolves from 'Aznar'.
    lopez1 = FakeDeputy("lopez-cano", "López Cano, Ignacio")
    lopez2 = FakeDeputy("lopez-alvarez", "López Álvarez, Patxi")
    aznar = make_person_entry("jose-maria-aznar", "former_pm", "Aznar López, José María",
                              aliases=["Aznar"], overrides_deputy=True)
    index = build_person_index([lopez1, lopez2], [aznar])
    assert resolve_mentions(["López"], index, 90) == []
    assert resolve_mentions(["Aznar"], index, 90)[0].person_type == "former_pm"


def test_deputy_wins_tie_over_nonoverride_nondeputy():
    # A bootstrapped minister shares a surname with a deputy ("Rego"): the deputy is the
    # primary referent and wins — a non-override non-deputy never blocks a deputy.
    deputy = FakeDeputy("rego-candamil-nestor", "Rego Candamil, Néstor")
    minister = make_person_entry("sira-rego", "minister", "Rego, Sira Abed")
    index = build_person_index([deputy], [minister])
    m = resolve_mentions(["Rego"], index, 90)
    assert (m[0].name, m[0].person_type) == ("Rego Candamil, Néstor", "deputy")


# --- curated deputy aliases (public names) ---------------------------------
# The public name of a deputy whose official catalog entry shares no token with it,
# so no threshold can reach it: it has to be curated.

TESLEM = FakeDeputy("andala-ubbi-teslem", "Andala Ubbi, Teslem")
TESH = {"andala-ubbi-teslem": ["Tesh Sidi"]}


def test_alias_resolves_a_mention_to_the_deputy():
    index = build_deputy_index([TESLEM], aliases=TESH)
    mention = resolve_mentions(["Tesh Sidi"], index, 90)[0]
    assert (mention.name, mention.person_id, mention.person_type) == (
        "Andala Ubbi, Teslem", "andala-ubbi-teslem", "deputy")


def test_alias_matches_under_the_same_normalization_as_any_span():
    index = build_deputy_index([TESLEM], aliases=TESH)
    assert resolve_person("la señora Tesh Sidi", index, 90).person_id == "andala-ubbi-teslem"
    assert resolve_person("TESH SIDI", index, 90).person_id == "andala-ubbi-teslem"


def test_alias_and_official_surfaces_collapse_into_one_mention():
    index = build_deputy_index([TESLEM], aliases=TESH)
    mentions = resolve_mentions(["Tesh Sidi", "Andala Ubbi"], index, 90)
    assert len(mentions) == 1
    assert mentions[0].count == 2
    assert mentions[0].surface_forms == ["Andala Ubbi", "Tesh Sidi"]


def test_index_is_unchanged_when_no_aliases_are_given():
    # The engine builds the index without aliases; that path must stay as it was.
    assert build_deputy_index([TESLEM]) == build_deputy_index([TESLEM], aliases={})
    assert resolve_mentions(["Tesh Sidi"], build_deputy_index([TESLEM]), 90) == []


def test_alias_for_an_unknown_deputy_id_is_a_no_op():
    index = build_deputy_index([TESLEM], aliases={"someone-else": ["Whoever"]})
    assert index == build_deputy_index([TESLEM])


def test_whole_public_name_covers_its_parts_without_swallowing_a_homonym():
    # token_set_ratio scores a subset at 100, so the curated full name already
    # resolves its parts — which is why aliases are curated whole: a bare "Sidi" key
    # would also match the Moroccan town "Sidi Ifni", a real corpus surface.
    index = build_deputy_index([TESLEM], aliases=TESH)
    assert resolve_person("Tesh", index, 90).person_id == "andala-ubbi-teslem"
    assert resolve_person("Sidi", index, 90).person_id == "andala-ubbi-teslem"
    assert resolve_person("Sidi Ifni", index, 90) is None


def test_build_person_index_threads_aliases():
    minister = make_person_entry("sira-rego", "minister", "Rego, Sira Abed")
    index = build_person_index([TESLEM], [minister], aliases=TESH)
    assert resolve_person("Tesh Sidi", index, 90).person_id == "andala-ubbi-teslem"


def test_gazetteer_takes_curated_public_names_verbatim():
    # A nickname has no surname group to count as distinctive, so curated surfaces go
    # in as given — the alias keys in the index have nothing to resolve unless the NER
    # spans them first.
    terms = build_surname_gazetteer([TESLEM], extra=["Tesh"])
    assert "Tesh" in terms
    assert terms == sorted(terms)
    # and it changes nothing when nothing is curated
    assert build_surname_gazetteer([TESLEM]) == build_surname_gazetteer(
        [TESLEM], extra=[])


def test_gazetteer_curated_surface_does_not_displace_a_surname():
    terms = build_surname_gazetteer([TESLEM], extra=["Tesh"])
    assert "Andala" in terms and "Ubbi" in terms


# --- gender from the courtesy form -----------------------------------------
# Spanish courtesy forms agree in gender with the person named, so "la señora Muñoz"
# rules out every male Muñoz. Evidence is pooled per speech: one honorific settles the
# bare occurrences too, which is the whole point (a speech says "señora Muñoz" once and
# "Muñoz" thirty times).

MUNOZ_F = FakeDeputy("m1", "Muñoz de la Iglesia, Ester", "Mujer")
MUNOZ_M = FakeDeputy("m2", "Muñoz Abrines, Pedro", "Hombre")
MUNOZ_INDEX = build_deputy_index([MUNOZ_F, MUNOZ_M])


@pytest.mark.parametrize("span, expected", [
    ("la señora Muñoz", "Mujer"),
    ("Señora Muñoz", "Mujer"),
    ("doña Ester", "Mujer"),
    ("Sra. Muñoz", "Mujer"),
    ("el señor Muñoz", "Hombre"),
    ("Señor Muñoz", "Hombre"),
    ("don Pedro", "Hombre"),
    ("Sr. Muñoz", "Hombre"),
    # no courtesy form at all
    ("Muñoz", None),
    ("Ester Muñoz de la Iglesia", None),
    # self-contradictory: assume nothing
    ("el señor y la señora Muñoz", None),
])
def test_span_gender(span, expected):
    assert span_gender(span) == expected


def test_feminine_courtesy_form_picks_the_female_holder():
    mentions = resolve_mentions(["la señora Muñoz"], MUNOZ_INDEX, 90)
    assert _names(mentions) == {"Muñoz de la Iglesia, Ester"}


def test_masculine_courtesy_form_picks_the_male_holder():
    mentions = resolve_mentions(["el señor Muñoz"], MUNOZ_INDEX, 90)
    assert _names(mentions) == {"Muñoz Abrines, Pedro"}


def test_bare_shared_surname_with_no_courtesy_form_still_drops():
    # The guard's default: nothing in the text says which Muñoz, so neither is claimed.
    assert resolve_mentions(["Muñoz"], MUNOZ_INDEX, 90) == []


def test_one_courtesy_form_settles_the_bare_occurrences_in_the_same_speech():
    # The reason the cue is pooled per speech rather than read per span: the honorific
    # appears once, the bare surname many times, and they are the same person.
    mentions = resolve_mentions(
        ["la señora Muñoz", "Muñoz", "Muñoz", "Muñoz"], MUNOZ_INDEX, 90)
    assert _names(mentions) == {"Muñoz de la Iglesia, Ester"}
    assert mentions[0].count == 4
    assert mentions[0].surface_forms == ["Muñoz", "la señora Muñoz"]


def test_contradicting_courtesy_forms_in_one_speech_drop_the_surname():
    # Both Muñoz are named, so the surname identifies neither.
    assert resolve_mentions(
        ["la señora Muñoz", "el señor Muñoz", "Muñoz"], MUNOZ_INDEX, 90) == []


def test_unknown_catalog_gender_never_rules_anybody_out():
    # Most bootstrapped speakers have no gender; treating that as a mismatch would break
    # resolutions that work today.
    index = build_deputy_index([FakeDeputy("u1", "Muñoz de la Iglesia, Ester"),
                                FakeDeputy("u2", "Muñoz Abrines, Pedro")])
    assert resolve_mentions(["la señora Muñoz"], index, 90) == []


def test_gender_does_not_promote_a_bare_given_name():
    # The surname gate. "Alberto" ties every Alberto and names none of them; a courtesy
    # form says nothing about WHICH. Measured on the gold set as a real false positive.
    albertos = build_deputy_index([
        FakeDeputy("a1", "Fabra Part, Alberto", "Hombre"),
        FakeDeputy("a2", "Catalán Higueras, Alberto", "Hombre"),
    ])
    assert resolve_mentions(["el señor Alberto", "Alberto"], albertos, 90) == []


def test_gender_still_drops_a_surname_shared_by_two_of_the_same_gender():
    two_women = build_deputy_index([
        FakeDeputy("w1", "Vázquez Blanco, Ana Belén", "Mujer"),
        FakeDeputy("w2", "Vázquez Jiménez, María del Mar", "Mujer"),
    ])
    assert resolve_mentions(["la señora Vázquez"], two_women, 90) == []


def test_courtesy_form_outranks_the_deputy_preference():
    # The M65 regression: "la señora Rego, ministra de Juventud" is the minister Sira
    # Rego, but the deputy preference handed the tie to the MALE deputy Néstor Rego.
    deputy = FakeDeputy("rego-candamil-nestor", "Rego Candamil, Néstor", "Hombre")
    minister = make_person_entry(
        "sira-rego", "minister", "Rego, Sira Abed", gender="Mujer")
    index = build_person_index([deputy], [minister])

    feminine = resolve_mentions(["la señora Rego"], index, 90)
    assert _names(feminine) == {"Rego, Sira Abed"}
    # and the masculine form still reaches the deputy
    masculine = resolve_mentions(["el señor Rego"], index, 90)
    assert _names(masculine) == {"Rego Candamil, Néstor"}
    # with no courtesy form the deputy preference is untouched
    assert _names(resolve_mentions(["Rego"], index, 90)) == {"Rego Candamil, Néstor"}


def test_gender_gate_can_be_switched_off():
    assert resolve_mentions(
        ["la señora Muñoz"], MUNOZ_INDEX, 90, gender_gate=False) == []


def test_gender_never_overrides_an_unambiguous_full_name():
    # A resolution that never needed the guard must not be second-guessed by a stray cue.
    mentions = resolve_mentions(["doña Pedro Sánchez Pérez-Castejón"], INDEX, 90)
    assert _names(mentions) == {"Sánchez Pérez-Castejón, Pedro"}


def test_an_all_conflicting_tie_is_left_to_the_ambiguity_guard():
    # If the cue contradicts every candidate the data is wrong somewhere; keep dropping
    # rather than inventing a winner.
    men = build_deputy_index([FakeDeputy("g1", "García López, Juan", "Hombre"),
                              FakeDeputy("g2", "García Ruiz, Luis", "Hombre")])
    assert resolve_mentions(["la señora García"], men, 90) == []


# --- the query path is unaffected ------------------------------------------
# resolve_person/match_person take no cue: a searcher types a bare name, so gender must
# not change what a query resolves to.

def test_query_resolution_is_unchanged_by_gender():
    assert resolve_person("Muñoz", MUNOZ_INDEX, 90) is None
    assert match_person("Muñoz", MUNOZ_INDEX, 90).candidate_names == [
        "Muñoz de la Iglesia, Ester", "Muñoz Abrines, Pedro"]


def test_a_courtesy_form_in_a_query_is_stripped_not_used():
    # Honorifics are normalized away on the query side, exactly as before.
    assert resolve_person("la señora Muñoz", MUNOZ_INDEX, 90) is None


# --- same-speech coreference -----------------------------------------------
# A surname the guard drops is decidable when the speech names exactly one of the tied
# people elsewhere in full: within one speech, one surname means one person.

def _counts(mentions):
    return {m.name: m.count for m in mentions}


def test_a_tied_surname_joins_the_person_the_speech_names_in_full():
    mentions = resolve_mentions(
        ["Ester Muñoz de la Iglesia", "señora Muñoz", "Muñoz"], MUNOZ_INDEX, 90)
    assert _counts(mentions) == {"Muñoz de la Iglesia, Ester": 3}


def test_nothing_attaches_when_the_speech_names_both_of_the_tied_people():
    # The hard case: two holders of the surname are each named in full, so the bare
    # occurrences are genuinely undecidable and must stay dropped.
    mentions = resolve_mentions(
        ["Ester Muñoz de la Iglesia", "Pedro Muñoz Abrines", "Muñoz"],
        MUNOZ_INDEX, 90, gender_gate=False)
    assert _counts(mentions) == {
        "Muñoz de la Iglesia, Ester": 1, "Muñoz Abrines, Pedro": 1}


def test_nothing_attaches_when_no_tied_person_is_named_elsewhere():
    assert resolve_mentions(["Muñoz", "Muñoz"], MUNOZ_INDEX, 90) == []


def test_coreference_does_not_rescue_a_bare_given_name():
    # "Pedro" ties two people and names neither of them; the surname gate keeps it
    # dropped even though one of them is named in full elsewhere.
    index = build_deputy_index([MUNOZ_M, FakeDeputy("p2", "Casares Hontañón, Pedro")])
    mentions = resolve_mentions(
        ["Pedro Muñoz Abrines", "Pedro"], index, 90, gender_gate=False)
    assert _counts(mentions) == {"Muñoz Abrines, Pedro": 1}


def test_coreference_does_not_rescue_a_span_that_missed_the_threshold():
    # A near miss carries candidates too, but it is a failed match, not a tie.
    mentions = resolve_mentions(
        ["Ester Muñoz de la Iglesia", "Muñoces"], MUNOZ_INDEX, 95)
    assert _counts(mentions) == {"Muñoz de la Iglesia, Ester": 1}


def test_coreference_still_honours_the_excluded_surnames():
    mentions = resolve_mentions(
        ["Ester Muñoz de la Iglesia", "Muñoz"], MUNOZ_INDEX, 90,
        frozenset({"muñoz"}))
    assert mentions == []


def test_coreference_off_leaves_the_tied_occurrences_dropped():
    mentions = resolve_mentions(
        ["Ester Muñoz de la Iglesia", "Muñoz"], MUNOZ_INDEX, 90, coreference=False)
    assert _counts(mentions) == {"Muñoz de la Iglesia, Ester": 1}


def test_coreference_does_not_chain_through_its_own_attachments():
    # Only first-pass resolutions are evidence, so the outcome cannot depend on the
    # order the spans arrive in.
    spans = ["Ester Muñoz de la Iglesia", "Muñoz", "Pedro Muñoz Abrines"]
    assert _counts(resolve_mentions(spans, MUNOZ_INDEX, 90, gender_gate=False)) == \
        _counts(resolve_mentions(list(reversed(spans)), MUNOZ_INDEX, 90,
                                 gender_gate=False))


# --- role apposition -------------------------------------------------------
# The office a speech names somebody by ("el presidente Sánchez") decides a surname the
# guard drops, including the case gender cannot touch: several men called Sánchez, only
# one of them the prime minister. The cue is read off the TEXT, since the role word is
# outside the span the NER returns.

def _offices(index, name, *families):
    """The office stamp ``attach_offices`` applies at catalog-assembly time."""
    return [replace(entry, offices=families) if entry.name == name else entry
            for entry in index]


SANCHEZ_SERNA = FakeDeputy("s2", "Sánchez Serna, Javier", "Hombre")
SANCHEZ_INDEX = _offices(
    build_deputy_index([FakeDeputy("s1", "Sánchez Pérez-Castejón, Pedro", "Hombre"),
                        SANCHEZ_SERNA]),
    "Sánchez Pérez-Castejón, Pedro", "presidente")


@pytest.mark.parametrize("text, expected", [
    # the tight apposition, with and without the article/courtesy form
    ("Lo dijo el presidente Sánchez.", {"sánchez": "presidente"}),
    ("Lo dijo presidente Sánchez.", {"sánchez": "presidente"}),
    ("Lo dijo el señor ministro Sánchez.", {"sánchez": "ministro"}),
    ("Lo dijo el vicepresidente Sánchez.", {"sánchez": "vicepresidente"}),
    # postposed, which is why the span is not the place to read this
    ("Se lo otorga al señor Sánchez, presidente del Gobierno.",
     {"sánchez": "presidente"}),
    # comma apposition: allowed because the office complement is there
    ("Habló la ministra de Vivienda y Agenda Urbana, la señora Sánchez.",
     {"sánchez": "ministro"}),
    # the vocative to the chair, which is the commonest role word in the corpus and
    # names nobody: the comma keeps it out
    ("Señor presidente, la señora Sánchez ha dicho…", {}),
    ("Muchas gracias, señora presidenta. Sánchez lo sabe.", {}),
    # a role reference that names nobody at all
    ("El presidente del Gobierno compareció ayer.", {}),
    # two offices on one surface: it names two people, so assume nothing
    ("El presidente Sánchez y la ministra Sánchez.", {}),
])
def test_role_cues(text, expected):
    assert role_cues(text, ["Sánchez", "la señora Sánchez"]) == expected


def test_a_capture_that_the_speech_never_spans_is_not_a_cue():
    # The gate that makes the loose tail harmless: "Gracias", "Señorías" and every
    # office complement get captured and then thrown away for naming no span.
    assert role_cues("Gracias, presidente Sánchez.", ["Montero"]) == {}


def test_role_apposition_picks_the_office_holder_out_of_a_tie():
    # Both are men, so gender says nothing and the guard drops the surname today.
    spans = ["Sánchez"]
    assert resolve_mentions(spans, SANCHEZ_INDEX, 90) == []
    mentions = resolve_mentions(spans, SANCHEZ_INDEX, 90,
                                text="Eso lo dijo el presidente Sánchez.")
    assert _names(mentions) == {"Sánchez Pérez-Castejón, Pedro"}


def test_one_role_apposition_settles_the_bare_occurrences_in_the_same_speech():
    # Why the cue is pooled per surface: the title is used once, the bare surname many
    # times, and within one speech they are the same person. This is the M65 class —
    # 18 occurrences hanging off a single "el presidente Sánchez".
    text = ("El presidente Sánchez lo anunció. Y el señor Sánchez insiste, "
            "porque Sánchez sabe que Sánchez no puede.")
    mentions = resolve_mentions(
        ["Sánchez", "el señor Sánchez", "Sánchez", "Sánchez"], SANCHEZ_INDEX, 90,
        text=text)
    assert _counts(mentions) == {"Sánchez Pérez-Castejón, Pedro": 4}


def test_role_apposition_outranks_the_deputy_preference():
    # A minister is not a deputy, so the deputy preference hands the tie to the deputy
    # who merely shares the surname. The office says otherwise.
    deputy = FakeDeputy("t1", "Torres Tejada, María", "Mujer")
    minister = make_person_entry("angel-victor-torres", "minister", "Torres Pérez, Ángel",
                                 gender="Hombre", offices=("ministro",))
    index = build_person_index([deputy], [minister])

    named = resolve_mentions(["Torres"], index, 90,
                             text="Lo firmó el ministro Torres.")
    assert _names(named) == {"Torres Pérez, Ángel"}
    # with no apposition the deputy preference is untouched
    assert _names(resolve_mentions(["Torres"], index, 90)) == {"Torres Tejada, María"}


def test_an_office_no_tied_person_holds_leaves_the_tie_alone():
    # "la ministra Maroto" is an ex-minister the catalog does not carry: nobody tied here
    # holds the office, so the role word is not evidence and the guard still drops it.
    assert resolve_mentions(["Sánchez"], SANCHEZ_INDEX, 90,
                            text="Lo dijo la ministra Sánchez.") == []


def test_role_apposition_does_not_promote_a_bare_given_name():
    # The surname gate again: "Pedro" ties both and names neither, whatever office the
    # sentence attaches to it.
    index = _offices(
        build_deputy_index([FakeDeputy("p1", "Sánchez Pérez-Castejón, Pedro"),
                            FakeDeputy("p2", "Casares Hontañón, Pedro")]),
        "Sánchez Pérez-Castejón, Pedro", "presidente")
    assert resolve_mentions(["Pedro"], index, 90,
                            text="Lo dijo el presidente Pedro.") == []


def test_role_apposition_never_overrides_an_unambiguous_name():
    # A resolution that never needed the guard must not be second-guessed by a stray
    # office: the deputy Sánchez Serna keeps his own occurrences.
    mentions = resolve_mentions(
        ["Sánchez Serna", "Sánchez"], SANCHEZ_INDEX, 90,
        text="El señor Sánchez Serna replicó al presidente Sánchez.")
    assert _counts(mentions) == {
        "Sánchez Serna, Javier": 1, "Sánchez Pérez-Castejón, Pedro": 1}


def test_role_apposition_is_inert_without_the_speech_text():
    # The spans alone carry no role word, so a caller that passes none — the query path,
    # and any caller predating this — gets exactly the old behaviour.
    assert resolve_mentions(["Sánchez"], SANCHEZ_INDEX, 90) == []


def test_role_apposition_can_be_switched_off():
    assert resolve_mentions(["Sánchez"], SANCHEZ_INDEX, 90,
                            text="Lo dijo el presidente Sánchez.",
                            role_apposition=False) == []


@pytest.mark.parametrize("role, expected", [
    ("Presidente del Gobierno", {"presidente"}),
    ("Presidenta del Congreso de los Diputados", {"presidente"}),
    ("Ministra de Vivienda y Agenda Urbana", {"ministro"}),
    # one person, two offices, both true of them
    ("Vicepresidenta Primera del Gobierno y Ministra de Hacienda",
     {"vicepresidente", "ministro"}),
    # "vicepresidenta" must not feed the "presidente" family: different people
    ("Vicepresidente Primero del Gobierno", {"vicepresidente"}),
    # no office to read
    ("Diputado", set()),
    ("Compareciente", set()),
    (None, set()),
])
def test_office_families(role, expected):
    assert office_families(role) == expected


def test_the_query_path_reads_no_office():
    # A searcher types a bare name with no speech around it, so the cue cannot reach
    # here: an ambiguous surname stays ambiguous.
    assert resolve_person("Sánchez", SANCHEZ_INDEX, 90) is None
    assert match_person("Sánchez", SANCHEZ_INDEX, 90).candidate_names == [
        "Sánchez Pérez-Castejón, Pedro", "Sánchez Serna, Javier"]


# --- role appositions as a detector seed ------------------------------------
# The same three shapes serve two readers: a cue about a span that exists, and a span the
# model never made. The offsets are what the second one needs.

def test_role_appositions_carry_the_names_offsets_and_family():
    text = "Lo dijo el presidente Sánchez ayer."
    assert role_appositions(text) == [
        (text.index("Sánchez"), text.index("Sánchez") + len("Sánchez"),
         "Sánchez", "presidente")]


def test_role_appositions_read_every_shape():
    found = {(name, family) for _s, _e, name, family in role_appositions(
        "El ministro Cuerpo y la ministra de Vivienda, Isabel Rodríguez, "
        "hablaron con el señor Sánchez, presidente del Gobierno.")}
    assert {("Cuerpo", "ministro"), ("Isabel Rodríguez", "ministro"),
            ("Sánchez", "presidente")} <= found


def test_role_appositions_are_ordered_by_position():
    text = "El ministro Cuerpo respondió al presidente Sánchez."
    offsets = [start for start, _e, _n, _f in role_appositions(text)]
    assert offsets == sorted(offsets)


def test_build_office_surfaces_maps_surnames_to_families():
    index = [
        make_person_entry("c", "minister", "Cuerpo Caballero, Carlos",
                          offices=("ministro", "vicepresidente")),
        make_person_entry("t", "minister", "Torres Pérez, Ángel Víctor",
                          offices=("ministro",)),
        make_person_entry("n", "minister", "Sin Cargo, Ana"),
    ]
    surfaces = build_office_surfaces(index)
    assert surfaces["cuerpo"] == ("ministro", "vicepresidente")
    assert surfaces["caballero"] == ("ministro", "vicepresidente")
    assert surfaces["torres"] == ("ministro",)
    # given names are not evidence ("el ministro Carlos" names nobody in particular)…
    assert "carlos" not in surfaces
    # …nor is a person with no recorded office
    assert "sin" not in surfaces and "cargo" not in surfaces


def test_build_office_surfaces_skips_surname_particles():
    index = [make_person_entry("m", "minister", "Muñoz de la Iglesia, Ester",
                               offices=("ministro",))]
    surfaces = build_office_surfaces(index)
    assert set(surfaces) == {"muñoz", "iglesia"}
