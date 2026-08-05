"""Unit tests for entity resolution — pure, with stubbed corpus + groups."""

from types import SimpleNamespace

import pytest

from qhld_ai.application.search.resolve_entities import EntityResolver
from qhld_ai.domain.ports.query_parser import ParsedQuery

pytestmark = pytest.mark.unit


CORPUS = {
    "speaker": {"Abascal Conde, Santiago", "Montero Cuadrado, María Jesús", "Aagesen Muñoz, Sara", None},
    "role": {"Diputado", "Ministra de Hacienda", "Ministra de Economía, Comercio y Empresa"},
    # Official catalog spellings, including every quirk class: parenthesized
    # article, bilingual "/", abbreviation, plain.
    "constituency": {"Málaga", "Cádiz", "Coruña (A)", "Rioja (La)", "Balears (Illes)",
                     "Alicante/Alacant", "S/C Tenerife", "Lleida", None},
    # Canonical keys as stamped by aggregate_entities at index time.
    "entities": {"eurovision", "festival de eurovision", "guerra de gaza",
                 "ucrania", "navantia", "ley de amnistia", None},
}

GROUPS = [
    SimpleNamespace(name="Grupo Parlamentario Socialista", shortname="GS", parties=["PSOE"]),
    SimpleNamespace(name="Grupo Parlamentario Popular", shortname="GP", parties=["PP"]),
    SimpleNamespace(name="Grupo Parlamentario Republicano", shortname="GR", parties=["ERC"]),
    SimpleNamespace(name="Grupo Parlamentario Mixto", shortname="GMx",
                    parties=["PODEMOS", "UPN", "BNG", "CCa", "PSOE"]),
]


class _FakeDeputy:
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_fullname(self):
        surname, given = (p.strip() for p in self.name.split(","))
        return f"{given} {surname}"


DEPUTIES = [
    _FakeDeputy("dep-montero", "Montero Cuadrado, María Jesús"),
    _FakeDeputy("dep-abascal", "Abascal Conde, Santiago"),
]


def _resolver(**overrides):
    # curated/nondeputy_speakers/curated_aliases/deputy_profiles injected → no
    # Mongo/data-file I/O happens in these unit tests.
    defaults = dict(
        distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS, deputies=DEPUTIES,
        curated=[], nondeputy_speakers=[], curated_aliases=[], deputy_profiles=[])
    return EntityResolver(**{**defaults, **overrides})


@pytest.fixture
def resolver():
    return _resolver()


def test_resolves_speaker_name_with_token_reordering(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Santiago Abascal"]))
    assert r.filters["speaker"] == "Abascal Conde, Santiago"


def test_unresolvable_speaker_blocks(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Fulano de Tal"]))
    assert "speaker" not in r.filters
    assert r.blocked
    entity = r.unresolved[0]
    assert (entity.field, entity.value, entity.blocking) == ("speaker", "Fulano de Tal", True)
    assert entity.suggestion  # the best sub-threshold corpus speaker, for triage
    assert any("unresolved" in note for note in r.notes)


def test_multiple_speakers_resolve_to_a_list(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", speakers=["Santiago Abascal", "María Jesús Montero"]))
    assert r.filters["speaker"] == [
        "Abascal Conde, Santiago", "Montero Cuadrado, María Jesús"]


def test_partially_resolved_speakers_keep_the_resolved_one(resolver):
    # Several speakers are an any-of list: dropping one member still honours the
    # query, so it is recorded but does not block.
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", speakers=["Santiago Abascal", "Fulano de Tal"]))
    assert r.filters["speaker"] == "Abascal Conde, Santiago"
    assert not r.blocked
    assert [(e.field, e.value, e.blocking) for e in r.unresolved] == [
        ("speaker", "Fulano de Tal", False)]


# --- collisions: resolved, but only by list order --------------------------


COLLIDING = {"speaker": {"Rueda Perelló, Patricia", "Rueda Pérez, Juan Carlos"}}


def test_a_shared_surname_keeps_every_bearer_instead_of_guessing():
    # Both Ruedas bear it as a FIRST surname and neither holds an office, so nothing
    # distinguishes them. Guessing one would make it a hard filter and return zero
    # results whenever the guess was wrong — indistinguishable from having no coverage.
    # So the filter keeps both and lets ranking decide what surfaces first.
    resolver = _resolver(distinct=lambda key: COLLIDING.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Rueda"]))

    assert r.filters["speaker"] == sorted(COLLIDING["speaker"])
    assert not r.blocked and not r.unresolved
    assert len(r.ambiguous) == 1
    match = r.ambiguous[0]
    assert (match.field, match.value) == ("speaker", "Rueda")
    assert sorted(match.tied) == sorted(COLLIDING["speaker"])
    assert sorted(match.kept) == sorted(COLLIDING["speaker"])
    # and the trace says so, because it is shown to the user as a chip
    assert any("names 2 people" in note for note in r.notes)


def test_a_tied_surname_is_settled_by_its_first_bearer():
    # "Bravo" is Juan Bravo, not Aitor Esteban Bravo: a surname names whoever carries it
    # FIRST, so this tie is breakable on evidence and no fail-open is needed.
    vocab = {"speaker": {"Bravo Baena, Juan", "Esteban Bravo, Aitor"}}
    resolver = _resolver(distinct=lambda key: vocab.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Bravo"]))

    assert r.filters["speaker"] == "Bravo Baena, Juan"
    assert r.ambiguous[0].kept == ["Bravo Baena, Juan"]


def test_the_motivating_case_is_settled_without_any_prior():
    # "Montero" is Montero Cuadrado's FIRST surname and Vaquero Montero's second, so the
    # surname alone identifies her. This is the query that used to return zero results,
    # and it resolves with no appeal to her office — which is why removing the office
    # tie-break does not regress it.
    vocab = {"speaker": {"Montero Cuadrado, María Jesús", "Vaquero Montero, Maribel"}}
    resolver = _resolver(distinct=lambda key: vocab.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Montero"]))

    assert r.filters["speaker"] == "Montero Cuadrado, María Jesús"
    assert r.ambiguous[0].kept == ["Montero Cuadrado, María Jesús"]


def test_an_office_holder_does_not_exclude_the_others_who_share_the_surname():
    # Tried the other way and measured: as a filter, office kept the prime minister (41
    # speeches) and threw away Sánchez Serna (59). Holding office makes somebody a likelier
    # referent, not the only possible one — a search for any of these people is plausible.
    vocab = {"speaker": {"Sánchez Pérez-Castejón, Pedro", "Sánchez Serna, Javier",
                         "Sánchez Torregrosa, Maribel"}}
    resolver = _resolver(
        distinct=lambda key: vocab.get(key, set()), deputies=[],
        speaker_offices=[{"speaker": "Sánchez Pérez-Castejón, Pedro",
                          "role": "Presidente del Gobierno"}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Sánchez"]))

    assert r.filters["speaker"] == sorted(vocab["speaker"])
    assert sorted(r.ambiguous[0].kept) == sorted(vocab["speaker"])


def test_the_office_holder_is_offered_first_among_the_candidates():
    # The one thing a prior may do: order the list the client offers as "did you mean".
    vocab = {"speaker": {"Sánchez Pérez-Castejón, Pedro", "Sánchez Serna, Javier",
                         "Sánchez Torregrosa, Maribel"}}
    resolver = _resolver(
        distinct=lambda key: vocab.get(key, set()), deputies=[],
        speaker_offices=[{"speaker": "Sánchez Pérez-Castejón, Pedro",
                          "role": "Presidente del Gobierno"}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Sánchez"]))

    assert r.ambiguous[0].kept[0] == "Sánchez Pérez-Castejón, Pedro"
    # ...and the rest alphabetically behind them
    assert r.ambiguous[0].kept[1:] == ["Sánchez Serna, Javier",
                                       "Sánchez Torregrosa, Maribel"]


def test_candidates_are_alphabetical_when_no_office_is_known():
    vocab = {"speaker": {"Núñez Guijarro, José Enrique", "Núñez Feijóo, Alberto",
                         "Núñez González, Noelia"}}
    resolver = _resolver(distinct=lambda key: vocab.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Núñez"]))

    assert r.ambiguous[0].kept == sorted(vocab["speaker"])


def test_a_surname_everybody_carries_second_keeps_the_whole_tie():
    # Nobody bears "Caballero" first, so narrowing has nothing to say. Letting office
    # decide here was measured and rejected — it yields "Caballero" -> Cuerpo Caballero.
    vocab = {"speaker": {"Cuerpo Caballero, Carlos", "Sierra Caballero, Francisco"}}
    resolver = _resolver(
        distinct=lambda key: vocab.get(key, set()), deputies=[],
        speaker_offices=[{"speaker": "Cuerpo Caballero, Carlos",
                          "role": "Ministro de Economía, Comercio y Empresa"}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Caballero"]))

    assert r.filters["speaker"] == sorted(vocab["speaker"])


def test_a_first_surname_behind_a_particle_still_counts():
    # "Del Valle" and "De los Santos" are first surnames; the particle is not the part
    # anybody says, so a query for "Valle" names him.
    vocab = {"speaker": {"Del Valle Rodríguez, Emilio Jesús", "Mellado Sierra, Valle"}}
    resolver = _resolver(distinct=lambda key: vocab.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Valle"]))

    assert r.filters["speaker"] == "Del Valle Rodríguez, Emilio Jesús"


def test_first_surname_matching_folds_accents_on_both_sides():
    resolver = _resolver(deputies=[])
    assert resolver._first_surname_match("Núñez", "Núñez Feijóo, Alberto")
    assert resolver._first_surname_match("nunez", "Núñez Feijóo, Alberto")
    assert not resolver._first_surname_match("Núñez", "Paniagua Núñez, Miguel Ángel")


# --- a name typed without its accents ---------------------------------------

NUNEZ = {"speaker": {"Núñez Feijóo, Alberto", "Núñez González, Noelia",
                     "Núñez Guijarro, José Enrique", "Sánchez Serna, Javier"}}


@pytest.mark.parametrize("typed", ["núñez", "nuñez", "nunez", "Núñez", "NUNEZ"])
def test_a_surname_resolves_the_same_however_its_accents_are_typed(typed):
    # The corpus reads accented transcripts; people type names without the accents. Before
    # folding, "nuñez" scored 29 against "Núñez Feijóo, Alberto" and the query resolved to
    # NOBODY, while "núñez" scored 100 — the same question answered two different ways.
    resolver = _resolver(distinct=lambda key: NUNEZ.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=[typed]))

    assert r.filters["speaker"] == sorted(
        {"Núñez Feijóo, Alberto", "Núñez González, Noelia", "Núñez Guijarro, José Enrique"})
    assert not r.unresolved


def test_folding_does_not_make_a_query_reach_an_unrelated_surname():
    # Folding is meant to add reach to the RIGHT name only: unrelated surnames score in the
    # 20s either way, so nothing new clears the threshold.
    resolver = _resolver(distinct=lambda key: NUNEZ.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["sanchez"]))

    assert r.filters["speaker"] == "Sánchez Serna, Javier"


def test_an_unknown_name_still_resolves_to_nobody():
    resolver = _resolver(distinct=lambda key: NUNEZ.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["xyzabc"]))

    assert "speaker" not in r.filters
    assert r.blocked


def test_a_role_resolves_without_its_accents_too():
    # Same fuzzy path, so the fix reaches roles as well.
    resolver = _resolver()
    r = resolver.resolve(
        ParsedQuery(semantic_query="x", speaker_title="ministra de economia"))

    assert r.filters["role"] == "Ministra de Economía, Comercio y Empresa"


def test_the_tied_pick_is_reproducible_across_processes():
    # The vocabulary arrives as a set, so before sorting the winner depended on string
    # hashing and the same query answered differently after a restart.
    vocab = {"speaker": {"Rueda Perelló, Patricia", "Rueda Pérez, Juan Carlos"}}
    runs = {
        tuple(_resolver(distinct=lambda key: vocab.get(key, set()), deputies=[])
              .resolve(ParsedQuery(semantic_query="x", speakers=["Rueda"]))
              .filters["speaker"])
        for _ in range(5)
    }
    assert len(runs) == 1


def test_an_unshared_win_is_not_ambiguous():
    resolver = _resolver(distinct=lambda key: COLLIDING.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Patricia Rueda"]))

    assert r.filters["speaker"] == "Rueda Perelló, Patricia"
    assert r.ambiguous == []


def test_a_sub_threshold_near_miss_is_not_ambiguous():
    # Only a win can be ambiguous: a miss is already reported through ``unresolved``,
    # and reporting it twice would double-count it downstream.
    resolver = _resolver(distinct=lambda key: COLLIDING.get(key, set()), deputies=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Fulano de Tal"]))

    assert r.ambiguous == []
    assert [e.field for e in r.unresolved] == ["speaker"]


# --- curated deputy aliases (public names) ---------------------------------
# A deputy the public knows by another name than the official one. "Tesh Sidi"
# shares no token with "Andala Ubbi, Teslem", so it scores far below any usable
# threshold and is unreachable without curation.

TESLEM = _FakeDeputy("andala-ubbi-teslem", "Andala Ubbi, Teslem")
TESH_ALIAS = [{"deputy_id": "andala-ubbi-teslem", "name": "Andala Ubbi, Teslem",
               "aliases": ["Tesh Sidi"]}]


def _alias_resolver(**overrides):
    """Resolver whose corpus has spoken for Teslem and whose alias file is curated."""
    corpus = {**CORPUS, "speaker": CORPUS["speaker"] | {"Andala Ubbi, Teslem"}}
    return _resolver(distinct=lambda key: corpus.get(key, set()),
                     deputies=[*DEPUTIES, TESLEM], deputy_profiles=TESH_ALIAS,
                     **overrides)


def test_speaker_alias_resolves_to_the_canonical_corpus_value():
    r = _alias_resolver().resolve(
        ParsedQuery(semantic_query="x", speakers=["Tesh Sidi"]))
    assert r.filters["speaker"] == "Andala Ubbi, Teslem"
    assert not r.blocked
    assert any("curated alias" in note for note in r.notes)


def test_speaker_alias_tolerates_honorifics_and_casing():
    r = _alias_resolver().resolve(
        ParsedQuery(semantic_query="x", speakers=["la señora TESH SIDI"]))
    assert r.filters["speaker"] == "Andala Ubbi, Teslem"


def test_speaker_alias_falls_through_when_corpus_lacks_that_speaker():
    # A curated deputy who has not spoken (or a stale curation) must not filter on a
    # name the corpus never had: the outcome stays exactly what it is today.
    r = _resolver(deputy_profiles=TESH_ALIAS, deputies=[*DEPUTIES, TESLEM]).resolve(
        ParsedQuery(semantic_query="x", speakers=["Tesh Sidi"]))
    assert "speaker" not in r.filters
    assert r.blocked
    assert r.unresolved[0].value == "Tesh Sidi"


def test_speaker_alias_works_without_the_deputies_catalog():
    # The speaker path needs no catalog, so a resolver built without one still
    # honours curated aliases.
    corpus = {**CORPUS, "speaker": CORPUS["speaker"] | {"Andala Ubbi, Teslem"}}
    resolver = EntityResolver(
        distinct=lambda key: corpus.get(key, set()), groups=GROUPS,
        curated_aliases=[], deputy_profiles=TESH_ALIAS)
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Tesh Sidi"]))
    assert r.filters["speaker"] == "Andala Ubbi, Teslem"


def test_curated_alias_does_not_disturb_fuzzy_speaker_matching():
    r = _alias_resolver().resolve(
        ParsedQuery(semantic_query="x", speakers=["Santiago Abascal"]))
    assert r.filters["speaker"] == "Abascal Conde, Santiago"
    # still the fuzzy note with its score, not the alias note
    assert any("(100)" in note for note in r.notes)
    assert not any("curated alias" in note for note in r.notes)


def test_mentioned_person_resolves_via_deputy_alias():
    r = _alias_resolver().resolve(
        ParsedQuery(semantic_query="sáhara", mentioned_persons=["Tesh Sidi"]))
    assert r.filters["mentions"] == "andala-ubbi-teslem"


def test_resolves_title_to_role(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speaker_title="ministra de economía"))
    assert r.filters["role"] == "Ministra de Economía, Comercio y Empresa"


def test_resolves_party_name_to_group_code(resolver):
    assert resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["PSOE"])).filters["group"] == "GS"
    assert resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["PP"])).filters["group"] == "GP"


def test_resolves_group_long_name(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["Grupo Socialista"]))
    assert r.filters["group"] == "GS"


def test_resolves_colloquial_group_names(resolver):
    # The phrasings a user swaps freely: party-word scaffolding and plural demonyms.
    for raw, code in [
        ("Partido Socialista", "GS"),
        ("los socialistas", "GS"),
        ("socialistas", "GS"),
        ("Partido Socialista Obrero Español", "GS"),
        ("los populares", "GP"),
        ("partido popular", "GP"),
        ("los republicanos", "GR"),
    ]:
        r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=[raw]))
        assert r.filters.get("group") == code, f"'{raw}' → {r.filters.get('group')}"


def test_curated_alias_resolves_when_code_is_in_catalog():
    resolver = _resolver(curated_aliases=[{"code": "GR", "aliases": ["Esquerra"]}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["esquerra"]))
    assert r.filters["group"] == "GR"


def test_curated_alias_for_absent_group_blocks():
    resolver = _resolver(curated_aliases=[{"code": "GCUP", "aliases": ["Esquerra"]}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["esquerra"]))
    assert "group" not in r.filters
    assert r.blocked


CATEGORIZED = [
    {"code": "GS", "aliases": [], "categories": ["izquierda"]},
    {"code": "GR", "aliases": ["Esquerra"], "categories": ["izquierda", "independentista"]},
    {"code": "GVOX", "aliases": [], "categories": ["derecha"]},  # not in the catalog
]


def test_ideological_category_expands_to_every_labelled_group():
    resolver = _resolver(curated_aliases=CATEGORIZED)
    for raw in ["izquierda", "la izquierda", "los partidos de izquierda"]:
        r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=[raw]))
        assert r.filters["group"] == ["GR", "GS"], f"'{raw}' → {r.filters.get('group')}"
        assert any("(category)" in note for note in r.notes)


def test_category_with_single_labelled_group_stays_scalar():
    resolver = _resolver(curated_aliases=CATEGORIZED)
    r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["los independentistas"]))
    assert r.filters["group"] == "GR"


def test_category_of_absent_group_blocks():
    # GVOX is labelled 'derecha' but is not in the catalog → nothing to expand to.
    resolver = _resolver(curated_aliases=CATEGORIZED)
    r = resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["la derecha"]))
    assert "group" not in r.filters
    assert r.blocked


def test_category_and_named_group_combine():
    resolver = _resolver(curated_aliases=CATEGORIZED)
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", groups_or_parties=["la izquierda", "PP"]))
    assert r.filters["group"] == ["GP", "GR", "GS"]


def test_multiple_groups_resolve_to_a_list(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", groups_or_parties=["Grupo Socialista", "Grupo Popular"]))
    assert r.filters["group"] == ["GP", "GS"]


def test_shared_party_prefers_single_party_group_over_mixto(resolver):
    # 'PSOE' is listed under both GS and the catch-all GMx; GS (fewer parties) wins —
    # for the verbatim alias and for the normalized colloquial forms alike.
    assert resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["PSOE"])).filters["group"] == "GS"
    assert resolver.resolve(ParsedQuery(semantic_query="x", groups_or_parties=["socialistas"])).filters["group"] == "GS"


def test_constituency_exact_value(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", constituencies=["Málaga"]))
    assert r.filters["constituency"] == "Málaga"
    assert not r.blocked


def test_constituency_variants_map_to_official_spelling(resolver):
    # Every quirk class of the official catalog values: parenthesized article,
    # bilingual "/", abbreviation, curated old Castilian name, spelling drift.
    for raw, official in [
        ("A Coruña", "Coruña (A)"),
        ("La Coruña", "Coruña (A)"),
        ("La Rioja", "Rioja (La)"),
        ("Alacant", "Alicante/Alacant"),
        ("Alicante", "Alicante/Alacant"),
        ("Tenerife", "S/C Tenerife"),
        ("Santa Cruz de Tenerife", "S/C Tenerife"),
        ("Lérida", "Lleida"),
        ("Islas Baleares", "Balears (Illes)"),
        ("Baleares", "Balears (Illes)"),
    ]:
        r = resolver.resolve(ParsedQuery(semantic_query="x", constituencies=[raw]))
        assert r.filters.get("constituency") == official, \
            f"'{raw}' → {r.filters.get('constituency')}"


def test_multiple_constituencies_resolve_to_a_list(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", constituencies=["Málaga", "Cádiz"]))
    assert r.filters["constituency"] == ["Cádiz", "Málaga"]


def test_unresolvable_constituency_blocks(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", constituencies=["Narnia"]))
    assert "constituency" not in r.filters
    assert r.blocked
    entity = r.unresolved[0]
    assert (entity.field, entity.value, entity.blocking) == ("constituency", "Narnia", True)


def test_partially_resolved_constituencies_keep_the_resolved_one(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", constituencies=["Málaga", "Narnia"]))
    assert r.filters["constituency"] == "Málaga"
    assert not r.blocked


def test_constituency_combines_with_group_filter(resolver):
    # "diputados del PSOE por Málaga" → both filters, ANDed by the store.
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", constituencies=["Málaga"], groups_or_parties=["PSOE"]))
    assert r.filters["constituency"] == "Málaga"
    assert r.filters["group"] == "GS"


def test_iso_dates_become_numeric_range(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", date_from="2025-04-03", date_to="2025-07-03"))
    assert r.filters["date"] == {"gte": 20250403, "lte": 20250703}


def test_open_ended_date_range(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", date_from="2025-01-01"))
    assert r.filters["date"] == {"gte": 20250101}


def test_lang_and_legislature_pass_through(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", lang="gl", legislature="15"))
    assert r.filters["lang"] == "gl"
    assert r.filters["legislature"] == "15"


def test_lang_names_and_variants_normalize_to_iso_code(resolver):
    # LLMs often emit the language name or an off-code ("Gallego", "cat").
    assert resolver.resolve(ParsedQuery(semantic_query="x", lang="Gallego")).filters["lang"] == "gl"
    assert resolver.resolve(ParsedQuery(semantic_query="x", lang="cat")).filters["lang"] == "ca"
    assert resolver.resolve(ParsedQuery(semantic_query="x", lang="euskera")).filters["lang"] == "eu"


def test_unknown_lang_blocks(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", lang="klingon"))
    assert "lang" not in r.filters
    assert r.blocked


def test_nothing_unresolved_means_not_blocked(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="x", speakers=["Santiago Abascal"]))
    assert not r.blocked
    assert r.unresolved == []


def test_no_filters_when_nothing_extracted(resolver):
    assert resolver.resolve(ParsedQuery(semantic_query="financiación")).filters == {}


def test_mentioned_person_resolves_to_deputy_id(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="vivienda", mentioned_persons=["Montero"]))
    assert r.filters["mentions"] == "dep-montero"
    assert any("mentions:" in note for note in r.notes)


def test_a_mention_carries_the_name_behind_its_id(resolver):
    # The filter has to be the id (that is how the payload is keyed), and nobody outside
    # this package can turn one back into a name — least of all its accents.
    r = resolver.resolve(ParsedQuery(semantic_query="vivienda", mentioned_persons=["Montero"]))
    assert r.labels["mentions"] == {"dep-montero": "Montero Cuadrado, María Jesús"}


def test_every_mentioned_person_is_named(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Abascal"]))
    assert set(r.labels["mentions"]) == set(r.filters["mentions"]["all"])

    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Abascal"], mentions_mode="any"))
    assert set(r.labels["mentions"]) == set(r.filters["mentions"])


def test_a_mention_that_filters_on_nothing_is_not_named(resolver):
    # Blocked: no mentions filter was emitted, so there is nothing to say back.
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Winston Churchill"]))
    assert "mentions" not in r.filters
    assert r.labels == {}


def test_unresolvable_mentioned_person_blocks(resolver):
    # The corpus is tagged with the same catalog: a person absent from it cannot
    # appear in any payload, so the query is unsatisfiable.
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_persons=["Winston Churchill"]))
    assert "mentions" not in r.filters
    assert r.blocked
    entity = r.unresolved[0]
    assert (entity.field, entity.value, entity.blocking) == ("mentions", "Winston Churchill", True)


def test_multiple_mentions_default_to_requiring_all(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Abascal"]))
    assert r.filters["mentions"] == {"all": ["dep-abascal", "dep-montero"]}


def test_multiple_mentions_any_mode_becomes_a_list(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Abascal"], mentions_mode="any"))
    assert r.filters["mentions"] == ["dep-abascal", "dep-montero"]


def test_partially_resolved_mentions_in_all_mode_block(resolver):
    # 'all' requires EVERY person to be mentioned; one unsatisfiable member makes
    # the whole conjunction unsatisfiable — no partial filter is emitted.
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Winston Churchill"]))
    assert "mentions" not in r.filters
    assert r.blocked


def test_partially_resolved_mentions_in_any_mode_keep_the_resolved_one(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Montero", "Winston Churchill"],
        mentions_mode="any"))
    assert r.filters["mentions"] == "dep-montero"
    assert not r.blocked
    assert [(e.field, e.blocking) for e in r.unresolved] == [("mentions", False)]


def test_wholly_unresolved_mentions_in_any_mode_block(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", mentioned_persons=["Winston Churchill", "Napoleón"],
        mentions_mode="any"))
    assert "mentions" not in r.filters
    assert r.blocked


def test_entity_resolves_to_canonical_key(resolver):
    r = resolver.resolve(ParsedQuery(semantic_query="Eurovisión", entities=["Eurovisión"]))
    assert r.filters["entities"] == "eurovision"
    assert any("entities: 'Eurovisión' → 'eurovision'" in note for note in r.notes)


def test_entity_normalization_strips_leading_article(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="la guerra de Gaza", entities=["la guerra de Gaza"]))
    assert r.filters["entities"] == "guerra de gaza"


def test_entity_fuzzy_fallback_absorbs_particle_drift(resolver):
    # "guerra en Gaza" normalizes to a key absent from the vocabulary; the fuzzy
    # fallback still lands on the corpus key.
    r = resolver.resolve(ParsedQuery(
        semantic_query="guerra en Gaza", entities=["guerra en Gaza"]))
    assert r.filters["entities"] == "guerra de gaza"


def test_multiple_entities_default_to_requiring_all(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="Gaza y Ucrania", entities=["Gaza", "Ucrania"]))
    assert r.filters["entities"] == {"all": ["guerra de gaza", "ucrania"]}


def test_multiple_entities_any_mode_becomes_a_list(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="Gaza o Ucrania", entities=["Gaza", "Ucrania"],
        entities_mode="any"))
    assert r.filters["entities"] == ["guerra de gaza", "ucrania"]


def test_unresolvable_entity_blocks_with_suggestion(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="cumbre del clima", entities=["cumbre del clima"]))
    assert "entities" not in r.filters
    assert r.blocked
    entity = r.unresolved[0]
    assert (entity.field, entity.value, entity.blocking) == (
        "entities", "cumbre del clima", True)


def test_partially_resolved_entities_in_any_mode_keep_the_resolved_one(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", entities=["Navantia", "cumbre del clima"],
        entities_mode="any"))
    assert r.filters["entities"] == "navantia"
    assert not r.blocked
    assert [(e.field, e.blocking) for e in r.unresolved] == [("entities", False)]


def test_partially_resolved_entities_in_all_mode_block(resolver):
    r = resolver.resolve(ParsedQuery(
        semantic_query="x", entities=["Navantia", "cumbre del clima"]))
    assert "entities" not in r.filters
    assert r.blocked


def test_furniture_entity_blocks(resolver):
    # "el Gobierno" normalizes to "" (stoplist): unsatisfiable rather than a
    # filter that would match every speech.
    r = resolver.resolve(ParsedQuery(semantic_query="x", entities=["el Gobierno"]))
    assert "entities" not in r.filters
    assert r.blocked


def test_ambiguous_mention_blocks_with_candidates():
    resolver = _resolver(deputies=[
        _FakeDeputy("dep-g1", "García López, Juan"),
        _FakeDeputy("dep-g2", "García Ruiz, Ana"),
    ])
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_persons=["García"]))
    assert "mentions" not in r.filters
    assert r.blocked
    assert "ambiguous" in r.unresolved[0].suggestion
    assert "García López, Juan" in r.unresolved[0].suggestion


def test_mentioned_person_ignored_without_deputies_catalog():
    # A resolver built without the catalog knowingly opts out of mention
    # filtering — that must NOT read as an unsatisfiable query.
    resolver = EntityResolver(
        distinct=lambda key: CORPUS.get(key, set()), groups=GROUPS, curated_aliases=[],
        deputy_profiles=[])
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_persons=["Montero"]))
    assert "mentions" not in r.filters
    assert not r.blocked
    assert any("no person catalog" in note for note in r.notes)


def test_mentioned_non_deputy_resolves_via_curated():
    # A curated non-deputy (Ayuso) resolves to her person id and is filtered.
    resolver = _resolver(
        curated=[{"person_id": "isabel-diaz-ayuso", "person_type": "regional_president",
                  "name": "Díaz Ayuso, Isabel", "aliases": ["Ayuso", "Díaz Ayuso"]}])
    r = resolver.resolve(ParsedQuery(semantic_query="vivienda", mentioned_persons=["Ayuso"]))
    assert r.filters["mentions"] == "isabel-diaz-ayuso"
    assert any("regional_president" in note for note in r.notes)


def test_mentioned_bootstrapped_minister_resolves():
    # A non-deputy speaker bootstrapped from the corpus (a minister) is resolvable too.
    resolver = _resolver(
        nondeputy_speakers=[{"speaker": "Aagesen Muñoz, Sara",
                             "role": "Vicepresidenta Tercera y Ministra"}])
    r = resolver.resolve(ParsedQuery(semantic_query="x", mentioned_persons=["Aagesen"]))
    assert r.filters["mentions"] == "aagesen-munoz-sara"
    assert any("minister" in note for note in r.notes)
