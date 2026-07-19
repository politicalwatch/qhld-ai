"""Pure tests for named-entity normalization and aggregation."""

import pytest

from qhld_ai.domain.entities import aggregate_entities, normalize_entity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("span", "key"),
    [
        ("Eurovisión", "eurovision"),
        ("la guerra de Gaza", "guerra de gaza"),   # leading article stripped
        ("guerra de Gaza", "guerra de gaza"),      # internal particle kept
        ("del Sáhara Occidental", "sahara occidental"),
        ("OTAN", "otan"),
        ("la UE", "ue"),                           # two-letter initialism survives
        ("«ley de amnistía»", "ley de amnistia"),  # punctuation dropped
        ("Agenda 2030", "agenda 2030"),
    ],
)
def test_normalize_entity(span, key):
    assert normalize_entity(span) == key


@pytest.mark.parametrize(
    "span",
    [
        "2030",            # digits-only
        "X",               # too short
        "la",              # nothing left after stripping
        "",
        "El Gobierno",     # parliamentary furniture (stoplist)
        "Congreso de los Diputados",
        "España",
        "Ustedes",         # discourse / forms of address (stoplist)
        "Buenos días",
        "Miren",
        "La Administración",
        "Arratsalde on",
    ],
)
def test_normalize_entity_drops_unusable(span):
    assert normalize_entity(span) == ""


def test_aggregate_entities_groups_surfaces_and_counts():
    spans = ["Eurovisión", "la guerra de Gaza", "guerra de Gaza",
             "Eurovisión", "eurovisión", "El Gobierno", "2030"]
    entities = aggregate_entities(spans)
    assert [(e.key, e.count) for e in entities] == [
        ("eurovision", 3), ("guerra de gaza", 2)]
    assert entities[0].surface_forms == ["Eurovisión", "eurovisión"]
    assert entities[1].surface_forms == ["guerra de Gaza", "la guerra de Gaza"]


def test_aggregate_entities_orders_by_count_then_key():
    entities = aggregate_entities(["OTAN", "Ucrania", "Ucrania", "Gaza"])
    assert [e.key for e in entities] == ["ucrania", "gaza", "otan"]


def test_aggregate_entities_empty():
    assert aggregate_entities([]) == []
