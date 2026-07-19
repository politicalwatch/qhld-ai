"""Light test for the spaCy NER adapter — loads the real es_core_news_lg model.

The model is a main dependency (mention NER runs in the daily extract), so this
runs by default; guarded by find_spec so a stripped env skips instead of erroring.
"""

from importlib.util import find_spec

import pytest

pytestmark = pytest.mark.unit

_HAS_MODEL = find_spec("spacy") and find_spec("es_core_news_lg")
pytestmark = [pytest.mark.unit,
              pytest.mark.skipif(not _HAS_MODEL, reason="spaCy model not installed")]


def test_person_spans_extracts_people_not_orgs():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    ner = create_ner_from_env(Settings())
    text = ("El señor Feijóo criticó a Pedro Sánchez. El Gobierno y el "
            "Partido Popular no se pusieron de acuerdo.")
    spans = ner.person_spans(text)
    # People are captured; the ORG mentions (Gobierno, Partido Popular) are not.
    assert any("Feijóo" in s for s in spans)
    assert any("Sánchez" in s for s in spans)
    assert not any("Partido Popular" == s for s in spans)


def test_empty_text_returns_no_spans():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    ner = create_ner_from_env(Settings())
    assert ner.person_spans("") == []


def test_gazetteer_tags_surname_the_model_misses():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    text = "La señora Vallugera intervino y también habló la señora Cruset."
    # These out-of-vocabulary surnames get tagged only once the gazetteer seeds them.
    seeded = create_ner_from_env(Settings(), gazetteer=["Vallugera", "Cruset"])
    spans = seeded.person_spans(text)
    assert any("Vallugera" in s for s in spans)
    assert any("Cruset" in s for s in spans)
    # An in-vocabulary common word offered to the gazetteer is filtered out, not tagged.
    only_common = create_ner_from_env(Settings(), gazetteer=["Madrid"])
    assert not any(s == "Madrid" for s in only_common.person_spans("Vivo en Madrid ahora."))


def test_entity_spans_extracts_non_persons_only():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    ner = create_ner_from_env(Settings())
    text = ("Pedro Sánchez defendió la participación de España en Eurovisión "
            "pese a la guerra de Gaza.")
    spans = ner.entity_spans(text)
    assert any("Eurovisión" in s for s in spans)
    assert any("Gaza" in s for s in spans)
    assert not any("Sánchez" in s for s in spans)
    assert ner.entity_spans("") == []


def test_entity_spans_excludes_gazetteer_rescued_surnames():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    # An out-of-vocabulary surname the model mislabels is rescued as a PERSON by
    # the gazetteer post-pass, so it must not leak into the entity pool.
    text = "La señora Vallugera habló sobre Eurovisión."
    seeded = create_ner_from_env(Settings(), gazetteer=["Vallugera"])
    assert any("Vallugera" in s for s in seeded.person_spans(text))
    entity_spans = seeded.entity_spans(text)
    assert not any("Vallugera" in s for s in entity_spans)
    assert any("Eurovisión" in s for s in entity_spans)


def test_entity_pos_gate_predicate_keeps_names_drops_clauses():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env
    from qhld_ai.infrastructure.ner.spacy import SpacyNer

    nlp = create_ner_from_env(Settings())._model()
    # Proper-noun phrases pass; discourse tokens (no PROPN) and verb-bearing spans fail.
    assert SpacyNer._is_entity_like(nlp("Unión Europea"))
    assert SpacyNer._is_entity_like(nlp("Eurovisión"))
    assert not SpacyNer._is_entity_like(nlp("Por tanto"))
    assert not SpacyNer._is_entity_like(nlp("Esta evidencia"))
    assert not SpacyNer._is_entity_like(nlp("aprobó la Ley de Industria"))


def test_entity_pos_gate_drops_verb_bearing_spans():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    # The base model glues a whole clause into one MISC span here; the gate must drop
    # any span carrying a verb, while the raw model (gate off) still emits it.
    text = "Esta evidencia es clara. La Unión Europea aprobó la Ley de Industria."
    on = create_ner_from_env(Settings())
    off = create_ner_from_env(Settings(ner_entity_pos_gate=False))
    assert any("aprobó" in s for s in off.entity_spans(text))
    assert not any("aprobó" in s for s in on.entity_spans(text))


def test_entity_pos_gate_keeps_genuine_entities():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    # Real proper-noun entities survive the gate (they are PROPN with no verb).
    ner = create_ner_from_env(Settings())
    spans = ner.entity_spans("Navantia construye buques en Cádiz.")
    assert any("Navantia" in s for s in spans)
    assert any("Cádiz" in s for s in spans)


def test_person_and_entity_spans_share_one_parse():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    ner = create_ner_from_env(Settings())
    text = "Pedro Sánchez habló de Eurovisión."
    ner.person_spans(text)
    parses = 0
    real_model = ner._model()

    class _CountingModel:
        def __call__(self, value):
            nonlocal parses
            parses += 1
            return real_model(value)

    ner._nlp = _CountingModel()
    ner.entity_spans(text)      # same text -> memo hit, no new parse
    assert parses == 0
    ner.entity_spans("Otro texto sobre la OTAN.")   # new text -> one parse
    assert parses == 1


def test_gazetteer_does_not_break_up_model_spans():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    # "Reyes Maroto" (the ex-minister) must survive as the model's own span even
    # with "Maroto" seeded (a deputy's distinctive second surname): the ruler runs
    # after the model and never splits an existing entity into an orphan surname
    # that would resolve to the wrong person.
    text = "La exministra Reyes Maroto anunció ayer las ayudas al sector."
    seeded = create_ner_from_env(Settings(), gazetteer=["Maroto"])
    spans = seeded.person_spans(text)
    assert "Reyes Maroto" in spans
    assert "Maroto" not in spans
