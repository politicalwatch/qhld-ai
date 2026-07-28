"""Light test for the spaCy NER adapter — loads the real es_core_news_lg model.

The model is a main dependency (mention NER runs in the daily extract), so this
runs by default; guarded by find_spec so a stripped env skips instead of erroring.
"""

import re
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


# --- courtesy words are absorbed into the span -----------------------------
# The model folds "señor"/"señora" in only sometimes, and the courtesy form carries the
# gender that tells two holders of a surname apart — so the adapter makes it uniform.

def _ner(**kw):
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    return create_ner_from_env(Settings(), **kw)


def test_courtesy_word_is_absorbed_into_the_span():
    spans = _ner().person_spans("Coincido con la señora Montero en ese punto.")
    assert "señora Montero" in spans
    # the article stays outside: "la" is nobody's name, and the span doubles as the
    # highlight target on the site
    assert not any(s.startswith("la ") or s.startswith("La ") for s in spans)


def test_courtesy_word_absorbed_for_a_gazetteer_rescued_surname():
    # The rescued path is where it matters most: a bare one-token match would otherwise
    # never carry the cue.
    spans = _ner(gazetteer=["Vallugera"]).person_spans(
        "Ha intervenido la señora Vallugera esta mañana.")
    assert "señora Vallugera" in spans


def test_a_name_without_a_courtesy_word_is_unchanged():
    spans = _ner().person_spans("Pedro Sánchez compareció ante el Pleno.")
    assert any(s == "Pedro Sánchez" for s in spans)


def test_abbreviated_courtesy_form_is_absorbed():
    # "Sra." tokenizes as "Sra" + "."; both are stepped over.
    spans = _ner(gazetteer=["Vallugera"]).person_spans("Lo dijo la Sra. Vallugera ayer.")
    assert any("Vallugera" in s and "Sra" in s for s in spans)


def test_a_role_word_is_not_absorbed():
    # Only courtesy forms, which inflect for gender. A role word would change what the
    # span means, not merely how polite it is.
    spans = _ner().person_spans("El ministro Bolaños respondió.")
    assert not any("ministro" in s for s in spans)


def test_a_leading_article_the_model_swallowed_is_trimmed():
    # Sentence-initial, where the model tends to take "El" into the entity because it
    # cannot tell a sentence capital from a name capital. The span still starts at the
    # courtesy word, so both casings agree.
    spans = _ner().person_spans("El señor Abascal intervino. Coincido con el señor Abascal.")
    assert [s for s in spans if "Abascal" in s] == ["señor Abascal", "señor Abascal"]


def test_no_span_begins_with_an_article():
    text = ("El señor Abascal y la señora Montero discutieron. "
            "La señora Montero respondió al señor Abascal.")
    for span in _ner().person_spans(text):
        assert not re.match(r"(?i)(el|la|los|las)\s", span), span


def test_a_contracted_article_is_trimmed_too():
    # "al" (a+el) and "del" (de+el) are as common before a courtesy form as the bare
    # article, and are equally not part of the name.
    spans = _ner().person_spans(
        "Se lo dije al señor Bolaños y también hablé del señor Bolaños con ella.")
    assert [s for s in spans if "Bolaños" in s] == ["señor Bolaños", "señor Bolaños"]


def test_an_article_that_belongs_to_the_name_is_kept():
    # The trim only fires when a courtesy word follows, so names whose article IS part of
    # them are untouched — the corpus really contains 'La Razón', 'El Tito Berni'.
    ner = _ner(gazetteer=["Berni"])
    for span in ner.person_spans("Lo publicó La Razón sobre el caso El Tito Berni."):
        assert "Razón" not in span or span.startswith("La")


# --- role apposition: the names the model mislabels -------------------------
# "El ministro Albares" comes back as MISC, "Cuerpo" as ORG. The role word is what says
# the occurrence names a person, so the office map can cover surnames the gazetteer must
# leave alone — these are all in vocabulary.

_OFFICES = {"cuerpo": ("ministro",), "torres": ("ministro",),
            "sánchez": ("presidente",)}


def test_apposition_tags_a_name_the_model_mislabels():
    text = "No sé si mañana el ministro Cuerpo hará alguna rectificación."
    assert not any("Cuerpo" in s for s in _ner().person_spans(text))
    spans = _ner(office_surfaces=_OFFICES).person_spans(text)
    # the name alone: the role word is nobody's name and the span is the highlight target
    assert "Cuerpo" in spans


def test_apposition_keeps_the_courtesy_word_it_finds():
    spans = _ner(office_surfaces=_OFFICES).person_spans(
        "Aquí está el presidente señor Sánchez, que interviene ahora.")
    assert any(s == "señor Sánchez" for s in spans)


def test_apposition_does_not_tag_an_ordinary_word():
    # Same surname, no office claimed for it: "cuerpo" stays an ordinary word.
    spans = _ner(office_surfaces=_OFFICES).person_spans(
        "El cuerpo del texto no dice nada.")
    assert not any("uerpo" in s for s in spans)


def test_apposition_needs_the_office_to_agree():
    # Carlos Cuerpo is a minister, not a president, so this phrase names somebody else —
    # and the model gives no PER span here, which leaves the gate as the only decider.
    spans = _ner(office_surfaces=_OFFICES).person_spans(
        "Lo dijo el presidente Cuerpo en su comparecencia.")
    assert not any("uerpo" in s for s in spans)


def test_the_office_gate_is_what_neutralises_the_vocative():
    # "Gracias, presidenta" is the commonest role word in the corpus (151 captures per 800
    # speeches) and names nobody. The patterns DO capture it — what makes it inert is that
    # no office holder is called "Gracias".
    from qhld_ai.domain.mentions import role_appositions

    text = "Gracias, presidenta."
    assert [name for _s, _e, name, _f in role_appositions(text)] == ["Gracias"]
    assert _ner(office_surfaces=_OFFICES).person_spans(text) == []


def test_apposition_never_overrides_a_model_person_span():
    # The model already tagged this one as PER; the pass must not double it.
    text = "Lo dijo el ministro Óscar Puente en la comisión."
    ner = _ner(office_surfaces={"puente": ("ministro",)})
    assert sum("Puente" in s for s in ner.person_spans(text)) == 1


def test_a_leading_role_word_is_trimmed_from_the_model_span():
    # The model does fold the role word in ("ministro Torres"); the span keeps the name,
    # for the same reason the article is trimmed.
    for span in _ner().person_spans(
            "Lo ha hecho el ministro Torres ante la comisión de esta Cámara."):
        assert not span.lower().startswith("ministro")


def test_entity_spans_excludes_an_apposed_name():
    # The wrong-label span is a person, so it must stop being a named entity — this is
    # what put bare surnames in the theme filter.
    text = "El ministro Cuerpo habló sobre Eurovisión."
    ner = _ner(office_surfaces=_OFFICES)
    assert "Cuerpo" in ner.person_spans(text)
    entities = ner.entity_spans(text)
    assert not any("Cuerpo" in s for s in entities)
    assert any("Eurovisión" in s for s in entities)


def test_no_office_map_leaves_the_adapter_as_it_was():
    text = "No sé si mañana el ministro Cuerpo hará alguna rectificación."
    assert _ner().person_spans(text) == _ner(office_surfaces={}).person_spans(text)


# --- courtesy forms: the names the model drops -------------------------------
# "señor Cuerpo" is how one speech addresses the finance minister fourteen times, and the
# model calls it an ORG every time. The courtesy word is the evidence, so this pass needs
# no catalog gate — whoever follows it is a person, catalogued or not.

def _off():
    from qhld_ai.infrastructure.config.settings import Settings
    from qhld_ai.infrastructure.ner.factory import create_ner_from_env

    return create_ner_from_env(Settings(ner_courtesy_form=False))


def test_courtesy_form_tags_a_name_the_model_mislabels():
    text = "Entro en materia, señor Cuerpo. Hace unos días lo advirtió el Eurogrupo."
    assert not any("Cuerpo" in s for s in _off().person_spans(text))
    # the courtesy word comes along, as in every other span: it carries the gender cue
    assert "señor Cuerpo" in _ner().person_spans(text)


def test_courtesy_form_does_not_read_across_a_sentence():
    # "señor." ends a sentence; what follows is a verb, not a name. Only the abbreviated
    # forms may be read across a full stop, because there it belongs to the word.
    text = "Aquí no entra cualquier cosa. No, señor. Están los puestos de inspección."
    assert _ner().person_spans(text) == _off().person_spans(text)
    assert any("Vallugera" in s for s in _ner(gazetteer=[]).person_spans(
        "Lo dijo la Sra. Vallugera en su intervención."))


def test_courtesy_form_never_doubles_a_name_already_tagged():
    # The model tags this one itself and the gazetteer would rescue it otherwise; either
    # way "la señora Vallugera" is one mention, not two overlapping spans.
    text = "Ha intervenido la señora Vallugera y después habló Pedro Sánchez."
    ner = _ner(gazetteer=["Vallugera"])
    assert sum("Vallugera" in s for s in ner.person_spans(text)) == 1
    assert sum("Sánchez" in s for s in ner.person_spans(text)) == 1


def test_courtesy_form_keeps_a_surname_written_with_particles():
    # "señora Álvarez" alone is three deputies and gets dropped as ambiguous; the particle
    # is what makes it one person.
    spans = _ner().person_spans("Estoy deseando, señora Álvarez de Toledo, que hablemos.")
    assert "señora Álvarez de Toledo" in spans


def test_courtesy_form_does_not_join_two_people():
    # "y" is not a name particle: these are two spans, whoever the model tags itself.
    for span in _ner().person_spans(
            "Discutieron el señor Feijóo y la señora Ayuso sobre la financiación."):
        assert not ("Feijóo" in span and "Ayuso" in span)


def test_entity_spans_excludes_a_courtesy_named_surname():
    # The whole point of the pass on the entity side: the bare surname stops being a
    # named entity, which is what kept "cuerpo" in the theme filter.
    text = "Entro en materia, señor Cuerpo. España participa en Eurovisión este año."
    assert any("Cuerpo" in s for s in _off().entity_spans(text))
    entities = _ner().entity_spans(text)
    assert not any("Cuerpo" in s for s in entities)
    assert any("Eurovisión" in s for s in entities)
