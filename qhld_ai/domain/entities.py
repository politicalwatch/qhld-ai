"""Aggregate raw NER non-person spans into named entities — pure, no I/O.

Unlike person mentions (``domain.mentions``), non-person entities — organizations
("Navantia"), events ("Eurovisión"), laws ("ley de amnistía"), conflicts ("guerra
de Gaza"), places a speech talks about — have no catalog to resolve against: the
identity IS the normalized text. ``normalize_entity`` produces that canonical
key; search filters match payload keys against query keys, so the same function
must run on both sides (tagging in the engine, query resolution in
``application.search.resolve_entities``).

The spaCy model spreads these spans erratically across its ORG/LOC/MISC labels
("Eurovisión" ORG, "Festival de Eurovisión" LOC, "guerra de Gaza" MISC), so all
non-person labels are pooled and the label is never kept.

``STOP_ENTITIES`` drops the parliamentary furniture named in virtually every
speech: as filter values they would select everything (a silently useless
constraint), and replicated per chunk they only bloat the payload. Other noise is
deliberately kept — a junk key in the payload is inert, since a filter only ever
fires on what a user actually asked for.
"""

import re
import unicodedata

from tipi_data.models.speech import NamedEntity

# Leading articles/prepositions spaCy often folds into a span ("la guerra de
# Gaza", "del Sáhara Occidental"). Only LEADING tokens are stripped: internal
# particles are part of the name ("guerra de gaza").
_LEADING_STOP = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "a", "al", "de", "del", "en",
}
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
# 2, not the 3 of person spans: two-letter initialisms are real referents here
# ("la UE") and must survive on both the tagging and the query side.
_MIN_LEN = 2

# Normalized keys named in virtually every sitting — chamber, state and
# procedure furniture. Curated editorial data, like the group alias file; revisit
# against a corpus frequency dump when tagging the full corpus. Deliberately NOT
# stoplisted: real recurring referents a user may filter by ("unión europea",
# party names — party queries route to ``groups_or_parties`` anyway).
STOP_ENTITIES = frozenset({
    "gobierno", "gobierno de espana", "camara", "congreso",
    "congreso de los diputados", "senado", "cortes", "cortes generales",
    "estado", "espana", "pleno", "mesa", "comision", "diario de sesiones",
    "constitucion", "constitucion espanola", "boletin oficial del estado",
    "presidencia", "legislatura", "senoria", "senorias",
})


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def normalize_entity(span: str) -> str:
    """Canonical entity key: lowercase, unaccent, drop punctuation, strip leading
    articles/prepositions, collapse whitespace — "la guerra de Gaza" →
    "guerra de gaza". Returns "" when nothing usable remains (too short,
    digits-only, or parliamentary furniture)."""
    cleaned = _PUNCT_RE.sub(" ", _strip_accents(span.lower()))
    tokens = cleaned.split()
    while tokens and tokens[0] in _LEADING_STOP:
        tokens.pop(0)
    key = " ".join(tokens)
    if len(key) < _MIN_LEN or key.isdigit() or key in STOP_ENTITIES:
        return ""
    return key


def aggregate_entities(spans: list[str]) -> list[NamedEntity]:
    """Collapse raw non-person NER spans into one ``NamedEntity`` per canonical
    key (spans normalizing to "" are dropped), collecting the distinct surface
    forms and their total count. Ordered by descending count then key, mirroring
    ``resolve_mentions``."""
    by_key: dict[str, dict] = {}
    for span in spans:
        key = normalize_entity(span)
        if not key:
            continue
        acc = by_key.setdefault(key, {"surface_forms": set(), "count": 0})
        acc["surface_forms"].add(span.strip())
        acc["count"] += 1
    entities = [
        NamedEntity(key=key,
                    surface_forms=sorted(acc["surface_forms"]),
                    count=acc["count"])
        for key, acc in by_key.items()
    ]
    entities.sort(key=lambda e: (-e.count, e.key))
    return entities
