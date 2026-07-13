"""Stenographer annotations: strip them from speech text and mine interruptions.

The Diario de Sesiones intersperses each intervention with parenthesized stage
directions — ``(Aplausos)``, ``(Rumores.―El señor Tellado Filgueira: Ábalos.
Cerdán en la cárcel…)``. They are the stenographers' voice, not the speaker's:
running NER over them credits the speaker with "mentioning" whoever interjected
from the floor (or whoever the interjector named). A corpus survey found the
parentheses are used for annotations essentially exclusively (the residue is
acronyms — "(PNV)", "(sic)" — which carry no person names), so
``strip_annotations`` removes them all before mention NER.

The annotations are themselves valuable: only floor activity notable enough to be
minuted lands in the Diario, so they double as a disruption record.
``parse_utterances`` recovers each interjection — who, and either the *quote*
(``El señor <Name>: <quote>``, also from unidentified interrupters: ``Un señor
diputado: ¡Sí, hombre!``) or the *reaction* the stenographer described (``Risas
del señor X``, ``hace signos negativos``, ``pronuncia palabras que no se
perciben``). ``resolve_interruptions`` then builds ``Interruption`` records: the
interrupter resolved against the person catalog, and the people named *inside*
their quotes resolved with the same NER + resolver used for mentions.

Pure like ``domain.mentions``: NER is injected as a plain ``quote_spans`` callable,
so everything here unit-tests offline.
"""

import re
from dataclasses import dataclass

from tipi_data.models.speech import Interruption

from qhld_ai.domain.mentions import match_person, resolve_mentions

_ANNOTATION_RE = re.compile(r"\(([^()]*)\)")
_SPACE_RE = re.compile(r"[ \t]{2,}")

# Annotations chain several directions with a dash: "Rumores.―Aplausos",
# "Aplausos.-El señor…". The horizontal bar / em dash always separates; a plain
# hyphen only after sentence punctuation, so hyphenated names ("Pérez-Castejón")
# and sigla ("EAJ-PNV") survive.
_SEGMENT_SPLIT_RE = re.compile(r"\s*[―—]\s*|(?<=[.…!?])\s*-\s*")

# "El señor <Name>…" — the determiner is case-insensitive (a chained segment can
# open lowercase: "aplausos.―el señor Tellado Filgueira: …") but the lookahead
# requires a capitalized name token, so office forms ("El señor vicepresidente,
# Rodríguez Gómez de Celis, ocupa la Presidencia" — a chair change, not an
# interruption) don't parse as a named interjection.
_ACTOR_PREFIX_RE = re.compile(
    r"^(?P<det>[Ee]l señor|[Ll]a señora|[Ll]os señores|[Ll]as señoras)"
    r"\s+(?=[A-ZÁÉÍÓÚÜÑ])")
# Lowercase tokens allowed inside a proper name ("Álvarez de Toledo Peralta-Ramos");
# "y" also chains two interrupters after a plural determiner.
_NAME_PARTICLES = {"de", "del", "la", "las", "los", "y", "e", "da", "dos", "san", "santa"}
_CAP_TOKEN_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][\w'’-]*$", re.UNICODE)

# Reaction-first wording: "Risas del señor Tellado Filgueira", "Denegaciones de la
# señora ministra de …, Saiz Delgado". Singular only — the plural forms ("de las
# señoras y los señores diputados") are whole-bench applause/protest, too routine
# to record. No capital required after: office tails resolve fuzzily.
_REACTION_OF_RE = re.compile(r"\b(?:del señor|de la señora)\s+", re.IGNORECASE)

# Gate for unidentified verbal interjections: the pre-colon part must read as the
# transcript's label for someone in the chamber ("Un señor diputado", "Varios
# Diputados del Grupo…"), not any sentence that happens to contain a colon
# ("Desde el teléfono móvil se escucha: …").
_COLLECTIVE_GATE_RE = re.compile(
    r"^(?:un|una|unos|unas|varios|varias|algunos|algunas|el|la|los|las)\b"
    r".*?\b(?:señor|diputad)", re.IGNORECASE)
_COLLECTIVE_LABEL_TOKENS = {
    "señor", "señores", "señora", "señoras",
    "diputado", "diputada", "diputados", "diputadas",
    "y", "e", "el", "la", "los", "las", "un", "una",
}
_CONNECTOR_TOKENS = {"y", "e", "el", "la", "los", "las", "un", "una"}


def strip_annotations(text: str) -> str:
    """The speech text with every parenthesized stage direction removed — what
    mention NER should see."""
    return _SPACE_RE.sub(" ", _ANNOTATION_RE.sub(" ", text or "")).strip()


def extract_annotations(text: str) -> list[str]:
    """The inner text of every parenthesized stage direction, in order."""
    return [m.group(1).strip() for m in _ANNOTATION_RE.finditer(text or "")]


@dataclass(frozen=True)
class Utterance:
    """One interjection recovered from an annotation. ``speaker`` is the surface to
    resolve against the person catalog; ``label`` is what to call the interrupter
    if resolution fails. Verbal interjections carry ``quote`` (what they said);
    recorded reactions carry ``reaction`` (the stenographer's description — "Risas",
    "hace signos negativos", "pronuncia palabras que no se perciben").
    ``drop_unresolved`` marks reaction-first wording, whose tail must resolve to a
    person to count (an unresolvable tail is a group/office, not an interrupter)."""

    speaker: str
    label: str
    quote: str | None = None
    reaction: str | None = None
    drop_unresolved: bool = False


def _name_span(text: str) -> tuple[str, str]:
    """Greedy leading proper-name span ("Álvarez de Toledo Peralta-Ramos dándose
    palmadas" → "Álvarez de Toledo Peralta-Ramos") and the remainder after it."""
    tokens = text.split()
    taken: list[str] = []
    pending: list[str] = []  # particles held until a capitalized token confirms them
    consumed = 0
    for i, token in enumerate(tokens):
        clean = token.rstrip(",;.")
        if _CAP_TOKEN_RE.match(clean):
            taken.extend(pending)
            pending = []
            taken.append(clean)
            consumed = i + 1
            if clean != token:  # trailing punctuation closes the name
                break
        elif clean.lower() in _NAME_PARTICLES and taken:
            pending.append(clean)
        else:
            break
    return " ".join(taken), " ".join(tokens[consumed:])


def _collective_label(pre: str) -> str:
    """The transcript's noun-phrase label for an unidentified interrupter: leading
    determiner plus señor/diputado tokens ("Una señora diputada canturrea, …" →
    "Una señora diputada", "Las señoras y los señores diputados del Grupo…" →
    "Las señoras y los señores diputados"). Office wording whose noun isn't in the
    whitelist ("El señor ministro de …") falls back to the first comma piece."""
    tokens = pre.split()
    kept = tokens[:1]
    for token in tokens[1:]:
        if token.rstrip(",").lower() not in _COLLECTIVE_LABEL_TOKENS:
            break
        kept.append(token.rstrip(","))
    while kept and kept[-1].lower() in _CONNECTOR_TOKENS:
        kept.pop()
    if len(kept) < 3:  # bare "El señor"-style stub: the noun phrase went elsewhere
        return pre.partition(",")[0].strip()
    return " ".join(kept)


def _actor_first(segment: str) -> list[Utterance]:
    """Utterances for an "El señor <Name>…" segment: a quote after a colon, else
    the action tail as a reaction ("se da palmaditas en la mejilla…")."""
    prefix = _ACTOR_PREFIX_RE.match(segment)
    if not prefix:
        return []
    rest = segment[prefix.end():]
    pre, _, quote = rest.partition(":")
    name, action = _name_span(pre.strip())
    if not name:
        return []
    quote = quote.strip() or None
    reaction = None if quote else (action.strip() or None)
    if not quote and not reaction:
        return []
    # A plural determiner chains interrupters: "Los señores Bravo Baena y
    # Hernando Fraile pronuncian palabras que no se perciben".
    names = ([n.strip() for n in re.split(r"\s+y\s+", name) if n.strip()]
             if prefix.group("det").lower() in ("los señores", "las señoras")
             else [name])
    return [Utterance(speaker=n, label=n, quote=quote, reaction=reaction)
            for n in names]


def _reaction_first(segment: str) -> list[Utterance]:
    """Utterance for "<Reacción> del señor / de la señora <person>" wording
    ("Risas del señor Tellado Filgueira" → reaction "Risas")."""
    match = _REACTION_OF_RE.search(segment)
    if not match:
        return []
    reaction = segment[:match.start()].strip().rstrip(",")
    tail = segment[match.end():].strip()
    if not reaction or not tail or ":" in segment:
        return []
    return [Utterance(speaker=tail, label=tail.partition(",")[0].strip(),
                      reaction=reaction, drop_unresolved=True)]


def parse_utterances(annotation: str) -> list[Utterance]:
    """Every interjection in one annotation's inner text. Segments that are pure
    stage direction (Aplausos, Rumores, the speaker's own gestures, whole-bench
    reactions…) yield nothing."""
    utterances: list[Utterance] = []
    for segment in _SEGMENT_SPLIT_RE.split(annotation or ""):
        segment = segment.strip().rstrip(".")
        if not segment:
            continue
        parsed = _actor_first(segment) or _reaction_first(segment)
        if parsed:
            utterances.extend(parsed)
            continue
        pre, has_quote, quote = segment.partition(":")
        pre = pre.strip()
        if has_quote and quote.strip() and _COLLECTIVE_GATE_RE.match(pre):
            utterances.append(Utterance(
                speaker=pre, label=_collective_label(pre), quote=quote.strip()))
    return utterances


def _resolve_surface(surface: str, index, threshold: int):
    """The catalog person an interrupter surface names, or ``None``. Tries the
    whole surface first, then its comma pieces — office wording buries the name
    behind tokens the fuzzy score can't absorb ("la señora vicepresidenta primera
    y ministra de Hacienda, Montero Cuadrado" only resolves as "Montero
    Cuadrado")."""
    entry = match_person(surface, index, threshold).entry
    if entry is not None:
        return entry
    pieces = [p.strip() for p in surface.split(",") if p.strip()]
    if len(pieces) < 2:
        return None
    for piece in pieces:
        entry = match_person(piece, index, threshold).entry
        if entry is not None:
            return entry
    return None


def resolve_interruptions(
        utterances, index, threshold: int, quote_spans,
        excluded_surnames: frozenset[str] = frozenset(),
        speaker_name: str | None = None) -> list[Interruption]:
    """Collapse parsed ``Utterance``s into ``Interruption`` records.

    Each interrupter surface is resolved against the person catalog with the same
    fuzzy match used for mentions (plus a comma-piece fallback for office wording,
    see ``_resolve_surface``); unresolved ones (collective labels,
    out-of-catalog names) group under their transcript label with
    ``person_id=None``, except ``drop_unresolved`` utterances, which are discarded.
    Interjections resolving to ``speaker_name`` are dropped too — the speech-closing
    "aplausos del señor presidente…, Sánchez Pérez-Castejón" is the speaker, not an
    interrupter. The people a group named while interrupting come from running
    ``quote_spans`` (the injected NER) over each quote and resolving with
    ``resolve_mentions`` — so an interruption's ``mentions`` behave exactly like
    speech mentions."""
    speaker = (match_person(speaker_name, index, threshold).entry
               if speaker_name else None)
    groups: dict[str, dict] = {}
    for utt in utterances:
        entry = _resolve_surface(utt.speaker, index, threshold)
        if entry is None and utt.drop_unresolved:
            continue
        if speaker is not None and entry is not None \
                and entry.person_id == speaker.person_id:
            continue
        key = entry.person_id if entry else f"label:{utt.label.lower()}"
        acc = groups.setdefault(key, {
            "person_id": entry.person_id if entry else None,
            "person_type": entry.person_type if entry else None,
            "name": entry.name if entry else utt.label,
            "surface_forms": set(), "count": 0, "quotes": [], "reactions": []})
        acc["surface_forms"].add(utt.speaker.strip())
        acc["count"] += 1
        if utt.quote:
            acc["quotes"].append(utt.quote)
        if utt.reaction:
            acc["reactions"].append(utt.reaction)

    interruptions = [
        Interruption(
            person_id=acc["person_id"],
            person_type=acc["person_type"],
            name=acc["name"],
            surface_forms=sorted(acc["surface_forms"]),
            count=acc["count"],
            quotes=acc["quotes"],
            reactions=acc["reactions"],
            mentions=resolve_mentions(
                [span for quote in acc["quotes"] for span in quote_spans(quote)],
                index, threshold, excluded_surnames))
        for acc in groups.values()
    ]
    interruptions.sort(key=lambda i: (-i.count, i.name))
    return interruptions
