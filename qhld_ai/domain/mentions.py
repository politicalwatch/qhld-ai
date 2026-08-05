"""Resolve raw NER person-spans to canonical people — pure, no I/O.

NER (``NerPort``) yields person
spans verbatim from a speech's Spanish text ("Sánchez", "el señor Sánchez",
"Pedro Sánchez"); this module normalizes them and fuzzy-matches each against a
person catalog (sitting deputies plus non-deputies — government ministers, the
King, regional presidents, foreign leaders), collapsing the many surface forms of
one person into a single ``Mention`` with an occurrence count.

The catalog is a flat list of ``PersonEntry`` — deputies and non-deputies scored
in ONE pass. A few non-deputies share a surname with a deputy ("Clavijo" = the
Canarias president vs the deputy Gamarra Ruiz-Clavijo's second surname); such
entries carry ``overrides_deputy`` so that, when tied with a deputy, they win.

Matching reuses the ``thefuzz.token_set_ratio`` + high-threshold trick the query
``EntityResolver`` relies on: ``token_set_ratio`` scores a subset match ~100
("sánchez" ⊆ "sánchez pérez-castejón, pedro") while unrelated names stay low, so
a surname alone resolves but noise does not. Because bare surnames collide across
deputies ("García"), an **ambiguity guard** drops any span whose top score is
shared by two or more deputies — favouring a missed mention over a wrong one.

Three cues recover what that guard would otherwise throw away, and all read the
whole speech rather than one span: the gender a courtesy form implies ("la señora
Muñoz" cannot be the male Muñoz), the office a role apposition names somebody by
("el presidente Sánchez" is the prime minister, not a deputy who shares the
surname) and, for what is still tied afterwards, the one tied person the speech
names elsewhere in full.

Kept pure (takes a prebuilt index, returns ``Mention`` objects) so it is unit-
testable offline with no Mongo, mirroring ``domain.speeches.segmentation``.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from thefuzz import fuzz

from tipi_data.models.speech import Mention

# Courtesy honorifics and articles spaCy often folds into a PER span. Stripped
# before matching so "el señor Sánchez" resolves like "Sánchez". Role words are
# stripped too, but they live with the office vocabulary below (see ``_STRIPPED``).
_HONORIFICS = {
    "el", "la", "los", "las", "un", "una", "al",
    "sr", "sra", "srs", "sras", "señor", "señora", "señores", "señoras",
    "don", "doña", "su", "señoria", "señorias", "señoría", "señorías",
    "excelentisimo", "excelentisima", "excelentísimo", "excelentísima",
    "ilustrisimo", "ilustrisima", "ilustrísimo", "ilustrísima",
}
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_MIN_LEN = 3
# Words a Spanish surname can open with that name nobody on their own. Only the ones
# long enough to survive ``_tokens`` need listing — "de", "la", "el", "y" are already
# below _MIN_LEN and drop out by themselves.
_NAME_PARTICLES = frozenset({"del", "las", "los", "van", "der", "dos", "das"})

# The gendered half of the same courtesy vocabulary. ``normalize_span`` throws these
# away before matching, but the courtesy form agrees in gender with the person named,
# so read it off the RAW span first: "la señora Muñoz" cannot be the male Muñoz. Values
# are the person catalog's own ("Hombre"/"Mujer") so no translation layer is needed.
_FEMININE_CUE = re.compile(r"\b(?:se[nñ]ora|se[nñ]oras|sra|sras|do[nñ]a)\b", re.I)
_MASCULINE_CUE = re.compile(r"\b(?:se[nñ]or|se[nñ]ores|sr|srs|don)\b", re.I)


# The offices a speech can name somebody by ("el presidente Sánchez"), mapped to the
# family the catalog records them under. Both grammatical genders map to one family, and
# tokens are matched WHOLE: "vicepresidenta" must not feed the "presidente" family, since
# they are different people. Kept to the offices the corpus actually records for a
# speaker, because an office nobody holds in the data can never be evidence.
_ROLE_FAMILIES = {
    "presidente": "presidente", "presidenta": "presidente",
    "vicepresidente": "vicepresidente", "vicepresidenta": "vicepresidente",
    "ministro": "ministro", "ministra": "ministro",
}
_ROLE_HEAD = "|".join(sorted(_ROLE_FAMILIES, key=len, reverse=True))
# "Vicepresidenta Primera del Gobierno", "Ministro de Política Territorial y Memoria
# Democrática" — the ordinal and the office complement that can stand between the role
# word and the name. The complement is bounded to six words so it cannot run off into the
# rest of the sentence, and its words may be comma-separated because real offices are
# ("Ministro de Derechos Sociales, Consumo y Agenda 2030"); the comma that ends the
# apposition is matched by the caller, so the longest complement is tried first.
_ORDINAL = r"(?:\s+(?:primer[oa]|segund[oa]|tercer[oa]|cuart[oa]))?"
_COMPLEMENT = r"(?:\s+(?:de|del|de\s+la|para|en)\s+[\w'’-]+(?:,?\s+[\w'’-]+){0,5})"
# What may sit between the role word and the name without breaking the apposition: the
# article and the courtesy form, and nothing else. No comma and no sentence boundary can
# appear here, which is what keeps the vocative apart from an apposition — "señor
# presidente, la señora Montero" addresses the chair and then names somebody else.
_LINK = r"\s+(?:(?:el|la)\s+)?(?:se[nñ]or(?:a)?|don|do[nñ]a)?\s*"
_NAMED = r"([A-ZÁÉÍÓÚÑ][\w'’-]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][\w'’-]{2,}){0,2})"
# "(el) presidente Sánchez", "el señor ministro Torres"
_ROLE_BEFORE = re.compile(rf"\b({_ROLE_HEAD})\b{_ORDINAL}{_LINK}{_NAMED}")
# "la ministra de Vivienda, Isabel Rodríguez" — a comma is allowed only when the role
# word carries its office complement, which is what a vocative never does.
_ROLE_APPOSED = re.compile(
    rf"\b({_ROLE_HEAD})\b{_ORDINAL}{_COMPLEMENT},\s*"
    r"(?:(?:el|la)\s+)?(?:se[nñ]or(?:a)?|don|do[nñ]a)?\s*" + _NAMED)
# "el señor Sánchez, presidente del Gobierno"
_ROLE_AFTER = re.compile(rf"{_NAMED}\s*[,\-–—]\s*(?:(?:el|la)\s+)?\b({_ROLE_HEAD})\b")

# Everything ``normalize_span`` throws away: courtesy forms, articles, and the role words
# above. A role word inside the span is not free — the model does fold one in
# ("ministro Torres"), and the extra token drops `token_set_ratio` to 71 against "Torres
# Pérez, Ángel Víctor", i.e. below any workable threshold, so the mention was silently
# lost. Measured over 800 speeches: 51 spans carry a role word, stripping resolves 4 that
# failed, empties 46 that name nobody at all ("Señora ministra"), and changes no other
# outcome. Nothing is lost by dropping the word here, because the office it names is read
# off the surrounding TEXT (see ``role_cues``), not off the span.
_STRIPPED = _HONORIFICS | set(_ROLE_FAMILIES)


def span_gender(span: str) -> str | None:
    """The gender the span's courtesy form implies, or ``None`` when it carries none (a
    bare "Muñoz") or is self-contradictory. Feminine is tested first because "señora"
    contains "señor" as a prefix, not as a word — the word boundaries keep them apart,
    but the order makes that independent of the regex engine's behaviour."""
    feminine = bool(_FEMININE_CUE.search(span or ""))
    masculine = bool(_MASCULINE_CUE.search(span or ""))
    if feminine == masculine:  # neither, or a span carrying both
        return None
    return "Mujer" if feminine else "Hombre"

# Common words spaCy sometimes tags as a person because they are also borne as a
# surname (typically sentence-initial "Bueno, …"). They never name a real person.
# (Famous non-deputies who share a surname with a deputy — Aznar, Suárez, Clavijo —
# are no longer dropped here: they now resolve to their own catalog entry, which
# wins the tie via ``overrides_deputy``.)
COMMON_WORD_SURNAMES = frozenset({"bueno"})


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens, split on whitespace and hyphens, keeping those long
    enough to discriminate (so "de"/"la" are ignored)."""
    return {t for t in re.split(r"[\s-]+", text.lower()) if len(t) >= _MIN_LEN}


def first_surname_tokens(name: str, *, skip_particles: bool = False) -> set[str]:
    """The tokens of a deputy's FIRST surname (the first whitespace element before the
    comma; hyphenated compounds like "Grande-Marlaska" split into both parts) — the
    core of their identity, used to tell a real mention from a homonym.

    ``skip_particles`` steps over the nobiliary/locative words a surname can open with,
    so "Del Valle Rodríguez" yields ``valle`` and "De los Santos González" yields
    ``santos`` instead of ``del`` and nothing at all. Callers that resolve a *query*
    want this: someone typing "Valle" means Del Valle, and the particle is not the part
    of the name anybody says.

    Off by default because the mentions path is calibrated without it and turning it on
    there moves span resolution — a change that has to be scored against the gold set,
    not assumed. See ``resolve_entities._first_surname_match`` for the query side."""
    surname_part = name.partition(",")[0]
    words = surname_part.split()
    if skip_particles:
        words = [w for w in words
                 if w.lower() not in _NAME_PARTICLES and len(w) >= _MIN_LEN] or words
    first = words[0] if words else ""
    return _tokens(first)


def _given_name_tokens(name: str) -> set[str]:
    """The tokens of the given name — everything after the comma in "Apellidos, Nombre",
    and nothing at all for a name written without one ("Felipe VI").

    Used to tell a span that identifies somebody from one that merely shares their
    surname, which is what decides whether a contradicting courtesy form is evidence or
    noise (see ``_gender_vetoes``)."""
    return _tokens(name.partition(",")[2])


# Offices a sitting deputy would not simultaneously hold: when the text introduces a
# person by one of these, the person named is not the deputy who happens to share the
# surname. Scanned per speech so the exclusion is scoped to that speech (a cue anywhere
# in it disqualifies the surname throughout — e.g. "expresidente Aznar" ⇒ every "Aznar").
_CONTEXT_CUE_PATTERNS = (
    # "<Apellido>, (actual) magistrado" / "magistrado(a) <Apellido>"
    r"([a-zá-úñ]+),?\s+(?:actual\s+)?magistrad[oa]",
    r"magistrad[oa]\s+(?:del?\s+\w+\s+)*([a-zá-úñ]+)",
    # "(el) juez/jueza <Apellido>", "(el) fiscal (general) <Apellido>"
    r"jueza?\s+([a-zá-úñ]+)",
    r"fiscal(?:\s+general)?\s+([a-zá-úñ]+)",
    # NB: no "expresidente(a) del Gobierno <Apellido>" cue — its only job was to keep
    # an ex-PM (Aznar, Zapatero) from resolving to a colliding deputy; the ex-PMs now
    # live in the person catalog and resolve there instead of being dropped.
    # NB: no dictatorship cue either — it used to drop "Franco" in speeches invoking the
    # dictatorship, which only ever deleted the wrong answer. Measured over the corpus it
    # was also leaky in both directions: it fired in 38 speeches, yet in none of the 30
    # that credited the deputy surnamed Franco with a mention — 29 of which were plainly
    # the dictator ("exhumar a Franco", "con Franco vivíamos mejor"). He is now a catalog
    # person who WINS the surname outright, so the reference resolves instead of vanishing.
)


def context_excluded_surnames(text: str) -> frozenset[str]:
    """Surnames the speech's own wording marks as non-deputies (a magistrate, judge or
    prosecutor). Speech-scoped.

    Still worth keeping after the judiciary entered the catalog, because it is what
    protects a deputy from an UNCURATED homonym on the bench; a curated judge needs no
    protecting, since ``resolve_mentions`` applies the exclusion to deputy resolutions
    only."""
    low = (text or "").lower()
    excluded = set()
    for pattern in _CONTEXT_CUE_PATTERNS:
        for match in re.finditer(pattern, low):
            excluded.add(match.group(1))
    return frozenset(excluded)


@dataclass(frozen=True)
class PersonEntry:
    """One person's canonical identity plus the lowercased keys a span is scored
    against (full "apellido, nombre", "nombre apellido", the bare surname, and any
    explicit aliases). ``person_type`` is ``"deputy"`` for catalog deputies, else the
    non-deputy kind. ``overrides_deputy`` marks a non-deputy who should win a tie
    against a deputy sharing the surname (e.g. "Clavijo").

    ``gender`` ("Hombre"/"Mujer", the catalog's own vocabulary) lets a gendered courtesy
    form in the span rule this person out: "la señora Muñoz" cannot be the male Muñoz.
    ``None`` means unknown and never rules anybody out — see ``_gender_conflicts``.

    ``offices`` are the role families this person holds ("presidente", "ministro"), so a
    speech that names somebody by their title — "el presidente Sánchez" — resolves to the
    holder rather than being dropped as an ambiguous surname. Empty means no office is
    recorded, which is the case for most people and never rules anybody out."""
    person_id: str
    person_type: str
    name: str
    keys: tuple[str, ...]
    overrides_deputy: bool = False
    gender: str | None = None
    offices: tuple[str, ...] = ()


def _gender_conflicts(cue: str | None, entry: PersonEntry) -> bool:
    """Whether ``cue`` rules this person out. Unknown on either side never rules anyone
    out — the catalog has no gender for most bootstrapped speakers, and treating that as
    a mismatch would drop resolutions that work today."""
    return bool(cue) and bool(entry.gender) and cue != entry.gender


def _narrow_by_gender(cue: str | None, tied: list[PersonEntry]) -> list[PersonEntry]:
    """Drop candidates the courtesy form contradicts, unless that would leave nothing —
    an all-conflicting tie means the cue or the catalog is wrong, so keep the tie intact
    and let the ambiguity guard have the last word rather than inventing a winner."""
    kept = [e for e in tied if not _gender_conflicts(cue, e)]
    return kept if kept else tied


def _gender_vetoes(cue: str | None, norm: str, entry: PersonEntry) -> bool:
    """Whether the courtesy form rules out the person the tie-break chain ended on.

    Narrowing (above) can only choose between candidates; when the catalog holds exactly
    ONE bearer of the surname it has nothing to choose from and hands back the person the
    form contradicts. That is how "el señor Marcos" became the deputy Milagros Marcos and
    "la señora Caballero" the deputy Francisco Sierra Caballero. Measured over the corpus,
    every such resolution names somebody outside the catalog, so the honest answer is to
    name nobody — and where that somebody is worth naming, the answer is to curate them,
    which is what removed the dictator Franco from this list.

    It fires only when the span identifies the person by SURNAME alone. A span carrying
    their given name has already identified them beyond doubt ("doña Pedro Sánchez
    Pérez-Castejón" is still the prime minister), so there the courtesy form is a
    transcription oddity rather than evidence — the same reasoning that keeps a bare given
    name out of ``_names_a_surname``, in the other direction."""
    if not _gender_conflicts(cue, entry):
        return False
    return not (_tokens(norm) & _given_name_tokens(entry.name))


def _narrow_by_office(cue: str, tied: list[PersonEntry]) -> list[PersonEntry]:
    """Keep the tied people who HOLD the cued office, unless that would leave nothing.

    Sharper than ``_narrow_by_gender``, which only removes candidates a courtesy form
    contradicts: an office is a positive claim about one person, so the holders are the
    only candidates left when any of them is a holder. When none is, the role word is not
    evidence about anybody here — the person named may be outside the catalog ("la
    ministra Maroto", who no longer sits) or their office simply unrecorded — so the tie
    is handed back untouched and the guard decides as if no role word had been read."""
    holders = [entry for entry in tied if cue in entry.offices]
    return holders if holders else tied


def office_families(role: str | None) -> frozenset[str]:
    """The role families a speaker's official title claims — "Vicepresidenta Primera del
    Gobierno y Ministra de Hacienda" holds both "vicepresidente" and "ministro". Read off
    whole tokens, so a title says nothing about the families it merely contains."""
    words = re.findall(r"[\w'’-]+", (role or "").lower())
    return frozenset(_ROLE_FAMILIES[word] for word in words if word in _ROLE_FAMILIES)


def _names_a_surname(norm: str, tied: list[PersonEntry]) -> bool:
    """Whether the span names a SURNAME of any tied candidate rather than a bare given
    name — the precondition for letting gender decide.

    Given names are the ambiguity guard's most valuable catch: "Pedro", "Laura",
    "Alberto" each tie a dozen people and name none of them, and a courtesy form does
    nothing to say which ("el señor Amador" is Alberto Amador, not the deputy whose GIVEN
    name is Amador). Measured on the gold set: without this, "Alberto" resolves to Fabra
    Part, Alberto — a false positive."""
    span_tokens = _tokens(norm)
    return any(span_tokens & first_surname_tokens(e.name) for e in tied)


def _name_keys(name: str) -> set[str]:
    """Match keys derived from an "Apellido(s), Nombre" name: the whole string, the
    bare first surname group, and the "Nombre Apellido" order."""
    keys = {name.lower(), name.split(",")[0].strip().lower()}
    parts = [p.strip() for p in name.split(",")]
    if len(parts) == 2 and parts[0] and parts[1]:
        keys.add(f"{parts[1]} {parts[0]}".lower())
    return {k for k in keys if k}


def make_person_entry(person_id, person_type, name, aliases=(), overrides_deputy=False,
                      gender=None, offices=()):
    """Build a non-deputy ``PersonEntry``. Keys come from the canonical ``name`` plus
    any ``aliases`` (nicknames, bare surname, role phrases like "su majestad"), each
    run through ``normalize_span`` so they match under the same normalization the
    corpus spans get."""
    keys = _name_keys(name)
    for alias in aliases:
        norm = normalize_span(alias)
        if norm:
            keys.add(norm)
    return PersonEntry(
        person_id=person_id, person_type=person_type, name=name,
        keys=tuple(sorted(keys)), overrides_deputy=overrides_deputy, gender=gender,
        offices=tuple(offices))


def build_deputy_index(deputies, *, aliases=None) -> list[PersonEntry]:
    """Build match entries from ``Deputy`` records (``name`` = 'Apellido, Nombre',
    ``get_fullname()`` = 'Nombre Apellido'). Deputies without a name are skipped.

    ``aliases`` maps a deputy id to the public names the chamber and the press use
    instead of the official one ("Tesh Sidi" for "Andala Ubbi, Teslem") — a curated
    surface the mechanical keys above cannot derive, since it shares no token with
    the catalog name. Each is run through ``normalize_span`` so it matches under the
    same normalization a queried or NER-extracted surface gets.

    Curate WHOLE public names, never a bare token: ``token_set_ratio`` scores a
    subset at 100, so "Tesh Sidi" already resolves "Tesh" and "Sidi" on its own,
    whereas a bare "Sidi" key would also swallow the Moroccan town "Sidi Ifni"."""
    aliases = aliases or {}
    index = []
    for deputy in deputies:
        name = getattr(deputy, "name", None)
        if not name:
            continue
        keys = {name.lower(), name.split(",")[0].strip().lower()}
        try:
            keys.add(deputy.get_fullname().lower())
        except (AttributeError, IndexError):
            pass
        for alias in aliases.get(getattr(deputy, "id", None), ()):
            keys.add(normalize_span(alias))
        index.append(PersonEntry(
            person_id=deputy.id, person_type="deputy", name=name,
            keys=tuple(k for k in keys if k), overrides_deputy=False,
            # getattr, not deputy.gender: callers duck-type this record (the query path
            # and the tests pass their own minimal stand-ins).
            gender=getattr(deputy, "gender", None)))
    return index


def build_person_index(deputies, extra=(), *, aliases=None) -> list[PersonEntry]:
    """The full match index: every deputy plus ``extra`` non-deputy ``PersonEntry``
    rows (curated catalog + corpus-bootstrapped speakers, assembled at the
    application layer). Scored together in one pass by the resolver."""
    return build_deputy_index(deputies, aliases=aliases) + list(extra)


# Connective particles inside a surname group ("Muñoz de la Iglesia", "Gil de
# Biedma") — never a name surface on their own, so never gazetteer candidates.
_SURNAME_PARTICLES = {
    "de", "del", "la", "las", "los", "y", "e", "i", "da", "dos", "san", "santa"}


def build_surname_gazetteer(deputies, *, extra=()) -> list[str]:
    """Distinctive surname surfaces to seed an NER gazetteer, so the model also tags
    the uncommon/compound surnames it otherwise misses. The WHOLE surname group
    contributes tokens — the chamber knows some deputies by their second surname
    ("Feijóo", from "Núñez Feijóo") — and hyphenated compounds contribute each part
    ("Grande-Marlaska" → "Grande", "Marlaska").

    A token qualifies when a bare span of it resolves deterministically: either one
    deputy bears it anywhere in the surname group, or exactly one bears it as the
    FIRST surname (``_break_tie`` awards a bare surname to its first-surname bearer,
    so "Montero" stays distinctive even when others carry it as a second surname).
    Tokens ambiguous even then ("García") are left out: the base model usually
    catches common ones, and they would only add spans the resolver drops anyway.
    Original casing is kept (names are Title-case in the Diario text).

    ``extra`` adds curated surfaces that no surname can produce — the public name a
    deputy is actually called in the chamber ("Tesh" for Andala Ubbi, Teslem). Those go
    in verbatim, bypassing the distinctiveness counting above (a nickname has no
    surname group to count), which is safe because they are hand-curated and still face
    the adapter's out-of-vocabulary gate. They must not NEST with each other or with a
    surname, though: two patterns matching the same text yield two overlapping spans and
    count one written mention twice. Note this list is about TAGGING only — being
    taggable and being a resolution key are separate decisions, curated separately (see
    ``application.persons_catalog``).

    False-positive exposure stays low downstream: the spaCy adapter only turns
    OUT-OF-VOCABULARY terms into entity-ruler patterns (a common word that doubles as
    someone's surname is in-vocabulary and never gets a rule), the patterns are
    case-sensitive, and the ruler never overrides the model's own entities."""
    total: Counter[str] = Counter()
    first: Counter[str] = Counter()
    surface: dict[str, str] = {}

    def _tokens(text):
        return {token for token in re.split(r"[-\s]+", text)
                if len(token) >= _MIN_LEN
                and token.lower() not in _SURNAME_PARTICLES}

    for deputy in deputies:
        name = getattr(deputy, "name", None)
        surname_part = name.partition(",")[0] if name else ""
        first_surname = surname_part.split()[0] if surname_part.split() else ""
        for token in _tokens(surname_part):
            total[token.lower()] += 1
            surface.setdefault(token.lower(), token)
        for token in _tokens(first_surname):
            first[token.lower()] += 1
    for term in extra:
        if term:
            surface.setdefault(term.lower(), term)
            total[term.lower()] = 1  # curated => distinctive by construction
    return sorted(surface[key] for key, count in total.items()
                  if count == 1 or first[key] == 1)


def build_office_surfaces(index: list[PersonEntry]) -> dict[str, tuple[str, ...]]:
    """Surname tokens borne by somebody the catalog records holding an office, mapped to
    the families they hold — what a detector needs to tell "el ministro Cuerpo" (a person)
    from "el cuerpo del texto" (not one).

    Same division of labour as ``build_surname_gazetteer``: the catalog says what may be
    LOOKED FOR, and the resolver still decides who was found. That is why a loose,
    token-level membership test is enough here — a created span that resolves to nobody is
    dropped like any other, so the gate only has to keep the detector away from ordinary
    words.

    Surnames only (the group before the comma), because a given name is not evidence: "el
    ministro Carlos" names nobody in particular, while every minister's given name would
    otherwise let a role word tag one. Distinctiveness is deliberately NOT required, unlike
    the gazetteer's: this detector fires only next to a role word, so the context supplies
    what distinctiveness supplies there — which is exactly why it reaches the common,
    in-vocabulary surnames the gazetteer must leave alone."""
    surfaces: dict[str, set[str]] = {}
    for entry in index:
        if not entry.offices:
            continue
        for token in _tokens(entry.name.partition(",")[0]):
            if token not in _SURNAME_PARTICLES:
                surfaces.setdefault(token, set()).update(entry.offices)
    return {token: tuple(sorted(families)) for token, families in surfaces.items()}


def normalize_span(span: str) -> str:
    """Lowercase, drop punctuation, courtesy honorifics/articles and role words. Returns
    the residual name, or "" when nothing usable remains ("Su Señoría", "Señora
    ministra")."""
    cleaned = _PUNCT_RE.sub(" ", span.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STRIPPED]
    residual = " ".join(tokens)
    return residual if len(residual) >= _MIN_LEN else ""


def _break_tie(norm: str, tied: list[PersonEntry]) -> list[PersonEntry]:
    """Narrow equally-scored people (``token_set_ratio`` gives a bare surname 100
    against everyone who carries it anywhere) to the one the span actually names:

    1. Prefer deputies whose FIRST surname the span matches — a surname resolves to
       whoever bears it first, not to someone who has it as a second surname or given
       name ("Bravo" → Juan Bravo, not Aitor Esteban Bravo).
    2. If several still qualify, prefer the closest exact-order match
       (``token_sort_ratio``), which separates a partial multi-token overlap from the
       real full surname ("Sánchez Pérez-Castejón" → Pedro, not Sánchez Pérez, César).

    Returns the surviving candidates; a still-tied result (e.g. two deputies sharing a
    first surname) is left for the caller to drop as genuinely ambiguous."""
    span_tokens = _tokens(norm)
    first = [e for e in tied if span_tokens & first_surname_tokens(e.name)]
    candidates = first if first else tied
    if len(candidates) == 1:
        return candidates
    # Only a fuller, multi-token reference can separate several deputies who share a
    # first surname; a bare surname borne by many stays ambiguous (caller drops it).
    if len(span_tokens) < 2:
        return candidates
    scored = [(max(fuzz.token_sort_ratio(norm, key) for key in e.keys), e)
              for e in candidates]
    top = max(score for score, _ in scored)
    return [e for score, e in scored if score == top]


@dataclass
class PersonMatch:
    """Outcome of matching one span against the catalog. ``entry`` is the resolved
    person, or ``None`` on failure; ``best_score`` and ``candidates`` then describe the
    failure — the top-scoring near misses when nothing cleared the threshold, or the
    still-tied people when a surname stayed ambiguous. Candidates are full entries, not
    just names, so a caller can carry on reasoning about who they are."""
    entry: PersonEntry | None
    best_score: int = 0
    candidates: list[PersonEntry] = field(default_factory=list)

    @property
    def candidate_names(self) -> list[str]:
        return [entry.name for entry in self.candidates]


def resolve_person(name: str, index: list[PersonEntry], threshold: int) -> PersonEntry | None:
    """Resolve a free-text person name (as typed in a search query, e.g. "Zapatero",
    "María Jesús Montero", "Ayuso") to a catalog person, or ``None`` if it does not clear
    the threshold or stays ambiguous. Runs the span through the SAME normalization + fuzzy
    match + ambiguity guard used to tag the corpus, so a query resolves consistently with
    what was indexed."""
    return match_person(name, index, threshold).entry


def match_person(name: str, index: list[PersonEntry], threshold: int) -> PersonMatch:
    """Like ``resolve_person`` but returns the full ``PersonMatch``, so a caller can
    tell WHY a name failed (out of catalog vs near miss vs ambiguous) and suggest the
    closest candidates."""
    norm = normalize_span(name)
    if not norm:
        return PersonMatch(None)
    return _match_one(norm, index, threshold)


def _match_one(norm: str, index: list[PersonEntry], threshold: int,
               gender=None, office=None, gender_veto: bool = True) -> PersonMatch:
    """``gender`` is the courtesy form's gender read off the RAW span (see
    ``span_gender``), which ``norm`` has already had stripped. ``office`` is the role
    family the speech appositions this surface with (see ``role_cues``). Both narrow an
    ambiguous tie, and the gender cue can additionally VETO the person the narrowing ended
    on (``gender_veto``, see ``_gender_vetoes``). Passing no ``gender`` — as every
    query-side caller does — leaves the outcome exactly as it was before either cue
    existed, veto included."""
    best_score = 0
    tied: list[PersonEntry] = []
    for entry in index:
        score = max(fuzz.token_set_ratio(norm, key) for key in entry.keys)
        if score > best_score:
            best_score, tied = score, [entry]
        elif score == best_score and best_score > 0:
            tied.append(entry)
    if best_score < threshold:
        return PersonMatch(None, best_score, tied)
    if len(tied) > 1 and (gender or office) and _names_a_surname(norm, tied):
        # Before any preference rule: what the text says about who is being named is a
        # hard fact, so it outranks the deputy preference below. Without this, "la señora
        # Rego, ministra de Juventud" ties the minister Sira Rego with the deputy Néstor
        # Rego and the deputy preference hands it to HIM; and "el presidente Sánchez" ties
        # the prime minister with the deputies who share his surname, all of them men, so
        # only the office tells them apart.
        if gender:
            tied = _narrow_by_gender(gender, tied)
        if office and len(tied) > 1:
            tied = _narrow_by_office(office, tied)
    if len(tied) > 1:
        # An override only applies when the span names the override's OWN first surname —
        # not when it merely shares a secondary token (the ex-PM "Aznar López" must not
        # hijack a bare "López" tie via his second surname).
        span_tokens = _tokens(norm)
        named_override = any(
            e.overrides_deputy and (first_surname_tokens(e.name) & span_tokens)
            for e in tied)
        if named_override:
            # A famous non-deputy tied with the deputy who merely shares the surname
            # wins: "Clavijo" is the Canarias president, not the deputy Gamarra
            # Ruiz-Clavijo. But only for a bare-surname reference — the ORDER-sensitive
            # score separates a fuller deputy match ("Gamarra Ruiz-Clavijo" → the deputy)
            # from a plain surname ("Clavijo" → the president).
            tied = _prefer_overrides(norm, tied)
        elif any(e.person_type == "deputy" for e in tied):
            # Deputies are the primary referents in the chamber: a non-deputy (a
            # bootstrapped minister, a non-override curated figure) never blocks or
            # steals a deputy resolution on a shared-surname tie ("Rego" → the deputy
            # Rego Candamil, not the minister Sira Rego). This keeps the deputy metric
            # identical to the deputies-only baseline.
            tied = [e for e in tied if e.person_type == "deputy"]
    if len(tied) > 1:
        tied = _break_tie(norm, tied)
    if len(tied) == 1:
        if gender_veto and _gender_vetoes(gender, norm, tied[0]):
            # Returned as a ONE-element candidate list on purpose: same-speech
            # coreference attaches a surface only when two or more people are still tied
            # for it, so a vetoed person cannot come back through it.
            return PersonMatch(None, best_score, tied)
        return PersonMatch(tied[0], best_score)
    return PersonMatch(None, best_score, tied)


def _prefer_overrides(norm: str, tied: list[PersonEntry]) -> list[PersonEntry]:
    """Resolve an override-vs-deputy tie by order-sensitive match: keep the entries the
    span matches most exactly (``token_sort_ratio``), then, within that group, keep the
    override(s) if any. So "Clavijo"/"Aznar" (bare surname) go to the non-deputy, while
    "Gamarra Ruiz-Clavijo" (the deputy's own full name) stays with the deputy."""
    scored = [(max(fuzz.token_sort_ratio(norm, key) for key in e.keys), e) for e in tied]
    top = max(score for score, _ in scored)
    best = [e for score, e in scored if score == top]
    overrides = [e for e in best if e.overrides_deputy]
    return overrides if overrides else best


def _is_excluded(norm: str, entry: PersonEntry, excluded: frozenset[str]) -> bool:
    """Whether a resolved span actually names a flagged non-deputy rather than the
    deputy it fuzzy-matched. Two cases:

    - *referent-homonym* — the deputy's OWN first surname is flagged (the ex-PM Aznar
      vs the deputy Aznar Teruel): the surname coincides, so drop it.
    - *mismatch* — the span carries a flagged token but resolved via a secondary
      surname (the Canarias president "Clavijo" fuzzy-matching Gamarra Ruiz-Clavijo):
      drop only when the deputy's first surname is absent from the span, so a genuine
      full-name mention ("Gamarra Ruiz-Clavijo") that merely contains the token survives.
    """
    if not excluded:
        return False
    span_tokens = _tokens(norm)
    if not (span_tokens & excluded):
        return False
    first = first_surname_tokens(entry.name)
    if first & excluded:
        return True
    return not (span_tokens & first)


def _gender_cues(spans) -> dict[str, str]:
    """Map each normalized surface to the gender its courtesy forms agree on, over the
    WHOLE speech.

    A speech addresses the same person both ways — "la señora Muñoz" once and a bare
    "Muñoz" thirty times — and ``normalize_span`` collapses those to one surface. Reading
    the cue per occurrence would rescue only the handful that carry the honorific, so the
    evidence is pooled per surface instead: one courtesy form anywhere in the speech
    settles who that surname refers to throughout it.

    Surfaces whose forms disagree are left out entirely — two genders on one surname mean
    it names two different people, which is exactly when nothing should be assumed."""
    seen: dict[str, set[str]] = {}
    for span in spans:
        norm = normalize_span(span)
        cue = span_gender(span) if norm else None
        if cue:
            seen.setdefault(norm, set()).add(cue)
    return {norm: next(iter(cues)) for norm, cues in seen.items() if len(cues) == 1}


def role_appositions(text: str) -> list[tuple[int, int, str, str]]:
    """Every role apposition in ``text`` as ``(start, end, name, family)``, ordered, where
    the offsets are the NAME's — so one caller can pool it as a cue about a span that
    exists (``role_cues``) and another can turn it into a span the model never made (the
    NER adapter's apposition pass).

    Both readings need the same three shapes, which is why they are compiled once here:
    the tight apposition ("el presidente Sánchez"), the comma apposition, which is allowed
    only when the role word carries its office complement because that is what a vocative
    never does ("la ministra de Vivienda, Isabel Rodríguez"), and the postposed form ("el
    señor Sánchez, presidente del Gobierno"). No comma and no sentence boundary may stand
    between role word and name in the tight shape — that alone separates an apposition
    from "señor presidente, la señora Montero", which addresses the chair and then names
    somebody else.

    Everything the shapes over-capture ("Gracias" after "señora presidenta", an office
    complement like "Autoridad Palestina") is left for the caller to reject: a cue must
    name a span the speech already has, and a created span must name somebody the catalog
    records holding that office."""
    found = []
    for pattern, name_group, role_group in (
            (_ROLE_BEFORE, 2, 1), (_ROLE_APPOSED, 2, 1), (_ROLE_AFTER, 1, 2)):
        for match in pattern.finditer(text or ""):
            family = _ROLE_FAMILIES.get(match.group(role_group).lower())
            if family:
                start, end = match.span(name_group)
                found.append((start, end, match.group(name_group), family))
    return sorted(found)


def role_cues(text: str, spans) -> dict[str, str]:
    """Map each normalized surface to the office the speech names it by — "el presidente
    Sánchez" says the Sánchez meant here is the one holding the presidency, which no
    amount of fuzzy matching can tell.

    Read off the TEXT rather than the spans, because the role word is outside the span:
    the model does not include it, and extending the span over it would break both the
    postposed form ("el señor Sánchez, presidente del Gobierno") and the surface the site
    highlights. A capture is kept only when it names a surface the speech actually spans,
    which is what makes the appositions that name no person ("el presidente del Gobierno
    …", "señora presidenta. Gracias") inert instead of dangerous.

    Pooled per surface over the whole speech for the same reason gender is (see
    ``_gender_cues``): the title is used once and the bare surname many times. Surfaces
    whose appositions disagree are dropped — two offices on one surname mean it names two
    people, which is exactly when nothing should be assumed."""
    surfaces = {normalize_span(span) for span in spans}
    surfaces.discard("")
    seen: dict[str, set[str]] = {}
    for _start, _end, name, family in role_appositions(text):
        norm = normalize_span(name)
        if norm in surfaces:
            seen.setdefault(norm, set()).add(family)
    return {norm: next(iter(families))
            for norm, families in seen.items() if len(families) == 1}


def _coreferents(matches: dict[str, PersonMatch], threshold: int) -> dict[str, PersonEntry]:
    """Map each still-ambiguous surface to the one tied person the speech names
    elsewhere — same-speech coreference.

    A bare "Muñoz" tied between two catalog Muñoz is undecidable on its own, but if the
    speech also says "Ester Muñoz de la Iglesia" and never names the other one, the
    ambiguity is only apparent: within one speech a surname is used for one person.
    Requiring EXACTLY one of the tied people to be named elsewhere is what makes this
    safe — when both are ("Patxi López" and "Óscar López" in the same speech), nothing
    is attached and the guard still drops the span.

    It can therefore only add occurrences to somebody the speech already names, never a
    new person; what it can get wrong is the count, when the other bearer of the surname
    is outside the catalog and so never resolves ("David Sánchez Pérez-Castejón").

    Near misses are skipped: ``candidates`` also holds the closest people when nothing
    reached the threshold, and those are failed matches, not ties."""
    resolved = {match.entry.person_id for match in matches.values() if match.entry}
    coreferents = {}
    for norm, match in matches.items():
        if match.entry is not None or len(match.candidates) < 2:
            continue
        if match.best_score < threshold:
            continue
        if not _names_a_surname(norm, match.candidates):
            continue
        named = [e for e in match.candidates if e.person_id in resolved]
        if len(named) == 1:
            coreferents[norm] = named[0]
    return coreferents


def resolve_mentions(
    spans, index: list[PersonEntry], threshold: int,
    excluded_surnames: frozenset[str] = frozenset(), *,
    gender_gate: bool = True, gender_veto: bool = True, coreference: bool = True,
    text: str | None = None, role_apposition: bool = True) -> list[Mention]:
    """Collapse raw NER ``spans`` (duplicates preserved) into canonical ``Mention``s.

    Each span is normalized then resolved (once per distinct normalized form).
    Occurrences that resolve to the same person are merged: ``count`` totals them and
    ``surface_forms`` keeps the distinct raw spans seen. ``excluded_surnames`` drops
    spans that name a flagged non-deputy homonym of a DEPUTY (see ``_is_excluded``);
    it never touches a resolved non-deputy. Returns mentions ordered by descending
    count then name.

    Three signals need the whole speech rather than one span, which is why the spans are
    matched first and counted second. ``gender_gate`` lets a gendered courtesy form
    settle a surname the ambiguity guard would otherwise drop ("la señora Muñoz" is the
    female Muñoz), and ``gender_veto`` lets the same form rule out the only bearer of a
    surname when it contradicts them ("el señor Marcos" is not the deputy Milagros
    Marcos); the veto needs the gate, since without pooled cues there is nothing to
    contradict. ``role_apposition`` lets the office a speech names somebody by settle
    one too ("el presidente Sánchez"); it is the only signal that needs the speech
    ``text``, because the role word falls outside the span, and stays inert without it.
    ``coreference`` then attaches the surnames still tied to the one tied person the
    speech names elsewhere (see ``_coreferents``). Coreference reads only the first pass,
    so no attachment can become evidence for another and the result does not depend on the
    order the spans arrive in."""
    cues = _gender_cues(spans) if gender_gate else {}
    offices = role_cues(text, spans) if role_apposition and text else {}
    matches: dict[str, PersonMatch] = {}
    for span in spans:
        norm = normalize_span(span)
        if norm and norm not in matches:
            matches[norm] = _match_one(norm, index, threshold, gender=cues.get(norm),
                                       office=offices.get(norm), gender_veto=gender_veto)
    coreferents = _coreferents(matches, threshold) if coreference else {}

    by_person: dict[str, dict] = {}
    for span in spans:
        norm = normalize_span(span)
        if not norm:
            continue
        entry = matches[norm].entry or coreferents.get(norm)
        if entry is None:
            continue
        # The homonym denylist / speech-scoped cues exist only to stop a famous
        # non-deputy being mistaken for a deputy — so they gate deputy resolutions
        # only. A resolved non-deputy is exactly who we want and is never excluded.
        if entry.person_type == "deputy" and _is_excluded(norm, entry, excluded_surnames):
            continue
        acc = by_person.setdefault(
            entry.person_id,
            {"name": entry.name, "person_type": entry.person_type,
             "surface_forms": set(), "count": 0})
        acc["surface_forms"].add(span.strip())
        acc["count"] += 1

    mentions = [
        Mention(
            person_id=person_id,
            person_type=acc["person_type"],
            name=acc["name"],
            surface_forms=sorted(acc["surface_forms"]),
            count=acc["count"])
        for person_id, acc in by_person.items()
    ]
    mentions.sort(key=lambda m: (-m.count, m.name))
    return mentions
