"""Assemble the person catalog used to resolve mentions — deputies + non-deputies.

Two data files here both hold a field called ``aliases``, and they do NOT reach the
same places. What a name is allowed to do depends on which list it is in:

===============================  ========  =======  ========
curated list                     mentions  speaker  NER tags
===============================  ========  =======  ========
persons_catalog ``aliases``      yes       no       no
deputy_aliases ``aliases``       yes       yes      no
deputy_aliases ``tag_surfaces``  no        no       yes
===============================  ========  =======  ========

- *mentions* — the alias becomes another key on that person's ``PersonEntry``, so a
  span resolves to them. Keys are scored as a MAX, never summed, so a list of variants
  cannot inflate a ``Mention.count``: Putin's four keys still count one span once.
  Append-anything-safe.
- *speaker* — deputy aliases additionally build an alias-ONLY index (``alias_index``)
  that the query speaker path scores against before its corpus fuzzy match. That is
  what lets "intervenciones de Tesh Sidi" find her, and it exists only for deputies
  because only they are corpus speakers under a public name.
- *NER tags* — ``tag_surfaces`` alone feeds the gazetteer. This is the one list that
  can create spans rather than merely resolve them, so it is the one with real
  precision risk, and it has its own rules (below).

Deputies come from the catalog plus ``deputy_aliases.json``, for the ones the public
knows by another name than the official one — "Tesh Sidi" for "Andala Ubbi, Teslem".
Such a name shares no token with the catalog entry, so no threshold can reach it; it
has to be curated. The file lives here rather than on the deputy document because the
extractor rewrites those wholesale (``replace_one``).

Curating an ``alias`` — the one rule no unit test can enforce, because the hazard is a
collision with something outside the catalog. ``token_set_ratio`` scores a subset at
100, so an alias also matches every surface it is contained in: curating "Sidi" would
resolve the Moroccan town "Sidi Ifni" to the deputy. A shape rule cannot catch that
("Sidi" is not a catalog person), so before adding one, check it against the real
corpus — confirm no existing speaker value or NER span starts resolving to that deputy.
Prefer whole public names: a subset match means "Tesh Sidi" already covers "Tesh".

Curating a ``tag_surface`` is stricter, and differently so. It makes every Title-cased
occurrence in the corpus a person span, so it must name nobody and nothing else —
"Tesh" qualifies (all 7 corpus occurrences are her), "Sidi" does not. Surfaces must
also not NEST: patterns for both "Tesh" and "Tesh Sidi" would each match the same
text and count one written mention twice (``_rescued`` does no longest-match
filtering). The spaCy adapter's out-of-vocabulary gate is a second line of defence —
an in-vocabulary token is silently ignored rather than tagged. Known gap: a surface
glued to a dash tokenizes as one token ("Tesh―") and a case-sensitive pattern cannot
match it, so those occurrences stay untagged.

Non-deputies come from two sources:

- a curated JSON data file (``persons_catalog.json``) for people who are named in
  debate but never speak in Congress — the King, regional presidents, former prime
  ministers, foreign leaders;
- the corpus itself: everyone who HAS spoken but is not a sitting deputy (government
  ministers, comparecencia witnesses), read from ``Speeches`` and fuzzy-deduped
  against the deputies (and curated) catalog so someone who is both — a minister who
  is also a deputy, a curated figure who once testified — is not listed twice. This
  tier grows on its own as more sessions are imported.

The matching itself (key building, fuzzy scoring) lives in the pure
``domain.speeches.mentions``; this module only does the I/O and the mapping of a
speaker's official role onto their entry (``person_type``, gender, and the offices they
hold), then hands a flat ``PersonEntry`` list to the resolver — mirroring how the
deputies list is passed into the domain today.
"""

import json
import re
from dataclasses import replace
from pathlib import Path

from qhld_ai.domain.mentions import (
    PersonEntry,
    build_deputy_index,
    make_person_entry,
    normalize_span,
    office_families,
    resolve_person,
)

CATALOG_FILE = Path(__file__).parent / "persons_catalog.json"
DEPUTY_ALIASES_FILE = Path(__file__).parent / "deputy_aliases.json"


def load_curated(path=CATALOG_FILE):
    """Read the curated non-deputy catalog (a JSON array of person records).

    Their ``aliases`` are resolution keys and nothing more — they never reach the NER
    gazetteer or the speaker path, so the list is safe to grow with any number of
    surface variants (see the reach table at the top of this module)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_deputy_aliases(path=DEPUTY_ALIASES_FILE):
    """Read the curated deputy-alias records (a JSON array of
    ``{deputy_id, name, aliases, tag_surfaces}``).

    Unlike the non-deputy catalog, these records reach two further places: ``aliases``
    also build the speaker-path alias index, and ``tag_surfaces`` — a SEPARATE, stricter
    list — feeds the NER gazetteer. See the reach table at the top of this module before
    adding to either."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def deputy_aliases_by_id(records):
    """``{deputy_id: (alias, ...)}`` — the shape ``build_deputy_index`` consumes, i.e.
    the *mentions* reach of ``aliases``. Deliberately ignores ``tag_surfaces``: a
    taggable surface is not automatically a resolution key, and vice versa."""
    return {row["deputy_id"]: tuple(row.get("aliases", ()))
            for row in (records or []) if row.get("deputy_id")}


def gazetteer_surfaces(records):
    """The curated ``tag_surfaces`` — what the NER may TAG, which is not the same as what
    a query may SAY (``aliases``).

    Tagging needs the surface the Diario actually prints ("Querida Tesh"), so a bare
    given-name-like token earns its place here; resolution then reaches the deputy
    because a curated alias covers its own subset. The lists stay separate precisely so a
    surface can be resolvable without being taggable: "Sidi" resolves (as a subset of
    "Tesh Sidi") but must never be tagged, or "la guerra de Sidi Ifni" becomes a mention
    of the deputy. This is also the only curated list that can CREATE a span rather than
    just match one, which is why its rules are the strict ones."""
    return tuple(surface
                 for row in (records or [])
                 for surface in row.get("tag_surfaces", ())
                 if surface)


def alias_index(records):
    """A match index holding ONLY the curated deputy aliases — one entry per record,
    keyed by its aliases and nothing else. This is the *speaker* reach of ``aliases``,
    which the non-deputy catalog does not have: curated non-deputies are named in debate
    but do not speak, so there is no corpus ``speaker`` value for an alias to name.

    The speaker path scores against this with the same ``match_person`` the mentions
    path uses, so the two agree by construction: a subset ("Tesh" for "Tesh Sidi") or a
    misspelling ("Tesh Sidí") resolves identically on both. Keying by aliases ALONE is
    what keeps it contained — an official name or surname scores ~0 here and falls
    through to the corpus fuzzy match untouched.

    The canonical ``name`` rides on the record rather than being looked up in the deputy
    catalog, because the speaker path has to work in resolvers built without one; it is
    re-checked against the live corpus vocabulary at match time, so a stale curation
    degrades to the fuzzy match instead of filtering on a name the corpus never had."""
    index = []
    for row in records or []:
        name = row.get("name")
        if not name:
            continue
        keys = {normalize_span(alias) for alias in row.get("aliases", ())}
        keys.discard("")
        if keys:
            index.append(PersonEntry(
                person_id=row.get("deputy_id"), person_type="deputy", name=name,
                keys=tuple(sorted(keys))))
    return index


def _curated_entries(curated):
    """Turn curated person records into ``PersonEntry`` rows."""
    return [
        make_person_entry(
            person_id=row["person_id"],
            person_type=row["person_type"],
            name=row["name"],
            aliases=row.get("aliases", ()),
            overrides_deputy=row.get("overrides_deputy", False),
            gender=row.get("gender"))
        for row in curated
    ]


def _type_from_role(role):
    """Coarse ``person_type`` from a speaker's official role: government offices →
    ``"minister"``; anyone else who speaks without a parliamentary group (agency
    directors and other comparecencia witnesses) → ``"official"``."""
    r = (role or "").lower()
    if "ministr" in r or "vicepresident" in r or "presidente del gobierno" in r:
        return "minister"
    return "official"


# Spanish office titles agree in gender with their holder, so the role a speaker is
# recorded under identifies them: "Ministra de Juventud e Infancia" is Sira Rego, not
# the deputy Néstor Rego. This is the only gender source for bootstrapped speakers —
# they come from the corpus, which carries no gender field — and it matters because
# ministers are exactly the people who collide with a deputy's surname. Matched at the
# START of the role so a trailing office ("…y Ministro de Economía") cannot override the
# leading one, and only for titles whose masculine/feminine forms actually differ.
_FEMININE_ROLE = re.compile(
    r"^\s*(?:ministra|vicepresidenta|presidenta|secretaria|directora|delegada|"
    r"comisionada|consejera|alcaldesa|interventora|subsecretaria|diputada)\b", re.I)
_MASCULINE_ROLE = re.compile(
    r"^\s*(?:ministro|vicepresidente|presidente|secretario|director|delegado|"
    r"comisionado|consejero|alcalde|interventor|subsecretario|diputado)\b", re.I)


def _gender_from_role(role):
    """"Mujer"/"Hombre" from the grammatical gender of a speaker's office, or ``None``
    when the title is unrecognised or gender-neutral in form ("Compareciente").

    ``diputado``/``diputada`` are included because the corpus really does distinguish
    them (2091 vs 1405 speeches), so a bootstrapped former deputy gets a gender too.

    The trailing ``\\b`` is what keeps a pair apart where one form is a prefix of the
    other: "alcalde" cannot match inside "Alcaldesa", because the boundary fails on the
    following "s". Order is therefore not load-bearing."""
    text = role or ""
    if _FEMININE_ROLE.match(text):
        return "Mujer"
    if _MASCULINE_ROLE.match(text):
        return "Hombre"
    return None


def _bootstrap_entries(speakers, known, threshold):
    """``PersonEntry`` rows for non-deputy speakers, skipping any that already resolve
    to a ``known`` person (a deputy or a curated figure) — that is how a minister who
    is also a deputy, or an ex-minister already curated, is de-duplicated.

    ONE row per person, keyed on the id: the source is grouped by (speaker, role), so
    anyone whose office was reworded or who was promoted arrives twice ("Ministro de
    Economía…" and later "Vicepresidente Primero del Gobierno y Ministro de Economía…").
    Two entries with identical keys are worse than none — they tie at 100 against every
    span naming them, and the ambiguity guard drops the mention, so a promotion would
    silently make a minister unmentionable. Roles that disagree resolve to the more
    specific type, since a government office is the informative label.

    ``gender`` comes from the grammatical gender of the office (``_gender_from_role``).
    Across a person's several roles it is taken wherever it is known; if two roles
    genuinely disagree the gender is dropped back to unknown, which merely disables the
    gender filter for them rather than betting on the wrong half of a contradiction."""
    from tipi_data.utils import generate_slug

    by_id = {}
    for row in speakers:
        speaker = row.get("speaker")
        if not speaker or resolve_person(speaker, known, threshold) is not None:
            continue
        person_id = generate_slug(speaker)
        person_type = _type_from_role(row.get("role"))
        gender = _gender_from_role(row.get("role"))
        current = by_id.get(person_id)
        if current is None:
            by_id[person_id] = (speaker, person_type, gender)
            continue
        name, current_type, current_gender = current
        if current_gender and gender and current_gender != gender:
            gender = None
        else:
            gender = current_gender or gender
        by_id[person_id] = (
            name, person_type if person_type == "minister" else current_type, gender)
    return [make_person_entry(person_id=person_id, person_type=person_type, name=name,
                              gender=gender)
            for person_id, (name, person_type, gender) in by_id.items()]


def attach_offices(index, speaker_offices, threshold):
    """Stamp each person's ``offices`` on their entry, from the offices the corpus records
    them speaking under ("Presidente del Gobierno", "Ministra de Vivienda y Agenda
    Urbana"). That is what lets a role apposition in a speech — "el presidente Sánchez" —
    pick the holder out of a tied surname.

    Applied to the ASSEMBLED index rather than while each tier is built, because an office
    holder can be in any of them: the prime minister and any minister who kept their seat
    are deputy records, the rest are bootstrapped speakers. Each speaker name is resolved
    with the same ``resolve_person`` the bootstrap dedup uses, so a person is matched here
    exactly as they would be if a speech named them; a name that does not resolve (or
    resolves ambiguously) records no office, which only leaves the cue with less to say.

    A person's offices ACCUMULATE over their roles, since the corpus lists one row per
    wording and per promotion ("Ministro de Economía…" then "Vicepresidente Primero del
    Gobierno y Ministro de Economía…") and both offices remain true of them."""
    families = {}
    for row in speaker_offices or []:
        speaker = row.get("speaker")
        found = office_families(row.get("role"))
        if not speaker or not found:
            continue
        entry = resolve_person(speaker, index, threshold)
        if entry is not None:
            families.setdefault(entry.person_id, set()).update(found)
    if not families:
        return index
    return [replace(entry, offices=tuple(sorted(families[entry.person_id])))
            if entry.person_id in families else entry
            for entry in index]


def load_person_index(deputies, threshold, *, curated=None, nondeputy_speakers=None,
                      deputy_aliases=None, speaker_offices=None):
    """The full match index: deputies + curated non-deputies + corpus-bootstrapped
    non-deputy speakers, scored together in one pass by the resolver, each carrying the
    offices the corpus records them holding.

    ``curated``, ``nondeputy_speakers``, ``deputy_aliases`` and ``speaker_offices`` can be
    injected (tests); otherwise they are read from the data files and from
    ``Speeches.distinct_nondeputy_speakers()`` / ``Speeches.distinct_speaker_offices()``.
    The two speaker queries differ on purpose: the bootstrap wants people who are NOT
    deputies, while offices are wanted for everybody who holds one — the prime minister
    included, and he sits in a parliamentary group.
    """
    if deputy_aliases is None:
        deputy_aliases = load_deputy_aliases()
    deputy_index = build_deputy_index(
        deputies, aliases=deputy_aliases_by_id(deputy_aliases))
    curated_entries = _curated_entries(load_curated() if curated is None else curated)
    if nondeputy_speakers is None or speaker_offices is None:
        from tipi_data.repositories.speeches import Speeches
        if nondeputy_speakers is None:
            nondeputy_speakers = Speeches.distinct_nondeputy_speakers()
        if speaker_offices is None:
            speaker_offices = Speeches.distinct_speaker_offices()
    bootstrap = _bootstrap_entries(
        nondeputy_speakers, deputy_index + curated_entries, threshold)
    return attach_offices(
        deputy_index + curated_entries + bootstrap, speaker_offices, threshold)
