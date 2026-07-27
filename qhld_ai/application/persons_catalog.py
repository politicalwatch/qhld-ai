"""Assemble the person catalog used to resolve mentions — deputies + non-deputies.

Deputies come from the catalog, plus a curated alias file (``deputy_aliases.json``)
for the ones the public knows by another name than the official one — "Tesh Sidi"
for "Andala Ubbi, Teslem". Such a name shares no token with the catalog entry, so no
threshold can reach it; it has to be curated. The file lives here rather than on the
deputy document because the extractor rewrites those wholesale (``replace_one``).

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
``domain.speeches.mentions``; this module only does the I/O and the role→type
mapping, then hands a flat ``PersonEntry`` list to the resolver — mirroring how the
deputies list is passed into the domain today.
"""

import json
from pathlib import Path

from qhld_ai.domain.mentions import (
    build_deputy_index,
    make_person_entry,
    normalize_span,
    resolve_person,
)

CATALOG_FILE = Path(__file__).parent / "persons_catalog.json"
DEPUTY_ALIASES_FILE = Path(__file__).parent / "deputy_aliases.json"


def load_curated(path=CATALOG_FILE):
    """Read the curated non-deputy catalog (a JSON array of person records)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_deputy_aliases(path=DEPUTY_ALIASES_FILE):
    """Read the curated deputy-alias records (a JSON array of
    ``{deputy_id, name, aliases}``)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def deputy_aliases_by_id(records):
    """``{deputy_id: (alias, ...)}`` — the shape ``build_deputy_index`` consumes."""
    return {row["deputy_id"]: tuple(row.get("aliases", ()))
            for row in (records or []) if row.get("deputy_id")}


def speaker_alias_map(records):
    """``{normalized alias: canonical name}`` for the query speaker path ("tesh sidi"
    -> "Andala Ubbi, Teslem"). Keys run through the SAME ``normalize_span`` a queried
    surface does, so the two sides meet under one normalization.

    The canonical ``name`` is carried by the record rather than looked up in the
    deputy catalog, because the speaker path has to work in resolvers built without
    one; it is re-checked against the live corpus vocabulary at match time, so a stale
    curation degrades to the fuzzy match instead of filtering on a name the corpus
    never had. First record wins a duplicated alias."""
    mapping = {}
    for row in records or []:
        name = row.get("name")
        if not name:
            continue
        for alias in row.get("aliases", ()):
            key = normalize_span(alias)
            if key:
                mapping.setdefault(key, name)
    return mapping


def _curated_entries(curated):
    """Turn curated person records into ``PersonEntry`` rows."""
    return [
        make_person_entry(
            person_id=row["person_id"],
            person_type=row["person_type"],
            name=row["name"],
            aliases=row.get("aliases", ()),
            overrides_deputy=row.get("overrides_deputy", False))
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


def _bootstrap_entries(speakers, known, threshold):
    """``PersonEntry`` rows for non-deputy speakers, skipping any that already resolve
    to a ``known`` person (a deputy or a curated figure) — that is how a minister who
    is also a deputy, or an ex-minister already curated, is de-duplicated."""
    from tipi_data.utils import generate_slug

    entries = []
    for row in speakers:
        speaker = row.get("speaker")
        if not speaker or resolve_person(speaker, known, threshold) is not None:
            continue
        entries.append(make_person_entry(
            person_id=generate_slug(speaker),
            person_type=_type_from_role(row.get("role")),
            name=speaker))
    return entries


def load_person_index(deputies, threshold, *, curated=None, nondeputy_speakers=None,
                      deputy_aliases=None):
    """The full match index: deputies + curated non-deputies + corpus-bootstrapped
    non-deputy speakers, scored together in one pass by the resolver.

    ``curated``, ``nondeputy_speakers`` and ``deputy_aliases`` can be injected (tests);
    otherwise they are read from the data files and from
    ``Speeches.distinct_nondeputy_speakers()``.
    """
    if deputy_aliases is None:
        deputy_aliases = load_deputy_aliases()
    deputy_index = build_deputy_index(
        deputies, aliases=deputy_aliases_by_id(deputy_aliases))
    curated_entries = _curated_entries(load_curated() if curated is None else curated)
    if nondeputy_speakers is None:
        from tipi_data.repositories.speeches import Speeches
        nondeputy_speakers = Speeches.distinct_nondeputy_speakers()
    bootstrap = _bootstrap_entries(
        nondeputy_speakers, deputy_index + curated_entries, threshold)
    return deputy_index + curated_entries + bootstrap
