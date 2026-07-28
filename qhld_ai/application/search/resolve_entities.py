"""Entity resolution for Query understanding: turn a ``ParsedQuery`` into concrete Qdrant
payload filters.

The LLM extracts what the user *said* (names, titles, parties, ISO dates); this
service maps that onto what the index actually *stores*:
- speaker names -> a curated deputy-alias lookup first (for the deputies the public
  knows by another name — "Tesh Sidi" for "Andala Ubbi, Teslem" — which shares no
  token with the catalog entry and so is unreachable by any threshold), then a fuzzy
  match against the corpus ``speaker`` values ("Apellido, Nombre"), so it works for
  deputies AND non-deputies (ministers etc.) alike.
- speaker title -> fuzzy match against the corpus ``role`` values (full office titles).
- groups/parties -> the payload ``group`` code (== ``ParliamentaryGroup.shortname``),
  resolved from an alias map over group short/long names, party names and a curated
  alias file, plus a token-normalized form that strips the generic words users swap
  freely ("grupo socialista" / "partido socialista" / "los socialistas"). A curated
  ideological/bloc category ("izquierda", "independentistas") expands to every group
  labelled with it — the parser passes categories through verbatim; the labels are
  editorial data in the alias file, not a model judgment.
- mentioned persons -> person ids (deputies, or non-deputies such as ministers, the
  King, regional presidents or foreign leaders), matched against the SAME person
  catalog that tags the corpus, then filtered on the payload ``mentions`` list.
- entities      -> canonical entity keys, matched against the payload ``entities``
  vocabulary (the speech-level non-person entities stamped at index time). The query
  value is normalized with the SAME ``normalize_entity`` that produced the payload
  keys, so an exact key hit is the common case; a fuzzy fallback absorbs particle
  drift ("guerra en Gaza" -> "guerra de gaza"). Combination honours the parsed
  ``entities_mode`` exactly like mentions.
- constituencies -> the payload ``constituency`` official province value (as recorded
  in the deputy catalog: "Coruña (A)", "Alicante/Alacant"). The parser already
  canonicalizes demonyms to a province proper name; here the user-facing spelling is
  mapped to the catalog spelling via mechanically-derived alias keys (bilingual "/"
  variants, unfolded parenthesized articles) plus a few curated old names ("Lérida"),
  with a fuzzy fallback.
- ISO dates     -> a numeric ``date`` range ({"gte"/"lte": YYYYMMDD}).

When several values resolve for one field, the filter value becomes a list (the
store treats it as any-of); mentioned persons instead honour the parsed
``mentions_mode`` — ``{"all": [ids]}`` requires every person to be mentioned,
a plain list accepts any of them.

Because every one of these fields resolves against the very values the corpus was
tagged/indexed with, a value that resolves to nothing is not ignorable noise — the
constraint is unsatisfiable and the honest answer is zero results. Failed values are
therefore recorded as structured ``UnresolvedEntity`` entries: *blocking* when the
query as asked cannot be satisfied (search should short-circuit to no hits), or
non-blocking when only some members of an any-of list dropped out. A single mention
in the default ``all`` mode blocks; an unresolved member of an ``any`` list only
blocks when no member resolved.

Corpus values are read via an injected ``distinct(key)`` callable (wrapping
``VectorStorePort.distinct_values`` on the target collection), so the resolver is
trivially testable with a stub. Each resolution is recorded in ``notes`` so the
CLI can show what was understood (and what could not be matched).
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from thefuzz import fuzz, process

from qhld_ai.application.persons_catalog import (
    alias_index,
    load_deputy_profiles,
    load_person_index,
)
from qhld_ai.domain.entities import normalize_entity
from qhld_ai.domain.ports.query_parser import ParsedQuery
from qhld_ai.domain.mentions import PersonMatch, match_person, normalize_span

# token_set_ratio scores a subset match ~100 ("María Jesús Montero" ⊆ "Montero
# Cuadrado, María Jesús", or a surname-only "Montero") while an unrelated name
# stays low (~25), so a high threshold is both forgiving and precise.
_SPEAKER_THRESHOLD = 90
_ROLE_THRESHOLD = 70      # "ministra de economía" ⊆ the full official title
_GROUP_THRESHOLD = 80
# High enough to keep provinces apart (they share few tokens), low enough for
# spelling drift the alias keys don't cover ("Baleares" vs the official "Balears").
_CONSTITUENCY_THRESHOLD = 85
# Entity keys usually hit exactly (query and payload share normalize_entity);
# the fuzzy fallback only needs to absorb particle drift ("guerra en Gaza" vs
# "guerra de gaza"), which token_set_ratio scores ~100.
_ENTITY_THRESHOLD = 90

# Map the many ways a language can be named (or mis-coded by an LLM: "Gallego",
# "cat") to the ISO code stored in the payload ``lang``. Payload uses es/ca/gl/eu.
_LANG_ALIASES = {
    "es": "es", "spa": "es", "castellano": "es", "español": "es", "espanol": "es",
    "ca": "ca", "cat": "ca", "catalán": "ca", "catalan": "ca", "català": "ca",
    "gl": "gl", "gal": "gl", "gallego": "gl", "galego": "gl",
    "eu": "eu", "eus": "eu", "euskera": "eu", "euskara": "eu", "vasco": "eu", "vascuence": "eu",
}

GROUP_ALIASES_FILE = Path(__file__).parent / "group_aliases.json"

# Words that carry no identity within a group/party name — the scaffolding users
# swap freely ("grupo socialista" / "partido socialista" / "los socialistas").
# Unaccented singular forms, matched after accent stripping and plural folding.
_GENERIC_GROUP_TOKENS = {
    "grupo", "parlamentario", "parlamentaria", "partido", "politico", "politica",
    "bloque", "el", "la", "los", "las", "de", "del", "per",
}

# Tokens that carry no identity within a province name — articles and the
# island/province scaffolding that official values fold into parentheses
# ("Rioja (La)", "Balears (Illes)") and users write out ("La Rioja", "Islas
# Baleares").
_CONSTITUENCY_STOP_TOKENS = {
    "el", "la", "los", "las", "a", "de", "del", "les", "illes", "islas",
}

# Curated spellings the mechanical alias keys cannot derive from the official
# catalog value: pre-normalization Castilian names and the abbreviated Tenerife.
_CONSTITUENCY_EXTRA_ALIASES = {
    "S/C Tenerife": ("Santa Cruz de Tenerife", "Tenerife"),
    "Lleida": ("Lérida",),
    "Girona": ("Gerona",),
    "Ourense": ("Orense",),
    "Bizkaia": ("Vizcaya",),
    "Gipuzkoa": ("Guipúzcoa",),
}


def load_curated_group_aliases(path=GROUP_ALIASES_FILE):
    """Read the curated group-alias records (a JSON array of {code, aliases})."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class UnresolvedEntity:
    """A query value that matched nothing in the catalog/corpus. ``blocking`` means
    the query as asked is unsatisfiable and search must return no hits; non-blocking
    entries are members dropped from an any-of list that still has resolved members.
    ``suggestion`` carries the closest sub-threshold candidate (or the tied names of
    an ambiguous surname), when known."""
    field: str
    value: str
    blocking: bool
    suggestion: str | None = None


@dataclass
class Resolution:
    """The store-ready filters plus a human-readable trace of how each field
    resolved (or why it did not)."""
    filters: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unresolved: list[UnresolvedEntity] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """True when some filter is unsatisfiable — the search must yield no hits."""
        return any(entity.blocking for entity in self.unresolved)


class EntityResolver:
    def __init__(self, distinct, groups, deputies=None, mention_threshold=90,
                 curated=None, nondeputy_speakers=None, curated_aliases=None,
                 deputy_profiles=None):
        """``distinct`` is ``callable(key) -> set`` over the target collection's
        payload; ``groups`` is the list of ``ParliamentaryGroup`` records. ``deputies``
        (the ``Deputy`` catalog) enables resolving mentioned persons; when given, the
        full person index (deputies + curated non-deputies + bootstrapped speakers) is
        built with the SAME assembler used to tag the corpus, so a query resolves to the
        same ids that were indexed. ``curated``/``nondeputy_speakers``/``curated_aliases``
        /``deputy_profiles`` may be injected (tests); otherwise they are read from the
        data files / ``Speeches``. Omit ``deputies`` => mentioned-person queries are left
        unfiltered, but curated deputy aliases still serve the speaker path, which needs
        no catalog."""
        self._distinct = distinct
        if curated_aliases is None:
            curated_aliases = load_curated_group_aliases()
        if deputy_profiles is None:
            deputy_profiles = load_deputy_profiles()
        (self._group_aliases, self._group_aliases_normalized,
         self._group_categories) = _build_group_aliases(groups, curated_aliases)
        self._alias_index = alias_index(deputy_profiles)
        self._person_index = (
            # No corpus offices: they exist for the index-time role-apposition cue ("el
            # presidente Sánchez"), which reads a speech's own wording. A query carries no
            # such cue, so the aggregation behind them would be paid for nothing and the
            # resolutions are identical without it. The curated ones ride along with the
            # data files that are loaded anyway, and are equally inert here.
            load_person_index(deputies, mention_threshold,
                              curated=curated, nondeputy_speakers=nondeputy_speakers,
                              deputy_profiles=deputy_profiles, speaker_offices=[])
            if deputies else [])
        self._mention_threshold = mention_threshold

    def resolve(self, parsed: ParsedQuery) -> Resolution:
        result = Resolution()
        if parsed.speakers:
            self._resolve_speakers(result, parsed.speakers)
        if parsed.speaker_title:
            self._resolve_role(result, parsed.speaker_title)
        if parsed.mentioned_persons:
            self._resolve_mentions(result, parsed.mentioned_persons, parsed.mentions_mode)
        if parsed.entities:
            self._resolve_entities(result, parsed.entities, parsed.entities_mode)
        if parsed.constituencies:
            self._resolve_constituencies(result, parsed.constituencies)
        if parsed.groups_or_parties:
            self._resolve_groups(result, parsed.groups_or_parties)
        self._resolve_dates(result, parsed)
        if parsed.lang:
            self._resolve_lang(result, parsed.lang)
        if parsed.legislature:
            result.filters["legislature"] = parsed.legislature
        return result

    def _resolve_speakers(self, result, raws):
        choices = [v for v in self._distinct("speaker") if v]
        vocab = _speaker_vocab(choices) if self._alias_index else {}
        matched, misses = [], []
        for raw in raws:
            # A curated public name ("Tesh Sidi") shares no token with the official
            # catalog name, so the fuzzy match below can never reach it; try the
            # alias map first.
            value, suggestion = self._alias_speaker(raw, vocab), None
            if value:
                result.notes.append(f"speaker: '{raw}' → '{value}' (curated alias)")
            else:
                value, suggestion = self._fuzzy_match(
                    result, "speaker", raw, choices, _SPEAKER_THRESHOLD)
            if value:
                if value not in matched:
                    matched.append(value)
            else:
                misses.append((raw, suggestion))
        for raw, suggestion in misses:
            _record_unresolved(result, "speaker", raw, blocking=not matched,
                               suggestion=suggestion)
        _set_filter(result, "speaker", matched)

    def _alias_speaker(self, raw, vocab):
        """The corpus ``speaker`` value a curated public name stands for ("Tesh Sidi",
        "Tesh", "Tesh Sidí" -> "Andala Ubbi, Teslem"), or ``None``.

        Matched with the SAME ``match_person`` the mentions path uses, against an index
        of aliases only — so an alias resolves identically however the user typed it,
        and an official name or surname (which scores ~0 against alias-only keys) falls
        straight through to the fuzzy match below.

        The curated canonical name is used only when the corpus vocabulary actually
        carries it, so a stale curation — or a deputy with no indexed speech — falls
        through too, and nothing that resolves today changes. The value returned is
        always the corpus's own string, never the curated spelling."""
        if not vocab:
            return None
        match = match_person(raw, self._alias_index, self._mention_threshold)
        return vocab.get(normalize_span(match.entry.name)) if match.entry else None

    def _resolve_role(self, result, raw):
        choices = [v for v in self._distinct("role") if v]
        value, suggestion = self._fuzzy_match(
            result, "role", raw, choices, _ROLE_THRESHOLD)
        if value:
            result.filters["role"] = value
        else:
            _record_unresolved(result, "role", raw, blocking=True,
                               suggestion=suggestion)

    def _resolve_mentions(self, result, raws, mode):
        if not self._person_index:
            # No person catalog was injected (see __init__): mentioned persons are
            # knowingly left unfiltered rather than treated as unsatisfiable.
            for raw in raws:
                result.notes.append(f"mentions: '{raw}' ignored — no person catalog")
            return
        ids, misses = [], []
        for raw in raws:
            match = match_person(raw, self._person_index, self._mention_threshold)
            if match.entry:
                result.notes.append(
                    f"mentions: '{raw}' → '{match.entry.name}' ({match.entry.person_type})")
                if match.entry.person_id not in ids:
                    ids.append(match.entry.person_id)
            else:
                misses.append((raw, self._person_suggestion(match)))
        # ``all`` requires every person, so a single miss is unsatisfiable; ``any``
        # survives on the resolved subset and only blocks when nobody resolved.
        blocking = bool(misses) if mode != "any" else not ids
        for raw, suggestion in misses:
            _record_unresolved(result, "mentions", raw, blocking=blocking,
                               suggestion=suggestion)
        if blocking or not ids:
            return
        if len(ids) == 1:
            result.filters["mentions"] = ids[0]
        elif mode == "any":
            result.filters["mentions"] = sorted(ids)
        else:
            result.filters["mentions"] = {"all": sorted(ids)}

    def _resolve_entities(self, result, raws, mode):
        vocab = {v for v in self._distinct("entities") if v}
        keys, misses = [], []
        for raw in raws:
            key, suggestion = self._match_entity(raw, vocab)
            if key:
                result.notes.append(f"entities: '{raw}' → '{key}'")
                if key not in keys:
                    keys.append(key)
            else:
                misses.append((raw, suggestion))
        # Same combination semantics as mentions: in ``all`` mode one miss makes
        # the query unsatisfiable; ``any`` survives on the resolved subset.
        blocking = bool(misses) if mode != "any" else not keys
        for raw, suggestion in misses:
            _record_unresolved(result, "entities", raw, blocking=blocking,
                               suggestion=suggestion)
        if blocking or not keys:
            return
        if len(keys) == 1:
            result.filters["entities"] = keys[0]
        elif mode == "any":
            result.filters["entities"] = sorted(keys)
        else:
            result.filters["entities"] = {"all": sorted(keys)}

    @staticmethod
    def _match_entity(raw, vocab):
        """Canonical corpus key for one query entity: normalize with the shared
        function (exact hits are the common case, both sides run it), fuzzy
        fallback for particle drift. ``(None, suggestion)`` when nothing clears
        the threshold — the constraint is unsatisfiable."""
        key = normalize_entity(raw)
        if not key:
            return None, None
        if key in vocab:
            return key, None
        match = process.extractOne(
            key, list(vocab), scorer=fuzz.token_set_ratio) if vocab else None
        if match and match[1] >= _ENTITY_THRESHOLD:
            return match[0], None
        return None, (f"'{match[0]}' ({match[1]})" if match else None)

    def _person_suggestion(self, match: PersonMatch) -> str | None:
        """Human-readable hint for a failed person match: the tied names when the
        span was ambiguous, the closest near-miss otherwise."""
        names = match.candidate_names
        if not names:
            return None
        if match.best_score >= self._mention_threshold:
            return "ambiguous: " + " / ".join(f"'{name}'" for name in names)
        return f"'{names[0]}' ({match.best_score})"

    def _resolve_constituencies(self, result, raws):
        aliases = {}
        for value in self._distinct("constituency"):
            if value:
                for key in _constituency_keys(value):
                    aliases.setdefault(key, value)
        matched, misses = [], []
        for raw in raws:
            value, suggestion = self._match_constituency(raw, aliases)
            if value:
                result.notes.append(f"constituency: '{raw}' → '{value}'")
                if value not in matched:
                    matched.append(value)
            else:
                misses.append((raw, suggestion))
        for raw, suggestion in misses:
            _record_unresolved(result, "constituency", raw, blocking=not matched,
                               suggestion=suggestion)
        _set_filter(result, "constituency", matched)

    @staticmethod
    def _match_constituency(raw, aliases):
        key = _normalize_constituency_key(raw)
        if not key:
            return None, None
        if key in aliases:
            return aliases[key], None
        match = process.extractOne(
            key, list(aliases), scorer=fuzz.token_set_ratio) if aliases else None
        if match and match[1] >= _CONSTITUENCY_THRESHOLD:
            return aliases[match[0]], None
        return None, (f"'{aliases[match[0]]}' ({match[1]})" if match else None)

    def _resolve_lang(self, result, raw):
        code = _LANG_ALIASES.get(raw.strip().lower())
        if code:
            result.filters["lang"] = code
            if code != raw:
                result.notes.append(f"lang: '{raw}' → '{code}'")
        else:
            _record_unresolved(result, "lang", raw, blocking=True)

    @staticmethod
    def _fuzzy_match(result, payload_key, raw, choices, threshold):
        """Best fuzzy match for one raw value, traced in ``notes`` when it clears
        ``threshold``; otherwise ``(None, suggestion)`` with the best sub-threshold
        candidate for the caller to report."""
        match = process.extractOne(
            raw, choices, scorer=fuzz.token_set_ratio) if choices else None
        if match and match[1] >= threshold:
            result.notes.append(f"{payload_key}: '{raw}' → '{match[0]}' ({match[1]})")
            return match[0], None
        return None, (f"'{match[0]}' ({match[1]})" if match else None)

    def _resolve_groups(self, result, raws):
        matched, misses = [], []
        for raw in raws:
            codes = self._group_categories.get(_normalize_group_key(raw))
            if codes:
                result.notes.append(f"group: '{raw}' → {', '.join(codes)} (category)")
            else:
                shortname = self._match_group(raw)
                if not shortname:
                    misses.append(raw)
                    continue
                result.notes.append(f"group: '{raw}' → '{shortname}'")
                codes = [shortname]
            matched.extend(code for code in codes if code not in matched)
        for raw in misses:
            _record_unresolved(result, "group", raw, blocking=not matched)
        _set_filter(result, "group", matched)

    def _match_group(self, raw):
        key = raw.strip().lower()
        if key in self._group_aliases:
            return self._group_aliases[key]
        normalized = _normalize_group_key(raw)
        if not normalized:
            return None
        if normalized in self._group_aliases_normalized:
            return self._group_aliases_normalized[normalized]
        match = process.extractOne(
            normalized, list(self._group_aliases_normalized),
            scorer=fuzz.token_set_ratio)
        if match and match[1] >= _GROUP_THRESHOLD:
            return self._group_aliases_normalized[match[0]]
        return None

    def _resolve_dates(self, result, parsed):
        bounds = {}
        if parsed.date_from:
            bounds["gte"] = _iso_to_int(parsed.date_from)
        if parsed.date_to:
            bounds["lte"] = _iso_to_int(parsed.date_to)
        bounds = {k: v for k, v in bounds.items() if v is not None}
        if bounds:
            result.filters["date"] = bounds
            result.notes.append(f"date: {bounds}")


def _record_unresolved(result, field_name, raw, blocking, suggestion=None):
    """Record one failed value both machine-readably (``unresolved``) and in the
    human trace (``notes``)."""
    result.unresolved.append(UnresolvedEntity(field_name, raw, blocking, suggestion))
    hint = f" (closest: {suggestion})" if suggestion else ""
    outcome = "no results" if blocking else "dropped from any-of"
    result.notes.append(f"{field_name}: '{raw}' unresolved{hint} — {outcome}")


def _set_filter(result, key, matched):
    """A single resolved value stays a scalar (exact match); several become a
    list (the store treats it as any-of)."""
    if matched:
        result.filters[key] = matched[0] if len(matched) == 1 else sorted(matched)


def _speaker_vocab(choices):
    """``{normalized speaker name: the corpus value}`` — the presence check behind a
    curated alias. Normalizing both sides absorbs case and comma drift between the
    curated canonical name and the indexed one; iterating sorted values keeps a
    (theoretical) normalization collision deterministic rather than set-ordered."""
    vocab = {}
    for value in sorted(choices):
        vocab.setdefault(normalize_span(value), value)
    return vocab


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def _fold_plural(token: str) -> str:
    """'populares' -> 'popular', 'socialistas' -> 'socialista'. Applied to aliases
    and queries alike, so any over-stripping stays symmetric and still matches."""
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _normalize_group_key(text: str) -> str:
    """Reduce a group/party name to its distinctive tokens: lowercase, unaccent,
    fold plurals, drop generic words. 'los socialistas' and 'Grupo Parlamentario
    Socialista' both reduce to 'socialista'; 'los partidos de izquierda' to
    'izquierda'."""
    tokens = re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))
    folded = (_fold_plural(t) for t in tokens)
    return " ".join(t for t in folded if t not in _GENERIC_GROUP_TOKENS)


def _normalize_constituency_key(text: str) -> str:
    """Reduce a province name to its distinctive tokens: lowercase, unaccent,
    drop articles/scaffolding. 'La Rioja', 'Rioja (La)' and 'rioja' all reduce
    to 'rioja'; 'Islas Baleares' to 'baleares'."""
    tokens = re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))
    return " ".join(t for t in tokens if t not in _CONSTITUENCY_STOP_TOKENS)


def _constituency_keys(value: str) -> set[str]:
    """Alias keys for one official constituency value: the value itself, its
    '/'-separated bilingual variants ('Alicante/Alacant'), the unfolded
    parenthesized-article form ('Coruña (A)' → 'a coruña'), and the curated
    extras ('Lérida' for 'Lleida')."""
    variants = [value]
    folded = re.match(r"(.+?)\s*\((.+)\)$", value)
    if folded:
        variants.append(f"{folded.group(2)} {folded.group(1)}")
    else:
        variants.extend(part for part in value.split("/") if len(part) > 1)
    variants.extend(_CONSTITUENCY_EXTRA_ALIASES.get(value, ()))
    return {key for key in map(_normalize_constituency_key, variants) if key}


def _build_group_aliases(groups, curated=None) -> tuple[dict, dict, dict]:
    """Three maps for group resolution: lowercased short/long/party/curated names
    verbatim -> code (``shortname``), their token-normalized forms -> code, and
    normalized curated category ('izquierda', 'independentista') -> every code
    labelled with it. Single-party groups win over the multi-party Mixto group on
    a party-name conflict (e.g. 'PSOE' -> GS, not GMx). Curated aliases and
    categories only apply to codes present in the current catalog."""
    curated_by_code = {row["code"]: row for row in (curated or [])}
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    categories: dict[str, list[str]] = {}
    for group in sorted(groups or [], key=lambda g: len(getattr(g, "parties", None) or [])):
        shortname = getattr(group, "shortname", None)
        if not shortname:
            continue
        row = curated_by_code.get(shortname, {})
        candidates = [shortname, getattr(group, "name", None),
                      *(getattr(group, "parties", None) or []),
                      *row.get("aliases", [])]
        for alias in candidates:
            if not alias:
                continue
            exact.setdefault(alias.lower(), shortname)
            key = _normalize_group_key(alias)
            if key:
                normalized.setdefault(key, shortname)
        for category in row.get("categories", []):
            key = _normalize_group_key(category)
            if key and shortname not in categories.setdefault(key, []):
                categories[key].append(shortname)
    return exact, normalized, categories


def _iso_to_int(iso: str) -> int | None:
    """'2025-04-03' -> 20250403. Returns None on a malformed value."""
    try:
        return int(iso.replace("-", ""))
    except (ValueError, AttributeError):
        return None
