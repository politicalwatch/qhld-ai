"""Application service for natural-language search over speeches.

Orchestrates parse -> resolve -> filtered retrieve, then delegates to the existing
``SearchSpeeches`` on the *residual* semantic query (topic only), applying the
resolved structured filters. A query that resolved to filters and no topic at all
("intervenciones de Pedro Sánchez") has nothing to rank by, so it takes
``SearchSpeeches``'s browse path instead: the newest matching speeches, no
scores, no highlights. Plain and injectable — parser, resolver and search
are all defaulted from env for production but overridable in tests. ``execute``
is the LangSmith root span (inert unless LANGSMITH_TRACING is set): the parser's
LLM call auto-nests under it via langchain, resolution and the search stages are
traced pass-throughs, and a rejected query surfaces as the span's error. The
LangGraph wrapper is still deferred; this near-linear flow is a plain service.

``today`` is passed in (from the CLI edge), never read from a wall-clock here, so
relative-date resolution stays deterministic and testable.
"""

import logging
from dataclasses import dataclass, field

from langsmith import traceable

from qhld_ai.application.search.resolve_entities import EntityResolver, Resolution
from qhld_ai.domain.entities import normalize_entity
from qhld_ai.domain.errors import NotASpeechQuery
from qhld_ai.domain.ports.query_parser import ParsedQuery
from qhld_ai.infrastructure.config.settings import get_settings
from qhld_ai.infrastructure.queryparsing.factory import create_query_parser_from_env
from qhld_ai.infrastructure.vectorstore.naming import collection_name

logger = logging.getLogger(__name__)

# Upper bound for the single-speech passage search. The ``speech_id`` filter
# limits candidates to one speech's chunks, so this ceiling is never the binding
# constraint — it exists only because Qdrant's query API requires a limit.
_PASSAGES_K = 1000

# Function words that connect entities inside a semantic query ("Navantia y
# UNRWA", "sequía en Málaga") without adding topical content of their own.
_CONNECTOR_TOKENS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "a", "al", "de", "del", "en", "y", "e", "o", "u",
    "sobre", "con", "por", "para",
}


def _entities_cover_topic(semantic: str, entities: list[str]) -> bool:
    """True when the semantic query adds nothing beyond the extracted entities —
    a PURE-entity query ("Eurovisión", "la guerra de Gaza"). The parser keeps
    entities inside ``semantic_query`` (the entity IS the topic), so comparing
    their normalized tokens tells the two query shapes apart: a leftover
    content token means the query asks something ABOUT the entity ("sequía en
    Málaga" leaves "sequía") and is topical."""
    entity_tokens = set()
    for raw in entities:
        entity_tokens.update(normalize_entity(raw).split())
    semantic_tokens = set(normalize_entity(semantic).split())
    return not (semantic_tokens - entity_tokens - _CONNECTOR_TOKENS)


@dataclass
class NaturalResult:
    """``semantic_query`` is the topic the parser extracted — empty when the query
    named only filters. ``browse`` says the hits came from the vector-free browse
    path that empty topic routes to: they are the newest matching speeches, not
    matches for anything, so nothing in them should be presented as a match."""
    parsed: ParsedQuery
    resolution: Resolution
    semantic_query: str
    hits: list = field(default_factory=list)
    grouped: bool = False
    browse: bool = False


class NaturalSearchSpeeches:
    def __init__(self, settings=None, parser=None, search=None, resolver=None):
        self.settings = settings or get_settings()
        self.parser = parser or create_query_parser_from_env(self.settings)
        self.search = search or self._default_search()
        self._resolver = resolver

    def _default_search(self):
        from qhld_ai.application.search.search_speeches import SearchSpeeches

        return SearchSpeeches(settings=self.settings)

    def _resolver_from_corpus(self) -> EntityResolver:
        """Build a resolver bound to the target (per-model) collection's payload:
        distinct speaker/role/group values come from that collection, group aliases
        from the ParliamentaryGroups repo, and the person catalog (deputies + curated
        and bootstrapped non-deputies) resolves a mentioned person to a person id
        (matching the payload ``mentions`` list)."""
        from tipi_data.repositories.deputies import Deputies
        from tipi_data.repositories.parliamentarygroups import ParliamentaryGroups
        from tipi_data.repositories.speeches import Speeches

        dim = len(self.search.embedder.embed_query("probe"))
        collection = collection_name(self.settings, dim)
        return EntityResolver(
            distinct=lambda key: self.search.store.distinct_values(collection, key),
            groups=ParliamentaryGroups.get_all(),
            deputies=Deputies.get_all(),
            # Who has spoken under a government office, which is what decides a tied
            # bare surname ("Montero"). Read once per resolver, not per query.
            speaker_offices=Speeches.distinct_speaker_offices(),
            # Who has spoken under a given group/constituency, asked per query — it
            # depends on what that query resolved, so unlike the offices above it cannot
            # be read once. Only a query that both ties a surname and names one of those
            # pays for it.
            speakers_under=lambda where: self.search.store.distinct_values(
                collection, "speaker", where),
            mention_threshold=self.settings.mention_match_threshold)

    def resolver(self) -> EntityResolver:
        if self._resolver is None:
            self._resolver = self._resolver_from_corpus()
        return self._resolver

    @traceable(name="resolve_entities", run_type="chain")
    def _resolve(self, parsed) -> Resolution:
        """Traced pass-through: the resolved filters and every unresolved entity
        show up per query in the search trace."""
        return self.resolver().resolve(parsed)

    def _prepare(self, query, parsed):
        """Shared prelude of ``execute`` and ``passages``: parse-gate, resolve,
        then derive the residual semantic query and the relevance-floor decision.
        Raises ``NotASpeechQuery`` for a non-search or empty parse; returns
        ``(resolution, filters, semantic, apply_floor)`` otherwise. An empty
        ``semantic`` is a pure-filter query, which the callers browse instead of
        searching."""
        if not parsed.is_speech_search:
            # Not a search at all (a command, an injection, a question to the
            # assistant): reject outright rather than retrieve on nonsense. The
            # flag rides on the parsed object, so a reused parse is covered too.
            raise NotASpeechQuery(query)
        resolution = self._resolve(parsed)
        # Unresolved values are the raw material for catalog curation (a missing
        # alias scores a near miss; an out-of-catalog person scores low), so keep
        # a trace of every one.
        for entity in resolution.unresolved:
            logger.info(
                "unresolved %s %r in query %r (blocking=%s, suggestion=%s)",
                entity.field, entity.value, query, entity.blocking, entity.suggestion)
        filters = resolution.filters or None
        # Search the topic only — never the raw query. A pure-filter query has no
        # topic, and searching its raw text instead ("intervenciones de Pedro
        # Sánchez") ranks the filtered set by resemblance to words the corpus
        # never contains and then presents the arbitrary result as matches. Such
        # a query browses instead (see ``execute``).
        semantic = parsed.semantic_query.strip() if parsed.semantic_query else ""
        # The relevance floor only makes sense for topical queries, where a low
        # reranked score means off-domain. Skip it for PURE-entity queries — the
        # semantic query is just the entity, so a speech referencing it only in
        # passing is a valid hit yet scores as low as junk; the entity filter
        # provides the precision there. Everything else keeps the floor: with a
        # topic beyond the entity ("sequía en Málaga"), a valid hit must discuss
        # that topic and its best passage scores well, while entity-filtered hits
        # about something else are junk. Speakers and mentioned persons never
        # exempt — they are stripped OUT of the semantic query, so any residual
        # topic is a genuine requirement ("vivienda que mencione a X" must be
        # about vivienda). A pure-filter query never reaches a floor at all: it
        # browses, and a browse ranks nothing.
        pure_entity = bool(parsed.entities) and _entities_cover_topic(semantic, parsed.entities)
        apply_floor = bool(semantic) and not pure_entity
        if not semantic and not filters and not resolution.blocked:
            # An empty parse — no residual topic and nothing resolved to filter
            # by — leaves nothing to search AND nothing to browse. Gibberish that
            # slips the parser's is_speech_search gate lands exactly here, so
            # treat it as the gate would.
            raise NotASpeechQuery(query)
        return resolution, filters, semantic, apply_floor

    @traceable(name="natural_search", run_type="chain")
    def execute(self, query, today, k=10, grouped=False, highlights=3,
                exclude=None, parsed=None) -> NaturalResult:
        """``exclude`` is the "load more" cursor of ``search_grouped``: the
        speech_ids already shown, skipped so the next call yields fresh speeches
        (grouped mode only — flat hits have no stateless pagination). ``parsed``
        lets a caller reuse a previous parse of the same query (the parse is an
        LLM call), skipping the parser entirely."""
        parsed = parsed or self.parser.parse(query, today)
        resolution, filters, semantic, apply_floor = self._prepare(query, parsed)
        if resolution.blocked:
            # Some filter is unsatisfiable (e.g. a mentioned person absent from the
            # catalog): the honest answer is zero hits. Searching without that
            # filter would return results that LOOK like matches, so skip retrieval.
            return NaturalResult(
                parsed=parsed, resolution=resolution, semantic_query=semantic,
                grouped=grouped)
        if not semantic:
            # Filters only, no topic: browse the matching speeches newest-first.
            hits = (
                self.search.browse_grouped(
                    page_size=k, filters=filters, exclude=exclude)
                if grouped
                else self.search.browse(k=k, filters=filters)
            )
            return NaturalResult(
                parsed=parsed, resolution=resolution, semantic_query=semantic,
                hits=hits, grouped=grouped, browse=True)
        if grouped:
            hits = self.search.search_grouped(
                semantic, page_size=k, highlights=highlights, filters=filters,
                exclude=exclude, apply_floor=apply_floor)
        else:
            hits = self.search.search(
                semantic, k=k, filters=filters, apply_floor=apply_floor)
        return NaturalResult(
            parsed=parsed, resolution=resolution, semantic_query=semantic,
            hits=hits, grouped=grouped)

    @traceable(name="natural_search_passages", run_type="chain")
    def passages(self, query, today, speech_id, parsed=None) -> NaturalResult:
        """Every relevance-floored passage of ONE speech for this query — the
        detail page highlights all matching passages, not just the few shown on a
        result card. An ungrouped ``search`` scoped to the speech via a
        ``speech_id`` payload filter; combining it with the query's resolved
        filters is sound because the speech was already a search result, so it
        satisfies them. Same floor gate as ``execute`` (the passages must match
        what the results page showed), and ``parsed`` reuses the memoized parse."""
        parsed = parsed or self.parser.parse(query, today)
        resolution, filters, semantic, apply_floor = self._prepare(query, parsed)
        if resolution.blocked:
            return NaturalResult(
                parsed=parsed, resolution=resolution, semantic_query=semantic)
        if not semantic:
            # A pure-filter query asked for no topic, so no passage of this speech
            # matches anything: the detail page must highlight none of them. The
            # results page browsed for the same reason (see ``execute``).
            return NaturalResult(
                parsed=parsed, resolution=resolution, semantic_query=semantic,
                browse=True)
        # k is a ceiling the Qdrant API requires, not a passage cap: the
        # speech_id filter narrows candidates to this one speech's chunks (dozens
        # at most), so a large k just means "return them all above the floor".
        scoped = {**(filters or {}), "speech_id": speech_id}
        hits = self.search.search(
            semantic, k=_PASSAGES_K, filters=scoped, apply_floor=apply_floor)
        return NaturalResult(
            parsed=parsed, resolution=resolution, semantic_query=semantic,
            hits=hits)
