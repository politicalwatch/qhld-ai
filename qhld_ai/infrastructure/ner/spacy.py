"""spaCy NER adapter for mention and entity extraction.

Loads ``es_core_news_lg`` (the same model the rule-based query parser uses) and
returns PER spans (``person_spans``) or everything else (``entity_spans``).
spaCy is lazy-imported and the model lazy-loaded on first use, so importing this
module (and factory self-registration) stays cheap; the model is loaded once per
adapter instance and reused across a whole extract/backfill run
(``MentionTagger`` holds a single instance).

The last parsed doc is memoized, so calling ``person_spans`` and
``entity_spans`` back to back on the same text — as the tagger does per speech —
costs a single parse.

We run NER only over the Spanish text block upstream, so a single Spanish model
covers monolingual and co-official speeches alike.

The optional gazetteer (distinctive deputy surnames) is applied as a
``PhraseMatcher`` post-pass over the parsed doc, NOT as an entity-ruler pipe: a
match is added only when its tokens fall outside every model PER span. Placed
before the model, a single-token pattern would claim its token and stop the model
from forming the fuller span around it ("Maroto" would break up "Reyes Maroto" —
the ex-minister — and the orphan surname then resolves to whichever deputy
carries it); placed after with no overwrites, it could not rescue the surnames
the model tags with the WRONG label ("El señor Feijóo" as MISC). The post-pass
does both: model PER spans always win, and everything else is fair game.
"""

from qhld_ai.domain.ports.ner import NerPort

from .factory import _register


class SpacyNer(NerPort):
    def __init__(self, settings, gazetteer=None):
        self.settings = settings
        self._gazetteer = tuple(gazetteer or ())
        self._nlp = None
        self._matcher = None
        self._last = None  # (text, doc) memo — one entry, see _doc

    def _model(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.settings.ner_model)
            # Only gazetteer surnames the model has NO representation for (out of
            # vocabulary). In-vocabulary surfaces — common words the model won't tag
            # as a person ("Madrid", "Torres") and common surnames it already knows
            # — are left to its context-sensitive judgement; overriding them with a
            # blunt rule tags every occurrence and wrecks precision.
            terms = [t for t in self._gazetteer if self._nlp.vocab[t.lower()].is_oov]
            if terms:
                from spacy.matcher import PhraseMatcher

                # Default ORTH matching = case-sensitive, so only the Title-case
                # name surface matches (names are Title-case in the Diario text).
                self._matcher = PhraseMatcher(self._nlp.vocab)
                self._matcher.add(
                    "SURNAME", [self._nlp.make_doc(term) for term in terms])
        return self._nlp

    def _doc(self, text):
        """Parse ``text``, reusing the previous doc when the text is unchanged:
        the tagger asks for person and entity spans of the same speech text in a
        row, and this makes that pair cost one parse."""
        if self._last is None or self._last[0] != text:
            self._last = (text, self._model()(text))
        return self._last[1]

    def _rescued(self, doc) -> list[tuple[int, int]]:
        """Gazetteer matches outside every model PER span: surnames the model
        missed or gave the wrong label. They count as person spans, so
        ``person_spans`` adds them and ``entity_spans`` excludes them."""
        if self._matcher is None:
            return []
        per = [(e.start, e.end) for e in doc.ents if e.label_ == "PER"]
        return [(start, end) for _, start, end in self._matcher(doc)
                if not any(s < end and start < e for s, e in per)]

    def person_spans(self, text: str) -> list[str]:
        if not text:
            return []
        doc = self._doc(text)
        spans = [(ent.start, ent.end, ent.text)
                 for ent in doc.ents if ent.label_ == "PER"]
        spans.extend(
            (start, end, doc[start:end].text) for start, end in self._rescued(doc))
        spans.sort()
        return [span_text for _, _, span_text in spans]

    def entity_spans(self, text: str) -> list[str]:
        if not text:
            return []
        doc = self._doc(text)
        rescued = self._rescued(doc)
        gate = getattr(self.settings, "ner_entity_pos_gate", True)
        return [ent.text for ent in doc.ents
                if ent.label_ != "PER"
                and not any(s < ent.end and ent.start < e for s, e in rescued)
                and (not gate or self._is_entity_like(ent))]

    @staticmethod
    def _is_entity_like(ent) -> bool:
        """A named entity is a proper-noun phrase, not a clause or a discourse token.
        The base model spreads clause-level span errors and function words across its
        MISC/ORG/LOC labels ("Por tanto", "Llama la atención la presencia de casos…");
        requiring a proper noun and rejecting any verb drops those at the source, while
        keeping real orgs/places/laws/events (all PROPN, no verb)."""
        pos = {tok.pos_ for tok in ent}
        return "PROPN" in pos and not (pos & {"VERB", "AUX"})


@_register("spacy")
def create(settings, gazetteer=None) -> SpacyNer:
    return SpacyNer(settings, gazetteer=gazetteer)
