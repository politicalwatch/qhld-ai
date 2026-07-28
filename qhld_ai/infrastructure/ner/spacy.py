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

Three post-passes add the people the model misses, all leaving its own PER spans
untouched and all excluded from ``entity_spans``: the gazetteer below; role
apposition (``_appositions``), which claims a name the text states an office for —
"El ministro Albares", which the model returns as MISC; and courtesy forms
(``_courtesies``), which claim the name a speech addresses politely — "señor
Cuerpo", returned as an ORG. They are complementary by construction: the gazetteer
covers surnames distinctive enough to tag anywhere but only if they are out of
vocabulary, while the other two cover any surname, however common, because the word
before it says that occurrence names a person.

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

from qhld_ai.domain.mentions import role_appositions
from qhld_ai.domain.ports.ner import NerPort

from .factory import _register

# Courtesy words a PER span is extended over, so the gender they carry is always
# available to the resolver. Kept to the forms that actually inflect for gender —
# a role word ("ministro") would change the span's meaning, not just its politeness.
_COURTESY_WORDS = frozenset({
    "señor", "señora", "señores", "señoras", "sr", "sra", "srs", "sras",
    "don", "doña",
})
# The courtesy forms written as an abbreviation, where the full stop belongs to the word
# rather than to the sentence — the only ones a name may be read ACROSS. "La Sra.
# Vallugera" names somebody; "a todo el mundo escucho decir que aquí entra cualquier cosa.
# No, señor. Están los puestos de inspección" does not, and the two differ by nothing else.
_COURTESY_ABBREVIATIONS = frozenset({"sr", "sra", "srs", "sras"})
# How much name a courtesy form may introduce: at most three tokens ("señor Rodríguez
# Zapatero"), each capitalised and long enough to discriminate — the same shape as the
# names the apposition patterns capture (``domain.mentions._NAMED``).
_MAX_NAME_TOKENS = 3
_MIN_NAME_LEN = 3
# Lowercase particles a surname may carry, crossed only when another capitalised token
# follows: "señora Álvarez de Toledo" is one deputy, while a bare "señora Álvarez" is three
# of them and gets dropped as ambiguous. "y"/"e" are deliberately absent — "el señor Feijóo
# y la señora Ayuso" are two people and must not become one span.
_NAME_PARTICLES = frozenset({"de", "del", "la", "las", "los", "da"})
# Definite articles the model sometimes swallows into the span, including the two forms
# contracted with a preposition ("al" = a+el, "del" = de+el — both as common before a
# courtesy form as the bare article: 915 and 958 occurrences against "la"'s 1304).
# Trimmed back off, so a span always begins at the courtesy word: the model's choice here
# tracks CAPITALISATION, not grammar — measured on the corpus, it keeps the article in 29
# of 29 sentence-initial cases ("El señor Abascal…") and drops it in 48 of 49 mid-sentence
# ones ("…con el señor Abascal"). Following the majority also drops an artifact, since no
# article is part of anyone's name.
_LEADING_ARTICLES = frozenset({"el", "la", "los", "las", "al", "del"})
# Role words, for span SHAPE only: a span the model began at one starts at the name
# instead ("ministro Torres" → "Torres"), for the same reason the article is trimmed — it
# is nobody's name, and the span doubles as the site's highlight. The resolution-side
# vocabulary is ``domain.mentions._ROLE_FAMILIES``, which is where the office a role word
# states is actually used; this copy exists because shaping a span is not resolving it
# (``_COURTESY_WORDS`` above is duplicated for the same reason).
_ROLE_WORDS = frozenset({
    "presidente", "presidenta", "vicepresidente", "vicepresidenta",
    "ministro", "ministra",
})


class SpacyNer(NerPort):
    def __init__(self, settings, gazetteer=None, office_surfaces=None):
        self.settings = settings
        self._gazetteer = tuple(gazetteer or ())
        self._office_surfaces = dict(office_surfaces or {})
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

    def _holds_office(self, name: str, family: str) -> bool:
        """Whether some token of the apposed name belongs to a catalog office holder of
        that same family. Deliberately a loose membership test, not a resolution: the
        catalog decides what may be looked for, the resolver decides who was found (see
        ``domain.mentions.build_office_surfaces``)."""
        return any(family in self._office_surfaces.get(token, ())
                   for token in name.lower().replace("-", " ").split()
                   if len(token) >= 3)

    def _appositions(self, doc) -> list[tuple[int, int]]:
        """Names the text states an office for but the model did not tag as people: "El
        ministro Albares" comes back as MISC, "Cuerpo" as ORG, "El señor Sánchez" as MISC.
        They are person spans, so ``person_spans`` adds them and ``entity_spans`` excludes
        them — the same contract, and the same overlap rule, as the gazetteer's
        ``_rescued``: a model PER span always wins, everything else is fair game.

        This is the half of role apposition the gazetteer cannot do. Its patterns only
        cover surnames that are distinctive AND out of vocabulary, and every surname here
        ("Sánchez", "Cuerpo", "Torres") is in vocabulary — a blunt rule on those would tag
        every ordinary use of the word. The role word is what makes it safe: it is local
        evidence that this occurrence names a person, so the office gate can be permissive
        where the gazetteer must not be."""
        if not self._office_surfaces:
            return []
        per = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PER"]
        bounds: list[tuple[int, int]] = []
        for start, end, name, family in role_appositions(doc.text):
            if any(s < end and start < e for s, e in per):
                continue
            if not self._holds_office(name, family):
                continue
            # expand: a capture can start mid-token when the text glues a name to a dash.
            span = doc.char_span(start, end, alignment_mode="expand")
            if span is not None and (span.start, span.end) not in bounds:
                bounds.append((span.start, span.end))
        return bounds

    @staticmethod
    def _name_token(token) -> bool:
        """Whether ``token`` can be part of a name: capitalised, long enough to
        discriminate, and made of letters — a hyphen or an apostrophe apart, which real
        surnames carry ("Grande-Marlaska", "O'Donnell")."""
        text = token.text
        return (len(text) >= _MIN_NAME_LEN and text[0].isupper()
                and all(char.isalpha() or char in "-'’" for char in text))

    def _name_after(self, doc, start: int) -> int:
        """Where the name beginning at ``start`` ends, or ``start`` when none does."""
        end = start
        while end < len(doc) and end - start < _MAX_NAME_TOKENS:
            if self._name_token(doc[end]):
                end += 1
            elif (doc[end].text.lower() in _NAME_PARTICLES and end > start
                  and end + 1 < len(doc) and self._name_token(doc[end + 1])):
                end += 2
            else:
                break
        return end

    def _courtesies(self, doc, claimed) -> list[tuple[int, int]]:
        """Names introduced by a courtesy form that nothing else tagged: "señor Cuerpo",
        which one speech uses fourteen times for the finance minister and the model returns
        as an ORG every time. Person spans like the two passes above, with the same overlap
        rule — a model PER span always wins — and likewise excluded from ``entity_spans``.

        A courtesy form is the same kind of local evidence a role word is, and a far
        commoner one: whatever follows it is a person, whoever that person turns out to be.
        Unlike ``_appositions`` this pass therefore takes no catalog gate. The office gate
        exists there because a role word also appears where it names nobody — in vocatives
        ("Gracias, presidenta") and in office complements — whereas a courtesy form is
        followed by a name essentially always. Measured over 400 speeches: 194 names
        claimed, of which the only ones naming nobody were two "don Quijote"; people from
        outside the catalog (Alfonso Rueda apart, a documented false positive) are inert
        rather than dangerous, because their surname reaches no catalog entry's threshold —
        "señor Carlos Moreno" scores 75, "señor Koldo García" 78, "señor Casado" 57.

        What must be exact is where the name STARTS: only an abbreviated courtesy form may
        be read across a full stop, or "No, señor. Están los puestos…" claims a verb.
        Returned bounds cover the name alone; ``_courtesy_start`` then extends the span
        back over the courtesy word, so these spans carry the gender cue and highlight the
        same surface as every other one, without this pass knowing about either."""
        if not getattr(self.settings, "ner_courtesy_form", True):
            return []
        taken = [(ent.start, ent.end) for ent in doc.ents if ent.label_ == "PER"]
        taken.extend(claimed)
        bounds: list[tuple[int, int]] = []
        for token in doc:
            word = token.text.lower().rstrip(".")
            if word not in _COURTESY_WORDS:
                continue
            start = token.i + 1
            if (word in _COURTESY_ABBREVIATIONS
                    and start < len(doc) and doc[start].text == "."):
                start += 1
            end = self._name_after(doc, start)
            if end == start:
                continue
            if any(s < end and start < e for s, e in taken):
                continue
            bounds.append((start, end))
        return bounds

    def _claimed(self, doc) -> list[tuple[int, int]]:
        """Every span the post-passes take for the person side.

        The courtesy pass is given what the other two claimed, because it is the one that
        would otherwise duplicate them: it fires on the same names from a different cue,
        and "la señora Vallugera" is one mention whether the gazetteer or the courtesy word
        found it, not two. The gazetteer and apposition passes cannot collide with each
        other by construction — the gazetteer only patterns out-of-vocabulary surnames and
        every office holder's is in vocabulary — and measured over 400 speeches they never
        did."""
        claimed = self._rescued(doc) + self._appositions(doc)
        return claimed + self._courtesies(doc, claimed)

    def _role_start(self, doc, start: int, end: int) -> int:
        """``start`` moved past a leading role word, so the span holds the name alone:
        "ministro Torres" → "Torres".

        Only a LEADING one can go: "señor ministro Torres" keeps its courtesy form, which
        carries the gender cue, and a role word in the middle of a span cannot be removed
        without breaking it. A role-only span ("Ministra") is left alone here and drops out
        later, when normalization finds no name in it."""
        if end - start > 1 and doc[start].text.lower() in _ROLE_WORDS:
            return start + 1
        return start

    def _courtesy_start(self, doc, start: int, end: int) -> int:
        """``start`` moved so the span begins at its courtesy word, in either direction.

        The model has no settled behaviour here: across the corpus it left the courtesy
        word OUT of the span 1,259 times and put it IN 1,160 — near a coin flip — and that
        inconsistency loses real information, because the courtesy form agrees in gender
        with the person named, which is what lets the resolver tell two holders of one
        surname apart. So it is normalised rather than trusted:

        - extend LEFT over an adjacent courtesy word the model dropped;
        - trim a leading article the model swallowed, so "El señor Abascal" and
          "el señor Abascal" both yield "señor Abascal".

        The article is excluded both ways because it is nobody's name, and because the
        span doubles as the highlight target on the site."""
        probe = start - 1
        if probe >= 0 and doc[probe].text == ".":      # "Sra." tokenizes as "Sra" + "."
            probe -= 1
        if probe >= 0 and doc[probe].text.lower().rstrip(".") in _COURTESY_WORDS:
            return probe
        # nothing to absorb — but the model may have taken the article in already
        if (end - start > 1
                and doc[start].text.lower() in _LEADING_ARTICLES
                and doc[start + 1].text.lower().rstrip(".") in _COURTESY_WORDS):
            return start + 1
        return start

    def person_spans(self, text: str) -> list[str]:
        if not text:
            return []
        doc = self._doc(text)
        bounds = [(ent.start, ent.end) for ent in doc.ents if ent.label_ == "PER"]
        bounds.extend(self._claimed(doc))
        spans = [(start, end, doc[self._role_start(
                     doc, self._courtesy_start(doc, start, end), end):end].text)
                 for start, end in bounds]
        spans.sort()
        return [span_text for _, _, span_text in spans]

    def entity_spans(self, text: str) -> list[str]:
        if not text:
            return []
        doc = self._doc(text)
        # The post-passes claim their spans for the person side, so none can also be an
        # entity: a surname the model mislabelled ("Cuerpo" as ORG, "El ministro Albares"
        # as MISC) is a person, and leaving it in the entity index is what put bare
        # surnames in the theme filter.
        claimed = self._claimed(doc)
        gate = getattr(self.settings, "ner_entity_pos_gate", True)
        return [ent.text for ent in doc.ents
                if ent.label_ != "PER"
                and not any(s < ent.end and ent.start < e for s, e in claimed)
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
def create(settings, gazetteer=None, office_surfaces=None) -> SpacyNer:
    return SpacyNer(settings, gazetteer=gazetteer, office_surfaces=office_surfaces)
