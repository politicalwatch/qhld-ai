"""Domain errors raised by the search flow."""


class SearchRefused(Exception):
    """Base for the refusals natural search raises before retrieving anything.

    Each subclass carries a ``reason`` that travels to the client, because the
    only useful thing a refused user can be told is WHY — and the reasons need
    different words. They are deliberately separate classes rather than one error
    with a flag: the measurement harness scores them as different outcomes, and a
    refusal delivered with the wrong explanation is its own defect."""

    reason = "refused"


class NotASpeechQuery(SearchRefused):
    """The input is not a search over parliamentary speeches (an instruction,
    a content-generation request, a question to the assistant, an attempt to
    change behaviour). Distinct from a search that legitimately finds nothing:
    it is a rejection, so callers surface it as an error rather than an empty
    result set."""

    reason = "not_a_speech_search"


class PromptInjection(SearchRefused):
    """An unambiguous attempt to make the system act rather than search —
    overriding its instructions, extracting its prompt or configuration,
    bypassing a filter or a limit, adopting an unrestricted persona, or
    extracting internal data.

    A strict subset of ``NotASpeechQuery`` in meaning, but a separate class
    because the CONSEQUENCE differs rather than the wording: this is the only
    refusal a caller can be banned for on a single occurrence. It is raised
    before the intent gate so the more serious classification wins.

    Deliberately narrow. An attack that is not recognised as one still fails the
    intent gate and is still refused, so the two classes compose and a miss is
    cheap; a false positive costs a real user their access outright. Recall is
    the property to give up here, never precision.

    ``reason`` never reaches the client: the API reports every refusal of this
    shape as ``not_a_speech_search``, so a caller is not told which classifier
    they tripped. It exists for what we record and what we count."""

    reason = "prompt_injection"


class UnsupportedLanguage(SearchRefused):
    """A genuine speech search, written in a language the product does not serve.

    Kept apart from ``NotASpeechQuery`` because the user did nothing wrong and the
    corpus may well cover their topic: the honest message is "ask in one of these
    languages", not "this is not a search". ``language`` is what the parser read,
    so a client can name it."""

    reason = "unsupported_language"

    def __init__(self, query, language=None):
        super().__init__(query)
        self.language = language
