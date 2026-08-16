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
