"""Domain errors raised by the search flow."""


class NotASpeechQuery(Exception):
    """The input is not a search over parliamentary speeches (an instruction,
    a content-generation request, a question to the assistant, an attempt to
    change behaviour). Distinct from a search that legitimately finds nothing:
    it is a rejection, so callers surface it as an error rather than an empty
    result set."""
