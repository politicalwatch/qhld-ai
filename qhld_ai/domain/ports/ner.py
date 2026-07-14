"""Port for named-entity recognition over speech text.

Deliberately minimal: the two consumers only need raw spans. The adapter returns
every span *verbatim* (including honorifics like "el señor Sánchez" or leading
articles like "la guerra de Gaza"); normalization and resolution are a separate
domain concern (``domain.mentions`` for persons, ``domain.entities`` for the
rest), kept out of the port so it stays a thin wrapper over whatever NER engine
backs it.
"""

from typing import Protocol


class NerPort(Protocol):
    def person_spans(self, text: str) -> list[str]:
        """Return the text of every person (PER) entity found in ``text``, in
        order of appearance and with duplicates preserved (callers count them)."""
        ...

    def entity_spans(self, text: str) -> list[str]:
        """Return the text of every NON-person entity found in ``text``, in order
        of appearance and with duplicates preserved. All non-PER labels are
        pooled: the model assigns ORG/LOC/MISC too erratically for the label to
        carry meaning (see ``domain.entities``)."""
        ...
