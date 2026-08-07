"""Port for a forced aligner: given audio and the words known to be spoken in it,
say when each word was said.

Forced alignment, not transcription. The words come from the Diario de Sesiones, so
the aligner is never asked what was said — only where in the audio it happened.
That is what keeps the subtitles as accurate as the stenographers' text, and why no
speech-to-text model appears anywhere in this flow.

Word level rather than cue level, even though only cue boundaries are stored: word
timings are what a CTC aligner produces natively, and grouping them into cues is a
pure text concern that belongs in ``domain.subtitles``. Implementations live in
``infrastructure/aligner/``.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WordTiming:
    """When one word was spoken, in seconds from the start of the audio."""

    start: float
    end: float


@dataclass(frozen=True)
class Alignment:
    """Timings for every word, plus how much the aligner trusts them.

    ``score`` runs 0-100 and answers "does the audio under these timings say what
    the transcript claims" — the question that catches a video paired with the wrong
    speech, which per-word confidence does not. It is reported rather than enforced:
    the check is made with the same acoustic model that produced the timings, so it
    inherits that model's blind spots and can be pessimistic about passages whose
    timings are in fact correct. ``model`` identifies the artifact responsible, so an
    alignment stays attributable after the model is changed."""

    words: list[WordTiming]
    score: float
    model: "ModelArtifact | None" = None


@dataclass(frozen=True)
class ModelArtifact:
    """Which weights produced an alignment: a name, the revision it was pinned at,
    and the digest of the bytes actually loaded."""

    id: str
    revision: str | None = None
    sha256: str | None = None


class AlignerPort(Protocol):
    def align(self, samples, sample_rate: int, words: list[str]) -> Alignment:
        """Time each of ``words`` against ``samples``.

        ``samples`` is mono float32 audio (see ``infrastructure.audio``). Returns one
        timing per input word, in the same order, monotonically non-decreasing —
        every word must be placed, because the transcript asserts that all of them
        were spoken. Audio the words do not account for (applause, an interjection
        from the floor, the chair's handover) is absorbed between them, so a caller
        should expect gaps but never reordering.
        """
        ...
