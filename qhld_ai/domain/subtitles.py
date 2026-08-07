"""Pure subtitle cue logic — no I/O, no deps.

Two halves of the same artifact. ``build_cues`` decides *what* each subtitle line
says, working from the transcript alone; an aligner then supplies *when*, and
``render_vtt`` writes the WebVTT a player's ``<track>`` loads.

Cues are built before alignment on purpose. A cue is the unit we time, so the
stenographers' departures from the words actually spoken — cleaned-up disfluencies,
expanded forms — stop mattering once they fall inside a cue whose boundaries land;
timing individual words would expose every one of them.

A cue carries character offsets rather than a copy of its text, so the text is
sliced back out at render time. Subtitles then cannot drift from the transcript,
and nothing stores the same prose twice. The offsets are into one ``SpeechText``
block's stored string — annotations included, since that is what readers see and
what search highlights are located in — which makes a highlight's offset and a
cue's offset directly comparable.

Segmentation is budgeted by characters and words, following the subtitling
conventions the cues have to satisfy: at most two lines of ~42 characters, and a
line short enough to read at ~21 characters per second.
"""

import re
from dataclasses import dataclass


# Cue boundaries, strongest first: a full stop is a better place to cut than a
# comma, so a cue closes at sentence punctuation and only falls back to length.
# Kept local so this module has no cross-domain import.
_SEGMENT_SPLIT = re.compile(r"[^.;:!?\n]+[.;:!?\n]?")
_WORD = re.compile(r"\S+")

# WebVTT requires this exact header, and timestamps in HH:MM:SS.mmm.
_VTT_HEADER = "WEBVTT"


@dataclass(frozen=True)
class CueSpan:
    """One subtitle line before it has been timed: the text and where it sits in the
    block it came from. ``text`` is carried for convenience while building, never
    stored — ``char_start``/``char_end`` are the record."""

    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Cue:
    """A timed cue. ``start``/``end`` are seconds from the start of the clip."""

    char_start: int
    char_end: int
    start: float
    end: float


def build_cues(text, max_chars=84, max_words=14):
    """Split ``text`` into subtitle-sized spans.

    Sentence-ish segments first (on ``.;:!?`` and newlines), then each segment is
    packed greedily until it would exceed ``max_chars`` or ``max_words``. Offsets
    are into ``text``; the whitespace between cues belongs to no cue, so slicing
    every cue never reproduces the original string exactly and is not meant to.
    """
    text = text or ""
    cues = []
    for segment in _SEGMENT_SPLIT.finditer(text):
        base = segment.start()
        words = list(_WORD.finditer(segment.group(0)))
        if not words:
            continue
        current = []
        for word in words:
            current.append(word)
            span = current[-1].end() - current[0].start()
            if span >= max_chars or len(current) >= max_words:
                cues.append(_span(segment.group(0), base, current))
                current = []
        if current:
            cues.append(_span(segment.group(0), base, current))
    return cues


def _span(segment, base, words):
    start, end = words[0].start(), words[-1].end()
    return CueSpan(text=segment[start:end],
                   char_start=base + start, char_end=base + end)


def word_spans(text):
    """Every whitespace-separated word of ``text`` with its offsets.

    The aligner works word by word while cues are cut from the same string, so both
    have to agree on what a word is and where it starts — hence one definition here
    rather than a second regex at the call site."""
    return [CueSpan(text=match.group(0),
                    char_start=match.start(), char_end=match.end())
            for match in _WORD.finditer(text or "")]


def render_vtt(cues, text):
    """The WebVTT track for ``cues``, with each cue's text sliced out of ``text``.

    Emitted rather than stored: the format is a projection of the offsets, so a
    transcript correction reaches the subtitles without a re-alignment, and a
    mismatch between the two is impossible by construction.
    """
    text = text or ""
    lines = [_VTT_HEADER, ""]
    for index, cue in enumerate(cues, start=1):
        body = " ".join(text[cue.char_start:cue.char_end].split())
        if not body:
            continue
        lines.append(str(index))
        lines.append(f"{_timestamp(cue.start)} --> {_timestamp(cue.end)}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def _timestamp(seconds):
    """``HH:MM:SS.mmm``. Milliseconds are truncated, not rounded, so a cue never
    starts later than the word it captions."""
    if seconds < 0:
        seconds = 0.0
    millis = int(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def cue_at(cues, char_offset):
    """The index of the cue covering ``char_offset``, or ``None``.

    What turns a search highlight into a seek: the highlight's position in the
    transcript is already known in these coordinates, so finding its cue is a
    lookup. Cues are ordered and non-overlapping, so this is a binary search; a
    miss means the offset fell in the whitespace between two cues, and the caller
    decides whether to round to the next one.
    """
    low, high = 0, len(cues) - 1
    while low <= high:
        mid = (low + high) // 2
        cue = cues[mid]
        if char_offset < cue.char_start:
            high = mid - 1
        elif char_offset >= cue.char_end:
            low = mid + 1
        else:
            return mid
    return None
