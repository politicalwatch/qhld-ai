"""Unit tests for the pure subtitle cue logic — no I/O, no model."""

from dataclasses import dataclass, field

import pytest

from qhld_ai.domain.subtitles import (
    Cue,
    aligned_text,
    build_cues,
    cue_at,
    render_vtt,
    subtitle_track,
    text_fingerprint,
)

pytestmark = pytest.mark.unit


# The stored shapes, structurally: the domain reads these by attribute and never
# imports them, so the tests need not drag tipi_data in either.

@dataclass
class Block:
    lang: str
    text: str


@dataclass
class StoredCue:
    start_ms: int
    end_ms: int
    char_start: int
    char_end: int


@dataclass
class Alignment:
    lang: str
    block_index: int
    text_sha256: str
    text_length: int
    cues: list = field(default_factory=list)


def _alignment(block, cues=(), block_index=0):
    digest, length = text_fingerprint(block.text)
    return Alignment(lang=block.lang, block_index=block_index,
                     text_sha256=digest, text_length=length, cues=list(cues))


# ---- building cues ----

def test_empty_text_yields_no_cues():
    assert build_cues("") == []
    assert build_cues("   ") == []


def test_short_sentence_is_a_single_cue():
    cues = build_cues("Muchas gracias, presidente.")

    assert len(cues) == 1
    assert cues[0].text == "Muchas gracias, presidente."


def test_offsets_index_the_original_text():
    text = "Muchas gracias, presidente. He escuchado con atención."
    for cue in build_cues(text):
        assert text[cue.char_start:cue.char_end] == cue.text


def test_splits_on_sentence_punctuation():
    cues = build_cues("Primera. Segunda; tercera: cuarta? quinta!")

    assert [cue.text for cue in cues] == [
        "Primera.", "Segunda;", "tercera:", "cuarta?", "quinta!"]


def test_long_sentence_is_split_by_word_budget():
    text = " ".join(["palabra"] * 30) + "."
    cues = build_cues(text, max_chars=999, max_words=10)

    assert len(cues) == 3
    assert all(len(cue.text.split()) <= 10 for cue in cues)


def test_long_sentence_is_split_by_char_budget():
    text = " ".join(["palabra"] * 30) + "."
    cues = build_cues(text, max_chars=40, max_words=999)

    assert len(cues) > 1
    assert all(len(cue.text) <= 48 for cue in cues)  # the word that trips the budget


def test_cues_are_ordered_and_non_overlapping():
    text = "Uno dos tres. Cuatro cinco seis. Siete ocho nueve."
    cues = build_cues(text, max_chars=12, max_words=2)

    for earlier, later in zip(cues, cues[1:]):
        assert earlier.char_end <= later.char_start


def test_annotations_stay_inside_the_offsets():
    # Cues span the stored text, stage directions included: the transcript readers
    # see is the string these offsets are for.
    text = "Muchas gracias. (Aplausos). Continúo."
    cues = build_cues(text)

    assert any("(Aplausos)" in cue.text for cue in cues)


# ---- rendering WebVTT ----

def _timed(text, cues, step=2.0):
    return [Cue(char_start=c.char_start, char_end=c.char_end,
                start=i * step, end=i * step + step)
            for i, c in enumerate(build_cues(text) if cues is None else cues)]


def test_vtt_has_the_required_header():
    vtt = render_vtt(_timed("Hola.", None), "Hola.")

    assert vtt.startswith("WEBVTT\n\n")


def test_vtt_carries_timestamps_and_sliced_text():
    text = "Muchas gracias, presidente."
    cues = [Cue(char_start=0, char_end=len(text), start=3.42, end=7.18)]

    vtt = render_vtt(cues, text)

    assert "00:00:03.420 --> 00:00:07.180" in vtt
    assert "Muchas gracias, presidente." in vtt


def test_vtt_timestamps_carry_hours_and_truncate_millis():
    cues = [Cue(char_start=0, char_end=4, start=3661.9999, end=3662.5)]

    vtt = render_vtt(cues, "hola")

    # Truncated, not rounded: a cue must never start after the word it captions.
    assert "01:01:01.999 --> 01:01:02.500" in vtt


def test_vtt_collapses_whitespace_in_cue_text():
    text = "Muchas\n  gracias."
    cues = [Cue(char_start=0, char_end=len(text), start=0.0, end=1.0)]

    assert "Muchas gracias." in render_vtt(cues, text)


def test_vtt_skips_cues_whose_slice_is_empty():
    # A stale offset pointing past the text must not emit a blank cue.
    cues = [Cue(char_start=100, char_end=120, start=0.0, end=1.0)]

    assert render_vtt(cues, "corto").strip() == "WEBVTT"


def test_vtt_escapes_markup_characters_in_cue_text():
    # "<" opens a cue span in WebVTT and "&" opens an entity, so raw ones would
    # swallow or garble the rest of the line.
    text = "El PP & Vox <interrumpen>."
    cues = [Cue(char_start=0, char_end=len(text), start=0.0, end=1.0)]

    vtt = render_vtt(cues, text)

    assert "El PP &amp; Vox &lt;interrumpen&gt;." in vtt


def test_vtt_numbers_cues_from_one():
    text = "Uno. Dos."
    spans = build_cues(text)
    cues = [Cue(char_start=s.char_start, char_end=s.char_end,
                start=float(i), end=float(i + 1))
            for i, s in enumerate(spans)]

    vtt = render_vtt(cues, text)

    assert "\n1\n" in vtt and "\n2\n" in vtt


# ---- highlight -> cue lookup ----

def test_cue_at_finds_the_covering_cue():
    text = "Uno dos tres. Cuatro cinco seis."
    spans = build_cues(text)
    cues = [Cue(char_start=s.char_start, char_end=s.char_end, start=0.0, end=1.0)
            for s in spans]

    assert cue_at(cues, spans[1].char_start) == 1
    assert cue_at(cues, spans[0].char_end - 1) == 0


def test_cue_at_returns_none_between_cues():
    cues = [Cue(char_start=0, char_end=5, start=0.0, end=1.0),
            Cue(char_start=10, char_end=15, start=1.0, end=2.0)]

    assert cue_at(cues, 7) is None
    assert cue_at(cues, 99) is None


def test_cue_at_on_empty_track():
    assert cue_at([], 0) is None


# ---- reading a stored alignment back ----

def test_fingerprint_is_stable_and_carries_the_length():
    digest, length = text_fingerprint("Muchas gracias.")

    assert (digest, length) == text_fingerprint("Muchas gracias.")
    assert length == 15
    assert text_fingerprint(None) == text_fingerprint("")


def test_aligned_text_returns_the_block_it_was_made_against():
    block = Block(lang="es", text="Muchas gracias, presidente.")
    alignment = _alignment(block)

    assert aligned_text([block], 0, alignment.lang,
                        alignment.text_sha256, alignment.text_length) == block.text


def test_aligned_text_rejects_a_transcript_that_has_changed():
    block = Block(lang="es", text="Muchas gracias, presidente.")
    alignment = _alignment(block)
    corrected = Block(lang="es", text="Muchas gracias, presidenta.")

    assert aligned_text([corrected], 0, alignment.lang,
                        alignment.text_sha256, alignment.text_length) is None


def test_aligned_text_rejects_a_block_of_another_language():
    # Same length, same position, different block: a re-ordered speech must not
    # caption the original with its translation.
    block = Block(lang="gl", text="Grazas, señora presidenta.")
    alignment = _alignment(block)
    translation = Block(lang="es", text="Gracias, señora presidenta")

    assert aligned_text([translation], 0, alignment.lang,
                        alignment.text_sha256, alignment.text_length) is None


def test_aligned_text_rejects_a_block_index_that_no_longer_exists():
    block = Block(lang="gl", text="Grazas.")
    alignment = _alignment(block, block_index=1)

    assert aligned_text([block], 1, alignment.lang,
                        alignment.text_sha256, alignment.text_length) is None


def test_subtitle_track_renders_the_stored_cues():
    block = Block(lang="es", text="Muchas gracias. He escuchado con atención.")
    cues = [StoredCue(start_ms=3420, end_ms=7180, char_start=0, char_end=15),
            StoredCue(start_ms=7180, end_ms=11000, char_start=16, char_end=42)]

    vtt = subtitle_track(_alignment(block, cues), [block])

    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:03.420 --> 00:00:07.180" in vtt
    assert "Muchas gracias." in vtt
    assert "He escuchado con atención." in vtt


def test_subtitle_track_reads_the_block_the_offsets_belong_to():
    # A co-official speech: the audio is the original, so the cues index block 0
    # and the Spanish translation that follows must not be sliced by them.
    original = Block(lang="gl", text="Grazas, señora presidenta.")
    translation = Block(lang="es", text="Gracias, señora presidenta.")
    cues = [StoredCue(start_ms=11980, end_ms=14000, char_start=0, char_end=26)]

    vtt = subtitle_track(_alignment(original, cues), [original, translation])

    assert "Grazas, señora presidenta." in vtt


def test_subtitle_track_withholds_a_stale_alignment():
    block = Block(lang="es", text="Muchas gracias.")
    alignment = _alignment(block, [StoredCue(0, 1000, 0, 15)])

    assert subtitle_track(alignment, [Block(lang="es", text="Otro texto.")]) is None
