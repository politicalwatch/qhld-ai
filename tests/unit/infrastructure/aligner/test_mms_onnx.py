"""Offline tests for the MMS aligner — no model is downloaded or loaded.

The acoustic model is replaced by a stub that emits a chosen character per frame, so
the alignment maths is checked against sequences whose correct answer is known by
construction rather than by trusting a model's output.
"""

import builtins

import numpy as np
import pytest

from qhld_ai.domain.ports.aligner import ModelArtifact
from qhld_ai.infrastructure.aligner.mms_onnx import (
    FRAME_SECS,
    MmsOnnxAligner,
    _fill_gaps,
    _normalise,
    _runs,
    _speech_mask,
    _verbalise,
)

pytestmark = pytest.mark.unit


# The MMS vocabulary's shape: a blank plus romanized letters, no word delimiter and
# no digits — which is what makes verbalizing numerals necessary.
VOCAB = {"<blank>": 0, "a": 1, "b": 2, "c": 3, "h": 4, "l": 5, "o": 6, "s": 7,
         "e": 8, "n": 9, "r": 10, "i": 11, "d": 12, "t": 13, "u": 14, "v": 15,
         "y": 16, "p": 17, "m": 18, "q": 19, "g": 20, "f": 21, "j": 22, "z": 23,
         "x": 24, "k": 25, "w": 26, "'": 27}
# A Spanish-specific vocabulary instead keeps the accents and adds a delimiter.
VOCAB_ES = dict(VOCAB, **{"|": 28, "ñ": 29, "ó": 30, "í": 31})


def _aligner(vocab=VOCAB):
    """An aligner with its lazy resources stubbed — no file is read, no session
    opened, and the artifact digest is not computed over a model that isn't there."""
    aligner = MmsOnnxAligner(model_path="/nowhere/model.onnx")
    aligner._vocab = vocab
    aligner._artifact = ModelArtifact(id="stub", revision="test", sha256="0" * 64)
    return aligner


def _emissions(text, vocab=VOCAB, frames_per_char=3):
    """Emissions that say ``text`` — each character certain for a few frames."""
    rows = []
    for character in text:
        index = vocab[character] if character != " " else 0
        for _ in range(frames_per_char):
            row = np.full(len(vocab), -20.0, dtype=np.float32)
            row[index] = 0.0
            rows.append(row)
    return np.array(rows, dtype=np.float32)


# ---- the alignment itself ----------------------------------------------------

def test_viterbi_places_each_token_where_it_is_spoken():
    aligner = _aligner()
    # "hola" then silence then "que": the second word must land after the gap.
    emissions = _emissions("hola   que")
    tokens, spans = aligner._tokenize(["hola", "que"])

    starts, ends = aligner._viterbi(emissions, np.array(tokens, dtype=np.int64))

    assert starts[spans[0][0]] == 0                 # 'h' in the first frame
    assert starts[spans[1][0]] >= 21                # 'q' after the silent frames
    # The final token's own duration is recovered, not collapsed to its onset.
    assert ends[spans[1][1] - 1] > starts[spans[1][1] - 1]


def test_align_returns_one_timing_per_word_in_order():
    aligner = _aligner()
    aligner._emissions = lambda samples: _emissions("hola   que")

    alignment = aligner.align(np.zeros(16000, dtype=np.float32), 16000,
                              ["hola", "que"])

    assert len(alignment.words) == 2
    assert alignment.words[0].start == pytest.approx(0.0)
    assert alignment.words[0].end < alignment.words[1].start
    assert alignment.words[1].end == pytest.approx(30 * FRAME_SECS)


def test_align_absorbs_audio_the_transcript_does_not_account_for():
    """Applause, an interjection or the chair's handover has no matching token, and
    the alignment must still place every word, in order.

    Where exactly the unaccounted audio is absorbed is NOT guaranteed: no token
    matches it, so every position costs the path the same and the choice among them
    is arbitrary. In real speech the words either side anchor it and the drift stays
    local, which is why the measured cue accuracy holds; a wildcard token exists for
    pathological cases and is deliberately not enabled by default.
    """
    aligner = _aligner()
    aligner._emissions = lambda samples: _emissions("hola" + "b" * 30 + "que")

    alignment = aligner.align(np.zeros(16000, dtype=np.float32), 16000,
                              ["hola", "que"])

    assert len(alignment.words) == 2
    assert alignment.words[0].start <= alignment.words[0].end
    assert alignment.words[0].end <= alignment.words[1].start
    assert alignment.words[1].start <= alignment.words[1].end


def test_align_rejects_the_wrong_sample_rate():
    with pytest.raises(ValueError, match="16 kHz"):
        _aligner().align(np.zeros(100, dtype=np.float32), 44100, ["hola"])


def test_align_rejects_a_transcript_too_long_for_the_audio():
    aligner = _aligner()
    aligner._emissions = lambda samples: _emissions("ho")

    with pytest.raises(ValueError, match="unlikely to match"):
        aligner.align(np.zeros(16000, dtype=np.float32), 16000, ["hola"] * 20)


def test_align_rejects_a_transcript_with_nothing_alignable():
    with pytest.raises(ValueError, match="no alignable tokens"):
        _aligner().align(np.zeros(16000, dtype=np.float32), 16000, ["...", "¿?"])


def test_confidence_is_high_when_the_audio_matches_the_words():
    aligner = _aligner()
    aligner._emissions = lambda samples: _emissions("hola   que")

    alignment = aligner.align(np.zeros(16000, dtype=np.float32), 16000,
                              ["hola", "que"])

    assert alignment.score > 90


def test_confidence_is_low_when_the_audio_says_something_else():
    aligner = _aligner()
    aligner._emissions = lambda samples: _emissions("bbbbbbbbbb")

    alignment = aligner.align(np.zeros(16000, dtype=np.float32), 16000,
                              ["hola", "que"])

    assert alignment.score < 50


# ---- tokenization -----------------------------------------------------------

def test_tokenize_gives_every_word_its_own_token_slice():
    aligner = _aligner()
    tokens, spans = aligner._tokenize(["hola", "que"])

    assert len(spans) == 2
    assert [tokens[a:b] for a, b in spans] == [
        [VOCAB[c] for c in "hola"], [VOCAB[c] for c in "que"]]


def test_tokenize_inserts_the_delimiter_when_the_vocabulary_has_one():
    aligner = _aligner(VOCAB_ES)
    tokens, spans = aligner._tokenize(["hola", "que"])

    assert VOCAB_ES["|"] in tokens
    # The delimiter belongs to no word, so the spans must skip over it.
    assert [tokens[a:b] for a, b in spans] == [
        [VOCAB_ES[c] for c in "hola"], [VOCAB_ES[c] for c in "que"]]


def test_tokenize_keeps_accents_when_the_vocabulary_has_them():
    assert _normalise("señor", VOCAB_ES, fold_accents=False) == "señor"
    assert _normalise("señor", VOCAB, fold_accents=True) == "senor"


def test_unalignable_word_still_gets_a_timing_slot():
    aligner = _aligner()
    tokens, spans = aligner._tokenize(["hola", "—", "que"])

    assert len(spans) == 3
    # The dash borrows the following word's opening rather than vanishing, so the
    # caller's word list and the returned timings stay index-for-index.
    assert spans[1][0] == spans[2][0]


def test_fill_gaps_handles_a_trailing_unalignable_word():
    assert _fill_gaps([(0, 4), None], 4) == [(0, 4), (0, 4)]


def test_fill_gaps_handles_nothing_alignable_at_all():
    assert _fill_gaps([None], 0) == [(0, 0)]


# ---- verbalizing numerals ---------------------------------------------------

@pytest.mark.parametrize("raw, spoken", [
    ("49", "cuarenta y nueve"),
    ("2017", "dos mil diecisiete"),
    ("118.885", "ciento dieciocho mil ochocientos ochenta y cinco"),
    ("47,5", "cuarenta y siete coma cinco"),
    ("%", "por ciento"),
])
def test_verbalise_reads_numerals_and_symbols_out(raw, spoken):
    assert _verbalise(raw) == spoken


def test_verbalise_strips_punctuation_around_a_numeral():
    assert _verbalise("(2017)") == "dos mil diecisiete"


def test_verbalise_leaves_words_alone():
    assert _verbalise("silicosis") == "silicosis"
    assert _verbalise("COVID-19") == "COVID-19"   # not a bare numeral


# ---- voice activity ---------------------------------------------------------

def test_speech_mask_marks_loud_frames_and_clears_silence():
    quiet = np.zeros(320 * 50, dtype=np.float32)
    loud = np.ones(320 * 50, dtype=np.float32) * 0.5
    samples = np.concatenate([quiet, loud])

    active = _speech_mask(samples, dilate=0)

    assert not active[:40].any()
    assert active[60:].all()


def test_speech_mask_dilates_so_quiet_onsets_stay_attached():
    samples = np.concatenate([np.zeros(320 * 50, dtype=np.float32),
                              np.ones(320 * 50, dtype=np.float32) * 0.5])

    dilated = _speech_mask(samples, dilate=10)

    assert dilated[45]          # the frames just before the onset are kept
    assert not dilated[:30].any()


def test_speech_mask_on_empty_audio():
    assert len(_speech_mask(np.zeros(0, dtype=np.float32))) == 0


def test_runs_finds_contiguous_speech_ranges():
    active = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)

    assert _runs(active) == [(1, 3), (5, 6)]


def test_missing_verbalizer_fails_loudly(monkeypatch):
    """A figure left unspoken degrades the alignment invisibly, so an incomplete
    install must raise rather than quietly drop the numeral."""
    import qhld_ai.infrastructure.aligner.mms_onnx as module

    monkeypatch.setattr(module, "_NUM2WORDS", None)
    real_import = builtins.__import__

    def deny(name, *args, **kwargs):
        if name == "num2words":
            raise ImportError("No module named 'num2words'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny)
    with pytest.raises(RuntimeError, match="num2words is required"):
        _verbalise("2017")
