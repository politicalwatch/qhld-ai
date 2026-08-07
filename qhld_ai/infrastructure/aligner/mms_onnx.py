"""Forced alignment with the MMS-300m CTC aligner, run through ONNX Runtime.

MMS is a multilingual wav2vec2 model fine-tuned for alignment rather than
recognition, which is what a corpus in Spanish, Galician, Catalan and Basque needs:
one model covers all four, so a co-official-language intervention takes the same
path as any other.

ONNX Runtime on the **CPU in fp32**, deliberately. Measured on Apple Silicon,
CoreML (GPU + Neural Engine) is about three times slower than the plain CPU
provider and dynamic int8 quantisation about twice as slow as fp32 — the optimised
fp32 kernels beat both. That the fastest configuration is also the one a CPU-only
server runs is what lets a bulk load and the daily increment share a single code
path and produce identical timings. The session (and onnxruntime itself) is loaded
lazily, so importing this module stays cheap for callers that never align.

The model artifact is downloaded pre-converted and pinned by revision, so PyTorch
is never a dependency here. That the conversion is faithful to the author's own
weights is not assumed: ``scripts/verify_aligner_model.py`` in qhld-engine
reproduces the comparison.
"""

import hashlib
import json
import re
import unicodedata

from qhld_ai.domain.ports.aligner import Alignment, AlignerPort, ModelArtifact, WordTiming
from qhld_ai.infrastructure.config.settings import Settings
from qhld_ai.logger import get_logger

from .factory import _register

_logger = get_logger(__name__)

BLANK = 0
FRAME_SECS = 0.02          # wav2vec2 downsamples 16 kHz audio by 320
_FRAME_SAMPLES = 320
# Attention is quadratic in the window, so a twelve-minute clip cannot go through in
# one pass. 20 s is comfortably inside the length the model was trained on.
_CHUNK_FRAMES = int(20.0 / FRAME_SECS)

# Silence is not merely wasted compute. A window dominated by it produces badly
# degraded emissions — measured on a real intervention that opens with the chair's
# off-mic handover, a 20 s window at 60% silence decoded to noise while the same
# audio trimmed to the speech decoded perfectly, and the degradation was monotonic
# in the silence fraction. So silence is kept out of the model entirely and filled
# with a blank-certain row, which is what CTC should see there anyway.
_VAD_WINDOW = _FRAME_SAMPLES
_VAD_RELATIVE_FLOOR = 0.15
_VAD_DILATE_FRAMES = 10
_SILENT_LOGPROB = -20.0

# Confidence sampling: how many stretches of the alignment are decoded back and
# compared against the words they claim, and how long each stretch is.
_CHECK_SAMPLES = 12
_CHECK_WORDS = 8


class MmsOnnxAligner(AlignerPort):
    def __init__(self, model_path="", repo="", revision="", model_file="",
                 vocab_file="vocab.json", threads=0):
        self._model_path = model_path
        self._repo = repo
        self._revision = revision
        self._model_file = model_file
        self._vocab_file = vocab_file
        self._threads = threads
        self._session = None
        self._vocab = None
        self._resolved = None
        self._artifact = None

    # ---- lazily acquired resources -----------------------------------------

    @property
    def paths(self):
        """``(model, vocab)`` on the local filesystem.

        An explicit ``model_path`` wins, which is how an air-gapped install or an
        artifact we publish ourselves is used; otherwise both files come from the
        pinned repository revision, so one pin covers them together.
        """
        if self._resolved is None:
            if self._model_path:
                from pathlib import Path

                model = Path(self._model_path)
                self._resolved = (str(model), str(model.parent / self._vocab_file))
            else:
                from huggingface_hub import hf_hub_download

                kwargs = {"revision": self._revision} if self._revision else {}
                self._resolved = (
                    hf_hub_download(self._repo, self._model_file, **kwargs),
                    hf_hub_download(self._repo, self._vocab_file, **kwargs),
                )
            _logger.debug("aligner model at %s", self._resolved[0])
        return self._resolved

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            if self._threads:
                options.intra_op_num_threads = self._threads
            self._session = ort.InferenceSession(
                self.paths[0], options, providers=["CPUExecutionProvider"])
        return self._session

    @property
    def vocab(self):
        if self._vocab is None:
            with open(self.paths[1]) as handle:
                self._vocab = json.load(handle)
        return self._vocab

    @property
    def artifact(self):
        """Which bytes are in use. The digest is what makes a silent change to a
        pinned upstream file detectable rather than merely unlikely.

        Hashed once per adapter — it reads the whole 1.2 GB file, which is nothing
        against a run but would be absurd once per speech in a bulk one."""
        if self._artifact is None:
            digest = hashlib.sha256()
            with open(self.paths[0], "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            self._artifact = ModelArtifact(id=self._model_path or self._repo,
                                           revision=self._revision or None,
                                           sha256=digest.hexdigest())
        return self._artifact

    # ---- the port -----------------------------------------------------------

    def align(self, samples, sample_rate, words):
        import numpy as np

        if sample_rate != 16000:
            raise ValueError(
                f"the aligner expects 16 kHz audio, got {sample_rate} Hz")
        tokens, spans = self._tokenize(words)
        if not tokens:
            raise ValueError("no alignable tokens in the transcript")

        emissions = self._emissions(samples)
        if len(emissions) < len(tokens):
            raise ValueError(
                f"{len(words)} words need more audio than {len(emissions) * FRAME_SECS:.0f}s "
                "can hold; the video and the transcript are unlikely to match")
        starts, ends = self._viterbi(emissions, np.array(tokens, dtype=np.int64))

        timings = [
            WordTiming(start=float(starts[first] * FRAME_SECS),
                       end=float((ends[last - 1] + 1) * FRAME_SECS))
            for first, last in spans
        ]
        score = self._confidence(emissions, words, spans, starts, ends)
        return Alignment(words=timings, score=score, model=self.artifact)

    # ---- text -> tokens -----------------------------------------------------

    def _tokenize(self, words):
        """Token ids for every word, plus each word's ``(first, last)`` slice of them.

        Words map to tokens rather than the reverse, so a word that expands (a
        numeral becoming several spoken words) still yields exactly one timing, and
        the caller's word list and the returned timings stay index-for-index.
        """
        vocab = self.vocab
        fold = "ñ" not in vocab
        delimiter = vocab.get("|")
        tokens, spans = [], []
        for word in words:
            normalised = _normalise(_verbalise(word), vocab, fold)
            if not normalised:
                # Punctuation, or a symbol this vocabulary has no token for. It is
                # skipped, not dropped: the span stays empty and the word inherits
                # its neighbours' boundary below.
                spans.append(None)
                continue
            if delimiter is not None and tokens:
                tokens.append(delimiter)
            spans.append((len(tokens), len(tokens) + len(normalised)))
            tokens.extend(vocab[character] for character in normalised)
        return tokens, _fill_gaps(spans, len(tokens))

    # ---- acoustics ----------------------------------------------------------

    def _emissions(self, samples):
        """Log-probabilities per 20 ms frame for the whole clip."""
        import numpy as np

        active = _speech_mask(samples)
        width = len(self.vocab)
        emissions = np.full((len(active), width), np.float32(_SILENT_LOGPROB),
                            dtype=np.float32)
        emissions[:, BLANK] = 0.0

        for first, last in _runs(active):
            for start in range(first, last, _CHUNK_FRAMES):
                stop = min(start + _CHUNK_FRAMES, last)
                window = samples[start * _FRAME_SAMPLES:stop * _FRAME_SAMPLES]
                if len(window) < _FRAME_SAMPLES * 10:
                    continue
                normalised = (window - window.mean()) / (window.std() + 1e-7)
                logits = self.session.run(
                    None, {"input_values": normalised[None, :].astype(np.float32)}
                )[0][0]
                produced = min(len(logits), stop - start)
                emissions[start:start + produced] = _log_softmax(logits[:produced])
        return emissions

    @staticmethod
    def _viterbi(emissions, tokens):
        """The most likely monotonic path through ``tokens``, frame by frame.

        Classic CTC forced alignment over the blank-extended target: every token must
        be consumed in order, so unexplained audio is absorbed by the blanks between
        them rather than allowed to reorder anything. Returns the first frame of each
        token.
        """
        import numpy as np

        frames = len(emissions)
        extended = np.zeros(2 * len(tokens) + 1, dtype=np.int64)
        extended[1::2] = tokens
        width = len(extended)
        # A two-step skip is only legal into a real token differing from the one two
        # back; otherwise CTC would collapse a doubled letter into one.
        skippable = np.zeros(width, dtype=bool)
        skippable[2:] = (extended[2:] != BLANK) & (extended[2:] != extended[:-2])

        unreachable = np.float32(-1e30)
        best = np.full(width, unreachable, dtype=np.float32)
        best[0] = emissions[0, BLANK]
        if width > 1:
            best[1] = emissions[0, extended[1]]
        # One byte per (frame, position) for the backtrace; the largest speeches make
        # this a few hundred megabytes, which is the cost to watch if interventions
        # ever get much longer than a quarter of an hour.
        choices = np.zeros((frames, width), dtype=np.uint8)

        stay = np.empty(width, dtype=np.float32)
        advance = np.empty(width, dtype=np.float32)
        skip = np.empty(width, dtype=np.float32)
        for frame in range(1, frames):
            stay[:] = best
            advance[0] = unreachable
            advance[1:] = best[:-1]
            skip[:2] = unreachable
            skip[2:] = np.where(skippable[2:], best[:-2], unreachable)
            options = np.stack((stay, advance, skip))
            choice = options.argmax(axis=0).astype(np.uint8)
            best = options[choice, np.arange(width)] + emissions[frame, extended]
            choices[frame] = choice

        position = width - 1 if best[-1] >= best[-2] else width - 2
        starts = np.zeros(len(tokens), dtype=np.int64)
        ends = np.zeros(len(tokens), dtype=np.int64)
        seen = np.zeros(len(tokens), dtype=bool)
        # Walking backwards, the first frame we see a token on is its last one and
        # the value ``starts`` settles on is its first. Both are needed: a word timed
        # only from where its final token *began* loses that token's duration, which
        # for a one-token word is the whole of it.
        for frame in range(frames - 1, 0, -1):
            if extended[position] != BLANK:
                token = (position - 1) // 2
                if not seen[token]:
                    ends[token] = frame
                    seen[token] = True
                starts[token] = frame
            position -= int(choices[frame, position])
        if extended[position] != BLANK:
            token = (position - 1) // 2
            starts[token] = 0
            if not seen[token]:
                ends[token] = 0
        return starts, ends

    # ---- confidence ---------------------------------------------------------

    def _confidence(self, emissions, words, spans, starts, ends):
        """Decode the audio under a sample of the alignment and compare it to the
        words placed there. A drifted alignment reads as gibberish; a sound one
        reproduces the transcript almost character for character."""
        from thefuzz import fuzz

        vocab = self.vocab
        fold = "ñ" not in vocab
        inverse = {index: token for token, index in vocab.items()}
        placed = [index for index, span in enumerate(spans) if span]
        if not placed:
            return 0.0

        step = max(1, len(placed) // _CHECK_SAMPLES)
        ratios = []
        for offset in range(0, len(placed), step):
            group = placed[offset:offset + _CHECK_WORDS]
            if not group:
                continue
            first, last = spans[group[0]][0], spans[group[-1]][1]
            frames = slice(int(starts[first]), int(ends[last - 1]) + 1)
            if frames.stop <= frames.start:
                continue
            heard = _greedy(emissions[frames], inverse).replace("|", "")
            expected = "".join(
                _normalise(_verbalise(words[index]), vocab, fold) for index in group)
            if expected:
                ratios.append(fuzz.ratio(heard, expected))
        return round(sum(ratios) / len(ratios), 1) if ratios else 0.0


# ---- pure helpers -----------------------------------------------------------

_NUMERAL_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$|^\d+(?:[.,]\d+)?$")
_SYMBOLS = {"%": "por ciento", "€": "euros", "$": "dólares", "&": "y"}


def _verbalise(word):
    """Numerals as the words they are read out as.

    The vocabulary has no digit tokens at all, so ``artículo 49`` would offer the
    aligner nothing for five spoken syllables. Two independent acoustic models were
    measured disagreeing on cue boundaries almost exclusively where the transcript
    carries a figure, which is what makes this a requirement rather than a polish.
    Spanish thousands separators and decimal commas are read the Spanish way:
    ``118.885`` is a hundred and eighteen thousand, ``47,5`` is forty-seven point
    five.
    """
    stripped = word.strip("«»\"'()[[]].,;:!?¡¿…")
    if stripped in _SYMBOLS:
        return _SYMBOLS[stripped]
    if not stripped or not any(character.isdigit() for character in stripped):
        return word
    if not _NUMERAL_RE.match(stripped):
        return word
    plain = stripped.replace(".", "") if "." in stripped and "," in stripped \
        else stripped.replace(".", "")
    plain = plain.replace(",", ".")
    try:
        number = float(plain) if "." in plain else int(plain)
        spoken = _num2words()(number, lang="es")
    except (ValueError, OverflowError):
        # Not a figure this can read out — leave it to be dropped by the vocabulary.
        return word
    # num2words reads a decimal point as "punto"; in Spain it is said "coma", which
    # is what the audio will contain.
    return spoken.replace(" punto ", " coma ")


_NUM2WORDS = None


def _num2words():
    """The verbalizer, imported on first use.

    A missing num2words must NOT degrade quietly to leaving figures unspoken: that
    silently reintroduces the drift this whole step exists to remove, and the
    resulting alignment looks healthy. So an incomplete install fails here instead.
    """
    global _NUM2WORDS
    if _NUM2WORDS is None:
        try:
            from num2words import num2words
        except ImportError as exc:
            raise RuntimeError(
                "num2words is required to align a transcript containing figures: "
                "the acoustic model has no digit tokens, so a numeral must be "
                "spelled out or its audio goes unaccounted for. Install the "
                "'align' extra."
            ) from exc
        _NUM2WORDS = num2words
    return _NUM2WORDS


def _normalise(word, vocab, fold_accents):
    """A word reduced to the characters this vocabulary actually has.

    MMS is trained on romanised targets, so accents fold away; a Spanish-specific
    vocabulary instead *contains* the accented vowels and ñ, and folding there would
    discard information the model uses. Which one applies is read off the vocabulary
    rather than configured, so the two cannot disagree.
    """
    lowered = word.lower()
    if fold_accents:
        decomposed = unicodedata.normalize("NFD", lowered)
        lowered = "".join(c for c in decomposed if not unicodedata.combining(c))
        lowered = lowered.replace("ñ", "n").replace("ç", "c").replace("ü", "u")
    return "".join(character for character in lowered if character in vocab)


def _fill_gaps(spans, total):
    """Give every word a token slice, including the ones that normalised to nothing.

    A word with no tokens still needs a timing, because the caller indexes timings by
    word. It borrows the following word's opening (or the previous word's close at
    the end), which places it at the moment it was passed over."""
    filled = list(spans)
    for index, span in enumerate(filled):
        if span:
            continue
        following = next((s for s in filled[index + 1:] if s), None)
        if following:
            filled[index] = (following[0], following[0] + 1)
            continue
        preceding = next((s for s in reversed(filled[:index]) if s), None)
        filled[index] = preceding if preceding else (0, min(1, total))
    return filled


def _speech_mask(samples, window=_VAD_WINDOW, relative=_VAD_RELATIVE_FLOOR,
                 dilate=_VAD_DILATE_FRAMES):
    """Which 20 ms frames carry speech, by energy relative to the clip's own loud
    passages. Dilated, so a quiet onset or coda stays attached to its word."""
    import numpy as np

    usable = len(samples) // window * window
    if usable == 0:
        return np.zeros(0, dtype=bool)
    frames = samples[:usable].reshape(-1, window)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    active = energy > relative * np.percentile(energy, 95)
    if dilate:
        active = np.convolve(active, np.ones(2 * dilate + 1, dtype=bool),
                             mode="same") > 0
    return active


def _runs(active):
    """Contiguous ``[start, stop)`` frame ranges where ``active`` is true."""
    import numpy as np

    edges = np.diff(np.concatenate(([0], active.view(np.int8), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def _log_softmax(logits):
    import numpy as np

    peak = logits.max(axis=1, keepdims=True)
    shifted = logits - peak
    return (shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            ).astype(np.float32)


def _greedy(emissions, inverse):
    """The CTC greedy decode of ``emissions`` — collapse repeats, drop blanks."""
    best = emissions.argmax(axis=1)
    out, previous = [], -1
    for index in best:
        if index != previous and index != BLANK:
            out.append(inverse.get(int(index), ""))
        previous = index
    return "".join(out)


@_register("mms_onnx")
def create(settings: Settings) -> MmsOnnxAligner:
    return MmsOnnxAligner(
        model_path=settings.aligner_model_path,
        repo=settings.aligner_model_repo,
        revision=settings.aligner_model_revision,
        model_file=settings.aligner_model_file,
        threads=settings.aligner_threads,
    )
