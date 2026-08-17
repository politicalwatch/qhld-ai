"""Offline tests for audio decoding — no network, no Congress video.

Test media is synthesised with PyAV itself into a temp file, so these run anywhere
and still exercise the real decode path rather than a mock of it. The predecessor
that shelled out to the ffmpeg binary had no tests at all, because asserting
anything about it meant asserting about a subprocess.
"""

import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from qhld_ai.infrastructure.audio.pyav import (
    SAMPLE_RATE,
    AudioDecodeError,
    DurationUnavailable,
    decode_pcm,
    probe_duration,
)

pytestmark = pytest.mark.unit


def _write_tone(path, seconds=1.0, rate=SAMPLE_RATE, channels=1, hz=440.0):
    """A wav of a sine tone, written with PyAV so the fixture needs no assets."""
    import av

    layout = "mono" if channels == 1 else "stereo"
    total = int(rate * seconds)
    time = np.arange(total, dtype=np.float32) / rate
    wave = (np.sin(2 * math.pi * hz * time) * 0.5 * 32767).astype(np.int16)
    # "s16" is packed, so every channel lives interleaved in a single plane —
    # shape (1, frames * channels), not one row per channel.
    samples = np.repeat(wave, channels)[None, :]

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(samples), format="s16", layout=layout)
        frame.rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def test_decodes_to_mono_float32_at_the_target_rate(tmp_path):
    source = _write_tone(tmp_path / "tone.wav", seconds=1.0)

    samples = decode_pcm(str(source))

    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert len(samples) == pytest.approx(SAMPLE_RATE, rel=0.02)
    assert np.abs(samples).max() <= 1.0


def test_resamples_a_different_input_rate(tmp_path):
    # Congress video carries 48 kHz AAC; the aligner only accepts 16 kHz.
    source = _write_tone(tmp_path / "48k.wav", seconds=1.0, rate=48000)

    samples = decode_pcm(str(source))

    assert len(samples) == pytest.approx(SAMPLE_RATE, rel=0.02)


def test_downmixes_stereo_to_mono(tmp_path):
    source = _write_tone(tmp_path / "stereo.wav", seconds=0.5, channels=2)

    samples = decode_pcm(str(source))

    assert samples.ndim == 1
    assert len(samples) == pytest.approx(SAMPLE_RATE * 0.5, rel=0.02)


def test_the_whole_signal_survives_including_the_tail(tmp_path):
    """The resampler buffers, so a missing flush silently truncates the end of a
    speech — which would misplace its last cues rather than fail."""
    seconds = 2.0
    source = _write_tone(tmp_path / "long.wav", seconds=seconds)

    samples = decode_pcm(str(source))

    assert len(samples) >= SAMPLE_RATE * seconds * 0.98
    # A tone runs to the end, so real signal must be present in the final 50 ms.
    assert np.abs(samples[-800:]).max() > 0.1


def test_an_explicit_sample_rate_is_honoured(tmp_path):
    source = _write_tone(tmp_path / "tone.wav", seconds=1.0)

    samples = decode_pcm(str(source), sample_rate=8000)

    assert len(samples) == pytest.approx(8000, rel=0.02)


def test_a_missing_file_raises_audio_decode_error(tmp_path):
    with pytest.raises(AudioDecodeError, match="could not decode"):
        decode_pcm(str(tmp_path / "does-not-exist.mp4"))


def test_a_file_that_is_not_media_raises_audio_decode_error(tmp_path):
    junk = tmp_path / "not-a-video.mp4"
    junk.write_bytes(b"this is not a container")

    with pytest.raises(AudioDecodeError):
        decode_pcm(str(junk))


# ---- undecodable packets -----------------------------------------------------
#
# The Congress publishes clips whose last packet is corrupt, and decoding the stream as
# one unit lost the whole speech over it. libav's real behaviour on those files is
# already measured (one bad packet in 13,525, at the very end, 99.8% and 100.0% of the
# audio readable); what these tests pin is our side of it — what we keep, and when we
# refuse to keep anything. A synthesised container cannot produce a mid-decode failure
# (a short wav simply decodes less, having no per-packet integrity), so the failure is
# injected instead.

class _FakePacket:
    def __init__(self, frames=(), error=None, pts=None, time_base=Fraction(1, 1000)):
        self._frames = frames
        self._error = error
        self.pts = pts
        self.time_base = time_base

    def decode(self):
        if self._error is not None:
            raise self._error
        return self._frames


def _invalid_data():
    import av

    return av.error.InvalidDataError(
        1094995529, "Invalid data found when processing input")


class _FakeContainer:
    """Just enough of a container for ``decode_pcm``, yielding real audio frames so the
    genuine resampler still runs — only the failure is a fake."""

    def __init__(self, packets, duration):
        self._packets = packets
        self.duration = None
        stream = SimpleNamespace(
            rate=SAMPLE_RATE, time_base=Fraction(1, 1000),
            duration=None if duration is None else int(duration * 1000))
        self.streams = SimpleNamespace(audio=[stream])

    def demux(self, stream):
        return iter(self._packets)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _frame(seconds, rate=SAMPLE_RATE):
    import av

    total = int(rate * seconds)
    samples = np.zeros((1, total), dtype=np.int16)
    samples[0, ::100] = 8000  # some signal, so silence-trimming logic can't hide a bug
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    frame.rate = rate
    return frame


def _patch_container(monkeypatch, packets, duration):
    import av

    monkeypatch.setattr(
        av, "open", lambda *a, **kw: _FakeContainer(packets, duration))


def _good(seconds, at_ms):
    return _FakePacket(frames=[_frame(seconds)], pts=at_ms)


def test_a_corrupt_final_packet_still_yields_the_speech(monkeypatch):
    # The 111 interventions this exists for: everything decodes but the last packet.
    packets = [_good(1.0, 0), _good(1.0, 1000), _good(1.0, 2000),
               _FakePacket(error=_invalid_data(), pts=3000)]
    _patch_container(monkeypatch, packets, duration=3.0)

    samples = decode_pcm("corrupt-tail.mp4")

    assert len(samples) == pytest.approx(SAMPLE_RATE * 3.0, rel=0.02)


def test_a_gap_in_the_middle_is_refused_where_the_same_loss_at_the_tail_is_not(
        monkeypatch):
    # Identical amount of audio lost; only the position differs. At the end it shifts no
    # cue, in the middle it drags every later cue earlier by the same amount.
    packets = [_good(1.0, 0), _FakePacket(error=_invalid_data(), pts=1000),
               _good(1.0, 1500)]
    _patch_container(monkeypatch, packets, duration=2.5)

    with pytest.raises(AudioDecodeError, match="mid-clip"):
        decode_pcm("gap-in-the-middle.mp4")


def test_audio_lost_beyond_the_tolerance_is_refused(monkeypatch):
    # Most of the clip is gone. Aligning the whole transcript against it would not fail,
    # it would return confident, wrong timings — so refusing is the only safe answer.
    packets = [_good(1.0, 0), _FakePacket(error=_invalid_data(), pts=9500)]
    _patch_container(monkeypatch, packets, duration=10.0)

    with pytest.raises(AudioDecodeError, match="too much to align against"):
        decode_pcm("mostly-broken.mp4")


def test_a_skipped_packet_without_a_declared_duration_is_refused(monkeypatch):
    # Nothing to check the remainder against, so the skip cannot be shown harmless.
    packets = [_good(1.0, 0), _FakePacket(error=_invalid_data(), pts=1000)]
    _patch_container(monkeypatch, packets, duration=None)

    with pytest.raises(AudioDecodeError, match="no declared duration"):
        decode_pcm("no-duration.mp4")


def test_a_shredded_stream_is_refused_however_little_time_it_loses(monkeypatch):
    # Each skip is a splice, and a splice is a transient the acoustic model reads as
    # speech, so the count matters independently of the seconds.
    packets = [_good(1.0, 0)]
    packets += [_FakePacket(error=_invalid_data(), pts=1000 + i) for i in range(6)]
    _patch_container(monkeypatch, packets, duration=1.0)

    with pytest.raises(AudioDecodeError, match="too damaged"):
        decode_pcm("shredded.mp4")


def test_a_dying_download_is_not_mistaken_for_a_corrupt_packet(monkeypatch):
    # The regression this guard exists to avoid: a stalled or reset connection is an
    # FFmpegError too, so tolerating that base class would turn a download dying at 80%
    # into silently truncated audio that a generous header would wave through.
    import av

    for error in (av.error.TimeoutError(110, "timed out"),
                  av.error.ConnectionResetError(104, "reset by peer"),
                  av.error.HTTPError(1, "bad gateway")):
        packets = [_good(1.0, 0), _FakePacket(error=error, pts=1000)]
        _patch_container(monkeypatch, packets, duration=1.5)

        with pytest.raises(AudioDecodeError, match="could not decode"):
            decode_pcm("dying-download.mp4")


def test_a_clean_decode_is_unaffected_by_the_tolerant_path(monkeypatch):
    # No packet fails, so no coverage check runs and the result is the plain decode.
    packets = [_good(0.5, i * 500) for i in range(4)]
    _patch_container(monkeypatch, packets, duration=2.0)

    samples = decode_pcm("clean.mp4")

    assert len(samples) == pytest.approx(SAMPLE_RATE * 2.0, rel=0.02)
    assert samples.dtype == np.float32


def test_probe_duration_reads_the_length_without_decoding(tmp_path):
    source = _write_tone(tmp_path / "tone.wav", seconds=2.0)

    assert probe_duration(str(source)) == pytest.approx(2.0, abs=0.05)


def test_probe_duration_is_independent_of_the_sample_rate(tmp_path):
    source = _write_tone(tmp_path / "48k.wav", seconds=1.5, rate=48000)

    assert probe_duration(str(source)) == pytest.approx(1.5, abs=0.05)


def test_probe_duration_raises_rather_than_reporting_zero(tmp_path):
    # A caller dividing a transcript's length by this must never be handed a 0 or a
    # None that reads as "the clip is empty".
    with pytest.raises(DurationUnavailable, match="could not open"):
        probe_duration(str(tmp_path / "does-not-exist.mp4"))


def test_probe_duration_on_something_that_is_not_media(tmp_path):
    junk = tmp_path / "not-a-video.mp4"
    junk.write_bytes(b"this is not a container")

    with pytest.raises(DurationUnavailable):
        probe_duration(str(junk))
