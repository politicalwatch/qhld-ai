"""Offline tests for audio decoding — no network, no Congress video.

Test media is synthesised with PyAV itself into a temp file, so these run anywhere
and still exercise the real decode path rather than a mock of it. The predecessor
that shelled out to the ffmpeg binary had no tests at all, because asserting
anything about it meant asserting about a subprocess.
"""

import math

import numpy as np
import pytest

from qhld_ai.infrastructure.audio.pyav import (
    SAMPLE_RATE,
    AudioDecodeError,
    decode_pcm,
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
