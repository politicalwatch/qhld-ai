"""Decode the audio of a video to mono PCM, without it ever touching disk.

The Congress publishes one mp4 per intervention and no audio-only rendition, so the
whole video has to be read to get at its audio track — ~10 MB per minute of speech.
Multiplied by a corpus that is meant to reach tens of thousands of interventions
that is terabytes, and none of it is worth keeping: the video is served to users
straight from the Congress, and we only need the samples long enough to align them.

So ffmpeg reads the URL and writes raw samples to a pipe, which are read into an
array and dropped when the caller is done. Nothing is written anywhere. A whole
twelve-minute intervention is about 24 MB in memory.

Raw ``s16le`` rather than a wav container: there is no header to parse or to lie
about a length that a pipe cannot know in advance.
"""

import subprocess

from qhld_ai.logger import get_logger

_logger = get_logger(__name__)

SAMPLE_RATE = 16000


class AudioDecodeError(Exception):
    """The audio of a video could not be decoded.

    Raised instead of returning empty samples: a caller that treats "no audio" as
    "nothing to align" would store an alignment of a speech it never heard.
    """


def decode_pcm(source, sample_rate=SAMPLE_RATE):
    """Mono float32 samples of ``source`` (a URL or a path), resampled.

    ``ffmpeg`` must be on PATH. Anything it cannot open — a video the Congress has
    not published yet, a link that 404s, a truncated file — raises
    ``AudioDecodeError`` with ffmpeg's own diagnostics attached, since they are far
    more specific than anything this layer could infer.
    """
    import numpy as np

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-vn",                        # the video stream is dead weight here
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le", "-",
    ]
    _logger.debug("decoding audio from %s", source)
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, check=False)
    except FileNotFoundError as exc:
        raise AudioDecodeError(
            "ffmpeg is not installed or not on PATH; it is required to read the "
            "audio of an intervention's video") from exc

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise AudioDecodeError(f"ffmpeg failed for {source}: {detail}")
    if not completed.stdout:
        raise AudioDecodeError(f"no audio stream decoded from {source}")

    samples = np.frombuffer(completed.stdout, dtype=np.int16)
    # A copy, not a view on the bytes object: the caller normalises in place and a
    # buffer-backed array is read-only.
    return (samples.astype(np.float32) / 32768.0)
