"""Decode the audio of a video to mono PCM, without it ever touching disk.

The Congress publishes one mp4 per intervention and no audio-only rendition, so the
whole video has to be read to get at its audio track — ~10 MB per minute of speech.
Multiplied by a corpus meant to reach tens of thousands of interventions that is
terabytes, and none of it is worth keeping: the video is served to users straight
from the Congress, and we only need the samples long enough to align them. So the
container is streamed and decoded frame by frame, and nothing is written anywhere.
A whole twelve-minute intervention is about 24 MB in memory.

PyAV binds the same libav* libraries the ffmpeg command line drives, and was chosen
over shelling out to that command for size rather than speed: the wheel is ~44 MB
against ~419 MB for Debian's ffmpeg package, and it removes a system dependency, so
a container and a laptop decode with the same code instead of merely similar
versions. It costs about 0.17 s per intervention (measured: 0.48 s against 0.31 s on
a nine-minute clip), which is under half a percent of what downloading and aligning
that intervention costs, and the samples are bit-identical either way — so the
choice moves no alignment.
"""

from qhld_ai.logger import get_logger

_logger = get_logger(__name__)

SAMPLE_RATE = 16000
# Generous, but bounded: a stalled connection to the Congress should fail a single
# intervention rather than hang a whole run.
_OPEN_TIMEOUT = 30.0
_READ_TIMEOUT = 60.0


class AudioDecodeError(Exception):
    """The audio of a video could not be decoded.

    Raised instead of returning empty samples: a caller that treats "no audio" as
    "nothing to align" would store an alignment of a speech it never heard.
    """


class DurationUnavailable(Exception):
    """How long a video runs could not be established.

    Separate from ``AudioDecodeError`` because the two have opposite consequences:
    failing to decode means an alignment cannot be made, while failing to read a
    duration means one number is missing. Raised rather than returning ``0.0`` or
    ``None`` so that "the clip is unknown" can never be mistaken for "the clip is
    empty" by a caller doing arithmetic on it.
    """


def probe_duration(source):
    """Seconds of ``source`` (a URL or a path), read from the container header.

    Only the header is fetched, so this costs a range request rather than a
    download — about 0.8 s against the Congress CDN, versus tens of seconds and
    ~10 MB per minute of speech to pull the whole file.

    Worth having stored: the Diario's text for an intervention should take roughly
    as long to say as its clip runs, so this is what lets extraction check its own
    output. A transcript that could not physically be spoken in the time available
    has been truncated, over-captured, or mis-split across languages, and none of
    those are visible from the text alone.
    """
    import av

    try:
        with av.open(str(source), timeout=(_OPEN_TIMEOUT, _READ_TIMEOUT)) as container:
            duration = container.duration
    except av.FFmpegError as exc:
        raise DurationUnavailable(f"could not open {source}: {exc}") from exc

    if not duration:
        # Streams without a header duration exist (live containers, some fragmented
        # mp4s). Saying so beats reporting the zero libav uses for "don't know".
        raise DurationUnavailable(f"no duration declared by {source}")
    return duration / av.time_base


def decode_pcm(source, sample_rate=SAMPLE_RATE):
    """Mono float32 samples of ``source`` (a URL or a path), resampled.

    Anything libav cannot open or read — a video the Congress has not published yet,
    a link that 404s, a truncated file, a stalled connection — surfaces as
    ``AudioDecodeError`` carrying libav's own diagnostics, which are far more
    specific than anything this layer could infer.
    """
    import av
    import numpy as np

    _logger.debug("decoding audio from %s", source)
    chunks = []
    try:
        with av.open(str(source),
                     timeout=(_OPEN_TIMEOUT, _READ_TIMEOUT)) as container:
            if not container.streams.audio:
                raise AudioDecodeError(f"no audio stream in {source}")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=sample_rate)
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            # The resampler buffers; flushing it is what makes the tail of the
            # speech arrive instead of being silently dropped.
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    except av.FFmpegError as exc:
        raise AudioDecodeError(f"could not decode {source}: {exc}") from exc

    if not chunks:
        raise AudioDecodeError(f"no audio decoded from {source}")
    return np.concatenate(chunks).astype(np.float32) / 32768.0
