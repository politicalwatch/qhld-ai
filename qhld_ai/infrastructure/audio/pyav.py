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
# How much of a clip may go undecoded and still be worth aligning against. Two budgets,
# because where the loss falls decides what it costs: audio missing from the very end
# shifts no cue at all, it only compresses the last words into what remains, while a gap
# in the middle pulls everything after it earlier by the same amount. So the tail is
# forgiven a second and the middle a quarter of one — the latter is about a word at
# parliamentary pace, and still an order of magnitude above the aligner's own 20 ms frame,
# which keeps tolerated corruption from becoming the largest error in the timing.
# Absolute seconds rather than a percentage: a second of unaccounted speech is as wrong in
# a half-minute question as in a two-hour debate.
_MAX_TAIL_LOSS_SECONDS = 1.0
_MAX_DRIFT_SECONDS = 0.25
# Each skipped packet is a splice, and a splice is a broadband transient the acoustic
# model reads as speech. A handful is survivable; a shredded stream is not, however little
# total time it adds up to. The clips this exists for fail exactly one packet.
_MAX_SKIPPED_PACKETS = 4


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

    **A few undecodable packets do not cost the whole speech.** The Congress publishes
    clips whose final packet is corrupt: measured over two of them, one packet in 13,525
    fails and it is the last one, leaving 99.8% and 100.0% of the audio perfectly
    readable. Decoding the stream as a single unit threw all of that away, so 111 of
    3,981 interventions had no subtitles because half a second at the very end was bad.
    Packets are therefore decoded one at a time and a failure skips only that packet.

    What makes skipping safe is the check afterwards, not the skipping: the audio kept
    must still account for the length the audio track declares. A transcript covers the
    whole speech, and a forced aligner asked to fit it into truncated audio does not
    fail — it finds a monotone path anyway and returns confident, wrong timings. So the
    result is refused unless the loss is small and, if it is not confined to the tail,
    smaller still; and it is refused outright when the track declares no length, because
    there is then nothing to verify the remainder against.

    **Only invalid data is tolerated, and only per packet.** A stalled connection, a
    reset, an HTTP error and a short read are all ``FFmpegError`` subclasses too, so
    catching that base class here would quietly turn a download dying at 80% into
    truncated audio; those keep failing loudly, as the promise above says they do. Nor is
    anything tolerated at the demuxing level: an exception escaping that generator closes
    it, so no packet after the fault is reachable anyway, and swallowing it would
    manufacture exactly the truncation this guards against.
    """
    import av
    import numpy as np

    _logger.debug("decoding audio from %s", source)
    chunks = []
    skipped = 0
    first_fault = None
    fault = None
    declared = 0.0
    try:
        with av.open(str(source),
                     timeout=(_OPEN_TIMEOUT, _READ_TIMEOUT)) as container:
            if not container.streams.audio:
                raise AudioDecodeError(f"no audio stream in {source}")
            stream = container.streams.audio[0]
            declared = _declared_seconds(stream, container)
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=sample_rate)
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError as exc:
                    skipped += 1
                    fault = fault or exc
                    if first_fault is None and packet.pts is not None:
                        first_fault = float(packet.pts * packet.time_base)
                    continue
                # Resampling sits outside the tolerant block on purpose: a filter-graph
                # failure is not corrupt input, and counting it as a skipped packet would
                # discard audio that decoded perfectly well.
                for frame in frames:
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
    samples = np.concatenate(chunks)
    if skipped:
        _check_coverage(source, len(samples) / sample_rate, declared, skipped,
                        first_fault, fault)
    return samples.astype(np.float32) / 32768.0


def _declared_seconds(stream, container):
    """How long the audio is supposed to run, ``0.0`` if nothing says.

    The audio track's own length in preference to the container's, which is the longest
    of all its streams: trailing video frames would otherwise read as missing audio and
    refuse a perfectly good clip.
    """
    import av

    if stream.duration is not None:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return container.duration / av.time_base
    return 0.0


def _check_coverage(source, decoded, declared, skipped, first_fault, fault):
    """Refuse audio that no longer accounts for the clip it came from.

    Separate from the decoding because it is the part that has to be right: above decides
    what to step over, this decides whether the result may be used at all.
    """
    def refuse(why):
        return AudioDecodeError(f"could not decode {source}: {why}")

    if skipped > _MAX_SKIPPED_PACKETS:
        raise refuse(f"{skipped} unreadable packets, too damaged to align against") \
            from fault
    if not declared:
        # Nothing to measure the remainder against, so the skip cannot be shown to be
        # harmless. Refusing beats aligning a whole transcript against an unknown
        # fraction of its audio.
        raise refuse(f"{skipped} unreadable packet(s) and no declared duration, so what "
                     "was decoded cannot be checked") from fault
    if decoded - declared > _MAX_TAIL_LOSS_SECONDS:
        # More audio than the track claims to hold, so the claim is not a usable
        # reference and the skip cannot be verified against it either.
        raise refuse(f"decoded {decoded:.1f}s against a declared {declared:.1f}s, so the "
                     "declared length cannot verify the packets skipped") from fault
    lost = declared - decoded
    # Loss confined to the end shifts no cue; anywhere else it drags everything after it.
    tail_only = first_fault is not None and declared - first_fault <= _MAX_TAIL_LOSS_SECONDS
    budget = _MAX_TAIL_LOSS_SECONDS if tail_only else _MAX_DRIFT_SECONDS
    if lost > budget:
        where = "at the tail" if tail_only else "mid-clip"
        raise refuse(f"{lost:.2f}s of {declared:.1f}s missing {where} after skipping "
                     f"{skipped} unreadable packet(s), too much to align against") \
            from fault
    _logger.warning(
        "%s: skipped %d unreadable packet(s) %s, %.2fs of %.1fs missing; aligning "
        "against the rest", source, skipped, "at the tail" if tail_only else "mid-clip",
        max(lost, 0.0), declared)
