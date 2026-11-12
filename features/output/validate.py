"""Decode a raw MP4 without changing its bytes, frame rate, or timestamps."""

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from features.media.ingest import MAX_INPUT_PIXELS, MediaError, _run


MAX_RAW_BYTES = 256 * 1024 * 1024
MAX_RAW_FRAMES = 2400
MP4_BRANDS = {"isom", "iso2", "iso3", "iso4", "iso5", "iso6", "iso7", "iso8", "iso9",
              "mp41", "mp42", "avc1", "dash", "cmfc", "cmfs", "M4V "}


@dataclass(frozen=True)
class RawVideo:
    video_bytes: bytes
    sha256: str
    codec: str
    width: int
    height: int
    duration_seconds: float
    frame_rate: str
    frame_seconds: tuple[float, ...]
    audio_present: bool
    timeline_origin_seconds: float = 0.0


def _check_origin(value: float) -> float:
    """Check a declared container origin without substituting its first frame."""
    try:
        if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
            raise ValueError("Container timeline origin must be finite and nonnegative.")
    except OverflowError:
        raise ValueError("Container timeline origin must be finite and nonnegative.") from None
    return value


def _probe_raw(path: Path, *, frames: bool = False) -> dict:
    entries = ("stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,channels,sample_rate:"
               "format=duration,start_time:format_tags=major_brand")
    arguments = ["-protocol_whitelist", "file,pipe", "-f", "mov"]
    if frames:
        entries = "frame=best_effort_timestamp_time,width,height"
        arguments += ["-select_streams", "v:0", "-show_frames"]
    return json.loads(_run("ffprobe", [*arguments, "-show_entries", entries, "-of", "json", str(path)]))


def validate_raw_video(contents: bytes) -> RawVideo:
    """Validate content, not provider provenance or visual quality.

    Accept one video stream and at most one audio stream. Limits are 256 MiB,
    20 seconds, 4096 pixels per axis, 8,847,360 pixels per frame, 120 FPS, and
    2400 decoded frames. Audio is limited to eight channels at 192 kHz.
    Every decoded frame must keep the declared dimensions.
    Preserve valid timing differences from the normalized 120-frame input.
    Bounds use elapsed time from the declared container origin; returned PTS
    remain native. The first video frame does not replace the container origin.
    """
    if type(contents) is not bytes or not 0 < len(contents) <= MAX_RAW_BYTES:
        raise MediaError("Raw video must contain 1 byte to 256 MiB of MP4 data.")
    try:
        with TemporaryDirectory(prefix="gen-reality-raw-") as folder:
            path = Path(folder) / "raw.mp4"
            path.write_bytes(contents)
            metadata = _probe_raw(path)
            if metadata["format"].get("tags", {}).get("major_brand") not in MP4_BRANDS:
                raise MediaError("Raw video must use a supported MP4 container.")
            streams = metadata["streams"]
            videos = [stream for stream in streams if stream["codec_type"] == "video"]
            audio = [stream for stream in streams if stream["codec_type"] == "audio"]
            if len(videos) != 1 or len(audio) > 1 or len(streams) != len(videos) + len(audio):
                raise MediaError("Raw MP4 must contain one video and at most one audio stream.")
            video = videos[0]
            width, height = video["width"], video["height"]
            if (type(width) is not int or type(height) is not int
                    or not 1 <= width <= 4096 or not 1 <= height <= 4096
                    or width * height > MAX_INPUT_PIXELS):
                raise MediaError("Raw video dimensions exceed the allowed size.")
            if video["codec_name"] not in ("h264", "hevc", "av1", "mpeg4"):
                raise MediaError("Raw MP4 uses an unsupported video codec.")
            duration = float(metadata["format"]["duration"])
            declared_origin = metadata["format"]["start_time"]
            if type(declared_origin) is not str:
                raise MediaError("Raw video must declare its container timeline origin.")
            origin = _check_origin(float(declared_origin))
            rate = video["avg_frame_rate"]
            if not math.isfinite(duration) or not 0 < duration <= 20:
                raise MediaError("Raw video duration must be positive and at most twenty seconds.")
            if type(rate) is not str or not 0 < Fraction(rate) <= 120:
                raise MediaError("Raw video frame rate must be positive and at most 120 FPS.")
            declared_frames = video.get("nb_frames")
            if declared_frames is not None and not 1 <= int(declared_frames) <= MAX_RAW_FRAMES:
                raise MediaError("Raw video declares too many frames.")
            if audio and (type(audio[0]["channels"]) is not int or not 1 <= audio[0]["channels"] <= 8
                          or not 1 <= int(audio[0]["sample_rate"]) <= 192_000):
                raise MediaError("Raw audio exceeds the allowed channel count or sample rate.")

            # Decode every video and audio packet; do not trim, filter, or encode.
            _run("ffmpeg", ["-xerror", "-nostdin", "-threads", "2", "-err_detect", "explode",
                            "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(path),
                            "-map", "0:v:0", "-map", "0:a?", "-sn", "-dn", "-f", "null", "-"])
            frames = _probe_raw(path, frames=True)["frames"]
            if any(frame.get("width") != width or frame.get("height") != height for frame in frames):
                raise MediaError("Raw video frame dimensions changed during decoding.")
            times = tuple(float(frame["best_effort_timestamp_time"]) for frame in frames)
            if (not 1 <= len(times) <= MAX_RAW_FRAMES
                    or any(not math.isfinite(value) or not 0 <= value - origin <= duration for value in times)
                    or any(later <= earlier for earlier, later in zip(times, times[1:]))):
                raise MediaError("Raw video has invalid frame timestamps or frame count.")
            if declared_frames is not None and len(times) != int(declared_frames):
                raise MediaError("Raw video did not decode all declared frames.")
    except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError) as error:
        if isinstance(error, MediaError):
            raise
        raise MediaError("Raw video metadata is incomplete or unreadable.") from None
    return RawVideo(contents, sha256(contents).hexdigest(), video["codec_name"], width, height,
                    duration, rate, times, bool(audio), origin)
