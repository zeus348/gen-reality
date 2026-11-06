"""Normalize a local MP4 or MOV with bounded FFmpeg commands."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_INPUT_PIXELS = 8_847_360
MAX_METADATA_BYTES = 1024 * 1024
MAX_ERROR_BYTES = 65_536


class MediaError(ValueError):
    """Media cannot be read, normalized, or verified."""


@dataclass(frozen=True)
class NormalizedVideo:
    original_bytes: bytes
    original_sha256: str
    video_bytes: bytes
    sha256: str
    frame_seconds: tuple[float, ...]


def _run(tool: str, arguments: list[str]) -> bytes:
    """Do not expose subprocess diagnostics, local paths, or input metadata."""
    def read_limited(pipe, limit):
        chunks = []
        size = 0
        while chunk := pipe.read(65_536):
            size += len(chunk)
            if size > limit:
                process.kill()
                return None
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        with subprocess.Popen(
            [tool, "-v", "error", "-max_alloc", "67108864", *arguments],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ) as process:
            with ThreadPoolExecutor(max_workers=2) as readers:
                output = readers.submit(read_limited, process.stdout, MAX_METADATA_BYTES)
                errors = readers.submit(read_limited, process.stderr, MAX_ERROR_BYTES)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise MediaError("Media tool exceeded the time limit.") from None
                stdout, stderr = output.result(), errors.result()
    except (OSError, subprocess.TimeoutExpired):
        raise MediaError("Media tools are unavailable or exceeded the time limit.") from None
    if stdout is None or stderr is None:
        raise MediaError("Media tool output exceeds the allowed size.")
    if process.returncode != 0 or stderr:
        raise MediaError("Media could not be decoded or processed safely.")
    return stdout


def _probe(path: Path, *, frames: bool = False) -> dict:
    entries = "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,duration,sample_aspect_ratio:format=duration"
    arguments = ["-protocol_whitelist", "file,pipe", "-f", "mov",
                 "-show_entries", entries, "-of", "json"]
    if frames:
        arguments += ["-select_streams", "v:0", "-read_intervals", "%+#121",
                      "-show_frames", "-show_entries", entries + ":frame=best_effort_timestamp_time"]
    arguments.append(str(path))
    try:
        return json.loads(_run("ffprobe", arguments))
    except (ValueError, RecursionError):
        raise MediaError("Media metadata is invalid.") from None


def verify_normalized(path: Path) -> tuple[float, ...]:
    """Fully decode and verify format, silence, duration, and all frame times."""
    metadata = _probe(path)
    try:
        streams = metadata["streams"]
        if len(streams) != 1:
            raise MediaError("Normalized video must contain one silent video stream.")
        stream = streams[0]
        required = {"codec_type": "video", "codec_name": "h264", "width": 1280,
                    "height": 720, "pix_fmt": "yuv420p", "avg_frame_rate": "24/1",
                    "sample_aspect_ratio": "1:1"}
        if any(stream.get(key) != value for key, value in required.items()):
            raise MediaError("Normalized video format does not match the fixed fixture.")
        duration = float(stream["duration"])
        metadata = _probe(path, frames=True)
        times = tuple(float(frame["best_effort_timestamp_time"])
                      for frame in metadata["frames"])
        if not math.isfinite(duration) or abs(duration - 5) > 0.0001:
            raise MediaError("Normalized video must be exactly five seconds.")
        if len(times) != 120 or any(not math.isfinite(time)
                                  or abs(time - index / 24) > 0.00001
                                  for index, time in enumerate(times)):
            raise MediaError("Normalized video must have 120 frames at 24 FPS from time zero.")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, MediaError):
            raise
        raise MediaError("Normalized video metadata is incomplete.") from None
    return times


def normalize_video(path: Path) -> NormalizedVideo:
    """Keep the original and normalize the first five seconds without cropping.

    MP4/MOV input must be 5–20 seconds, at most 256 MiB, at most 4096 pixels on
    either axis, and at most 8,847,360 pixels per frame. FFmpeg applies rotation
    metadata before scaling. Display aspect ratio includes non-square pixels.
    """
    path = Path(path)
    if path.suffix.lower() not in (".mp4", ".mov"):
        raise MediaError("Source must be an MP4 or MOV file.")
    try:
        with path.open("rb") as file:
            original = file.read(MAX_INPUT_BYTES + 1)
    except OSError:
        raise MediaError("Source file cannot be read.") from None
    if len(original) > MAX_INPUT_BYTES:
        raise MediaError("Source file exceeds 256 MiB.")
    with TemporaryDirectory(prefix="gen-reality-media-") as folder:
        source = Path(folder) / "source.mp4"
        output = Path(folder) / "normalized.mp4"
        source.write_bytes(original)
        metadata = _probe(source)
        try:
            stream = next(stream for stream in metadata["streams"] if stream["codec_type"] == "video")
            duration = float(stream.get("duration", metadata["format"]["duration"]))
            width, height = stream["width"], stream["height"]
            if not math.isfinite(duration) or not 5 <= duration <= 20:
                raise MediaError("Source duration must be from five to twenty seconds.")
            if not (1 <= width <= 4096 and 1 <= height <= 4096
                    and width * height <= MAX_INPUT_PIXELS):
                raise MediaError("Source dimensions exceed the allowed size.")
        except (KeyError, TypeError, ValueError, StopIteration) as error:
            if isinstance(error, MediaError):
                raise
            raise MediaError("Source video metadata is incomplete.") from None

        input_args = ["-nostdin", "-threads", "2", "-protocol_whitelist", "file,pipe",
                      "-f", "mov", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn"]
        # Decode the whole original, including the portion beyond the trim.
        _run("ffmpeg", ["-xerror", *input_args, "-f", "null", "-"])
        filters = ("fps=24:start_time=0,"
                   "scale=w='max(2,trunc(min(1280,720*dar)/2)*2)':"
                   "h='max(2,trunc(min(720,1280/dar)/2)*2)',"
                   "setsar=1,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setpts=PTS-STARTPTS")
        _run("ffmpeg", ["-xerror", *input_args, "-t", "5", "-frames:v", "120",
                        "-vf", filters, "-c:v", "libx264", "-preset", "veryfast",
                        "-threads", "2", "-pix_fmt", "yuv420p", "-map_metadata", "-1",
                        "-r", "24", "-fps_mode", "cfr", "-video_track_timescale", "24000",
                        "-movflags", "+faststart", str(output)])
        times = verify_normalized(output)
        video = output.read_bytes()
    return NormalizedVideo(original, sha256(original).hexdigest(), video,
                           sha256(video).hexdigest(), times)
