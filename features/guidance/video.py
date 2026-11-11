"""Compile matching Mode A and B guidance from the same source and placement."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from features.asset.ingest import ValidatedAsset
from features.guidance.keyframe import compose_frame
from features.media.ingest import MediaError, NormalizedVideo, _run, verify_normalized
from features.placement.math import Placement


FRAME_BYTES = 1280 * 720 * 3
DEFAULT_PROMPT = ("Make the placed creature subtly turn its head. "
                  "Preserve the original camera, room and object appearance.")


class GuidanceError(ValueError):
    """The matching guidance artifacts cannot be compiled."""


@dataclass(frozen=True)
class Guidance:
    keyframe_png: bytes
    keyframe_sha256: str
    guide_mp4: bytes
    guide_sha256: str
    prompt: str


def compile_guidance(
    video: NormalizedVideo,
    asset: ValidatedAsset,
    placement: Placement,
    prompt: str = DEFAULT_PROMPT,
) -> Guidance:
    """Render all 120 RGB frames before encoding, keeping original inputs intact.

    Raw frame files use about 633 MiB of temporary disk space. Only one source
    frame and its rendered result are held in memory at a time. No provider is
    contacted. The same frozen prompt belongs to both modes.
    """
    if type(prompt) is not str or not prompt.strip() or len(prompt) > 1000:
        raise GuidanceError("Prompt must contain from 1 to 1000 characters.")
    try:
        with TemporaryDirectory(prefix="gen-reality-guidance-") as folder:
            root = Path(folder)
            source = root / "source.mp4"
            decoded = root / "source.rgb"
            rendered = root / "guide.rgb"
            output = root / "guide.mp4"
            source.write_bytes(video.video_bytes)
            verify_normalized(source)
            _run("ffmpeg", ["-xerror", "-nostdin", "-threads", "2",
                            "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(source),
                            "-map", "0:v:0", "-frames:v", "121", "-an", "-sn", "-dn",
                            "-c:v", "rawvideo", "-threads", "2", "-pix_fmt", "rgb24",
                            "-f", "rawvideo", str(decoded)])
            if decoded.stat().st_size != 120 * FRAME_BYTES:
                raise GuidanceError("Source must decode to exactly 120 RGB frames.")
            keyframe = BytesIO()
            with decoded.open("rb") as frames, rendered.open("wb") as target:
                for index in range(120):
                    frame = Image.frombytes("RGB", (1280, 720), frames.read(FRAME_BYTES))
                    result = compose_frame(frame, asset, placement)
                    if index == 0:
                        result.save(keyframe, format="PNG")
                    target.write(result.tobytes())
            _run("ffmpeg", ["-nostdin", "-threads", "2", "-protocol_whitelist", "file,pipe",
                            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x720",
                            "-framerate", "24", "-i", str(rendered), "-frames:v", "120", "-an",
                            "-vf", "setsar=1",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "2",
                            "-pix_fmt", "yuv420p", "-r", "24", "-fps_mode", "cfr",
                            "-video_track_timescale", "24000", "-movflags", "+faststart", str(output)])
            verify_normalized(output)
            guide = output.read_bytes()
    except (OSError, MediaError):
        raise GuidanceError("Guidance could not be compiled safely.") from None
    png = keyframe.getvalue()
    return Guidance(png, sha256(png).hexdigest(), guide, sha256(guide).hexdigest(), prompt)
