"""Extract one decoded video frame as a PNG without changing the source."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

from PIL import Image

from features.asset.ingest import MAX_FILE_BYTES as MAX_PNG_BYTES
from features.media.ingest import MAX_INPUT_PIXELS, _run
from features.output.validate import MAX_RAW_BYTES, MAX_RAW_FRAMES


class FrameExtractionError(ValueError):
    """A requested frame could not be extracted and verified safely."""


def _read_png(path: Path, width: int, height: int) -> bytes:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_PNG_BYTES:
        raise FrameExtractionError("Extracted PNG must be a regular file within the size limit.")
    with path.open("rb") as file:
        contents = file.read(MAX_PNG_BYTES + 1)
    if not 0 < len(contents) <= MAX_PNG_BYTES:
        raise FrameExtractionError("Extracted PNG exceeds its size limit.")
    # Pillow can expose 16-bit truecolor PNGs as RGB after reducing bit depth.
    if (len(contents) < 33 or contents[:8] != b"\x89PNG\r\n\x1a\n"
            or contents[12:16] != b"IHDR" or contents[24:26] != b"\x08\x02"):
        raise FrameExtractionError("Extracted PNG must contain 8-bit RGB pixels.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(contents), formats=["PNG"]) as image:
            if (image.format != "PNG" or image.mode != "RGB" or image.size != (width, height)
                    or getattr(image, "n_frames", 1) != 1):
                raise FrameExtractionError("Extracted PNG must be one RGB image with the expected dimensions.")
            image.verify()
        with Image.open(BytesIO(contents), formats=["PNG"]) as image:
            image.load()
    return contents


def extract_frame_png(contents: bytes, frame_index: int, *, width: int,
                      height: int, frame_count: int) -> bytes:
    """Extract one exact decoded index from already validated MP4 bytes.

    The caller supplies matching trusted dimensions and frame count. Full video
    validation and attempt identity remain caller responsibilities. Decode in
    original frame order without automatic rotation, scaling, or frame-rate
    conversion. Return a derived 8-bit RGB PNG; the caller retains original PTS.
    Use private temporary files because image data can exceed the tool wrapper's
    stdout limit. This helper publishes no evidence and assigns no verdict.
    """
    if type(contents) is not bytes or not 0 < len(contents) <= MAX_RAW_BYTES:
        raise FrameExtractionError("Frame extraction requires 1 byte to 256 MiB of validated MP4 data.")
    if type(frame_count) is not int or not 1 <= frame_count <= MAX_RAW_FRAMES:
        raise FrameExtractionError("Frame count exceeds the allowed range.")
    if type(frame_index) is not int or not 0 <= frame_index < frame_count:
        raise FrameExtractionError("Frame index must select an existing declared frame.")
    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= 4096 or not 1 <= height <= 4096
            or width * height > MAX_INPUT_PIXELS):
        raise FrameExtractionError("Frame dimensions exceed the allowed size.")
    try:
        with TemporaryDirectory(prefix="gen-reality-frame-") as folder:
            root = Path(folder)
            source = root / "source.mp4"
            output = root / "frames"
            output.mkdir(mode=0o700)
            source.write_bytes(contents)
            _run("ffmpeg", ["-xerror", "-nostdin", "-threads", "2", "-err_detect", "explode",
                            "-protocol_whitelist", "file,pipe", "-f", "mov", "-noautorotate",
                            "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
                            "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1",
                            "-fps_mode", "passthrough", "-map_metadata", "-1",
                            "-c:v", "png", "-pix_fmt", "rgb24", "-threads", "2",
                            "-f", "image2", str(output / "frame-%02d.png")])
            expected = output / "frame-01.png"
            if list(output.iterdir()) != [expected]:
                raise FrameExtractionError("Frame extraction must produce exactly one PNG.")
            return _read_png(expected, width, height)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise FrameExtractionError("Video frame could not be extracted and verified safely.") from None
