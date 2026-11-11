"""Validate a prepared PNG without changing its original bytes or alpha."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError


MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_777_216
MAX_DIMENSION = 16_384


class AssetError(ValueError):
    """The asset does not meet the prepared PNG contract."""


@dataclass(frozen=True)
class ValidatedAsset:
    """Original PNG and decoded RGBA bytes, with the original content hash."""

    original_bytes: bytes
    sha256: str
    width: int
    height: int
    rgba: bytes


def load_asset(path: Path, *, non_human: bool) -> ValidatedAsset:
    """Require the caller's non-human declaration; no subject detection is done.

    Accept one 8-bit RGBA PNG with at least one fully transparent pixel and one
    nontransparent pixel. Keep binary and partial alpha values unchanged.
    """
    if non_human is not True:
        raise AssetError("Confirm that the asset is non-human.")
    try:
        with Path(path).open("rb") as asset_file:
            contents = asset_file.read(MAX_FILE_BYTES + 1)
    except OSError:
        raise AssetError("Asset file cannot be read.") from None
    if len(contents) > MAX_FILE_BYTES:
        raise AssetError("Asset file exceeds 32 MiB.")

    # IHDR records bit depth and color type. Reject silent decoder conversion.
    if (len(contents) < 33 or contents[:8] != b"\x89PNG\r\n\x1a\n"
            or contents[12:16] != b"IHDR" or contents[24:26] != b"\x08\x06"):
        raise AssetError("Asset must be an 8-bit RGBA PNG.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(contents)) as image:
                width, height = image.size
                if (not 1 <= width <= MAX_DIMENSION
                        or not 1 <= height <= MAX_DIMENSION
                        or width * height > MAX_PIXELS):
                    raise AssetError("Asset dimensions exceed the allowed size.")
                if image.mode != "RGBA" or getattr(image, "n_frames", 1) != 1:
                    raise AssetError("Asset must be a single RGBA image.")
                image.verify()
            with Image.open(BytesIO(contents)) as image:
                image.load()
                minimum, maximum = image.getchannel("A").getextrema()
                if minimum != 0 or maximum == 0:
                    raise AssetError("Asset needs transparent and nontransparent pixels.")
                rgba = image.tobytes()
    except AssetError:
        raise
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise AssetError("Asset PNG cannot be decoded safely.") from None
    return ValidatedAsset(contents, sha256(contents).hexdigest(), width, height, rgba)
