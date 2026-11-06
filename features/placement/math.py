"""Convert a normalized placement to a rectangle. No images are allocated."""

from dataclasses import dataclass
import math


def _check_fraction(name: str, value: float, *, allow_zero: bool) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number.")
    if not 0 <= value <= 1 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} is outside its allowed range.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


@dataclass(frozen=True)
class Placement:
    """Center fractions and displayed height relative to the frame height."""

    u: float
    v: float
    height_frac: float
    rotation_deg: float = 0

    def __post_init__(self):
        _check_fraction("u", self.u, allow_zero=True)
        _check_fraction("v", self.v, allow_zero=True)
        _check_fraction("height_frac", self.height_frac, allow_zero=False)
        if type(self.rotation_deg) not in (int, float) or self.rotation_deg != 0:
            raise ValueError("rotation_deg must be zero.")


@dataclass(frozen=True)
class SpriteBounds:
    """Unclipped pixel rectangle. Right and bottom edges are excluded."""

    left: int
    top: int
    width: int
    height: int


def sprite_bounds(
    placement: Placement,
    asset_width: int,
    asset_height: int,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> SpriteBounds:
    """Keep the asset aspect ratio and round pixel values to nearest, ties to even.

    A positive height occupies at least one pixel. The center error is at most
    half a pixel. Keep negative edges so the renderer can clip at the viewport.
    """
    for dimension in (asset_width, asset_height, frame_width, frame_height):
        if type(dimension) is not int or not 1 <= dimension <= 16_384:
            raise ValueError("Image dimensions must be integers from 1 to 16384.")
    height = max(1, round(placement.height_frac * frame_height))
    width = max(1, round(height * asset_width / asset_height))
    left = round(placement.u * frame_width - width / 2)
    top = round(placement.v * frame_height - height / 2)
    return SpriteBounds(left, top, width, height)
