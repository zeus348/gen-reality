"""Render a static asset over an RGB frame before video encoding."""

from PIL import Image

from features.asset.ingest import ValidatedAsset
from features.placement.math import Placement, sprite_bounds


def compose_frame(frame: Image.Image, asset: ValidatedAsset, placement: Placement) -> Image.Image:
    """Return a new RGB frame; keep source and asset data unchanged.

    Resize with Pillow's bilinear filter. Sample only the output viewport so an
    extreme asset aspect ratio cannot allocate a huge intermediate image.
    Blend each channel as round((alpha * asset + (255-alpha) * source) / 255).
    """
    if frame.mode != "RGB" or frame.size != (1280, 720):
        raise ValueError("Source frame must be RGB with dimensions 1280 by 720.")
    bounds = sprite_bounds(placement, asset.width, asset.height)
    scale_x = asset.width / bounds.width
    scale_y = asset.height / bounds.height
    sprite = Image.frombytes("RGBA", (asset.width, asset.height), asset.rgba)
    if (asset.width, asset.height) == (bounds.width, bounds.height):
        # Avoid resampling: conversion to premultiplied alpha can round colors.
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        overlay.paste(sprite, (bounds.left, bounds.top))
    else:
        overlay = sprite.transform(
            frame.size,
            Image.Transform.AFFINE,
            (scale_x, 0, -bounds.left * scale_x, 0, scale_y, -bounds.top * scale_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0, 0),
        )
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
