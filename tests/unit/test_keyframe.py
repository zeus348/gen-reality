"""Compare rendered pixels with independent integer alpha calculations."""

from hashlib import sha256
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from features.asset.ingest import load_asset
from features.guidance.keyframe import compose_frame
from features.placement.math import Placement


class KeyframeTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "asset.png"
        self.source = Image.new("RGB", (1280, 720), (20, 100, 200))

    def asset(self, size=(2, 2), pixels=None):
        image = Image.new("RGBA", size, (220, 50, 10, 255))
        image.putpixel((0, 0), (0, 0, 0, 0))
        if pixels is not None:
            image.putdata(pixels)
        image.save(self.path)
        return load_asset(self.path, non_human=True)

    def test_exact_alpha_blend_matches_independent_reference(self):
        pixels = [(200, 30, 70, 0), (220, 50, 10, 64),
                  (10, 200, 50, 128), (255, 255, 255, 255)]
        asset = self.asset(pixels=pixels)
        result = compose_frame(self.source, asset, Placement(0.5, 0.5, 2 / 720))
        reference = self.source.copy()
        for index, pixel in enumerate(pixels):
            alpha = pixel[3]
            rgb = tuple((alpha * color + (255 - alpha) * source + 127) // 255
                        for color, source in zip(pixel[:3], (20, 100, 200)))
            reference.putpixel((639 + index % 2, 359 + index // 2), rgb)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (1280, 720))
        self.assertEqual(result.tobytes(), reference.tobytes())

    def test_edge_clipping_keeps_visible_pixels(self):
        asset = self.asset()
        result = compose_frame(self.source, asset, Placement(0, 0, 2 / 720))
        self.assertEqual(result.getpixel((0, 0)), (220, 50, 10))
        self.assertEqual(result.getpixel((1, 0)), self.source.getpixel((1, 0)))
        self.assertEqual(result.getpixel((0, 1)), self.source.getpixel((0, 1)))

    def test_repeated_render_has_identical_png_hash(self):
        asset = self.asset()
        hashes = []
        for _ in range(2):
            image = compose_frame(self.source, asset, Placement(0.5, 0.5, 0.25))
            output = BytesIO()
            image.save(output, format="PNG")
            hashes.append(sha256(output.getvalue()).hexdigest())
        self.assertEqual(hashes[0], hashes[1])

    def test_source_and_asset_are_unchanged(self):
        asset = self.asset()
        source_bytes = self.source.tobytes()
        original = self.path.read_bytes()
        rgba = asset.rgba
        result = compose_frame(self.source, asset, Placement(0.5, 0.5, 0.25))
        self.assertIsNot(result, self.source)
        self.assertEqual(self.source.tobytes(), source_bytes)
        self.assertEqual(asset.rgba, rgba)
        self.assertEqual(self.path.read_bytes(), original)

    def test_invalid_source_format_is_rejected(self):
        asset = self.asset()
        for source in [Image.new("RGB", (10, 10)), Image.new("RGBA", (1280, 720))]:
            with self.subTest(mode=source.mode, size=source.size):
                with self.assertRaisesRegex(ValueError, "Source frame"):
                    compose_frame(source, asset, Placement(0.5, 0.5, 0.25))

    def test_extreme_aspect_ratio_uses_only_viewport_size(self):
        asset = self.asset(size=(16_384, 1))
        def guard_size(function):
            def checked(*args, **kwargs):
                width, height = kwargs.get("size", args[1] if len(args) > 1 else None)
                self.assertLessEqual(width * height, 1280 * 720,
                                     "Intermediate image exceeds viewport.")
                return function(*args, **kwargs)
            return checked

        with ExitStack() as checks:
            for owner, name in [(Image, "new"), (Image, "frombytes"),
                                (Image.Image, "resize"), (Image.Image, "transform")]:
                original = getattr(owner, name)
                checks.enter_context(patch.object(owner, name, autospec=True,
                                                 side_effect=guard_size(original)))
            result = compose_frame(self.source, asset, Placement(0.5, 0.5, 1))
        self.assertEqual(result.size, (1280, 720))
        self.assertEqual(result.getpixel((640, 360)), (220, 50, 10))
