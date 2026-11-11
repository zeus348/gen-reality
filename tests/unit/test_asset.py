"""Validate real PNG bytes and preserve the original image and alpha."""

from hashlib import sha256
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
import zlib

from PIL import Image

from features.asset.ingest import AssetError, load_asset


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "asset.png"

    def write_image(self, mode="RGBA", alpha=(0, 128, 255)):
        image = Image.new(mode, (3, 1))
        if mode == "RGBA":
            image.putdata([(10, 20, 30, value) for value in alpha])
        image.save(self.path)
        return image

    def test_original_hash_and_partial_alpha_are_preserved(self):
        image = self.write_image()
        original = self.path.read_bytes()
        asset = load_asset(self.path, non_human=True)
        self.assertEqual(asset.original_bytes, original)
        self.assertEqual(asset.sha256, sha256(original).hexdigest())
        self.assertEqual((asset.width, asset.height), (3, 1))
        self.assertEqual(asset.rgba, image.tobytes())
        self.assertEqual(self.path.read_bytes(), original)

    def test_binary_alpha_is_valid(self):
        image = self.write_image(alpha=(0, 0, 255))
        self.assertEqual(load_asset(self.path, non_human=True).rgba, image.tobytes())

    def test_non_human_declaration_is_required(self):
        self.write_image()
        for value in [False, None, 1, "true"]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(AssetError, "non-human"):
                    load_asset(self.path, non_human=value)

    def test_opaque_empty_and_nonzero_minimum_alpha_are_rejected(self):
        for alpha in [(255, 255, 255), (0, 0, 0), (1, 128, 255)]:
            with self.subTest(alpha=alpha):
                self.write_image(alpha=alpha)
                with self.assertRaisesRegex(AssetError, "transparent"):
                    load_asset(self.path, non_human=True)

    def test_rgb_palette_and_wrong_format_are_rejected(self):
        for mode in ["RGB", "P"]:
            with self.subTest(mode=mode):
                self.write_image(mode=mode)
                with self.assertRaisesRegex(AssetError, "8-bit RGBA PNG"):
                    load_asset(self.path, non_human=True)
        Image.new("RGB", (3, 1)).save(self.path, format="JPEG")
        with self.assertRaises(AssetError):
            load_asset(self.path, non_human=True)

    def test_missing_and_corrupt_files_have_safe_errors(self):
        with self.assertRaisesRegex(AssetError, "cannot be read"):
            load_asset(self.path, non_human=True)
        for contents in [b"dummy-secret", b"", b"\x89PNG\r\n\x1a\n"]:
            with self.subTest(contents=contents):
                self.path.write_bytes(contents)
                with self.assertRaises(AssetError) as caught:
                    load_asset(self.path, non_human=True)
                self.assertNotIn("dummy-secret", str(caught.exception))
        self.write_image()
        self.path.write_bytes(self.path.read_bytes()[:-20])
        with self.assertRaises(AssetError):
            load_asset(self.path, non_human=True)

    def test_oversized_file_is_rejected_before_decode(self):
        with self.path.open("wb") as file:
            file.truncate(32 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(AssetError, "exceeds 32 MiB"):
            load_asset(self.path, non_human=True)

    def test_oversized_dimensions_are_rejected_before_decode(self):
        self.write_image()
        contents = bytearray(self.path.read_bytes())
        for width, height in [(16_385, 1), (1, 16_385), (4097, 4096)]:
            with self.subTest(width=width, height=height):
                contents[16:24] = struct.pack(">II", width, height)
                contents[29:33] = struct.pack(">I", zlib.crc32(contents[12:29]))
                self.path.write_bytes(contents)
                with self.assertRaisesRegex(AssetError, "dimensions"):
                    load_asset(self.path, non_human=True)

    def test_16_bit_rgba_is_rejected(self):
        self.write_image()
        contents = bytearray(self.path.read_bytes())
        contents[24] = 16
        contents[29:33] = struct.pack(">I", zlib.crc32(contents[12:29]))
        self.path.write_bytes(contents)
        with self.assertRaisesRegex(AssetError, "8-bit RGBA PNG"):
            load_asset(self.path, non_human=True)

    def test_animated_png_is_rejected(self):
        first = self.write_image()
        second = Image.new("RGBA", (3, 1), (100, 0, 0, 255))
        first.save(self.path, save_all=True, append_images=[second], duration=100, loop=0)
        with self.assertRaisesRegex(AssetError, "single RGBA image"):
            load_asset(self.path, non_human=True)
