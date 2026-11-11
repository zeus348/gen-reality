"""Verify exact frame pixels and bounded PNG extraction with local fixtures."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

from PIL import Image

from features.output import extract
from features.output.validate import validate_raw_video


class FrameExtractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.root = Path(cls.folder.name)
        cls.frames = []
        for index, color in enumerate(((240, 20, 30), (10, 230, 40), (20, 30, 220), (180, 140, 60))):
            frame = Image.new("RGB", (64, 36), color)
            frame.putpixel((0, 0), (11 + index, 37, 255 - index))
            frame.putpixel((63, 35), (23, 61 + index, 151))
            cls.frames.append(frame.tobytes())
        source = cls.root / "source.mp4"
        cls.encode(source, b"".join(cls.frames), 64, 36)
        cls.contents = source.read_bytes()
        cls.video = validate_raw_video(cls.contents)
        shifted = cls.root / "shifted.mp4"
        cls.command(["-i", str(source), "-map", "0:v:0", "-c", "copy",
                     "-output_ts_offset", "0.01", "-video_track_timescale", "30000", str(shifted)])
        cls.shifted = validate_raw_video(shifted.read_bytes())
        rotated = cls.root / "rotated.mp4"
        cls.command(["-display_rotation:v:0", "90", "-i", str(source),
                     "-map", "0:v:0", "-c", "copy", str(rotated)])
        cls.rotated = validate_raw_video(rotated.read_bytes())
        rotation = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(rotated)],
                                  capture_output=True, check=True, timeout=30)
        cls.rotation_metadata = json.loads(rotation.stdout)
        cls.noise = random.Random(14).randbytes(1024 * 512 * 3)
        large = cls.root / "large.mp4"
        cls.encode(large, cls.noise, 1024, 512)
        cls.large = validate_raw_video(large.read_bytes())

    @staticmethod
    def command(arguments):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", *arguments],
                       capture_output=True, check=True, timeout=30)

    @classmethod
    def encode(cls, target, pixels, width, height):
        raw = target.with_suffix(".rgb")
        raw.write_bytes(pixels)
        cls.command(["-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                     "-framerate", "30", "-i", str(raw), "-c:v", "libx264rgb", "-crf", "0",
                     "-preset", "ultrafast", "-pix_fmt", "rgb24", "-threads", "2",
                     "-video_track_timescale", "30000", str(target)])

    def frame(self, index=0, video=None, **changes):
        video = self.video if video is None else video
        arguments = {"width": video.width, "height": video.height, "frame_count": len(video.frame_seconds)}
        arguments.update(changes)
        return extract.extract_frame_png(video.video_bytes, index, **arguments)

    def assert_pixels(self, png, expected, size=(64, 36)):
        with Image.open(BytesIO(png)) as image:
            self.assertEqual((image.format, image.mode, image.size, image.n_frames), ("PNG", "RGB", size, 1))
            self.assertEqual(image.tobytes(), expected)

    @staticmethod
    def image_bytes(mode="RGB", format="PNG", *, animated=False):
        output = BytesIO()
        image = Image.new(mode, (64, 36), "red")
        options = {"save_all": True, "append_images": [Image.new(mode, (64, 36), "blue")]} if animated else {}
        image.save(output, format=format, **options)
        return output.getvalue()

    @staticmethod
    def write_outputs(contents, *, count=1):
        def run(tool, arguments):
            pattern = arguments[-1]
            for number in range(1, count + 1):
                Path(pattern.replace("%02d", f"{number:02d}")).write_bytes(contents)
            return b""
        return run

    @staticmethod
    def rgb_png(bit_depth, compressed_pixels):
        def chunk(name, contents):
            return (len(contents).to_bytes(4, "big") + name + contents
                    + zlib.crc32(name + contents).to_bytes(4, "big"))

        header = (64).to_bytes(4, "big") + (36).to_bytes(4, "big") + bytes((bit_depth, 2, 0, 0, 0))
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                + chunk(b"IDAT", compressed_pixels) + chunk(b"IEND", b""))

    def test_distinct_lossless_frames_are_selected_by_exact_decoded_index(self):
        original_hash = sha256(self.contents).hexdigest()
        self.assertEqual(self.video.frame_rate, "30/1")
        for index in (0, 2, 3):
            with self.subTest(index=index):
                self.assert_pixels(self.frame(index), self.frames[index])
        self.assertEqual(sha256(self.contents).hexdigest(), original_hash)
        self.assertEqual((self.root / "source.mp4").read_bytes(), self.contents)

    def test_shifted_timestamps_do_not_change_decoded_index_selection(self):
        original_times = self.shifted.frame_seconds
        self.assertAlmostEqual(original_times[0], 0.01)
        self.assert_pixels(self.frame(2, self.shifted), self.frames[2])
        self.assertEqual(self.shifted.frame_seconds, original_times)

    def test_rotation_metadata_does_not_rotate_or_resize_the_frame(self):
        side_data = self.rotation_metadata["streams"][0]["side_data_list"]
        self.assertTrue(any(item.get("rotation") == 90 for item in side_data))
        self.assertEqual((self.rotated.width, self.rotated.height), (64, 36))
        self.assert_pixels(self.frame(1, self.rotated), self.frames[1])

    def test_png_above_stdout_limit_is_returned_through_temporary_files(self):
        png = self.frame(0, self.large)
        self.assertGreater(len(png), 1024 * 1024)
        self.assert_pixels(png, self.noise, (1024, 512))

    def test_temporary_files_are_private_contain_exact_input_and_are_removed(self):
        run = extract._run
        roots = []

        def inspect_files(tool, arguments):
            source = Path(arguments[arguments.index("-i") + 1])
            output = Path(arguments[-1]).parent
            roots.append(source.parent)
            self.assertEqual(source.read_bytes(), self.contents)
            self.assertEqual(source.parent.stat().st_mode & 0o077, 0)
            self.assertEqual(output.stat().st_mode & 0o077, 0)
            return run(tool, arguments)

        with patch.object(extract, "_run", side_effect=inspect_files):
            self.assert_pixels(self.frame(1), self.frames[1])
        self.assertEqual(len(roots), 1)
        self.assertFalse(roots[0].exists())

    def test_invalid_bytes_are_rejected_before_media_tools(self):
        for contents in (None, "private fixture detail", bytearray(b"video"), b""):
            with self.subTest(type=type(contents).__name__), patch.object(extract, "_run") as run:
                with self.assertRaises(extract.FrameExtractionError):
                    extract.extract_frame_png(contents, 0, width=64, height=36, frame_count=4)
                run.assert_not_called()
        with patch.object(extract, "MAX_RAW_BYTES", 1), patch.object(extract, "_run") as run:
            with self.assertRaises(extract.FrameExtractionError):
                extract.extract_frame_png(b"12", 0, width=64, height=36, frame_count=4)
            run.assert_not_called()

    def test_invalid_indexes_counts_and_geometry_are_rejected_before_media_tools(self):
        cases = []
        for value in (True, False, None, "1", 1.0, -1, 4, 10 ** 1000):
            cases.append({"frame_index": value})
        for value in (True, False, None, "4", 4.0, 0, -1, 2401, 10 ** 1000):
            cases.append({"frame_count": value})
        for field in ("width", "height"):
            for value in (True, False, None, "64", 64.0, 0, -1, 4097, 10 ** 1000):
                cases.append({field: value})
        cases.append({"width": 4096, "height": 2161})
        for changes in cases:
            arguments = {"frame_index": 0, "width": 64, "height": 36, "frame_count": 4}
            arguments.update(changes)
            with self.subTest(changes=str(changes)[:60]), patch.object(extract, "_run") as run:
                with self.assertRaises(extract.FrameExtractionError):
                    extract.extract_frame_png(self.contents, **arguments)
                run.assert_not_called()

    def test_actual_missing_frame_or_wrong_dimensions_cannot_pass(self):
        with self.assertRaises(extract.FrameExtractionError):
            self.frame(4, frame_count=5)
        with self.assertRaises(extract.FrameExtractionError):
            self.frame(width=63)

    def test_missing_or_multiple_outputs_are_rejected(self):
        png = self.image_bytes()
        for count in (0, 2):
            with self.subTest(count=count), patch.object(extract, "_run", side_effect=self.write_outputs(png, count=count)):
                with self.assertRaises(extract.FrameExtractionError):
                    self.frame()

    def test_output_size_limit_is_checked_before_image_decoding(self):
        png = self.image_bytes()
        with patch.object(extract, "_run", side_effect=self.write_outputs(png)), \
                patch.object(extract, "MAX_PNG_BYTES", len(png) - 1), patch.object(extract.Image, "open") as open_image:
            with self.assertRaises(extract.FrameExtractionError):
                self.frame()
            open_image.assert_not_called()

    def test_wrong_format_mode_animation_and_corrupt_png_are_rejected(self):
        png = self.image_bytes()
        damaged = bytearray(png)
        damaged[29] ^= 1
        for contents in (b"", self.image_bytes(format="BMP"), self.image_bytes(mode="RGBA"),
                         self.image_bytes(animated=True), bytes(damaged), png[:-15]):
            with self.subTest(size=len(contents)), patch.object(extract, "_run", side_effect=self.write_outputs(contents)):
                with self.assertRaises(extract.FrameExtractionError):
                    self.frame()

    def test_png_with_valid_checksums_must_still_decode_all_pixels(self):
        contents = self.rgb_png(8, b"invalid compressed pixels")
        with Image.open(BytesIO(contents)) as image:
            image.verify()
        with patch.object(extract, "_run", side_effect=self.write_outputs(contents)):
            with self.assertRaises(extract.FrameExtractionError):
                self.frame()

    def test_16_bit_truecolor_png_is_rejected_even_when_pillow_reports_rgb(self):
        row = b"\x00" + b"\x12\x00\x34\x00\x56\x00" * 64
        contents = self.rgb_png(16, zlib.compress(row * 36))
        with Image.open(BytesIO(contents)) as image:
            image.load()
            self.assertEqual((image.mode, image.getpixel((0, 0))), ("RGB", (18, 52, 86)))
        with patch.object(extract, "_run", side_effect=self.write_outputs(contents)):
            with self.assertRaises(extract.FrameExtractionError):
                self.frame()

    def test_corrupt_media_and_tool_errors_do_not_disclose_details(self):
        with self.assertRaises(extract.FrameExtractionError) as caught:
            extract.extract_frame_png(b"private fixture detail", 0, width=64, height=36, frame_count=1)
        self.assertNotIn("private fixture detail", str(caught.exception))
        for error in (OSError("private fixture detail"), ValueError("private fixture detail")):
            with self.subTest(error=type(error).__name__), patch.object(extract, "_run", side_effect=error):
                with self.assertRaises(extract.FrameExtractionError) as caught:
                    self.frame()
                self.assertNotIn("private fixture detail", str(caught.exception))
