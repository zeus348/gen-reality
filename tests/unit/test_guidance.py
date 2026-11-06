"""Verify both guidance artifacts and every frame before encoding."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from features.asset.ingest import load_asset
from features.guidance import video as compiler
from features.guidance.video import DEFAULT_PROMPT, GuidanceError, compile_guidance
from features.media.ingest import normalize_video, verify_normalized
from features.placement.math import Placement


class GuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.root = Path(cls.folder.name)
        cls.source = cls.root / "source.mp4"
        cls.bad_codec = cls.root / "mpeg4.mp4"
        for path, size, codec in [(cls.source, "64x36", "libx264"),
                                  (cls.bad_codec, "1280x720", "mpeg4")]:
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            f"testsrc2=s={size}:r=24:d=5", "-c:v", codec,
                            "-threads", "2", "-pix_fmt", "yuv420p", str(path)],
                           capture_output=True, check=True, timeout=30)
        cls.video = normalize_video(cls.source)
        image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        for y in range(1, 7):
            for x in range(1, 7):
                image.putpixel((x, y), (240, 120, 20, 128))
        cls.asset_path = cls.root / "asset.png"
        image.save(cls.asset_path)
        cls.asset = load_asset(cls.asset_path, non_human=True)
        cls.placement = Placement(0.5, 0.5, 8 / 720)

    def test_real_compilation_preserves_inputs_and_renders_every_frame(self):
        source_bytes = self.source.read_bytes()
        asset_bytes = self.asset_path.read_bytes()
        original = compiler.compose_frame
        run_tool = compiler._run
        frame_count = 0
        first = None
        frame_hashes = []
        encoder_checked = False

        def check_frame(frame, asset, placement):
            nonlocal frame_count, first
            reference = frame.copy()
            for y in range(1, 7):
                for x in range(1, 7):
                    position = (636 + x, 356 + y)
                    background = frame.getpixel(position)
                    rgb = tuple((128 * color + 127 * source + 127) // 255
                                for color, source in zip((240, 120, 20), background))
                    reference.putpixel(position, rgb)
            result = original(frame, asset, placement)
            self.assertEqual(result.tobytes(), reference.tobytes())
            if frame_count == 0:
                first = reference.tobytes()
            frame_hashes.append(sha256(reference.tobytes()).hexdigest())
            frame_count += 1
            return result

        def check_encoder(tool, arguments):
            nonlocal encoder_checked
            input_path = Path(arguments[arguments.index("-i") + 1])
            if input_path.suffix == ".rgb":
                self.assertEqual(len(frame_hashes), 120)
                with input_path.open("rb") as frames:
                    for expected_hash in frame_hashes:
                        self.assertEqual(sha256(frames.read(1280 * 720 * 3)).hexdigest(),
                                         expected_hash)
                    self.assertEqual(frames.read(1), b"")
                encoder_checked = True
            return run_tool(tool, arguments)

        with patch.object(compiler, "compose_frame", side_effect=check_frame), \
                patch.object(compiler, "_run", side_effect=check_encoder):
            guidance = compile_guidance(self.video, self.asset, self.placement)
        self.assertEqual(frame_count, 120)
        self.assertTrue(encoder_checked)
        with Image.open(BytesIO(guidance.keyframe_png)) as keyframe:
            self.assertEqual(keyframe.mode, "RGB")
            self.assertEqual(keyframe.size, (1280, 720))
            self.assertEqual(keyframe.tobytes(), first)
        path = self.root / "guide.mp4"
        path.write_bytes(guidance.guide_mp4)
        self.assertEqual(len(verify_normalized(path)), 120)
        decoded = self.root / "decoded-guide.png"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                        "-threads", "2", str(decoded)], capture_output=True, check=True,
                       timeout=30)
        with Image.open(decoded) as frame:
            expected = Image.frombytes("RGB", (1280, 720), first).getpixel((640, 360))
            for actual, reference in zip(frame.getpixel((640, 360)), expected):
                self.assertLessEqual(abs(actual - reference), 25)
        self.assertEqual(guidance.keyframe_sha256, sha256(guidance.keyframe_png).hexdigest())
        self.assertEqual(guidance.guide_sha256, sha256(guidance.guide_mp4).hexdigest())
        self.assertEqual(guidance.prompt, DEFAULT_PROMPT)
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(self.asset_path.read_bytes(), asset_bytes)

    def test_repeated_compilation_has_identical_artifact_hashes(self):
        first = compile_guidance(self.video, self.asset, self.placement)
        second = compile_guidance(self.video, self.asset, self.placement)
        self.assertEqual(first.keyframe_sha256, second.keyframe_sha256)
        self.assertEqual(first.guide_sha256, second.guide_sha256)

    def test_invalid_prompts_are_rejected(self):
        for prompt in ["", "   ", "x" * 1001, None, 1]:
            with self.subTest(prompt=str(prompt)[:20]):
                with self.assertRaisesRegex(GuidanceError, "Prompt"):
                    compile_guidance(self.video, self.asset, self.placement, prompt)

    def test_invalid_normalized_input_is_rejected(self):
        for contents in [b"broken", self.bad_codec.read_bytes()]:
            with self.subTest(size=len(contents)):
                video = replace(self.video, video_bytes=contents)
                with self.assertRaises(GuidanceError):
                    compile_guidance(video, self.asset, self.placement)

    def test_incomplete_raw_decode_is_rejected(self):
        def short_decode(tool, arguments):
            Path(arguments[-1]).write_bytes(b"short")
            return b""
        with patch.object(compiler, "_run", side_effect=short_decode):
            with self.assertRaisesRegex(GuidanceError, "exactly 120 RGB frames"):
                compile_guidance(self.video, self.asset, self.placement)

    def test_compile_errors_do_not_echo_diagnostics(self):
        with patch.object(compiler, "_run", side_effect=OSError("dummy-secret")):
            with self.assertRaises(GuidanceError) as caught:
                compile_guidance(self.video, self.asset, self.placement)
        self.assertNotIn("dummy-secret", str(caught.exception))
