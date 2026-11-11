"""Use real FFmpeg fixtures to check trim, format, aspect ratio, and timebase."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from features.media import ingest
from features.media.ingest import MediaError, normalize_video, verify_normalized


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.root = Path(cls.folder.name)
        cls.source = cls.root / "source.mp4"
        cls.portrait = cls.root / "portrait.mov"
        cls.command(["-f", "lavfi", "-i", "color=red:s=64x36:r=24:d=5",
                     "-f", "lavfi", "-i", "color=blue:s=64x36:r=24:d=1",
                     "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
                     "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                     "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-c:a", "aac",
                     "-pix_fmt", "yuv420p", str(cls.source)])
        cls.command(["-f", "lavfi", "-i", "color=white:s=36x64:r=24:d=5",
                     "-vf", "drawbox=x=0:y=0:w=36:h=8:color=red:t=fill,"
                     "drawbox=x=0:y=56:w=36:h=8:color=blue:t=fill,"
                     "drawbox=x=0:y=0:w=4:h=64:color=green:t=fill,"
                     "drawbox=x=32:y=0:w=4:h=64:color=yellow:t=fill",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.portrait)])
        cls.original = cls.source.read_bytes()
        cls.normalized = normalize_video(cls.source)
        cls.normalized_portrait = normalize_video(cls.portrait)
        cls.output = cls.root / "normalized.mp4"
        cls.output.write_bytes(cls.normalized.video_bytes)
        cls.portrait_output = cls.root / "portrait-normalized.mp4"
        cls.portrait_output.write_bytes(cls.normalized_portrait.video_bytes)

    @staticmethod
    def command(arguments):
        return subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *arguments],
                              capture_output=True, check=True, timeout=30).stdout

    def pixel(self, path, seconds, width, x, y):
        pixels = self.command(["-ss", str(seconds), "-i", str(path), "-frames:v", "1",
                               "-threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        offset = (y * width + x) * 3
        return tuple(pixels[offset:offset + 3])

    def test_real_output_has_exact_media_format_and_timebase(self):
        normalized = normalize_video(self.source)
        output = self.root / "format-check.mp4"
        output.write_bytes(normalized.video_bytes)
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                 "-count_frames", "-of", "json", str(output)],
                                capture_output=True, check=True, timeout=30)
        metadata = json.loads(result.stdout)
        self.assertEqual(len(metadata["streams"]), 1)
        video = metadata["streams"][0]
        for key, expected in {"codec_name": "h264", "codec_type": "video", "width": 1280,
                              "height": 720, "pix_fmt": "yuv420p", "r_frame_rate": "24/1",
                              "nb_read_frames": "120", "sample_aspect_ratio": "1:1"}.items():
            with self.subTest(field=key):
                self.assertEqual(video[key], expected)
        self.assertAlmostEqual(float(metadata["format"]["duration"]), 5, places=5)
        self.assertEqual(len(normalized.frame_seconds), 120)
        for index, timestamp in enumerate(normalized.frame_seconds):
            self.assertAlmostEqual(timestamp, index / 24, places=5)

    def test_original_bytes_and_hashes_are_preserved(self):
        normalized = normalize_video(self.source)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(normalized.original_bytes, self.original)
        self.assertEqual(normalized.original_sha256, sha256(self.original).hexdigest())
        self.assertEqual(normalized.sha256, sha256(normalized.video_bytes).hexdigest())

    def test_only_the_first_five_seconds_are_used(self):
        output = self.root / "trim-check.mp4"
        output.write_bytes(normalize_video(self.source).video_bytes)
        self.assertGreater(self.pixel(self.source, 5.5, 64, 32, 18)[2], 240)
        for seconds in [0, 1, 2, 3, 4, 4.9]:
            with self.subTest(seconds=seconds):
                red, green, blue = self.pixel(output, seconds, 1280, 640, 360)
                self.assertGreater(red, 240)
                self.assertLess(green + blue, 20)

    def test_portrait_is_pillarboxed_without_crop(self):
        output = self.root / "portrait-check.mp4"
        output.write_bytes(normalize_video(self.portrait).video_bytes)
        top = self.pixel(output, 0, 1280, 640, 10)
        bottom = self.pixel(output, 0, 1280, 640, 710)
        left = self.pixel(output, 0, 1280, 445, 360)
        right = self.pixel(output, 0, 1280, 835, 360)
        self.assertGreater(top[0], 240)
        self.assertLess(top[1] + top[2], 20)
        self.assertGreater(bottom[2], 240)
        self.assertLess(bottom[0] + bottom[1], 20)
        self.assertGreater(left[1], 100)
        self.assertLess(left[0] + left[2], 20)
        self.assertGreater(min(right[:2]), 240)
        self.assertLess(right[2], 20)
        self.assertLess(max(self.pixel(output, 0, 1280, 10, 360)), 10)
        self.assertLess(max(self.pixel(output, 0, 1280, 1270, 360)), 10)

    def test_short_and_long_inputs_are_rejected(self):
        for seconds in [4, 21]:
            path = self.root / f"duration-{seconds}.mp4"
            self.command(["-f", "lavfi", "-i", f"color=red:s=16x16:r=24:d={seconds}",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
            with self.subTest(seconds=seconds):
                with self.assertRaisesRegex(MediaError, "duration"):
                    normalize_video(path)

    def test_bad_paths_and_corrupt_media_have_safe_errors(self):
        for name, contents in [("wrong.avi", self.original), ("corrupt.mp4", b"dummy-secret")]:
            path = self.root / name
            path.write_bytes(contents)
            with self.subTest(name=name):
                with self.assertRaises(MediaError) as caught:
                    normalize_video(path)
                self.assertNotIn("dummy-secret", str(caught.exception))
        with self.assertRaisesRegex(MediaError, "cannot be read"):
            normalize_video(self.root / "missing.mp4")

    def test_invalid_normalized_video_is_rejected(self):
        with self.assertRaises(MediaError):
            verify_normalized(self.source)
        path = self.root / "truncated.mp4"
        path.write_bytes(self.normalized.video_bytes[:len(self.normalized.video_bytes) // 2])
        with self.assertRaises(MediaError):
            verify_normalized(path)

    def test_process_errors_are_sanitized(self):
        for failure in ["missing", "timeout", "error"]:
            with self.subTest(failure=failure):
                process = self.fake_process(stderr=b"dummy-secret" if failure == "error" else b"")
                if failure == "timeout":
                    process.wait.side_effect = [subprocess.TimeoutExpired("dummy-secret", 90), -9]
                options = {"side_effect": OSError("dummy-secret")} if failure == "missing" else {"return_value": process}
                with patch.object(ingest.subprocess, "Popen", **options):
                    with self.assertRaises(MediaError) as caught:
                        ingest._run("ffprobe", [])
                    self.assertNotIn("dummy-secret", str(caught.exception))

    def test_probe_uses_local_container_and_bounded_process(self):
        process = self.fake_process(stdout=b"{}")
        with patch.object(ingest.subprocess, "Popen", return_value=process) as run:
            ingest._probe(self.source, frames=True)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-protocol_whitelist") + 1], "file,pipe")
        self.assertEqual(command[command.index("-f") + 1], "mov")
        self.assertEqual(command[command.index("-max_alloc") + 1], "67108864")
        self.assertEqual(command[command.index("-read_intervals") + 1], "%+#121")
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(process.wait.call_args.kwargs["timeout"], 90)

    @staticmethod
    def fake_process(stdout=b"", stderr=b""):
        process = MagicMock()
        process.__enter__.return_value = process
        process.stdout = BytesIO(stdout)
        process.stderr = BytesIO(stderr)
        process.returncode = 0
        return process

    def test_oversized_process_output_is_killed_and_rejected(self):
        for stream, limit in [("stdout", 1024 * 1024), ("stderr", 65_536)]:
            with self.subTest(stream=stream):
                process = self.fake_process(**{stream: b"x" * (limit + 1)})
                with patch.object(ingest.subprocess, "Popen", return_value=process):
                    with self.assertRaisesRegex(MediaError, "output exceeds"):
                        ingest._run("ffprobe", [])
                process.kill.assert_called()

    def test_size_and_dimension_limits_are_enforced(self):
        with patch.object(ingest, "MAX_INPUT_BYTES", 4):
            with self.assertRaisesRegex(MediaError, "exceeds 256 MiB"):
                normalize_video(self.source)
        metadata = {"streams": [{"codec_type": "video", "duration": "6", "width": 4097, "height": 1}],
                    "format": {"duration": "6"}}
        with patch.object(ingest, "_probe", return_value=metadata):
            with self.assertRaisesRegex(MediaError, "dimensions"):
                normalize_video(self.source)

    def test_bad_frame_times_and_duration_are_rejected(self):
        for change in ["time", "count", "duration"]:
            metadata = ingest._probe(self.output, frames=True)
            if change == "time":
                metadata["frames"][1]["best_effort_timestamp_time"] = "0.1"
            elif change == "count":
                metadata["frames"].pop()
            else:
                metadata["streams"][0]["duration"] = "4.9"
            with self.subTest(change=change):
                with patch.object(ingest, "_probe", return_value=metadata):
                    with self.assertRaises(MediaError):
                        verify_normalized(self.output)
