"""Use real MP4 fixtures to verify unchanged bytes and complete decoding."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from features.media.ingest import MediaError
from features.output import validate as raw_module
from features.output.validate import validate_raw_video


class RawVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        root = Path(cls.folder.name)
        cls.path = root / "raw.mp4"
        cls.audio_path = root / "with-audio.mp4"
        cls.mov_path = root / "not-mp4.mov"
        cls.command(["-f", "lavfi", "-i", "color=blue:s=64x36:r=30:d=5",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.path)])
        cls.command(["-f", "lavfi", "-i", "color=red:s=64x36:r=24:d=5",
                     "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
                     "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(cls.audio_path)])
        cls.command(["-i", str(cls.path), "-c", "copy", str(cls.mov_path)])
        cls.contents = cls.path.read_bytes()
        cls.metadata = raw_module._probe_raw(cls.path)
        cls.frame_metadata = raw_module._probe_raw(cls.path, frames=True)
        packets = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets",
                                  "-show_entries", "packet=pos,size,pts_time", "-of", "json", str(cls.audio_path)],
                                 capture_output=True, check=True, timeout=30)
        packet = next(item for item in json.loads(packets.stdout)["packets"] if float(item["pts_time"]) > 3)
        damaged = bytearray(cls.audio_path.read_bytes())
        offset, size = int(packet["pos"]), int(packet["size"])
        damaged[offset:offset + size] = b"\x00" * size
        cls.damaged_audio = bytes(damaged)
        cls.damaged_audio_path = root / "damaged-audio.mp4"
        cls.damaged_audio_path.write_bytes(cls.damaged_audio)
        parts = []
        for index, size in enumerate(["64x36", "4200x8"]):
            part = root / f"resolution-{index}.h264"
            cls.command(["-f", "lavfi", "-i", f"color=red:s={size}:r=24:d=1",
                         "-c:v", "libx264", "-x264-params", "bframes=0", "-threads", "2",
                         "-pix_fmt", "yuv420p", "-f", "h264", str(part)])
            parts.append(part.read_bytes())
        joined = root / "joined.h264"
        joined.write_bytes(b"".join(parts))
        cls.changed_size_path = root / "changed-resolution.mp4"
        cls.command(["-r", "24", "-i", str(joined), "-c", "copy", str(cls.changed_size_path)])

    @staticmethod
    def command(arguments):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", *arguments],
                       capture_output=True, check=True, timeout=30)

    def test_bytes_and_actual_150_frame_timing_are_preserved(self):
        raw = validate_raw_video(self.contents)
        self.assertEqual(raw.video_bytes, self.contents)
        self.assertEqual(raw.sha256, sha256(self.contents).hexdigest())
        self.assertEqual((raw.codec, raw.width, raw.height, raw.frame_rate, raw.audio_present),
                         ("h264", 64, 36, "30/1", False))
        self.assertEqual(raw.duration_seconds, 5.0)
        self.assertEqual(len(raw.frame_seconds), 150)
        for index, seconds in enumerate(raw.frame_seconds):
            self.assertAlmostEqual(seconds, index / 30, places=5)
        self.assertEqual(self.path.read_bytes(), self.contents)

    def test_valid_audio_is_retained_in_the_original_bytes(self):
        contents = self.audio_path.read_bytes()
        raw = validate_raw_video(contents)
        self.assertTrue(raw.audio_present)
        self.assertEqual(raw.video_bytes, contents)
        self.assertEqual(raw.frame_rate, "24/1")
        self.assertEqual(len(raw.frame_seconds), 120)

    def test_corrupt_and_non_mp4_content_are_rejected_with_safe_errors(self):
        for contents in [b"", "not bytes", b"private fixture detail", self.contents[:100], self.mov_path.read_bytes()]:
            with self.subTest(size=len(contents)):
                with self.assertRaises(MediaError) as caught:
                    validate_raw_video(contents)
                self.assertNotIn("private fixture detail", str(caught.exception))

    def test_byte_limit_blocks_media_tools(self):
        with patch.object(raw_module, "MAX_RAW_BYTES", 1), patch.object(raw_module, "_probe_raw") as probe:
            with self.assertRaises(MediaError):
                validate_raw_video(b"12")
            probe.assert_not_called()

    def test_metadata_limits_block_full_decode(self):
        variants = []
        for key, value in [("width", 4097), ("height", 4097), ("avg_frame_rate", "121/1"), ("nb_frames", "2401")]:
            data = deepcopy(self.metadata)
            data["streams"][0][key] = value
            variants.append(data)
        data = deepcopy(self.metadata)
        data["streams"][0].update(width=4096, height=4096)
        variants.append(data)
        data = deepcopy(self.metadata)
        data["format"]["duration"] = "21"
        variants.append(data)
        for channels, rate in [(9, "48000"), (2, "192001")]:
            data = deepcopy(self.metadata)
            data["streams"].append({"codec_type": "audio", "channels": channels, "sample_rate": rate})
            variants.append(data)
        for data in variants:
            with self.subTest(metadata=data), patch.object(raw_module, "_probe_raw", return_value=data):
                with patch.object(raw_module, "_run") as run:
                    with self.assertRaises(MediaError):
                        validate_raw_video(self.contents)
                    run.assert_not_called()

    def test_full_decoder_failure_cannot_be_ignored(self):
        original = raw_module._run
        decodes = []

        def fail_decode(tool, arguments):
            if tool == "ffmpeg":
                decodes.append(arguments)
                raise MediaError("fixture full-decode failure")
            return original(tool, arguments)

        with patch.object(raw_module, "_run", side_effect=fail_decode):
            with self.assertRaisesRegex(MediaError, "full-decode failure"):
                validate_raw_video(self.contents)
        self.assertEqual(len(decodes), 1)

    def test_invalid_or_nonmonotonic_timestamps_are_rejected(self):
        original = raw_module._probe_raw
        for value in ["NaN", "-1", "6", "0"]:
            data = deepcopy(self.frame_metadata)
            data["frames"][50]["best_effort_timestamp_time"] = value

            def bad_frames(path, *, frames=False):
                return data if frames else original(path)

            with self.subTest(value=value), patch.object(raw_module, "_probe_raw", side_effect=bad_frames):
                with self.assertRaises(MediaError):
                    validate_raw_video(self.contents)

    def test_all_declared_frames_must_decode(self):
        original = raw_module._probe_raw
        data = deepcopy(self.metadata)
        data["streams"][0]["nb_frames"] = "149"

        def bad_count(path, *, frames=False):
            return original(path, frames=True) if frames else data

        with patch.object(raw_module, "_probe_raw", side_effect=bad_count):
            with self.assertRaisesRegex(MediaError, "all declared frames"):
                validate_raw_video(self.contents)

    def test_corrupt_late_audio_is_rejected_even_when_video_and_metadata_pass(self):
        # These independent probes pass: only full audio decoding finds the damage.
        self.assertEqual(raw_module._probe_raw(self.damaged_audio_path)["format"]["duration"], "5.000000")
        self.assertEqual(len(raw_module._probe_raw(self.damaged_audio_path, frames=True)["frames"]), 120)
        with self.assertRaises(MediaError):
            validate_raw_video(self.damaged_audio)

    def test_later_frames_cannot_exceed_the_declared_dimensions(self):
        metadata = raw_module._probe_raw(self.changed_size_path)
        self.assertEqual((metadata["streams"][0]["width"], metadata["streams"][0]["height"]), (64, 36))
        frames = raw_module._probe_raw(self.changed_size_path, frames=True)["frames"]
        self.assertEqual({(frame["width"], frame["height"]) for frame in frames}, {(64, 36), (4200, 8)})
        with self.assertRaisesRegex(MediaError, "dimensions changed"):
            validate_raw_video(self.changed_size_path.read_bytes())
