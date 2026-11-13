"""Check fixed preview bytes and historical guides without publishing new work."""

import asyncio
from dataclasses import FrozenInstanceError, fields
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from PIL import Image, PngImagePlugin

from app import prepare, preparation
from core import budget
from features.asset import ingest as asset_validation
from features.guidance import keyframe, video as guidance
from features.inference import submission
from features.media import ingest as media_validation
from features.output import extract
from features.placement.math import Placement


class PreparationPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        root = Path(cls.fixture.name)
        cls.original_source = root / "distinct.mp4"
        cls.original_asset = root / "asset.png"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=24:d=5",
            "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(cls.original_source),
        ], capture_output=True, check=True, timeout=30)
        image = Image.new("RGBA", (18, 14), (230, 90, 35, 255))
        image.putpixel((0, 0), (0, 0, 0, 0))
        image.putpixel((8, 7), (30, 210, 230, 127))
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("preview-fixture", "Retain this harmless original PNG metadata.")
        image.save(cls.original_asset, pnginfo=metadata)
        cls.video = media_validation.normalize_video(cls.original_source)
        cls.asset = asset_validation.load_asset(cls.original_asset, non_human=True)
        cls.first_placement = Placement(0.37, 0.63, 0.21)
        cls.second_placement = Placement(0.72, 0.31, 0.18)
        cls.frame_zero = extract.extract_frame_png(cls.video.video_bytes, 0, width=1280, height=720, frame_count=120)
        cls.frame_one = extract.extract_frame_png(cls.video.video_bytes, 1, width=1280, height=720, frame_count=120)
        if cls.frame_zero == cls.frame_one:
            raise AssertionError("The fixture must distinguish decoded frame zero from frame one.")
        cls.expected_placement = {}
        cls.guides = {}
        for placement in (cls.first_placement, cls.second_placement):
            with Image.open(BytesIO(cls.frame_zero)) as frame:
                expected = keyframe.compose_frame(frame, cls.asset, placement)
                encoded = BytesIO()
                expected.save(encoded, format="PNG")
                cls.expected_placement[placement] = (expected.tobytes(), encoded.getvalue())
            cls.guides[placement] = guidance.compile_guidance(cls.video, cls.asset, placement)

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.state = self.root / "private"
        self.state.mkdir(mode=0o700)
        self.outputs = self.root / "outputs"

    def service(self):
        ledger = budget.BudgetLedger(self.state / "budget.sqlite", clock=lambda: 1000.0)
        owner = submission.SubmissionCoordinator(ledger)
        self.addCleanup(owner.close)
        service = preparation.PreparationService(self.state, self.outputs, owner, clock=lambda: 1000.0)
        self.addCleanup(service.close)
        return service

    @staticmethod
    async def chunks(contents):
        yield contents[:31]
        yield contents[31:]

    def import_input(self, service, run, kind):
        async def operation():
            async with service.media_operation(run.run_id, expected_revision=run.revision) as media:
                if kind == "source":
                    return await media.import_source(self.chunks(self.video.original_bytes), source_suffix=".mp4",
                                                     clean_source=True, source_kind="synthetic_fixture")
                return await media.import_asset(self.chunks(self.asset.original_bytes), non_human=True)

        with patch.object(preparation, "normalize_video", return_value=self.video), \
                patch.object(preparation, "load_asset", return_value=self.asset):
            return asyncio.run(operation())

    def ready(self, service):
        run = self.import_input(service, service.create_run(), "source")
        run = self.import_input(service, run, "asset")
        return service.set_placement(run.run_id, expected_revision=run.revision, placement=self.first_placement)

    def build(self, service, run):
        async def operation():
            async with service.media_operation(run.run_id, expected_revision=run.revision) as media:
                return await media.build_guides()

        with patch.object(prepare, "compile_guidance", return_value=self.guides[run.placement]):
            return asyncio.run(operation())

    @staticmethod
    def preview(service, run, role, **changes):
        arguments = {"expected_revision": run.revision, "role": role} | changes
        return asyncio.run(service.preview(run.run_id, **arguments))

    @staticmethod
    def historical(service, run, attempt_id, role):
        return asyncio.run(service.prepared_media(run.run_id, attempt_id, role=role))

    def assert_media(self, result, contents, media_type):
        self.assertEqual({field.name for field in fields(result)}, {"contents", "media_type", "sha256"})
        self.assertEqual((result.contents, result.media_type, result.sha256),
                         (contents, media_type, sha256(contents).hexdigest()))
        with self.assertRaises(FrozenInstanceError):
            result.sha256 = "a" * 64

    def assert_code(self, code, operation):
        with self.assertRaises(preparation.ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)

    def files(self):
        return {path.relative_to(self.outputs).as_posix(): path.read_bytes()
                for path in self.outputs.rglob("*") if path.is_file()}

    def database_rows(self):
        records = {}
        for name in ("preparation.sqlite", "budget.sqlite"):
            with sqlite3.connect(self.state / name) as connection:
                records[name] = tuple(connection.iterdump())
        return records

    def test_source_is_exact_frame_zero_with_no_revalidation_publication_or_state_change(self):
        service = self.service()
        run = self.ready(service)
        before_files, before_rows = self.files(), self.database_rows()
        service.clock = Mock(side_effect=AssertionError("A preview cannot restart the clock."))
        original_open = Path.open

        def open_file(path, *arguments, **keywords):
            if path in (self.original_source, self.original_asset):
                raise AssertionError("A preview reopened an original upload path.")
            return original_open(path, *arguments, **keywords)

        with patch.object(preparation, "normalize_video", side_effect=AssertionError("No normalization.")) as normalize, \
                patch.object(preparation, "load_asset", side_effect=AssertionError("No validation.")) as validate, \
                patch.object(preparation, "prepare_validated_attempt", side_effect=AssertionError("No attempt.")) as publish, \
                patch.object(prepare, "compile_guidance", side_effect=AssertionError("No guide compilation.")) as compile_guide, \
                patch.object(Path, "open", open_file):
            result = self.preview(service, run, "source")
        self.assert_media(result, self.frame_zero, "image/png")
        self.assertNotEqual(result.contents, self.frame_one)
        for call in (normalize, validate, publish, compile_guide, service.clock):
            call.assert_not_called()
        self.assertEqual(service.get_run(run.run_id), run)
        self.assertEqual(self.files(), before_files)
        self.assertEqual(self.database_rows(), before_rows)

    def test_asset_preview_preserves_original_alpha_and_ancillary_png_bytes(self):
        service = self.service()
        run = self.ready(service)
        before = self.files()
        result = self.preview(service, run, "asset")
        self.assert_media(result, self.asset.original_bytes, "image/png")
        with Image.open(BytesIO(result.contents)) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getpixel((0, 0))[3], 0)
            self.assertEqual(image.getpixel((8, 7))[3], 127)
            self.assertEqual(image.info["preview-fixture"], "Retain this harmless original PNG metadata.")
        self.assertEqual(self.files(), before)

    def test_placement_pixels_use_the_saved_nondefault_placement(self):
        service = self.service()
        run = self.ready(service)
        before = self.files()
        result = self.preview(service, run, "placement")
        pixels, expected_png = self.expected_placement[self.first_placement]
        self.assert_media(result, expected_png, "image/png")
        with Image.open(BytesIO(result.contents)) as image:
            self.assertEqual((image.mode, image.size), ("RGB", (1280, 720)))
            self.assertEqual(image.tobytes(), pixels)
        self.assertEqual(service.get_run(run.run_id), run)
        self.assertEqual(self.files(), before)

    def test_placement_preview_uses_captured_revision_when_a_new_edit_wins(self):
        service = self.service()
        run = self.ready(service)
        before = self.files()
        entered, release = Event(), Event()
        real_worker = service._preview_media

        def paused_worker(snapshot, role):
            entered.set()
            if not release.wait(5):
                raise ValueError("The test did not release its preview worker.")
            return real_worker(snapshot, role)

        async def scenario():
            task = asyncio.create_task(service.preview(
                run.run_id, expected_revision=run.revision, role="placement"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5), "Preview worker did not start.")
                changed = service.set_placement(run.run_id, expected_revision=run.revision,
                                                 placement=self.second_placement)
            finally:
                release.set()
                try:
                    result = await asyncio.wait_for(task, timeout=15)
                except BaseException:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise
            return result, changed

        with patch.object(service, "_preview_media", side_effect=paused_worker):
            result, changed = asyncio.run(scenario())
        self.assert_media(result, self.expected_placement[self.first_placement][1], "image/png")
        self.assertEqual(changed.placement, self.second_placement)
        self.assertEqual(changed.revision, run.revision + 1)
        self.assertEqual(service.get_run(run.run_id), changed)
        self.assertEqual(self.files(), before)

    def test_missing_inputs_bad_roles_and_stale_revisions_have_exact_codes(self):
        service = self.service()
        run = service.create_run()
        with patch.object(preparation, "extract_frame_png", side_effect=ValueError("Unexpected extraction.")) as frame:
            for role in ("source", "asset", "placement"):
                self.assert_code("INPUT_REQUIRED", lambda: self.preview(service, run, role))
            run = self.import_input(service, run, "source")
            for role in ("asset", "placement"):
                self.assert_code("INPUT_REQUIRED", lambda: self.preview(service, run, role))
            run = self.import_input(service, run, "asset")
            self.assert_code("INPUT_REQUIRED", lambda: self.preview(service, run, "placement"))
            for role in (None, True, [], "", "Source", "guide-b", "../asset"):
                with self.subTest(role=role):
                    self.assert_code("INVALID_INPUT", lambda: self.preview(service, run, role))
            self.assert_code("REVISION_CONFLICT", lambda: self.preview(service, run, "source", expected_revision=0))
            self.assert_code("INVALID_INPUT", lambda: self.preview(service, run, "source", expected_revision=True))
            frame.assert_not_called()
        service.coordinator.close()
        self.assert_code("SERVICE_CLOSED", lambda: self.preview(service, run, "asset"))

    def test_changed_cache_records_or_selected_bytes_fail_before_rendering(self):
        service = self.service()
        run = self.ready(service)
        root = self.outputs / run.run_id
        record_path = root / run.source.record.path
        original_record = record_path.read_bytes()
        record_path.write_bytes(original_record[:-1] + b" ")
        with patch.object(preparation, "extract_frame_png", return_value=self.frame_zero) as frame:
            self.assert_code("STATE_UNAVAILABLE", lambda: self.preview(service, run, "source"))
            frame.assert_not_called()
        record_path.write_bytes(original_record)
        for record, role in ((run.source.normalized, "source"), (run.asset.original, "asset")):
            with self.subTest(role=role):
                path = root / record.path
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                before = self.files()
                with patch.object(preparation, "extract_frame_png", return_value=self.frame_zero) as frame, \
                        patch.object(preparation, "compose_frame", side_effect=ValueError("Unexpected composition.")) as compose:
                    self.assert_code("STATE_UNAVAILABLE", lambda: self.preview(service, run, role))
                    frame.assert_not_called()
                    compose.assert_not_called()
                self.assertEqual(self.files(), before)
                path.write_bytes(original)

    def test_source_input_cap_is_checked_before_byte_reconstruction(self):
        service = self.service()
        run = self.ready(service)
        with patch.object(preparation, "MAX_PREVIEW_SOURCE_BYTES", len(self.video.video_bytes)):
            self.assert_media(self.preview(service, run, "source"), self.frame_zero, "image/png")
        real_load = preparation.preparation_media.load_source
        byte_reads = []

        def load(*arguments, **keywords):
            if keywords.get("with_bytes"):
                byte_reads.append(True)
            return real_load(*arguments, **keywords)

        with patch.object(preparation, "MAX_PREVIEW_SOURCE_BYTES", len(self.video.video_bytes) - 1), \
                patch.object(preparation.preparation_media, "load_source", side_effect=load), \
                patch.object(preparation, "extract_frame_png", return_value=self.frame_zero) as frame:
            for role in ("source", "placement"):
                self.assert_code("STATE_UNAVAILABLE", lambda: self.preview(service, run, role))
            self.assertEqual(byte_reads, [])
            frame.assert_not_called()

    def test_preview_and_historical_response_caps_allow_equality_and_reject_one_byte_more(self):
        service = self.service()
        run = self.ready(service)
        expected = {"source": self.frame_zero, "asset": self.asset.original_bytes,
                    "placement": self.expected_placement[self.first_placement][1]}
        for role, contents in expected.items():
            with self.subTest(role=role):
                with patch.object(preparation, "MAX_PREVIEW_PNG_BYTES", len(contents)):
                    self.assert_media(self.preview(service, run, role), contents, "image/png")
                with patch.object(preparation, "MAX_PREVIEW_PNG_BYTES", len(contents) - 1):
                    self.assert_code("STATE_UNAVAILABLE", lambda: self.preview(service, run, role))
        completed = self.build(service, run)
        attempt = completed.attempts[0].attempt_id
        for role, limit, contents, mime in (
            ("keyframe-a", "MAX_PREVIEW_PNG_BYTES", self.guides[self.first_placement].keyframe_png, "image/png"),
            ("guide-b", "MAX_PREVIEW_VIDEO_BYTES", self.guides[self.first_placement].guide_mp4, "video/mp4"),
        ):
            with self.subTest(role=role):
                with patch.object(preparation, limit, len(contents)):
                    self.assert_media(self.historical(service, completed, attempt, role), contents, mime)
                with patch.object(preparation, limit, len(contents) - 1), \
                        patch.object(preparation, "read_verified", side_effect=ValueError("Unexpected final payload read.")) as read:
                    self.assert_code("STATE_UNAVAILABLE", lambda: self.historical(service, completed, attempt, role))
                    read.assert_not_called()

    def test_tool_errors_and_empty_or_wrong_type_outputs_are_safe_and_publish_nothing(self):
        service = self.service()
        run = self.ready(service)
        before_files, before_rows = self.files(), self.database_rows()
        for output in (b"", bytearray(self.frame_zero), None):
            with self.subTest(output_type=type(output).__name__):
                with patch.object(preparation, "extract_frame_png", return_value=output):
                    self.assert_code("STATE_UNAVAILABLE", lambda: self.preview(service, run, "source"))
        with patch.object(preparation, "extract_frame_png", side_effect=ValueError("private-media-path")):
            self.assert_code("STATE_UNAVAILABLE", lambda: self.preview(service, run, "source"))
        with patch.object(preparation, "compose_frame", side_effect=ValueError("private-asset-path")):
            self.assert_code("STATE_UNAVAILABLE", lambda: self.preview(service, run, "placement"))
        self.assertEqual(self.files(), before_files)
        self.assertEqual(self.database_rows(), before_rows)

    def test_old_prepared_media_survives_new_edits_and_corrupt_current_draft(self):
        service = self.service()
        first = self.build(service, self.ready(service))
        first_attempt = first.attempts[0].attempt_id
        changed = service.set_placement(first.run_id, expected_revision=first.revision, placement=self.second_placement)
        second = self.build(service, changed)
        self.assertNotEqual(first.prepared, second.prepared)
        draft = service.set_placement(first.run_id, expected_revision=second.revision, placement=self.first_placement)
        path = self.outputs / draft.run_id / draft.asset.record.path
        original = path.read_bytes()
        path.write_bytes(original[:-1] + b" ")
        self.assert_code("STATE_UNAVAILABLE", lambda: service.get_run(draft.run_id))
        before_files, before_rows = self.files(), self.database_rows()
        service.clock = Mock(side_effect=AssertionError("History must not need a fresh clock."))
        with patch.object(service, "_selection", side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as selection, \
                patch.object(service, "get_run", side_effect=preparation.ServiceError("STATE_UNAVAILABLE")) as current:
            for role, contents, mime in (
                ("keyframe-a", self.guides[self.first_placement].keyframe_png, "image/png"),
                ("guide-b", self.guides[self.first_placement].guide_mp4, "video/mp4"),
            ):
                self.assert_media(self.historical(service, draft, first_attempt, role), contents, mime)
            selection.assert_not_called()
            current.assert_not_called()
        service.clock.assert_not_called()
        self.assertEqual(self.files(), before_files)
        self.assertEqual(self.database_rows(), before_rows)

    def test_unknown_foreign_or_unfinished_attempts_and_bad_roles_have_exact_codes(self):
        service = self.service()
        completed = self.build(service, self.ready(service))
        attempt = completed.attempts[0].attempt_id
        other = service.create_run()
        self.assert_code("ATTEMPT_NOT_FOUND", lambda: self.historical(service, other, attempt, "guide-b"))
        unknown = "att-" + "f" * 32
        orphan = self.outputs / completed.run_id / "attempts" / unknown
        orphan.mkdir()
        (orphan / "manifest.json").write_bytes(b"{}")
        self.assert_code("ATTEMPT_NOT_FOUND", lambda: self.historical(service, completed, unknown, "keyframe-a"))
        for role in (None, True, [], "", "source", "guide-B", "../guide-b"):
            with self.subTest(role=role):
                self.assert_code("INVALID_INPUT", lambda: self.historical(service, completed, attempt, role))
        self.assert_code("INVALID_INPUT", lambda: self.historical(service, completed, "../attempt", "guide-b"))
        with sqlite3.connect(self.state / "preparation.sqlite") as connection:
            row = connection.execute("SELECT source_id,asset_id,placement FROM preparation_runs WHERE run_id=?",
                                     (completed.run_id,)).fetchone()
            for index, status in enumerate(("BUILDING", "FAILED", "INTERRUPTED"), 1):
                pending = "att-" + str(index) * 32
                error = {"BUILDING": None, "FAILED": "WORK_FAILED", "INTERRUPTED": "WORK_INTERRUPTED"}[status]
                connection.execute("INSERT INTO preparation_attempts VALUES (?,?,?,?,?,?,?,?,NULL,?)",
                    (pending, completed.run_id, completed.revision, *row, guidance.DEFAULT_PROMPT, status, error))
        for index in (1, 2, 3):
            pending = "att-" + str(index) * 32
            self.assert_code("INPUT_REQUIRED", lambda: self.historical(service, completed, pending, "guide-b"))

    def test_changed_manifest_or_frozen_payload_cannot_be_returned_as_history(self):
        service = self.service()
        completed = self.build(service, self.ready(service))
        attempt = completed.attempts[0].attempt_id
        root = self.outputs / completed.run_id
        manifest_path = root / completed.prepared.path
        manifest_bytes = manifest_path.read_bytes()
        document = json.loads(manifest_bytes)
        manifest_path.write_bytes(manifest_bytes[:-1] + b" ")
        self.assert_code("STATE_UNAVAILABLE", lambda: self.historical(service, completed, attempt, "keyframe-a"))
        manifest_path.write_bytes(manifest_bytes)
        for role, key in (("keyframe-a", "frozen/keyframe_A.png"), ("guide-b", "frozen/guide_B.mp4")):
            with self.subTest(role=role):
                path = root / document["artifacts"][key]["path"]
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                self.assert_code("STATE_UNAVAILABLE", lambda: self.historical(service, completed, attempt, role))
                path.write_bytes(original)
        manifest_path.unlink()
        self.assert_code("STATE_UNAVAILABLE", lambda: self.historical(service, completed, attempt, "guide-b"))

    def test_historical_response_uses_verified_bytes_without_reopening_the_payload(self):
        service = self.service()
        completed = self.build(service, self.ready(service))
        attempt = completed.attempts[0].attempt_id
        verified_read = preparation.read_verified
        changed = []

        def read_then_change(root, record):
            contents = verified_read(root, record)
            path = root / record.path
            path.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
            changed.append(record.path)
            return contents

        with patch.object(preparation, "read_verified", side_effect=read_then_change) as read:
            result = self.historical(service, completed, attempt, "keyframe-a")
        read.assert_called_once()
        self.assertEqual(changed, [f"attempts/{attempt}/frozen/keyframe_A.png"])
        self.assert_media(result, self.guides[self.first_placement].keyframe_png, "image/png")
