"""Keep native raw timestamps while aligning review against container elapsed time."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.budget import BudgetLedger
from core.storage import StorageError, StoredArtifact, read_verified
from features.inference.submission import SubmissionCoordinator
from features.media.ingest import MediaError
from features.output import align, complete, review as output_review, review_contract, validate as raw_module, view
from features.review import bind, binding_contract, review_records
import test_archive_review as archive_fixtures
import test_scorecard as rubric_fixtures


RUN, ATTEMPT = archive_fixtures.RUN, archive_fixtures.ATTEMPT_A
SOURCE_TIMES = tuple(index / 24 for index in range(120))
BASE_COMPLETE_MODULE = complete
BASE_COMPLETER = complete.OutputCompleter
BASE_PUBLISH = complete._publish_or_verify


class RawOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        root = Path(folder.name)
        cls.shifted_path, cls.audio_path = root / "shifted.mp4", root / "audio-first.mp4"
        cls.encode(["-f", "lavfi", "-i", "color=blue:s=64x36:r=4:d=1", "-c:v", "libx264", "-bf", "0",
                    "-output_ts_offset", "1.25", str(cls.shifted_path)])
        cls.encode(["-f", "lavfi", "-i", "color=red:s=64x36:r=30:d=1", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=2", "-filter:v", "setpts=PTS+1/TB", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-bf", "0", "-c:a", "aac", str(cls.audio_path)])
        cls.shifted_bytes, cls.audio_bytes = cls.shifted_path.read_bytes(), cls.audio_path.read_bytes()
        # Probe fixtures independently of the validator's requested fields.
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                                 str(cls.shifted_path)], capture_output=True, check=True, timeout=30)
        cls.shifted_metadata = json.loads(result.stdout)
        cls.shifted_frames = {"frames": [{"width": 64, "height": 36, "best_effort_timestamp_time": str(value)}
                                       for value in (1.25, 1.5, 1.75, 2.0)]}
        fixture = type("OriginArchiveFixture", (archive_fixtures.ArchiveReviewTests,), {})
        fixture.setUpClass()
        cls.addClassCleanup(fixture.doClassCleanups)
        cls.archive_fixture = fixture
        node = fixture
        while node is not None:
            if hasattr(node, "base"):
                cls.base = node.base
            if hasattr(node, "completed"):
                cls.media_fixture = node
            node = getattr(node, "fixture", None)

    @staticmethod
    def encode(arguments):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", *arguments],
                       capture_output=True, check=True, timeout=30)

    def setUp(self):
        self.archived = self.archive_fixture()
        self.archived.setUp()
        self.addCleanup(self.archived.doCleanups)

    def shifted_candidate(self, *, origin=1.25, change_review=None, change_metadata=None):
        def metadata(data):
            data["timelineOriginSeconds"] = origin
            data["frameSeconds"] = [value + origin for value in data["frameSeconds"]]
            if change_metadata is not None:
                change_metadata(data)

        def review(data):
            data["rawTimelineOriginSeconds"] = origin
            for sample in data["samples"]:
                sample["raw"]["ptsSeconds"] += origin
            if change_review is not None:
                change_review(data)

        return self.archived.candidate(change_metadata=metadata, change_review=review)

    def verify_archive(self, candidate):
        return review_records.verify_export_review(self.archived.root, candidate.archive, candidate.inventory,
                                                   run_id=RUN, attempt_id=ATTEMPT)

    def local_copy(self, candidate=None):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        shutil.copytree(self.base, root, dirs_exist_ok=True)
        ledger = BudgetLedger(root / "budget.sqlite", clock=lambda: 5000.0)
        owner = SubmissionCoordinator(ledger)
        self.addCleanup(owner.close)
        if candidate is not None:
            for name, contents in candidate.members.items():
                if name != "inventory.json":
                    path = root / RUN / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(contents)
            with ledger._connection() as connection:
                connection.execute("UPDATE completed_outputs SET metadata_sha256=?, metadata_size=? WHERE attempt_id=?",
                                   (candidate.metadata.sha256, candidate.metadata.size, ATTEMPT))
        return root, owner

    @staticmethod
    def card(*, start=None, end=None):
        data = rubric_fixtures.card(attemptId=ATTEMPT, reviewedAtUtc="1970-01-01T01:23:20Z")
        if start is not None:
            data["hardFailures"] = [rubric_fixtures.event(start=start, end=end)]
        return json.dumps(data, indent=2).encode()

    def bind_local(self, root, owner, candidate, contents=None):
        return bind.bind_scorecard(owner, root, candidate.review, self.card() if contents is None else contents,
                                   run_id=RUN, attempt_id=ATTEMPT)

    def interrupted_legacy_completion(self, contents):
        """Use real durable publication and SQL rollback, with old serialization only."""
        root, owner = self.local_copy()
        output = root / RUN / f"attempts/{ATTEMPT}/output"
        shutil.rmtree(output)
        with owner.ledger._connection() as connection:
            connection.execute("DELETE FROM completed_outputs WHERE attempt_id=?", (ATTEMPT,))
            connection.execute("UPDATE submission_jobs SET state='QUEUED' WHERE attempt_id=?", (ATTEMPT,))
            connection.execute("""CREATE TRIGGER reject_origin_completion BEFORE UPDATE OF state
                ON submission_jobs WHEN NEW.state='SUCCEEDED'
                BEGIN SELECT RAISE(ABORT, 'fixture completion transaction failure'); END""")

        def old_metadata(run_root, path, details):
            details = dict(details)
            details.pop("timelineOriginSeconds", None)
            body = (json.dumps(details, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            return BASE_PUBLISH(run_root, path, body)

        with patch.object(BASE_COMPLETE_MODULE, "_publish_metadata", side_effect=old_metadata):
            with self.assertRaisesRegex(sqlite3.IntegrityError, "fixture completion transaction failure"):
                BASE_COMPLETER(owner, root).complete(ATTEMPT, lambda *_: contents)
        self.assertEqual(owner.get(ATTEMPT).state, "QUEUED")
        with owner.ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM completed_outputs WHERE attempt_id=?",
                                                (ATTEMPT,)).fetchone()[0], 0)
            connection.execute("DROP TRIGGER reject_origin_completion")
        old_bytes = (output / "metadata.json").read_bytes()
        self.assertNotIn("timelineOriginSeconds", json.loads(old_bytes))
        expected = StoredArtifact(f"attempts/{ATTEMPT}/output/metadata.json", sha256(old_bytes).hexdigest(), len(old_bytes))
        owner.close()
        restarted = SubmissionCoordinator(BudgetLedger(root / "budget.sqlite", clock=lambda: 5001.0))
        self.addCleanup(restarted.close)
        return root, restarted, output, expected

    def test_real_shifted_mp4_preserves_native_bytes_pts_and_declared_origin(self):
        video = raw_module.validate_raw_video(self.shifted_bytes)
        self.assertEqual((video.timeline_origin_seconds, video.duration_seconds), (1.25, 1.0))
        self.assertEqual(video.frame_seconds, (1.25, 1.5, 1.75, 2.0))
        self.assertEqual(video.video_bytes, self.shifted_bytes)
        self.assertEqual(video.sha256, sha256(self.shifted_bytes).hexdigest())
        self.assertEqual(self.shifted_path.read_bytes(), self.shifted_bytes)
        with self.assertRaises(FrozenInstanceError):
            video.timeline_origin_seconds = 0
        legacy = raw_module.RawVideo(b"x", "a" * 64, "h264", 64, 36, 1.0, "4/1", (0.0,), False)
        self.assertEqual(legacy.timeline_origin_seconds, 0.0)

    def test_audio_before_video_keeps_container_origin_zero_and_initial_gap(self):
        video = raw_module.validate_raw_video(self.audio_bytes)
        self.assertEqual(video.timeline_origin_seconds, 0.0)
        self.assertTrue(video.audio_present)
        self.assertEqual(video.frame_seconds[0], 1.0)
        matches = align.plan_review_frames(SOURCE_TIMES, video.frame_seconds,
            raw_duration_seconds=video.duration_seconds, raw_timeline_origin_seconds=video.timeline_origin_seconds)
        self.assertEqual((matches[0].raw_seconds, matches[0].offset_seconds, matches[0].raw_span_position),
                         (1.0, 1.0, "before_first"))
        self.assertEqual(video.video_bytes, self.audio_bytes)

    def test_raw_probe_requests_container_start_time_without_stream_origin_inference(self):
        calls = []

        def run(tool, arguments):
            calls.append((tool, arguments))
            return b"{}"
        with patch.object(raw_module, "_run", side_effect=run):
            self.assertEqual(raw_module._probe_raw(Path("fixture.mp4")), {})
        entries = calls[0][1][calls[0][1].index("-show_entries") + 1]
        self.assertIn("start_time", next(part for part in entries.split(":") if part.startswith("format=")))
        self.assertEqual(calls[0][0], "ffprobe")

    def test_missing_or_malformed_fresh_origin_blocks_full_decode(self):
        for value in (None, True, -0.01, "-0.01", "NaN", "Infinity", "private-origin-sentinel", []):
            metadata = deepcopy(self.shifted_metadata)
            if value is None:
                metadata["format"].pop("start_time", None)
            else:
                metadata["format"]["start_time"] = value
            with self.subTest(origin=value), patch.object(raw_module, "_probe_raw", return_value=metadata), \
                    patch.object(raw_module, "_run") as decoder:
                with self.assertRaises(MediaError) as caught:
                    raw_module.validate_raw_video(self.shifted_bytes)
                self.assertNotIn("private-origin-sentinel", str(caught.exception))
                decoder.assert_not_called()

    def test_raw_validation_checks_elapsed_bounds_and_never_rebases_returned_pts(self):
        for first, last, valid in ((1.25, 2.25, True), (1.249, 2.0, False), (1.25, 2.251, False)):
            frames = deepcopy(self.shifted_frames)
            frames["frames"][0]["best_effort_timestamp_time"] = str(first)
            frames["frames"][-1]["best_effort_timestamp_time"] = str(last)
            with self.subTest(first=first, last=last), \
                    patch.object(raw_module, "_probe_raw", side_effect=[self.shifted_metadata, frames]), \
                    patch.object(raw_module, "_run", return_value=b""):
                if valid:
                    video = raw_module.validate_raw_video(self.shifted_bytes)
                    self.assertEqual(video.frame_seconds, (1.25, 1.5, 1.75, 2.25))
                else:
                    with self.assertRaises(MediaError):
                        raw_module.validate_raw_video(self.shifted_bytes)

    def test_alignment_uses_elapsed_nearest_ties_span_and_source_epsilon(self):
        source = (-0.000001,) + SOURCE_TIMES[1:]
        matches = align.plan_review_frames(source, (1.25, 1.75, 2.75, 3.75),
            raw_duration_seconds=3.0, raw_timeline_origin_seconds=1.25)
        self.assertEqual([match.raw_index for match in matches], [0, 1, 2, 3, 3])
        self.assertEqual([match.raw_seconds for match in matches], [1.25, 1.75, 2.75, 3.75, 3.75])
        self.assertEqual([match.offset_seconds for match in matches], [0.000001, -0.5, -0.5, -0.5, -1.5])
        self.assertEqual([match.raw_span_position for match in matches],
                         ["before_first", "within_span", "within_span", "after_last", "after_last"])
        self.assertEqual([match.within_declared_duration for match in matches], [False, True, True, False, False])

    def test_alignment_rejects_bad_origin_and_elapsed_out_of_bounds(self):
        for origin in (True, -1, float("nan"), float("inf"), "1.25", None):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                align.plan_review_frames(SOURCE_TIMES, (1.25, 1.5), raw_duration_seconds=1,
                                         raw_timeline_origin_seconds=origin)
        for times in ((1.24, 1.5), (1.25, 2.251)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                align.plan_review_frames(SOURCE_TIMES, times, raw_duration_seconds=1,
                                         raw_timeline_origin_seconds=1.25)

    def test_completion_and_review_publish_origin_with_native_raw_bytes_and_samples(self):
        root, owner = self.local_copy()
        prefix = root / RUN / f"attempts/{ATTEMPT}"
        shutil.rmtree(prefix / "output")
        shutil.rmtree(prefix / "review")
        with owner.ledger._connection() as connection:
            connection.execute("DELETE FROM completed_outputs WHERE attempt_id=?", (ATTEMPT,))
            connection.execute("UPDATE submission_jobs SET state='QUEUED' WHERE attempt_id=?", (ATTEMPT,))
        completed = complete.OutputCompleter(owner, root).complete(ATTEMPT, lambda *_: self.shifted_bytes)
        metadata = json.loads(read_verified(root / RUN, completed.metadata))
        self.assertEqual((metadata["timelineOriginSeconds"], metadata["durationSeconds"], metadata["frameSeconds"]),
                         (1.25, 1.0, [1.25, 1.5, 1.75, 2.0]))
        self.assertEqual(read_verified(root / RUN, completed.raw), self.shifted_bytes)
        manifest = output_review.build_review_evidence(owner, root, self.media_fixture.manifests[ATTEMPT],
            self.media_fixture.checks[ATTEMPT].evidence, run_id=RUN, attempt_id=ATTEMPT)
        review = json.loads(read_verified(root / RUN, manifest))
        self.assertEqual(review["rawTimelineOriginSeconds"], 1.25)
        self.assertEqual([sample["raw"]["frameIndex"] for sample in review["samples"]], [0, 3, 3, 3, 3])
        self.assertEqual([sample["raw"]["ptsSeconds"] for sample in review["samples"]], [1.25, 2.0, 2.0, 2.0, 2.0])
        self.assertEqual([sample["raw"]["offsetSeconds"] for sample in review["samples"]], [0.0, -0.25, -1.25, -2.25, -3.25])
        self.assertEqual(review["samples"][0]["raw"]["spanPosition"], "within_span")
        self.assertFalse(review["samples"][1]["raw"]["withinDeclaredDuration"])
        self.assertEqual(review["rawOutput"], asdict(completed.raw))
        self.assertEqual((review["visualEffectDemonstrated"], review["automatedWorkflowComplete"]), ("NOT_RUN", "NOT_RUN"))
        # A fresh declared zero is emitted too; only absent legacy fields stay absent.
        shutil.rmtree(prefix / "output")
        shutil.rmtree(prefix / "review")
        with owner.ledger._connection() as connection:
            connection.execute("DELETE FROM completed_outputs WHERE attempt_id=?", (ATTEMPT,))
            connection.execute("UPDATE submission_jobs SET state='QUEUED' WHERE attempt_id=?", (ATTEMPT,))
        completed = complete.OutputCompleter(owner, root).complete(ATTEMPT, lambda *_: self.audio_bytes)
        metadata = json.loads(read_verified(root / RUN, completed.metadata))
        self.assertEqual(metadata["timelineOriginSeconds"], 0.0)
        self.assertEqual(read_verified(root / RUN, completed.raw), self.audio_bytes)
        manifest = output_review.build_review_evidence(owner, root, self.media_fixture.manifests[ATTEMPT],
            self.media_fixture.checks[ATTEMPT].evidence, run_id=RUN, attempt_id=ATTEMPT)
        review = json.loads(read_verified(root / RUN, manifest))
        self.assertEqual(review["rawTimelineOriginSeconds"], 0.0)
        self.assertEqual((review["samples"][0]["raw"]["ptsSeconds"], review["samples"][0]["raw"]["offsetSeconds"],
                          review["samples"][0]["raw"]["spanPosition"]), (1.0, 1.0, "before_first"))

    def test_legacy_missing_origin_keeps_zero_basis_and_exact_rebuilt_manifest_bytes(self):
        candidate = self.archived.candidate(
            change_metadata=lambda data: data.pop("timelineOriginSeconds", None),
            change_review=lambda data: data.pop("rawTimelineOriginSeconds", None))
        self.assertEqual(self.verify_archive(candidate).review, candidate.review)
        root, owner = self.local_copy(candidate)
        expected = (json.dumps(candidate.review_data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        (root / RUN / candidate.review.path).unlink()
        record = output_review.build_review_evidence(owner, root, self.media_fixture.manifests[ATTEMPT],
            self.media_fixture.checks[ATTEMPT].evidence, run_id=RUN, attempt_id=ATTEMPT)
        self.assertEqual(read_verified(root / RUN, record), expected)
        self.assertNotIn("rawTimelineOriginSeconds", json.loads(expected))
        candidate.review = record
        receipt = self.bind_local(root, owner, candidate)
        self.assertEqual(bind.read_scorecard_binding(owner, root, receipt, run_id=RUN, attempt_id=ATTEMPT).record, receipt)

    def test_shared_sample_contract_preserves_native_values_beyond_duration_and_legacy_zero(self):
        for origin in (1.25, 25.0):
            with self.subTest(origin=origin):
                candidate = self.shifted_candidate(origin=origin)
                times = review_contract._raw_times(candidate.metadata_data)
                self.assertEqual(times, tuple(candidate.metadata_data["frameSeconds"]))
                self.assertGreater(times[-1], candidate.metadata_data["durationSeconds"])
                self.assertIsNone(review_contract._check_raw_samples(candidate.review_data, candidate.metadata_data))
                records = review_contract._view_records(candidate.review_data, self.archived.prefix)
                self.assertIn(candidate.review_data["samples"][0]["raw"]["image"]["path"], records)
        legacy = self.archived.candidate(change_metadata=lambda data: data.pop("timelineOriginSeconds", None),
            change_review=lambda data: data.pop("rawTimelineOriginSeconds", None))
        self.assertIsNone(review_contract._check_raw_samples(legacy.review_data, legacy.metadata_data))

    def test_saved_origins_are_strict_numeric_and_metadata_frames_use_elapsed_bounds(self):
        candidate = self.shifted_candidate()
        for value in (True, -1.0, "1.25", None, float("nan"), float("inf")):
            with self.subTest(origin=value):
                metadata = deepcopy(candidate.metadata_data)
                metadata["timelineOriginSeconds"] = value
                with self.assertRaises(review_contract.ReviewContractError):
                    review_contract._raw_times(metadata)
                review = deepcopy(candidate.review_data)
                review["rawTimelineOriginSeconds"] = value
                with self.assertRaises(review_contract.ReviewContractError):
                    review_contract._view_records(review, self.archived.prefix)
        for native in (1.249, 6.251):
            metadata = deepcopy(candidate.metadata_data)
            metadata["frameSeconds"][0 if native < 1.25 else -1] = native
            with self.subTest(native=native), self.assertRaises(review_contract.ReviewContractError):
                review_contract._raw_times(metadata)

    def test_archive_origin_join_and_elapsed_labels_are_checked_after_coherent_rehash(self):
        valid = self.shifted_candidate()
        self.assertEqual(self.verify_archive(valid).review, valid.review)

        def mismatched_origin(data):
            data.pop("rawTimelineOriginSeconds")
            for sample in data["samples"]:
                sample["raw"]["offsetSeconds"] += 1.25

        for change in (mismatched_origin,
                lambda data: data["samples"][2]["raw"].update(spanPosition="before_first"),
                lambda data: data["samples"][2]["raw"].update(withinDeclaredDuration=False),
                lambda data: data["samples"][2]["raw"].update(offsetSeconds=1.25)):
            candidate = self.shifted_candidate(change_review=change)
            with self.subTest(change=change), self.assertRaises(review_records.ExportReviewError):
                self.verify_archive(candidate)

    def test_filesystem_binding_rechecks_metadata_origin_and_elapsed_sample_labels(self):
        for change in (lambda data: data.update(rawTimelineOriginSeconds=0.0),
                       lambda data: data["samples"][2]["raw"].update(spanPosition="after_last")):
            candidate = self.shifted_candidate(change_review=change)
            if candidate.review_data["rawTimelineOriginSeconds"] == 0:
                # Isolate the origin join: the view-only offset equation still holds.
                def zero_basis(data):
                    data["rawTimelineOriginSeconds"] = 0.0
                    for sample in data["samples"]:
                        sample["raw"]["offsetSeconds"] += 1.25
                candidate = self.shifted_candidate(change_review=zero_basis)
            root, owner = self.local_copy(candidate)
            with self.subTest(change=change), self.assertRaises(bind.ScorecardBindingError):
                self.bind_local(root, owner, candidate)

    def test_scorecard_endpoints_remain_exact_native_pts_and_never_elapsed_or_terminal(self):
        candidate = self.shifted_candidate()
        root, owner = self.local_copy(candidate)
        native = binding_contract._exact_native_times(candidate.members[candidate.metadata.path])
        self.assertIn(Decimal("1.25"), native)
        self.assertNotIn(Decimal("0"), native)
        contents = self.card(start=1.25, end=1.55)
        receipt = self.bind_local(root, owner, candidate, contents)
        verified = bind.read_scorecard_binding(owner, root, receipt, run_id=RUN, attempt_id=ATTEMPT)
        self.assertEqual(verified.assessment.rubric_verdict, "FAIL")
        self.assertEqual(read_verified(root / RUN, verified.scorecard), contents)
        for start, end in ((0.0, 0.3), (1.250000001, 1.55), (6.216667, 6.25)):
            with self.subTest(start=start, end=end), self.assertRaises(bind.ScorecardBindingError):
                self.bind_local(root, owner, candidate, self.card(start=start, end=end))

    def test_review_captions_distinguish_elapsed_and_native_without_changing_legacy_text(self):
        shifted = self.shifted_candidate()
        rendered = view._render(shifted.review_data, self.archived.prefix).decode()
        self.assertIn("Raw output: frame 0; PTS 1.250000 s. Elapsed from container origin: 0.000000 s.", rendered)
        self.assertIn("Declared container timeline origin is 1.250000 s.", rendered)
        self.assertIn("elapsed", rendered.lower())
        self.assertIn("native", rendered.lower())
        legacy = deepcopy(self.archived.review_data)
        legacy.pop("rawTimelineOriginSeconds", None)
        legacy_html = view._render(legacy, self.archived.prefix).decode()
        self.assertIn("Raw output: frame 0; PTS 0.000000 s.", legacy_html)
        self.assertIn("Raw offset is raw PTS minus source PTS.", legacy_html)
        self.assertNotIn("elapsed", legacy_html.lower())

    def test_interrupted_legacy_completion_resumes_after_restart_without_rewriting_artifacts(self):
        contents = self.media_fixture.raw_bytes
        root, owner, output, expected = self.interrupted_legacy_completion(contents)
        before = {name: ((output / name).read_bytes(), (output / name).stat().st_mtime_ns)
                  for name in ("raw.mp4", "metadata.json")}
        completer = complete.OutputCompleter(owner, root)
        self.assertIsNone(completer.get(ATTEMPT))
        with patch.object(complete, "publish_bytes", wraps=complete.publish_bytes) as publisher:
            completed = completer.complete(ATTEMPT, lambda *_: contents)
        proposed_metadata = [call.args[2] for call in publisher.call_args_list
                             if call.args[1].endswith("/output/metadata.json")]
        self.assertEqual(len(proposed_metadata), 1)
        self.assertEqual(json.loads(proposed_metadata[0])["timelineOriginSeconds"], 0.0)
        self.assertNotEqual(proposed_metadata[0], before["metadata.json"][0])
        self.assertEqual(completed.metadata, expected)
        self.assertEqual(read_verified(root / RUN, completed.metadata), before["metadata.json"][0])
        self.assertEqual(completed.raw.sha256, sha256(contents).hexdigest())
        self.assertEqual(owner.get(ATTEMPT).state, "SUCCEEDED")
        self.assertEqual(completer.get(ATTEMPT), completed)
        for name, (body, modified) in before.items():
            with self.subTest(artifact=name):
                self.assertEqual((output / name).read_bytes(), body)
                self.assertEqual((output / name).stat().st_mtime_ns, modified)

    def test_legacy_resume_rejects_changed_fields_and_contradictory_explicit_origins(self):
        for change in (lambda data: data.update(width=65),
                       lambda data: data.update(timelineOriginSeconds=1.25)):
            contents = self.media_fixture.raw_bytes
            root, owner, output, _ = self.interrupted_legacy_completion(contents)
            metadata = json.loads((output / "metadata.json").read_bytes())
            change(metadata)
            body = (json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            (output / "metadata.json").write_bytes(body)
            before = (output / "metadata.json").stat().st_mtime_ns
            with self.subTest(change=change), self.assertRaises(StorageError):
                complete.OutputCompleter(owner, root).complete(ATTEMPT, lambda *_: contents)
            self.assertEqual((output / "metadata.json").read_bytes(), body)
            self.assertEqual((output / "metadata.json").stat().st_mtime_ns, before)
            self.assertEqual(owner.get(ATTEMPT).state, "QUEUED")
            self.assertIsNone(complete.OutputCompleter(owner, root).get(ATTEMPT))

    def test_legacy_metadata_cannot_mask_a_publication_failure(self):
        contents = self.media_fixture.raw_bytes
        root, owner, output, expected = self.interrupted_legacy_completion(contents)
        before = ((output / "metadata.json").read_bytes(), (output / "metadata.json").stat().st_mtime_ns)
        current_publish = complete.publish_bytes

        def failed_metadata_publication(run_root, path, body):
            if path.endswith("/output/metadata.json"):
                raise StorageError("fixture metadata publication failure")
            return current_publish(run_root, path, body)

        with self.subTest(publication_failure=True), \
                patch.object(complete, "publish_bytes", side_effect=failed_metadata_publication):
            with self.assertRaisesRegex(StorageError, "fixture metadata publication failure"):
                complete.OutputCompleter(owner, root).complete(ATTEMPT, lambda *_: contents)
        self.assertEqual(read_verified(root / RUN, expected), before[0])
        self.assertEqual((output / "metadata.json").stat().st_mtime_ns, before[1])
        self.assertEqual(owner.get(ATTEMPT).state, "QUEUED")
        self.assertIsNone(complete.OutputCompleter(owner, root).get(ATTEMPT))

    def test_legacy_resume_requires_native_timeline_valid_on_the_original_zero_basis(self):
        root, owner, output, expected = self.interrupted_legacy_completion(self.shifted_bytes)
        before = (output / "metadata.json").read_bytes()
        metadata = json.loads(before)
        self.assertGreater(metadata["frameSeconds"][0], metadata["durationSeconds"])
        with self.assertRaises(StorageError):
            complete.OutputCompleter(owner, root).complete(ATTEMPT, lambda *_: self.shifted_bytes)
        self.assertEqual(read_verified(root / RUN, expected), before)
        self.assertEqual(owner.get(ATTEMPT).state, "QUEUED")
        self.assertIsNone(complete.OutputCompleter(owner, root).get(ATTEMPT))


if __name__ == "__main__":
    unittest.main()
