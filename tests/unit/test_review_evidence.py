"""Build real review samples from cached preparation and mocked SDK requests."""

from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from PIL import Image
import runwayml.resources.uploads as sdk_uploads

from app.prepare import prepare_attempt
from core.budget import BudgetLedger
from core.storage import StorageError, StoredArtifact, publish_bytes, read_verified
from features.inference.account import check_account_once
from features.inference.frozen import FrozenCreate
from features.inference.polling import TaskPoller
from features.inference.preflight import check_prepared_request
from features.inference.prepared import read_prepared_attempt
from features.inference.pricing import record_price_quote
from features.inference.retrieve import retrieve_task_once
from features.inference.send import submit_checked
from features.inference.submission import SubmissionCoordinator
from features.inference.upload import upload_frozen_artifact
from features.output.complete import OutputCompleter
from features.output.fetch import download_task_output
from features.output import review
from features.placement.math import Placement


RUN = "review-run"
ATTEMPT_A = "att-" + "a" * 32
ATTEMPT_B = "att-" + "b" * 32
KEY = "review-fixture-api-secret"
OUTPUT_URL = "https://outputs.example.com/raw.mp4?signature=private-output-token"


class ReviewEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.base = Path(cls.folder.name)
        source = cls.base / "source.mp4"
        cls.colors = ((20, 40, 180), (30, 170, 60), (180, 30, 50),
                      (140, 40, 160), (20, 150, 160), (180, 160, 30))
        source_pixels = b"".join(Image.new("RGB", (128, 72), color).tobytes() * 24 for color in cls.colors)
        cls.encode(source, source_pixels, 128, 72, 24)
        asset = cls.base / "asset.png"
        image = Image.new("RGBA", (32, 32), "white")
        image.putpixel((0, 0), (0, 0, 0, 0))
        image.save(asset)
        raw = cls.base / "fixture-output.mp4"
        cls.raw_frames = [Image.new("RGB", (64, 36), (index, index * 3 % 256, index * 7 % 256)).tobytes()
                          for index in range(150)]
        cls.encode(raw, b"".join(cls.raw_frames), 64, 36, 30)
        cls.raw_bytes = raw.read_bytes()
        cls.manifests, cls.prepared, cls.checks, cls.completed, cls.tasks = {}, {}, {}, {}, {}
        run_root = cls.base / RUN
        evidence = cls.base / "provider-evidence"
        source_record = publish_bytes(evidence, "fixture-price.txt", b"Synthetic price evidence; no live provider call.")
        price = record_price_quote(evidence, source_record, checked_by="fixture operator", checked_unix=1000.0,
            credits_per_second=28, minimum_credits=56, usd_per_credit="0.01", maximum_additional_cents=0,
            clock=lambda: 1000.0)
        account_body = {"creditBalance": 500,
            "tier": {"models": {"aleph2": {"maxConcurrentGenerations": 5, "maxDailyGenerations": 20}}},
            "usage": {"models": {"aleph2": {"dailyGenerations": 0}}}}
        account = check_account_once(KEY, 140, evidence, clock=lambda: 1000.0,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=account_body)))
        ledger = BudgetLedger(cls.base / "budget.sqlite", clock=lambda: 1001.0)
        ledger.approve_run(RUN, approved_by="fixture operator", limit_cents=1000, max_attempts=6, cost_cents=140)
        with SubmissionCoordinator(ledger) as owner:
            for attempt, mode, u in ((ATTEMPT_A, "A", 0.25), (ATTEMPT_B, "B", 0.75)):
                result = prepare_attempt(source, asset, cls.base, RUN, attempt, Placement(u, 0.5, 0.2),
                    non_human=True, clean_source=True, source_kind="synthetic_fixture")
                manifest = StoredArtifact(**result["manifest"])
                prepared = read_prepared_attempt(run_root, manifest, run_id=RUN, attempt_id=attempt)
                video, keyframe = prepared.media_for_mode(mode)
                uploaded_video = cls.upload(run_root, video, evidence)
                uploaded_keyframe = cls.upload(run_root, keyframe, evidence) if keyframe else None
                checked = check_prepared_request(run_root, manifest, run_id=RUN, attempt_id=attempt, mode=mode,
                    video=uploaded_video, keyframe=uploaded_keyframe, price=price, account=account,
                    evidence_root=evidence, clock=lambda: 1001.0)
                ledger.reserve(RUN, "fixture:" + attempt, checked.frozen.payload_sha256, attempt_id=attempt)
                task = "fixture-task-" + mode.lower()
                sent = submit_checked(owner, checked, cls.base, KEY,
                    transport=httpx.MockTransport(lambda _, task=task: httpx.Response(200, json={"id": task})))
                if sent.state != "QUEUED":
                    raise AssertionError("Fixture did not pass the checked SDK submission path.")
                cls.manifests[attempt], cls.prepared[attempt], cls.checks[attempt] = manifest, prepared, checked
                cls.tasks[attempt] = task
                cls.completed[attempt] = cls.complete(cls.base, owner, attempt, task)

    @staticmethod
    def encode(path, contents, width, height, fps):
        pixels = path.with_suffix(".rgb")
        pixels.write_bytes(contents)
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-f", "rawvideo",
                        "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
                        "-i", str(pixels), "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast",
                        "-threads", "2", "-video_track_timescale", "24000", str(path)],
                       capture_output=True, check=True, timeout=30)

    @staticmethod
    def upload(run_root, source, evidence_root):
        def storage_client(**arguments):
            return httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(204)), **arguments)
        sdk_httpx = SimpleNamespace(**(vars(httpx) | {"Client": storage_client}))
        body = {"runwayUri": "runway://" + source.sha256, "uploadUrl": "https://uploads.example.com/file", "fields": {}}
        with patch.object(sdk_uploads, "httpx", sdk_httpx):
            return upload_frozen_artifact(run_root, source, KEY, evidence_root,
                allowed_upload_hosts=frozenset({"uploads.example.com"}), clock=lambda: 1000.0,
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))

    @classmethod
    def complete(cls, root, owner, attempt, task):
        body = {"id": task, "status": "SUCCEEDED", "output": [OUTPUT_URL]}
        TaskPoller(owner).poll_once(attempt, lambda known_task: retrieve_task_once(known_task, KEY, root / RUN,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))),
            deadline_unix=owner.ledger._now() + 30)
        return OutputCompleter(owner, root).complete(attempt, lambda known_task, evidence:
            download_task_output(known_task, evidence, root / RUN, allowed_hosts=frozenset({"outputs.example.com"}),
                transport=httpx.MockTransport(lambda _: httpx.Response(200, content=cls.raw_bytes))))

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)
        self.run_root = self.root / RUN
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 5000.0)
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)

    def build(self, attempt=ATTEMPT_A, **changes):
        arguments = {"preparation_manifest": self.manifests[attempt], "request_check": self.checks[attempt].evidence,
                     "run_id": RUN, "attempt_id": attempt} | changes
        return review.build_review_evidence(self.owner, self.root, **arguments)

    def data(self, record):
        return json.loads(read_verified(self.run_root, record))

    def assert_no_review(self):
        self.assertFalse((self.run_root / f"attempts/{ATTEMPT_A}/review").exists())

    def changed_check(self, **changes):
        data = self.data(self.checks[ATTEMPT_A].evidence)
        data.update(changes)
        if "frozen" in changes:
            data["payloadSha256"] = FrozenCreate(**data["frozen"]).payload_sha256
        body = json.dumps(data, sort_keys=True).encode()
        return publish_bytes(self.run_root, f"attempts/{ATTEMPT_A}/provider_checks/{sha256(body).hexdigest()}.json", body)

    def write_metadata(self, contents):
        path = self.run_root / self.completed[ATTEMPT_A].metadata.path
        path.write_bytes(contents)
        with self.ledger._connection() as connection:
            connection.execute("UPDATE completed_outputs SET metadata_sha256=?, metadata_size=? WHERE attempt_id=?",
                               (sha256(contents).hexdigest(), len(contents), ATTEMPT_A))

    def test_mode_a_reuses_actual_static_keyframe_and_preserves_original_source_reference(self):
        manifest = self.data(self.build())
        prepared = self.prepared[ATTEMPT_A]
        self.assertEqual((manifest["runId"], manifest["attemptId"], manifest["mode"]), (RUN, ATTEMPT_A, "A"))
        self.assertEqual(manifest["sourceOriginal"], asdict(prepared.source_original))
        self.assertEqual(manifest["sourceNormalized"], asdict(prepared.normalized_source))
        self.assertNotEqual(manifest["sourceOriginal"]["sha256"], manifest["sourceNormalized"]["sha256"])
        self.assertEqual(manifest["providerVideoInput"], asdict(prepared.normalized_source))
        self.assertEqual(manifest["guide"], {"kind": "static_keyframe", "artifact": asdict(prepared.keyframe_a)})
        self.assertEqual(len(manifest["samples"]), 5)
        for sample in manifest["samples"]:
            self.assertEqual(sample["guide"], {"kind": "static_keyframe", "referenceSeconds": 0.0,
                                               "image": asdict(prepared.keyframe_a)})
        self.assertFalse((self.run_root / f"attempts/{ATTEMPT_A}/review/guide").exists())
        self.assertEqual(manifest["sourceKind"], "synthetic_fixture")
        self.assertEqual(manifest["visualEffectDemonstrated"], "NOT_RUN")
        self.assertEqual(manifest["automatedWorkflowComplete"], "NOT_RUN")
        self.assertIs(manifest["sampleAutomaticRotationApplied"], False)
        for name in ("request", "frozenRequest", "providerSuccess", "rawOutput", "rawMetadata"):
            read_verified(self.run_root, StoredArtifact(**manifest[name]))
        self.assertNotIn(KEY, json.dumps(manifest))
        self.assertNotIn("private-output-token", json.dumps(manifest))
        self.assertNotIn("https://", json.dumps(manifest))

    def test_mode_b_samples_real_24_fps_guide_and_30_fps_raw_frames(self):
        manifest = self.data(self.build(ATTEMPT_B))
        self.assertEqual(manifest["guide"], {"kind": "video", "artifact": asdict(self.prepared[ATTEMPT_B].guide_b)})
        self.assertEqual(manifest["providerVideoInput"], asdict(self.prepared[ATTEMPT_B].guide_b))
        for second, sample in enumerate(manifest["samples"]):
            self.assertEqual(sample["requestedSeconds"], second)
            self.assertEqual((sample["source"]["frameIndex"], sample["guide"]["frameIndex"], sample["raw"]["frameIndex"]),
                             (second * 24, second * 24, second * 30))
            self.assertEqual((sample["source"]["ptsSeconds"], sample["guide"]["ptsSeconds"], sample["raw"]["ptsSeconds"]),
                             (second, second, second))
            self.assertEqual(sample["raw"]["offsetSeconds"], 0)
            self.assertEqual(sample["raw"]["spanPosition"], "within_span")
            self.assertIs(sample["raw"]["withinDeclaredDuration"], True)
            source = read_verified(self.run_root, StoredArtifact(**sample["source"]["image"]))
            guide = read_verified(self.run_root, StoredArtifact(**sample["guide"]["image"]))
            raw = read_verified(self.run_root, StoredArtifact(**sample["raw"]["image"]))
            with Image.open(BytesIO(source)) as image:
                for actual, expected in zip(image.getpixel((10, 10)), self.colors[second]):
                    self.assertLessEqual(abs(actual - expected), 5)
            with Image.open(BytesIO(guide)) as image:
                self.assertTrue(all(channel >= 240 for channel in image.getpixel((960, 360))))
            with Image.open(BytesIO(raw)) as image:
                self.assertEqual(image.tobytes(), self.raw_frames[second * 30])

    def test_mixed_run_attempt_manifest_and_check_are_rejected_before_publication(self):
        cases = [{"run_id": "other-run"}, {"attempt_id": ATTEMPT_B},
                 {"preparation_manifest": self.manifests[ATTEMPT_B]},
                 {"request_check": self.checks[ATTEMPT_B].evidence},
                 {"preparation_manifest": replace(self.manifests[ATTEMPT_A], sha256="f" * 64)}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(review.ReviewEvidenceError):
                self.build(**changes)
            self.assert_no_review()

    def test_changed_preparation_prompt_source_kind_and_mode_cannot_rebind_the_attempt(self):
        frozen = asdict(self.checks[ATTEMPT_A].frozen)
        variants = [{"frozen": frozen | {"preparation_sha256": "f" * 64}},
                    {"frozen": frozen | {"prompt": "Different action."}}, {"sourceKind": "clean_camera"},
                    {"frozen": frozen | {"mode": "B", "keyframe_uri": None,
                                          "video_uri": self.checks[ATTEMPT_B].frozen.video_uri}}]
        for changes in variants:
            record = self.changed_check(**changes)
            with self.subTest(changes=changes), self.assertRaises(review.ReviewEvidenceError):
                self.build(request_check=record)
            self.assert_no_review()

    def test_reservation_payload_cost_and_run_must_match(self):
        self.ledger.approve_run("other-run", approved_by="fixture", limit_cents=140, max_attempts=1, cost_cents=140)
        for field, value, original in (("payload_sha256", "f" * 64, self.checks[ATTEMPT_A].frozen.payload_sha256),
                                        ("cost_cents", 141, 140), ("run_id", "other-run", RUN)):
            with self.ledger._connection() as connection:
                connection.execute(f"UPDATE reservations SET {field}=? WHERE attempt_id=?", (value, ATTEMPT_A))
            try:
                with self.subTest(field=field), self.assertRaises(review.ReviewEvidenceError):
                    self.build()
                self.assert_no_review()
            finally:
                with self.ledger._connection() as connection:
                    connection.execute(f"UPDATE reservations SET {field}=? WHERE attempt_id=?", (original, ATTEMPT_A))

    def test_saved_request_bytes_and_frozen_record_must_match_before_publication(self):
        for name in ("request.json", "frozen_request.json"):
            path = self.run_root / f"attempts/{ATTEMPT_A}/provider/{name}"
            original = path.read_bytes()
            path.write_bytes(b"changed submitted fixture")
            with self.subTest(name=name), self.assertRaises(review.ReviewEvidenceError):
                self.build()
            self.assertEqual(path.read_bytes(), b"changed submitted fixture")
            self.assert_no_review()
            path.write_bytes(original)

    def test_bare_coordinator_submission_cannot_replace_saved_request_evidence(self):
        with self.ledger._transaction() as connection:
            for table in ("completed_outputs", "poll_events", "submission_jobs"):
                connection.execute(f"DELETE FROM {table} WHERE attempt_id=?", (ATTEMPT_A,))
        for name in ("request.json", "frozen_request.json"):
            (self.run_root / f"attempts/{ATTEMPT_A}/provider/{name}").unlink()
        self.owner.submit_once(ATTEMPT_A, lambda: self.tasks[ATTEMPT_A])
        self.complete(self.root, self.owner, ATTEMPT_A, self.tasks[ATTEMPT_A])
        with self.assertRaises(review.ReviewEvidenceError):
            self.build()
        self.assert_no_review()

    def test_output_metadata_identity_raw_record_and_timing_are_checked_before_publication(self):
        original = read_verified(self.run_root, self.completed[ATTEMPT_A].metadata)
        data = json.loads(original)
        variants = [{"runId": "other-run"}, {"attemptId": ATTEMPT_B}, {"providerTaskId": self.tasks[ATTEMPT_B]},
                    {"raw": asdict(self.completed[ATTEMPT_B].raw)}, {"width": True},
                    {"frameSeconds": [0, float("nan")]}, {"frameSeconds": "0,1,2"}, {"rawContentValidation": "NOT_RUN"}]
        for changes in variants:
            self.write_metadata(json.dumps(data | changes).encode())
            with self.subTest(changes=changes), self.assertRaises(review.ReviewEvidenceError):
                self.build()
            self.assert_no_review()
        self.write_metadata(original)

    def test_completed_raw_path_cannot_point_to_another_attempt(self):
        other = self.completed[ATTEMPT_B].raw
        data = self.data(self.completed[ATTEMPT_A].metadata)
        self.write_metadata(json.dumps(data | {"raw": asdict(other)}).encode())
        with self.ledger._connection() as connection:
            connection.execute("UPDATE completed_outputs SET raw_path=? WHERE attempt_id=?", (other.path, ATTEMPT_A))
        with self.assertRaises(review.ReviewEvidenceError):
            self.build()
        self.assert_no_review()

    def test_provider_success_body_is_bounded_hashed_and_matches_the_task(self):
        task = self.tasks[ATTEMPT_A]
        bodies = [json.dumps({"id": self.tasks[ATTEMPT_B], "status": "SUCCEEDED"}).encode(),
                  json.dumps({"id": task, "status": "FAILED"}).encode(),
                  ('{"id":"' + task + '","status":"FAILED","status":"SUCCEEDED"}').encode(),
                  b" " * (1024 * 1024 + 1)]
        original_hash = self.completed[ATTEMPT_A].provider_evidence_sha256
        for body in bodies:
            digest = sha256(body).hexdigest()
            publish_bytes(self.run_root, f"provider_responses/{digest}.json", body)
            with self.ledger._connection() as connection:
                connection.execute("UPDATE completed_outputs SET provider_evidence_sha256=? WHERE attempt_id=?", (digest, ATTEMPT_A))
            with self.subTest(size=len(body)), self.assertRaises(review.ReviewEvidenceError):
                self.build()
            self.assert_no_review()
        with self.ledger._connection() as connection:
            connection.execute("UPDATE completed_outputs SET provider_evidence_sha256=? WHERE attempt_id=?", (original_hash, ATTEMPT_A))
        path = self.run_root / f"provider_responses/{original_hash}.json"
        path.write_bytes(b"damaged saved success response")
        with self.assertRaises(review.ReviewEvidenceError):
            self.build()
        self.assert_no_review()

    def test_unknown_or_incomplete_task_cannot_publish_review_evidence(self):
        for state in ("SUBMISSION_UNKNOWN", "QUEUED"):
            with self.ledger._connection() as connection:
                connection.execute("UPDATE submission_jobs SET state=? WHERE attempt_id=?", (state, ATTEMPT_A))
            with self.subTest(state=state), self.assertRaises(review.ReviewEvidenceError):
                self.build()
            self.assert_no_review()

    def test_repeat_and_other_attempt_keep_existing_review_bytes_unchanged(self):
        first = self.build()
        folder = self.run_root / f"attempts/{ATTEMPT_A}"
        before = {path.relative_to(folder): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in folder.rglob("*") if path.is_file()}
        self.assertEqual(self.build(), first)
        second = self.data(self.build(ATTEMPT_B))
        self.assertEqual(second["attemptId"], ATTEMPT_B)
        self.assertNotEqual(second["guide"]["artifact"], self.data(first)["guide"]["artifact"])
        for name, expected in before.items():
            path = folder / name
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)

    def test_interrupted_publication_resumes_without_replacing_existing_samples(self):
        publish = review._publish_or_verify

        def fail_after_two(root, path, contents):
            if path.endswith("source/frame-000048.png"):
                raise StorageError("private fixture path")
            return publish(root, path, contents)

        with patch.object(review, "_publish_or_verify", side_effect=fail_after_two):
            with self.assertRaises(review.ReviewEvidenceError) as caught:
                self.build()
        self.assertNotIn("private fixture path", str(caught.exception))
        folder = self.run_root / f"attempts/{ATTEMPT_A}/review"
        self.assertFalse((folder / "manifest.json").exists())
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in folder.rglob("*.png")}
        self.assertEqual(len(before), 2)
        result = self.build()
        self.assertEqual(len(self.data(result)["samples"]), 5)
        for path, expected in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)

    def test_changed_existing_sample_is_rejected_without_repair(self):
        result = self.build()
        data = self.data(result)
        path = self.run_root / data["samples"][0]["raw"]["image"]["path"]
        path.write_bytes(b"changed existing sample")
        with self.assertRaises(review.ReviewEvidenceError):
            self.build()
        self.assertEqual(path.read_bytes(), b"changed existing sample")
        self.assertEqual(self.data(result), data)
