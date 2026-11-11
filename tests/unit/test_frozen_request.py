"""Check frozen evidence and the reservation at the real SDK send boundary."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from core.budget import BudgetError, BudgetExceeded, BudgetLedger, IdempotencyConflict
from core.storage import publish_bytes
from features.guidance.video import DEFAULT_PROMPT
from features.inference.frozen import FrozenCreate, submit_frozen
from features.inference.runway import RunwayCreateAdapter
from features.inference.submission import SubmissionCoordinator, SubmissionError


TASK_ID = "c26d7283-d514-4492-8589-760dfe6f0289"
API_KEY = "fixture-credential"


class FrozenRequestTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.output = self.root / "runs"
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1000.0)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=140,
                               max_attempts=1, cost_cents=140)
        self.frozen = FrozenCreate("A", "runway://source", DEFAULT_PROMPT,
                                   sha256(b"preparation manifest").hexdigest(),
                                   sha256(b"verified account and price evidence").hexdigest(),
                                   140, "runway://keyframe")
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.requests = []

    def reserve(self, frozen=None):
        return self.ledger.reserve("run-1", "fixture-key", (frozen or self.frozen).payload_sha256).attempt_id

    def provider_folder(self, attempt):
        return self.output / "run-1" / "attempts" / attempt / "provider"

    def send(self, attempt, frozen=None, *, key=API_KEY, check=None):
        def handle(request):
            self.requests.append(request)
            if check:
                check(request)
            return httpx.Response(200, json={"id": TASK_ID})

        return submit_frozen(self.owner, attempt, frozen or self.frozen, self.output, key,
                              transport=httpx.MockTransport(handle))

    def test_each_frozen_input_changes_the_reservation_fingerprint(self):
        variants = [
            replace(self.frozen, video_uri="runway://other-source"),
            replace(self.frozen, prompt=DEFAULT_PROMPT + " Keep the light fixed."),
            replace(self.frozen, keyframe_uri="runway://other-keyframe"),
            replace(self.frozen, preparation_sha256="1" * 64),
            replace(self.frozen, pricing_sha256="2" * 64),
            replace(self.frozen, estimated_cents=141),
            replace(self.frozen, mode="B", keyframe_uri=None),
        ]
        hashes = {item.payload_sha256 for item in [self.frozen, *variants]}
        self.assertEqual(len(hashes), len(variants) + 1)
        self.assertEqual(self.frozen.payload_sha256, sha256(self.frozen.record_bytes).hexdigest())
        with self.assertRaises(FrozenInstanceError):
            self.frozen.estimated_cents = 1

    def test_request_and_evidence_are_published_before_the_post(self):
        attempt = self.reserve()
        folder = self.provider_folder(attempt)

        def check(request):
            self.assertEqual(self.owner.get(attempt).state, "SUBMITTING")
            self.assertEqual((folder / "request.json").read_bytes(), request.content)
            self.assertEqual(request.content, self.frozen.request_body)
            record_bytes = (folder / "frozen_request.json").read_bytes()
            self.assertEqual(sha256(record_bytes).hexdigest(), self.frozen.payload_sha256)
            self.assertEqual(json.loads(record_bytes)["requestBodySha256"], sha256(request.content).hexdigest())
            self.assertNotIn(API_KEY.encode(), record_bytes + request.content)

        result = self.send(attempt, check=check)
        self.assertEqual((result.state, result.provider_task_id), ("QUEUED", TASK_ID))
        self.assertEqual(len(self.requests), 1)

    def test_mode_b_uses_its_frozen_guide_and_omits_keyframes(self):
        frozen = replace(self.frozen, mode="B", video_uri="runway://guide-B", keyframe_uri=None)
        attempt = self.reserve(frozen)
        self.assertEqual(self.send(attempt, frozen).state, "QUEUED")
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["videoUri"], "runway://guide-B")
        self.assertNotIn("keyframes", body)
        self.assertEqual(self.requests[0].content, frozen.request_body)

    def test_changed_payload_cannot_use_an_existing_reservation(self):
        attempt = self.reserve()
        changed = replace(self.frozen, prompt="Different frozen prompt.")
        with self.assertRaises(IdempotencyConflict):
            self.send(attempt, changed)
        self.assertEqual(self.owner.get(attempt).state, "RESERVED")
        self.assertEqual(self.requests, [])
        self.assertFalse(self.output.exists())

    def test_frozen_cost_must_match_the_server_quote(self):
        changed = replace(self.frozen, estimated_cents=100)
        attempt = self.reserve(changed)
        with self.assertRaisesRegex(BudgetError, "approved quote"):
            self.send(attempt, changed)
        self.assertEqual(self.owner.get(attempt).state, "RESERVED")
        self.assertEqual(self.requests, [])

    def test_changed_sdk_body_is_blocked_before_network_or_publication(self):
        attempt = self.reserve()
        original = RunwayCreateAdapter.create_task

        def change_prompt(adapter, video_uri, prompt, **arguments):
            return original(adapter, video_uri, prompt + " Changed during send.", **arguments)

        with patch.object(RunwayCreateAdapter, "create_task", change_prompt):
            result = self.send(attempt)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.requests, [])
        self.assertFalse(self.output.exists())

    def test_recording_collision_keeps_prior_bytes_and_blocks_the_post(self):
        attempt = self.reserve()
        relative = f"attempts/{attempt}/provider/request.json"
        publish_bytes(self.output / "run-1", relative, b"prior evidence")
        result = self.send(attempt)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual((self.provider_folder(attempt) / "request.json").read_bytes(), b"prior evidence")
        self.assertEqual(self.requests, [])
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "another-key", self.frozen.payload_sha256)

    def test_restart_returns_the_task_without_credentials_or_artifact_rewrites(self):
        attempt = self.reserve()
        first = self.send(attempt)
        folder = self.provider_folder(attempt)
        records = {file.name: file.read_bytes() for file in folder.iterdir()}
        self.owner.close()
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.assertEqual(self.send(attempt, key=""), first)
        self.assertEqual({file.name: file.read_bytes() for file in folder.iterdir()}, records)
        self.assertEqual(len(self.requests), 1)
        with self.assertRaises(IdempotencyConflict):
            self.send(attempt, replace(self.frozen, preparation_sha256="0" * 64), key="")

    def test_concurrent_duplicates_publish_one_request_and_make_one_post(self):
        attempt = self.reserve()
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(lambda _: self.send(attempt), range(20)))
        for result in results:
            self.assertEqual(result.attempt_id, attempt)
            self.assertIn(result.state, ["SUBMITTING", "QUEUED"])
            if result.state == "QUEUED":
                self.assertEqual(result.provider_task_id, TASK_ID)
        self.assertEqual(self.owner.get(attempt).state, "QUEUED")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(sorted(file.name for file in self.provider_folder(attempt).iterdir()),
                         ["frozen_request.json", "request.json"])

    def test_invalid_frozen_values_are_rejected(self):
        invalid = [{"mode": "C"}, {"keyframe_uri": None}, {"mode": "B"},
                   {"preparation_sha256": "bad"}, {"pricing_sha256": "bad"},
                   {"estimated_cents": 0}, {"estimated_cents": True}, {"estimated_cents": 1001}]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    replace(self.frozen, **values)

    def test_missing_reservation_cannot_create_artifacts_or_a_task(self):
        with self.assertRaisesRegex(SubmissionError, "no matching reservation"):
            self.send("att-" + "f" * 32)
        self.assertEqual(self.requests, [])
        self.assertFalse(self.output.exists())
