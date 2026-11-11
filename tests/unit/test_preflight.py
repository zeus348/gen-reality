"""Join the real evidence writers with fake provider transports."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import runwayml.resources.uploads as sdk_uploads

from app import prepare
from core.budget import BudgetLedger
from core.storage import StorageError, StoredArtifact, publish_bytes, read_verified
from features.inference.account import check_account_once
from features.inference.frozen import FrozenCreate, submit_frozen
from features.inference import preflight as preflight_module
from features.inference.preflight import PreflightError, check_prepared_request
from features.inference.prepared import read_prepared_attempt
from features.inference.pricing import record_price_quote
from features.inference.submission import SubmissionCoordinator
from features.inference.upload import upload_frozen_artifact
from features.placement.math import Placement


ATTEMPT = "att-" + "a" * 32
KEY = "fixture-api-secret"


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.run = self.root / "run-1"
        self.evidence = self.root / "evidence"
        video = SimpleNamespace(original_bytes=b"original" * 128, video_bytes=b"normalized" * 128)
        asset = SimpleNamespace(original_bytes=b"asset" * 128)
        guide = SimpleNamespace(keyframe_png=b"keyframe" * 128, guide_mp4=b"guide" * 128, prompt="Animate the creature.")
        with patch.object(prepare, "normalize_video", return_value=video), \
                patch.object(prepare, "load_asset", return_value=asset), \
                patch.object(prepare, "compile_guidance", return_value=guide):
            result = prepare.prepare_attempt(self.root / "source.mp4", self.root / "asset.png", self.root,
                "run-1", ATTEMPT, Placement(0.5, 0.5, 0.25), non_human=True, clean_source=True,
                source_kind="synthetic_fixture")
        self.manifest = StoredArtifact(**result["manifest"])
        self.prepared = read_prepared_attempt(self.run, self.manifest, run_id="run-1", attempt_id=ATTEMPT)
        self.price_source = publish_bytes(self.evidence, "price-source.txt", b"Synthetic operator evidence.")
        self.price = self.make_price()
        self.account = self.make_account()
        self.video = self.make_upload(self.prepared.normalized_source)
        self.keyframe = self.make_upload(self.prepared.keyframe_a)
        self.guide = self.make_upload(self.prepared.guide_b)

    def make_price(self, now=1000.0):
        return record_price_quote(self.evidence, self.price_source, checked_by="fixture operator", checked_unix=now,
            credits_per_second=28, minimum_credits=56, usd_per_credit="0.01", maximum_additional_cents=0,
            clock=lambda: now)

    def make_account(self, now=1000.0, required=140, balance=500):
        body = {"creditBalance": balance,
                "tier": {"models": {"aleph2": {"maxConcurrentGenerations": 5, "maxDailyGenerations": 20}}},
                "usage": {"models": {"aleph2": {"dailyGenerations": 0}}}}
        return check_account_once(KEY, required, self.evidence, clock=lambda: now,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))

    def make_upload(self, source):
        def storage_client(**arguments):
            return httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(204)), **arguments)
        sdk_httpx = SimpleNamespace(**(vars(httpx) | {"Client": storage_client}))
        body = {"runwayUri": "runway://" + source.sha256, "uploadUrl": "https://uploads.example.com/file", "fields": {}}
        with patch.object(sdk_uploads, "httpx", sdk_httpx):
            return upload_frozen_artifact(self.run, source, KEY, self.evidence,
                allowed_upload_hosts=frozenset({"uploads.example.com"}), clock=lambda: 1000.0,
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))

    def check(self, now=1001.0, **changes):
        arguments = {"run_id": "run-1", "attempt_id": ATTEMPT, "mode": "A", "video": self.video,
                     "keyframe": self.keyframe, "price": self.price, "account": self.account,
                     "evidence_root": self.evidence, "clock": lambda: now} | changes
        return check_prepared_request(self.run, self.manifest, **arguments)

    def test_full_frozen_fields_and_evidence_links_are_saved_before_reservation(self):
        result = self.check()
        self.assertEqual((result.run_id, result.attempt_id, result.source_kind), ("run-1", ATTEMPT, "synthetic_fixture"))
        self.assertEqual((result.frozen.video_uri, result.frozen.keyframe_uri), (self.video.uri, self.keyframe.uri))
        self.assertEqual(result.frozen.preparation_sha256, self.manifest.sha256)
        self.assertEqual((result.frozen.pricing_sha256, result.frozen.estimated_cents), (self.price.record.sha256, 140))
        data = json.loads(read_verified(self.run, result.evidence))
        self.assertEqual(FrozenCreate(**data["frozen"]), result.frozen)
        self.assertEqual(data["payloadSha256"], result.frozen.payload_sha256)
        self.assertEqual(data["account"], asdict(self.account.record))
        self.assertEqual(data["videoUpload"], asdict(self.video.evidence))
        self.assertEqual(data["keyframeUpload"], asdict(self.keyframe.evidence))
        self.assertIs(data["spendingAuthorized"], False)
        self.assertNotIn(KEY, json.dumps(data))
        result.require_current(1002.0)

    def test_mode_b_uses_its_guide_and_keyframe_presence_is_strict(self):
        result = self.check(mode="B", video=self.guide, keyframe=None)
        self.assertEqual(result.frozen.video_uri, self.guide.uri)
        self.assertNotIn("keyframes", json.loads(result.frozen.request_body))
        for changes in [{"mode": "A", "keyframe": None}, {"mode": "B", "video": self.guide},
                        {"mode": "C"}]:
            with self.subTest(changes=changes), self.assertRaises(PreflightError):
                self.check(**changes)

    def test_wrong_media_or_upload_uri_cannot_bind_to_the_request(self):
        for changes in [{"video": self.guide}, {"keyframe": self.video},
                        {"video": replace(self.video, uri="runway://different")},
                        {"video": replace(self.video, source=replace(self.video.source, path="other.mp4"))}]:
            with self.subTest(changes=changes), self.assertRaises(PreflightError):
                self.check(**changes)

    def test_every_linked_artifact_is_verified_without_repair(self):
        records = [(self.run, self.manifest), (self.run, self.prepared.normalized_source),
                   *[(self.evidence, item) for item in [self.price.source, self.price.record, self.account.record,
                       self.account.response, self.video.evidence, self.keyframe.evidence]]]
        for root, record in records:
            with self.subTest(path=record.path):
                path = root / record.path
                original = path.read_bytes()
                path.write_bytes(b"damaged fixture")
                with self.assertRaises(PreflightError):
                    self.check()
                self.assertEqual(path.read_bytes(), b"damaged fixture")
                path.write_bytes(original)

    def test_in_memory_values_must_match_saved_evidence(self):
        for changes in [{"price": replace(self.price, reserved_cents=1)},
                        {"price": replace(self.price, checked_unix=1001.0)},
                        {"account": replace(self.account, credit_balance=999)},
                        {"account": replace(self.account, checked_unix=1001.0)},
                        {"video": replace(self.video, expires_unix=999999.0)}]:
            with self.subTest(changes=changes), self.assertRaises(PreflightError):
                self.check(**changes)

    def test_account_credit_requirement_and_raw_response_are_checked(self):
        for account in [self.make_account(required=1), self.make_account(balance=100)]:
            with self.subTest(account=account), self.assertRaises(PreflightError):
                self.check(account=account)
        poor = self.make_account(balance=99)
        forged = replace(self.account, response=poor.response)
        data = json.loads(read_verified(self.evidence, self.account.record))
        data["response"] = asdict(poor.response)
        record = publish_bytes(self.evidence, "forged-account.json", json.dumps(data).encode())
        forged = replace(forged, record=record)
        with self.assertRaises(PreflightError):
            self.check(account=forged)

    def test_expired_or_future_evidence_and_slow_checks_are_rejected(self):
        for changes in [{"price": self.make_price(now=600.0)}, {"account": self.make_account(now=900.0)},
                        {"now": 999.0}, {"now": 87400.0, "price": self.make_price(now=87400.0),
                                         "account": self.make_account(now=87400.0)}]:
            with self.subTest(changes=changes), self.assertRaises(PreflightError):
                self.check(**changes)
        times = iter([1001.0, 1060.0])
        with self.assertRaises(PreflightError):
            self.check(clock=lambda: next(times))
        result = self.check()
        for now in [999.0, 1060.0, True, float("nan"), float("inf"), 10**500]:
            with self.subTest(now=now), self.assertRaises(PreflightError):
                result.require_current(now)

    def test_account_refresh_preserves_the_reserved_payload(self):
        first = self.check()
        second = self.check(now=1006.0, account=self.make_account(now=1005.0))
        self.assertNotEqual(first.evidence.sha256, second.evidence.sha256)
        self.assertEqual(first.frozen, second.frozen)
        ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1006.0)
        ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=140, max_attempts=1, cost_cents=140)
        reservation = ledger.reserve("run-1", "stable-key", first.frozen.payload_sha256, attempt_id=ATTEMPT)
        self.assertEqual(ledger.reserve("run-1", "stable-key", second.frozen.payload_sha256, attempt_id=ATTEMPT), reservation)

    def test_repeated_check_verifies_the_existing_record(self):
        result = self.check()
        path = self.run / result.evidence.path
        modified = path.stat().st_mtime_ns
        self.assertEqual(self.check(), result)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        path.write_bytes(b"damaged fixture check")
        with self.assertRaises(PreflightError):
            self.check()
        self.assertEqual(path.read_bytes(), b"damaged fixture check")

    def test_publication_failure_returns_no_checked_request(self):
        with patch.object(preflight_module, "publish_bytes", side_effect=StorageError("fixture secret path")):
            with self.assertRaises(PreflightError) as caught:
                self.check()
        self.assertEqual(str(caught.exception), "Prepared request or its provider evidence could not be verified.")

    def test_large_or_duplicate_evidence_cannot_support_a_request(self):
        data = json.loads(read_verified(self.evidence, self.price.record))
        bodies = [json.dumps(data | {"extra": "x" * (64 * 1024)}).encode(),
                  json.dumps(data).replace('"reservedCents": 140', '"reservedCents":0,"reservedCents":140').encode()]
        for index, body in enumerate(bodies):
            record = publish_bytes(self.evidence, f"invalid-price-{index}.json", body)
            with self.subTest(index=index), self.assertRaises(PreflightError):
                self.check(price=replace(self.price, record=record))

    def test_checked_mode_b_request_reaches_the_sdk_once_after_reservation(self):
        checked = self.check(mode="B", video=self.guide, keyframe=None)
        ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1001.0)
        ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=140, max_attempts=1, cost_cents=140)
        ledger.reserve("run-1", "stable-key", checked.frozen.payload_sha256, attempt_id=ATTEMPT)
        requests = []

        def create(request):
            requests.append(request)
            return httpx.Response(200, json={"id": "fixture-provider-task"})

        with SubmissionCoordinator(ledger) as coordinator:
            result = submit_frozen(coordinator, ATTEMPT, checked.frozen, self.root, KEY,
                                   transport=httpx.MockTransport(create))
            self.assertEqual(result.state, "QUEUED")
            self.assertEqual(submit_frozen(coordinator, ATTEMPT, checked.frozen, self.root, ""), result)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(requests[0].content, checked.frozen.request_body)
        self.assertEqual(json.loads(requests[0].content)["videoUri"], self.guide.uri)
        self.assertNotIn("keyframes", json.loads(requests[0].content))
