"""Check SDK GET requests, saved response bytes, and polling recovery."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import httpx

from core.budget import BudgetLedger
from core.storage import StorageError
from features.inference import retrieve as retrieve_module
from features.inference.polling import TaskPoller
from features.inference.retrieve import retrieve_task_once
from features.inference.runway import ProviderError
from features.inference.submission import SubmissionCoordinator


TASK_ID = "c26d7283-d514-4492-8589-760dfe6f0289"
API_KEY = "fixture-credential-never-record"
BODY = json.dumps({"id": TASK_ID, "status": "RUNNING", "progress": 0.5}, indent=2).encode()


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.evidence = self.root / "evidence"
        self.requests = []

    def retrieve(self, response=None, *, task_id=TASK_ID, **arguments):
        def handle(request):
            self.requests.append(request)
            if callable(response):
                return response(request)
            return response or httpx.Response(200, content=BODY, headers={"content-type": "application/json"})

        return retrieve_task_once(task_id, API_KEY, self.evidence,
                                  transport=httpx.MockTransport(handle), **arguments)

    def poller(self):
        ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1000.0)
        ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=140,
                          max_attempts=1, cost_cents=140)
        attempt = ledger.reserve("run-1", "first", "1" * 64).attempt_id
        owner = SubmissionCoordinator(ledger)
        self.addCleanup(owner.close)
        create = Mock(return_value=TASK_ID)
        owner.submit_once(attempt, create)
        return owner, TaskPoller(owner), attempt, create

    def test_sdk_sends_one_get_and_saves_exact_response_bytes(self):
        result = self.retrieve(timeout_seconds=1.25)
        self.assertEqual((result.task_id, result.status), (TASK_ID, "RUNNING"))
        self.assertEqual(result.evidence_sha256, sha256(BODY).hexdigest())
        path = self.evidence / "provider_responses" / (result.evidence_sha256 + ".json")
        self.assertEqual(path.read_bytes(), BODY)
        self.assertNotIn(API_KEY.encode(), path.read_bytes())
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request.method, "GET")
        self.assertEqual(str(request.url), "https://api.dev.runwayml.com/v1/tasks/" + TASK_ID)
        self.assertEqual(request.headers["authorization"], "Bearer " + API_KEY)
        self.assertEqual(set(request.extensions["timeout"].values()), {1.25})

    def test_identical_responses_reuse_verified_evidence(self):
        first = self.retrieve()
        path = self.evidence / "provider_responses" / (first.evidence_sha256 + ".json")
        before = path.stat().st_mtime_ns
        self.assertEqual(self.retrieve(), first)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_damaged_prior_evidence_is_rejected_without_replacement(self):
        first = self.retrieve()
        path = self.evidence / "provider_responses" / (first.evidence_sha256 + ".json")
        path.write_bytes(b"damaged fixture evidence")
        with self.assertRaises(ProviderError):
            self.retrieve()
        self.assertEqual(path.read_bytes(), b"damaged fixture evidence")

    def test_large_success_body_stops_streaming_and_closes_response(self):
        class LargeStream(httpx.SyncByteStream):
            chunks_read = 0
            closed = False

            def __iter__(self):
                for _ in range(30):
                    self.chunks_read += 1
                    yield b" " * (64 * 1024)

            def close(self):
                self.closed = True

        stream = LargeStream()
        with self.assertRaises(ProviderError):
            self.retrieve(httpx.Response(200, stream=stream))
        self.assertLessEqual(stream.chunks_read, 17)
        self.assertTrue(stream.closed)
        self.assertFalse(self.evidence.exists())

    def test_invalid_or_credential_bearing_body_cannot_be_saved(self):
        invalid = [b"not JSON", b"[]", json.dumps({"id": "wrong", "status": "RUNNING"}).encode(),
                   json.dumps({"id": TASK_ID, "status": "NEW_UNKNOWN_STATUS"}).encode(),
                   json.dumps({"id": TASK_ID, "status": "RUNNING", "debug": API_KEY}).encode(),
                   json.dumps({"id": TASK_ID, "status": "RUNNING", "debug": API_KEY})
                       .replace(API_KEY, "\\u0066" + API_KEY[1:]).encode()]
        for body in invalid:
            with self.subTest(body=body):
                with self.assertRaises(ProviderError) as caught:
                    self.retrieve(httpx.Response(200, content=body))
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertFalse(self.evidence.exists())

    def test_duplicate_fields_cannot_hide_credentials_in_saved_bytes(self):
        escaped_key = "\\u0066" + API_KEY[1:]
        hidden_values = [
            '"debug":"' + API_KEY + '","debug":"safe"',
            '"debug":"' + API_KEY + '","\\u0064ebug":"safe"',
            '"debug":"' + escaped_key + '","debug":"safe"',
            '"details":{"debug":"' + API_KEY + '","debug":"safe"}',
        ]
        for fields in hidden_values:
            with self.subTest(fields=fields):
                body = ('{"id":"' + TASK_ID + '","status":"RUNNING",' + fields + '}').encode()
                with self.assertRaises(ProviderError) as caught:
                    self.retrieve(httpx.Response(200, content=body))
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertFalse(self.evidence.exists())

    def test_http_errors_have_no_retries_or_saved_error_details(self):
        for status in [401, 408, 429, 500]:
            with self.subTest(status=status):
                self.requests.clear()
                with self.assertRaises(ProviderError) as caught:
                    self.retrieve(httpx.Response(status, json={"error": API_KEY}))
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertEqual(len(self.requests), 1)
                self.assertEqual(self.requests[0].method, "GET")
                self.assertFalse(self.evidence.exists())

    def test_redirect_is_not_followed(self):
        with self.assertRaises(ProviderError):
            self.retrieve(httpx.Response(307, headers={"Location": "https://example.com/other"}))
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(self.evidence.exists())

    def test_invalid_task_or_timeout_cannot_reach_network(self):
        transport = httpx.MockTransport(Mock())
        for task_id in [None, "", "../task", "a" * 129]:
            with self.subTest(task_id=task_id):
                with self.assertRaises(ValueError):
                    retrieve_task_once(task_id, API_KEY, self.evidence, transport=transport)
        for timeout in [True, 0, -1, 31, float("nan"), float("inf")]:
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    retrieve_task_once(TASK_ID, API_KEY, self.evidence, timeout_seconds=timeout, transport=transport)
        transport.handler.assert_not_called()

    def test_sdk_get_timeout_resumes_after_restart_without_another_create(self):
        owner, poller, attempt, create = self.poller()

        def timeout(request):
            raise httpx.ReadTimeout(API_KEY, request=request)

        first = poller.poll_once(attempt, lambda task: self.retrieve(timeout, task_id=task), deadline_unix=1100.0)
        self.assertEqual((first.state, first.poll_status), ("QUEUED", "POLL_DEADLINE_REACHED"))
        owner.close()
        restarted = SubmissionCoordinator(owner.ledger)
        self.addCleanup(restarted.close)
        result = TaskPoller(restarted).poll_once(attempt, lambda task: self.retrieve(task_id=task), deadline_unix=1100.0)
        self.assertEqual((result.state, result.poll_status), ("RUNNING", "OBSERVED"))
        self.assertEqual([request.method for request in self.requests], ["GET", "GET"])
        self.assertTrue(all(request.url.path.endswith(TASK_ID) for request in self.requests))
        create.assert_called_once_with()

    def test_response_recording_failure_keeps_job_queued_and_can_resume_get(self):
        owner, poller, attempt, create = self.poller()
        with patch.object(retrieve_module, "publish_bytes", side_effect=StorageError("fixture write failure")):
            result = poller.poll_once(attempt, lambda task: self.retrieve(task_id=task), deadline_unix=1100.0)
        self.assertEqual((result.state, result.poll_status), ("QUEUED", "POLL_ERROR"))
        self.assertFalse(self.evidence.exists())
        self.assertEqual(poller.poll_once(attempt, lambda task: self.retrieve(task_id=task), deadline_unix=1100.0).state, "RUNNING")
        self.assertEqual([request.method for request in self.requests], ["GET", "GET"])
        create.assert_called_once_with()
