"""Check account evidence through the installed SDK with fake HTTP only."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tracemalloc
import unittest
from unittest.mock import patch

import httpx

from core.storage import StorageError, read_verified
from features.inference import account as account_module
from features.inference import runway as runway_module
from features.inference.account import check_account_once
from features.inference.runway import ProviderError


API_KEY = "fixture-provider-credential"
DATA = {"creditBalance": 500,
        "tier": {"maxMonthlyCreditSpend": 10000,
                 "models": {"aleph2": {"maxConcurrentGenerations": 5, "maxDailyGenerations": 20}}},
        "usage": {"models": {"aleph2": {"dailyGenerations": 3}}}}
BODY = json.dumps(DATA, indent=2).encode()


class TrackedStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.chunks_read = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.chunks_read += 1
            yield chunk

    def close(self):
        self.closed = True


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "evidence"
        self.requests = []
        self.response = lambda _: httpx.Response(200, content=BODY)
        self.times = [1000.0, 1001.0]

    def handle(self, request):
        self.requests.append(request)
        return self.response(request)

    def check(self, required=140, **arguments):
        now = iter(self.times)
        return check_account_once(API_KEY, required, self.root,
                                  transport=httpx.MockTransport(self.handle),
                                  clock=lambda: next(now), **arguments)

    def test_one_sdk_get_saves_response_and_summary_without_spending_authority(self):
        result = self.check()
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual((request.method, str(request.url)), ("GET", "https://api.dev.runwayml.com/v1/organization"))
        self.assertEqual(request.headers["authorization"], "Bearer " + API_KEY)
        self.assertEqual(request.headers["accept-encoding"], "identity")
        self.assertEqual(set(request.extensions["timeout"].values()), {10.0})
        self.assertEqual((result.checked_unix, result.credit_balance, result.required_credits), (1000.0, 500, 140))
        self.assertEqual(result.blocked_reasons, ())
        result.require_current(1001.0)
        self.assertEqual(read_verified(self.root, result.response), BODY)
        summary = read_verified(self.root, result.record)
        data = json.loads(summary)
        self.assertEqual(data["response"]["sha256"], result.response.sha256)
        self.assertEqual((data["sdkVersion"], data["model"]), ("5.24.0", "aleph2"))
        self.assertIs(data["spendingAuthorized"], False)
        self.assertNotIn(API_KEY.encode(), summary)
        self.assertNotIn(API_KEY.encode(), read_verified(self.root, result.response))

    def test_balance_and_model_limits_block_the_account_check(self):
        cases = []
        low = deepcopy(DATA)
        low["creditBalance"] = 139
        cases.append((low, "INSUFFICIENT_CREDITS"))
        unavailable = deepcopy(DATA)
        unavailable["tier"]["models"] = {}
        cases.append((unavailable, "MODEL_UNAVAILABLE"))
        disabled = deepcopy(DATA)
        disabled["tier"]["models"]["aleph2"]["maxConcurrentGenerations"] = 0
        cases.append((disabled, "MODEL_CONCURRENCY_DISABLED"))
        exhausted = deepcopy(DATA)
        exhausted["usage"]["models"]["aleph2"]["dailyGenerations"] = 20
        cases.append((exhausted, "MODEL_DAILY_LIMIT_REACHED"))
        for data, reason in cases:
            with self.subTest(reason=reason):
                self.response = lambda _, data=data: httpx.Response(200, json=data)
                result = self.check()
                self.assertEqual(result.blocked_reasons, (reason,))
                with self.assertRaises(ProviderError):
                    result.require_current(1001.0)
                self.assertEqual(json.loads(read_verified(self.root, result.record))["blockedReasons"], [reason])

    def test_explicit_null_limits_are_unlimited(self):
        data = deepcopy(DATA)
        data["creditBalance"] = 140
        data["tier"]["models"]["aleph2"] = {"maxConcurrentGenerations": None, "maxDailyGenerations": None}
        data["usage"]["models"]["aleph2"]["dailyGenerations"] = 1_000_000
        self.response = lambda _: httpx.Response(200, json=data)
        result = self.check()
        self.assertEqual(result.blocked_reasons, ())
        result.require_current(1001.0)

    def test_invalid_quote_key_or_sdk_version_cannot_reach_network(self):
        for required in [True, 0, -1, 1.5, "140", 2**63]:
            with self.subTest(required=required), self.assertRaises(ProviderError):
                self.check(required)
        with patch.dict("os.environ", {"RUNWAYML_API_SECRET": API_KEY}), self.assertRaises(ProviderError):
            check_account_once("", 140, self.root, transport=httpx.MockTransport(self.handle))
        with patch.object(runway_module, "version", return_value="wrong"), self.assertRaises(ProviderError):
            self.check()
        self.assertEqual(self.requests, [])

    def test_invalid_counts_or_missing_model_usage_fail_closed(self):
        cases = []
        for value in [True, -1, 2**63, "500", 500.5, None]:
            data = deepcopy(DATA)
            data["creditBalance"] = value
            cases.append(data)
        for field in ["maxConcurrentGenerations", "maxDailyGenerations"]:
            data = deepcopy(DATA)
            data["tier"]["models"]["aleph2"][field] = True
            cases.append(data)
            data = deepcopy(DATA)
            del data["tier"]["models"]["aleph2"][field]
            cases.append(data)
        missing = deepcopy(DATA)
        missing["usage"]["models"] = {}
        cases.append(missing)
        invalid_usage = deepcopy(DATA)
        invalid_usage["usage"]["models"]["aleph2"]["dailyGenerations"] = -1
        cases.append(invalid_usage)
        for data in [[], {}, *cases]:
            with self.subTest(data=data):
                self.response = lambda _, data=data: httpx.Response(200, json=data)
                with self.assertRaises(ProviderError):
                    self.check()
                self.assertFalse(self.root.exists())

    def test_ambiguous_nonfinite_or_credential_bearing_json_is_not_saved(self):
        bodies = [b"not JSON", BODY.replace(b'"creditBalance": 500', b'"creditBalance":0,"creditBalance":500'),
                  json.dumps({**DATA, "debug": API_KEY}).encode(),
                  json.dumps({**DATA, "debug": API_KEY}).replace(API_KEY, "\\u0066" + API_KEY[1:]).encode(),
                  json.dumps({**DATA, "extra": float("nan")}).encode()]
        for body in bodies:
            with self.subTest(body=body):
                self.response = lambda _, body=body: httpx.Response(200, content=body)
                with self.assertRaises(ProviderError) as caught:
                    self.check()
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertFalse(self.root.exists())

    def test_large_response_stops_streaming_and_closes_without_evidence(self):
        stream = TrackedStream([b" " * (64 * 1024)] * 30)
        self.response = lambda _: httpx.Response(200, stream=stream)
        with self.assertRaises(ProviderError):
            self.check()
        self.assertLessEqual(stream.chunks_read, 17)
        self.assertTrue(stream.closed)
        self.assertFalse(self.root.exists())

    def test_encoded_response_is_rejected_before_reading(self):
        stream = TrackedStream([gzip.compress(BODY)])
        self.response = lambda _: httpx.Response(200, stream=stream, headers={"Content-Encoding": "gzip"})
        with self.assertRaises(ProviderError):
            self.check()
        self.assertEqual(stream.chunks_read, 0)
        self.assertTrue(stream.closed)
        self.assertFalse(self.root.exists())

    def test_small_chunks_obey_the_elapsed_deadline(self):
        stream = TrackedStream([b" "] * 100)
        self.response = lambda _: httpx.Response(200, stream=stream)
        now = iter([0.0, 1.0, 2.0] + [11.0] * 110)
        with self.assertRaises(ProviderError):
            self.check(elapsed_clock=lambda: next(now))
        self.assertEqual(stream.chunks_read, 3)
        self.assertTrue(stream.closed)
        self.assertFalse(self.root.exists())

    def test_fragmented_response_does_not_amplify_buffer_memory(self):
        body = json.dumps({**DATA, "extra": "x" * 100_000}).encode()
        stream = TrackedStream(body[index:index + 1] for index in range(len(body)))
        self.response = lambda _: httpx.Response(200, stream=stream)
        tracemalloc.start()
        try:
            result = self.check()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result.blocked_reasons, ())
        self.assertTrue(stream.closed)
        # A list and join needs over 8 MiB for these one-byte chunks.
        self.assertLess(peak, 4 * 1024 * 1024)

    def test_http_errors_and_timeouts_have_no_retry_or_unsafe_error_text(self):
        for status in [307, 401, 408, 429, 500]:
            with self.subTest(status=status):
                self.requests.clear()
                self.response = lambda _, status=status: httpx.Response(status, content=API_KEY.encode(),
                                                                       headers={"Location": "https://example.com"})
                with self.assertRaises(ProviderError) as caught:
                    self.check()
                self.assertEqual(len(self.requests), 1)
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertFalse(self.root.exists())

        def timeout(request):
            raise httpx.ReadTimeout(API_KEY, request=request)

        self.requests.clear()
        self.response = timeout
        with self.assertRaises(ProviderError) as caught:
            self.check()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(str(caught.exception), "Account check or evidence recording failed; no generation was submitted.")

    def test_summary_write_failure_cannot_return_a_check_and_can_resume_get(self):
        original = account_module.publish_bytes

        def fail_summary(root, path, contents):
            if "/checks/" in path:
                raise StorageError("fixture summary write failure")
            return original(root, path, contents)

        with patch.object(account_module, "publish_bytes", side_effect=fail_summary), self.assertRaises(ProviderError):
            self.check()
        responses = list((self.root / "provider_account/responses").iterdir())
        self.assertEqual(len(responses), 1)
        modified = responses[0].stat().st_mtime_ns
        result = self.check()
        result.require_current(1001.0)
        self.assertEqual(responses[0].stat().st_mtime_ns, modified)
        self.assertEqual([request.method for request in self.requests], ["GET", "GET"])

    def test_repeated_evidence_is_verified_without_replacement(self):
        result = self.check()
        self.assertEqual(self.check(), result)
        for artifact in [result.response, result.record]:
            path = self.root / artifact.path
            original = path.read_bytes()
            path.write_bytes(b"damaged fixture evidence")
            with self.assertRaises(ProviderError):
                self.check()
            self.assertEqual(path.read_bytes(), b"damaged fixture evidence")
            path.write_bytes(original)

    def test_invalid_clock_and_stale_or_future_checks_cannot_be_used(self):
        for started in [True, -1, float("nan"), float("inf")]:
            with self.subTest(started=started):
                self.times = [started]
                with self.assertRaises(ProviderError):
                    self.check()
        self.assertEqual(self.requests, [])
        for completed in [True, 999.0, 1061.0, float("nan"), float("inf")]:
            with self.subTest(completed=completed):
                self.times = [1000.0, completed]
                with self.assertRaises(ProviderError):
                    self.check()
                self.assertFalse(self.root.exists())
        self.times = [1000.0, 1001.0]
        result = self.check()
        result.require_current(1060.0)
        for now in [True, 999.0, 1060.001, float("nan"), float("inf")]:
            with self.subTest(now=now), self.assertRaises(ProviderError):
                result.require_current(now)
