"""Inspect requests from the installed SDK without contacting Runway."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import httpx

from core.budget import BudgetExceeded, BudgetLedger
from core.config import load_config
from features.guidance.video import DEFAULT_PROMPT
from features.inference import runway as adapter_module
from features.inference.runway import ProviderError, RunwayCreateAdapter
from features.inference.submission import SubmissionCoordinator


VIDEO_URI = "runway://fixture-video"
KEYFRAME_URI = "runway://fixture-keyframe"
TASK_ID = "c26d7283-d514-4492-8589-760dfe6f0289"
API_KEY = "test-credential-do-not-record"


class RunwayTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.bodies = []

    def adapter(self, response=None, recorder=None):
        def handle(request):
            self.requests.append(request)
            if callable(response):
                return response(request)
            return response or httpx.Response(200, json={"id": TASK_ID})

        adapter = RunwayCreateAdapter(API_KEY, record_request=recorder or self.bodies.append,
                                       transport=httpx.MockTransport(handle))
        self.addCleanup(adapter.close)
        return adapter

    def test_modes_send_exact_payloads_through_the_pinned_sdk(self):
        self.assertEqual(version("runwayml"), "5.24.0")
        self.assertEqual(load_config()["sdk"]["version"], version("runwayml"))
        adapter = self.adapter()
        self.assertEqual(adapter.create_task(VIDEO_URI, DEFAULT_PROMPT, keyframe_uri=KEYFRAME_URI), TASK_ID)
        self.assertEqual(adapter.create_task(VIDEO_URI, DEFAULT_PROMPT), TASK_ID)
        expected = {"model": "aleph2", "videoUri": VIDEO_URI, "promptText": DEFAULT_PROMPT}
        self.assertEqual(json.loads(self.bodies[0]),
                         {**expected, "keyframes": [{"uri": KEYFRAME_URI, "seconds": 0}]})
        self.assertEqual(json.loads(self.bodies[1]), expected)
        for request in self.requests:
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), "https://api.dev.runwayml.com/v1/video_to_video")
            self.assertEqual(request.headers["authorization"], "Bearer " + API_KEY)

    def test_recorder_gets_exact_body_before_send_without_credentials(self):
        def respond(request):
            self.assertEqual(self.bodies, [request.content])
            self.assertNotIn(API_KEY.encode(), self.bodies[0])
            self.assertNotIn(b"Authorization", self.bodies[0])
            self.assertEqual(sha256(self.bodies[0]).digest(), sha256(request.content).digest())
            return httpx.Response(200, json={"id": TASK_ID})

        adapter = self.adapter(response=respond)
        self.assertEqual(adapter.create_task(VIDEO_URI, DEFAULT_PROMPT), TASK_ID)

    def test_retryable_http_errors_make_one_post_each(self):
        for status in [408, 409, 429, 500, 503]:
            with self.subTest(status=status):
                self.requests.clear()
                adapter = self.adapter(httpx.Response(status, json={"error": API_KEY}))
                with self.assertRaises(ProviderError) as caught:
                    adapter.create_task(VIDEO_URI, DEFAULT_PROMPT)
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertEqual(len(self.requests), 1)

    def test_connection_failures_and_timeouts_make_one_post_each(self):
        for error in [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout]:
            with self.subTest(error=error.__name__):
                self.requests.clear()

                def fail(request):
                    raise error(API_KEY, request=request)

                adapter = self.adapter(response=fail)
                with self.assertRaises(ProviderError) as caught:
                    adapter.create_task(VIDEO_URI, DEFAULT_PROMPT)
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertEqual(len(self.requests), 1)

    def test_redirect_cannot_repeat_the_paid_post(self):
        def redirect_once(request):
            if len(self.requests) == 1:
                return httpx.Response(307, headers={"Location": "/v1/another-create"})
            return httpx.Response(200, json={"id": TASK_ID})

        adapter = self.adapter(response=redirect_once)
        with self.assertRaises(ProviderError):
            adapter.create_task(VIDEO_URI, DEFAULT_PROMPT)
        self.assertEqual(len(self.requests), 1)

    def test_invalid_inputs_cannot_reach_the_transport(self):
        adapter = self.adapter()
        invalid_uris = [None, "", "https://example.com/video", "runway://", "runway://bad uri",
                        "runway://id?secret=value", "runway://" + "a" * 2001]
        for uri in invalid_uris:
            with self.subTest(video_uri=uri):
                with self.assertRaises(ProviderError):
                    adapter.create_task(uri, DEFAULT_PROMPT)
            if uri is not None:
                with self.subTest(keyframe_uri=uri):
                    with self.assertRaises(ProviderError):
                        adapter.create_task(VIDEO_URI, DEFAULT_PROMPT, keyframe_uri=uri)
        for prompt in [None, "", " ", "x" * 1001]:
            with self.subTest(prompt=prompt):
                with self.assertRaises(ProviderError):
                    adapter.create_task(VIDEO_URI, prompt)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.bodies, [])

    def test_unverified_sdk_version_is_blocked(self):
        with patch.object(adapter_module, "version", return_value="unverified"):
            with self.assertRaisesRegex(ProviderError, "pinned Runway SDK"):
                self.adapter()
        self.assertEqual(self.requests, [])

    def test_missing_key_is_blocked_without_using_environment_credentials(self):
        with patch.dict("os.environ", {"RUNWAYML_API_SECRET": "fixture-env-key"}):
            for key in [None, "", " "]:
                with self.subTest(key=key):
                    with self.assertRaisesRegex(ProviderError, "API key is required"):
                        adapter = RunwayCreateAdapter(key, record_request=Mock(),
                                                       transport=httpx.MockTransport(Mock()))
                        self.addCleanup(adapter.close)

    def test_recording_failure_prevents_network_submission(self):
        recorder = Mock(side_effect=OSError(API_KEY))
        adapter = self.adapter(recorder=recorder)
        with self.assertRaises(ProviderError) as caught:
            adapter.create_task(VIDEO_URI, DEFAULT_PROMPT)
        self.assertNotIn(API_KEY, str(caught.exception))
        recorder.assert_called_once()
        self.assertEqual(self.requests, [])

    def test_missing_provider_task_id_is_uncertain(self):
        for response in [{}, {"id": None}, {"id": ""}]:
            with self.subTest(response=response):
                self.requests.clear()
                adapter = self.adapter(httpx.Response(200, json=response))
                with self.assertRaises(ProviderError):
                    adapter.create_task(VIDEO_URI, DEFAULT_PROMPT)
                self.assertEqual(len(self.requests), 1)

    def test_sdk_timeout_stays_unknown_across_coordinator_restart(self):
        def timeout(request):
            raise httpx.ReadTimeout("fixture timeout", request=request)

        adapter = self.adapter(response=timeout)
        with TemporaryDirectory() as folder:
            ledger = BudgetLedger(Path(folder) / "budget.sqlite", clock=lambda: 1000.0)
            ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=140,
                               max_attempts=1, cost_cents=140)
            attempt = ledger.reserve("run-1", "fixture-key", sha256(b"frozen inputs").hexdigest())
            create = lambda: adapter.create_task(VIDEO_URI, DEFAULT_PROMPT)
            for _ in range(2):
                with SubmissionCoordinator(ledger) as owner:
                    result = owner.submit_once(attempt.attempt_id, create)
                    self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
            self.assertEqual(len(self.requests), 1)
            with self.assertRaises(BudgetExceeded):
                ledger.reserve("run-1", "another-key", attempt.payload_sha256)
