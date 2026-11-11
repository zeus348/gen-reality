"""Check output GET boundaries with HTTPX and local synthetic video."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import tracemalloc
import unittest
from unittest.mock import Mock, patch

import httpx

from core.budget import BudgetLedger
from core.storage import publish_bytes, read_verified
from features.inference.polling import TaskPoller
from features.inference.retrieve import retrieve_task_once
from features.inference.submission import SubmissionCoordinator
from features.output.complete import OutputCompleter, OutputError
from features.output import fetch as fetch_module
from features.output.fetch import DownloadError, download_task_output


TASK_ID = "c26d7283-d514-4492-8589-760dfe6f0289"
HOST = "output.example.com"
URL = "https://output.example.com/raw.mp4?signature=fixture-private-token"
API_KEY = "fixture-provider-credential"


class TrackedStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.chunks_read = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.chunks_read += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


class OutputFetchTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.evidence = self.root / "evidence"
        self.requests = []
        self.data = {"id": TASK_ID, "status": "SUCCEEDED", "output": [URL]}

    def save(self, data=None, *, body=None):
        if body is None:
            body = json.dumps(self.data if data is None else data).encode()
        digest = sha256(body).hexdigest()
        publish_bytes(self.evidence, f"provider_responses/{digest}.json", body)
        return digest

    def download(self, digest, response, **arguments):
        def handle(request):
            self.requests.append(request)
            return response(request) if callable(response) else response

        return download_task_output(TASK_ID, digest, self.evidence,
                                    allowed_hosts=frozenset({HOST}),
                                    transport=httpx.MockTransport(handle), **arguments)

    def test_one_get_uses_first_output_and_sends_no_provider_credentials(self):
        self.data["output"].append("https://output.example.com/second.mp4")
        digest = self.save()
        stream = TrackedStream([b"first\x00", b"second\xff"])
        with patch.dict("os.environ", {"RUNWAYML_API_SECRET": API_KEY}):
            result = self.download(digest, httpx.Response(200, stream=stream))
        self.assertEqual(result, b"first\x00second\xff")
        self.assertTrue(stream.closed)
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual((request.method, str(request.url)), ("GET", URL))
        self.assertNotIn("authorization", request.headers)
        self.assertNotIn("cookie", request.headers)
        self.assertNotIn(API_KEY, str(request.headers))
        self.assertEqual(request.headers["accept-encoding"], "identity")
        self.assertEqual(request.extensions["timeout"], {"connect": 10.0, "read": 30.0, "write": 30.0, "pool": 30.0})

    def test_damaged_or_linked_evidence_cannot_reach_network(self):
        digest = self.save()
        path = self.evidence / "provider_responses" / (digest + ".json")
        original = path.read_bytes()
        path.write_bytes(original.replace(b"fixture-private-token", b"altered-private-token"))
        with self.assertRaises(DownloadError):
            self.download(digest, httpx.Response(200, content=b"video"))
        path.unlink()
        target = self.root / "linked.json"
        target.write_bytes(original)
        path.symlink_to(target)
        with self.assertRaises(DownloadError):
            self.download(digest, httpx.Response(200, content=b"video"))
        self.assertEqual(self.requests, [])

    def test_invalid_success_evidence_cannot_reach_network(self):
        invalid = [[], {**self.data, "id": "other-task"}, {**self.data, "status": "RUNNING"},
                   {**self.data, "output": []}, {**self.data, "output": URL},
                   {**self.data, "output": [None]}]
        for data in invalid:
            with self.subTest(data=data):
                with self.assertRaises(DownloadError):
                    self.download(self.save(data), httpx.Response(200, content=b"video"))
        duplicate = ('{"id":"wrong","id":"' + TASK_ID + '","status":"SUCCEEDED","output":["' + URL + '"]}').encode()
        for body in [b"not JSON", duplicate, b"", b" " * (fetch_module.MAX_TASK_RESPONSE_BYTES + 1)]:
            with self.subTest(body_length=len(body)):
                with self.assertRaises(DownloadError):
                    self.download(self.save(body=body), httpx.Response(200, content=b"video"))
        for digest in [None, "../evidence", "a" * 63]:
            with self.subTest(digest=digest):
                with self.assertRaises(DownloadError):
                    self.download(digest, httpx.Response(200, content=b"video"))
        self.assertEqual(self.requests, [])

    def test_unapproved_or_unsafe_urls_cannot_reach_network(self):
        urls = ["http://output.example.com/raw.mp4", "https://localhost/raw.mp4",
                "https://output.example.com.attacker.test/raw.mp4", "//output.example.com/raw.mp4",
                "https://output.example.com:444/raw.mp4", "https://user:secret@output.example.com/raw.mp4",
                URL + "#fragment", "https://output.example.com/raw\n.mp4", URL + "x" * 8192,
                "https://127.0.0.1/raw.mp4", "https://[::1]/raw.mp4"]
        transport = httpx.MockTransport(Mock(return_value=httpx.Response(200, content=b"video")))
        for url in urls:
            with self.subTest(url=url):
                digest = self.save({**self.data, "output": [url]})
                with self.assertRaises(DownloadError):
                    download_task_output(TASK_ID, digest, self.evidence,
                                         allowed_hosts=frozenset({HOST, "127.0.0.1", "::1"}), transport=transport)
        transport.handler.assert_not_called()

    def test_host_allowlist_must_be_an_explicit_server_set(self):
        digest = self.save()
        transport = httpx.MockTransport(Mock(return_value=httpx.Response(200, content=b"video")))
        for allowed in [HOST, [HOST], frozenset(), frozenset({HOST, None})]:
            with self.subTest(allowed=allowed):
                with self.assertRaises(DownloadError):
                    download_task_output(TASK_ID, digest, self.evidence, allowed_hosts=allowed, transport=transport)
        transport.handler.assert_not_called()

    def test_http_errors_and_redirects_have_no_retries_or_error_details(self):
        digest = self.save()
        for status in [206, 301, 302, 307, 308, 401, 403, 408, 429, 500, 503]:
            with self.subTest(status=status):
                self.requests.clear()
                stream = TrackedStream([URL.encode()])
                with self.assertRaises(DownloadError) as caught:
                    self.download(digest, httpx.Response(status, stream=stream, headers={"Location": URL}))
                self.assertEqual(len(self.requests), 1)
                self.assertEqual(stream.chunks_read, 0)
                self.assertTrue(stream.closed)
                self.assertNotIn("fixture-private-token", str(caught.exception))
                self.assertIsNone(caught.exception.__cause__)

    def test_encoding_and_declared_size_are_checked_before_reading(self):
        digest = self.save()
        headers = [{"Content-Encoding": "gzip"}, {"Content-Encoding": "deflate"},
                   {"Content-Length": "0"}, {"Content-Length": "-1"},
                   {"Content-Length": "65"}, {"Content-Length": "invalid"}]
        for header in headers:
            with self.subTest(header=header), patch.object(fetch_module, "MAX_RAW_BYTES", 64):
                stream = TrackedStream([b"video"])
                with self.assertRaises(DownloadError):
                    self.download(digest, httpx.Response(200, stream=stream, headers=header))
                self.assertEqual(stream.chunks_read, 0)
                self.assertTrue(stream.closed)

    def test_stream_size_limit_stops_reading_and_closes_response(self):
        digest = self.save()
        stream = TrackedStream([b"x" * 8] * 10)
        with patch.object(fetch_module, "MAX_RAW_BYTES", 32), self.assertRaises(DownloadError):
            self.download(digest, httpx.Response(200, stream=stream))
        self.assertEqual(stream.chunks_read, 5)
        self.assertTrue(stream.closed)

    def test_many_small_chunks_do_not_amplify_buffer_memory(self):
        digest = self.save()
        stream = TrackedStream(b"x" for _ in range(100_000))
        tracemalloc.start()
        try:
            result = self.download(digest, httpx.Response(200, stream=stream))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result, b"x" * 100_000)
        self.assertTrue(stream.closed)
        # A list and join used over 8 MiB for this 100 KiB fragmented body.
        self.assertLess(peak, 4 * 1024 * 1024)

    def test_deadline_is_checked_between_small_chunks(self):
        digest = self.save()
        stream = TrackedStream([b"x"] * 10)
        now = iter([0.0, 1.0, 2.0] + [61.0] * 20)
        with self.assertRaises(DownloadError):
            self.download(digest, httpx.Response(200, stream=stream), clock=lambda: next(now))
        self.assertEqual(stream.chunks_read, 3)
        self.assertTrue(stream.closed)

    def test_empty_incomplete_or_late_body_is_rejected(self):
        digest = self.save()
        for chunks, headers, times in [([], {}, [0.0, 1.0]),
                                      ([b"short"], {"Content-Length": "6"}, [0.0, 1.0, 2.0]),
                                      ([b"late"], {}, [0.0, 1.0, 61.0])]:
            with self.subTest(chunks=chunks, headers=headers, times=times):
                stream = TrackedStream(chunks)
                now = iter(times)
                with self.assertRaises(DownloadError):
                    self.download(digest, httpx.Response(200, stream=stream, headers=headers), clock=lambda: next(now))
                self.assertTrue(stream.closed)

    def test_stream_timeout_closes_response_and_hides_signed_url(self):
        digest = self.save()
        stream = TrackedStream([b"part", httpx.ReadTimeout(URL)])
        with self.assertRaises(DownloadError) as caught:
            self.download(digest, httpx.Response(200, stream=stream))
        self.assertEqual(str(caught.exception), "Output GET failed; resume the same task.")
        self.assertTrue(stream.closed)
        self.assertEqual(len(self.requests), 1)

    def test_failed_download_resumes_saved_success_and_completes_without_another_create(self):
        path = self.root / "fixture.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                        "color=green:s=64x36:r=30:d=5", "-c:v", "libx264", "-threads", "2",
                        "-pix_fmt", "yuv420p", str(path)], capture_output=True, check=True, timeout=30)
        video = path.read_bytes()
        ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1000.0)
        ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=140, max_attempts=1, cost_cents=140)
        attempt = ledger.reserve("run-1", "first", "1" * 64).attempt_id
        owner = SubmissionCoordinator(ledger)
        self.addCleanup(owner.close)
        create = Mock(return_value=TASK_ID)
        owner.submit_once(attempt, create)
        observation = retrieve_task_once(TASK_ID, API_KEY, self.evidence,
                                        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=self.data)))
        TaskPoller(owner).poll_once(attempt, lambda _: observation, deadline_unix=1100.0)
        completer = OutputCompleter(owner, self.root / "runs")

        def download(task, evidence):
            self.assertEqual((task, evidence), (TASK_ID, observation.evidence_sha256))
            return self.download(evidence, response)

        response = httpx.Response(403, content=b"expired fixture URL")
        with self.assertRaises(OutputError):
            completer.complete(attempt, download)
        self.assertEqual(owner.get(attempt).state, "QUEUED")
        self.assertIsNone(completer.get(attempt))
        response = httpx.Response(200, stream=TrackedStream([video]))
        result = completer.complete(attempt, download)
        self.assertEqual(owner.get(attempt).state, "SUCCEEDED")
        self.assertEqual(result.provider_evidence_sha256, observation.evidence_sha256)
        self.assertEqual(read_verified(self.root / "runs/run-1", result.raw), video)
        metadata = json.loads(read_verified(self.root / "runs/run-1", result.metadata))
        self.assertEqual(metadata["automatedWorkflowComplete"], "NOT_RUN")
        self.assertEqual(metadata["visualEffectDemonstrated"], "NOT_RUN")
        with ledger._connection() as connection:
            self.assertEqual(ledger._usage(connection, "run-1"), (1, 140, False))
        self.assertEqual([request.method for request in self.requests], ["GET", "GET"])
        self.assertTrue(all(str(request.url) == URL for request in self.requests))
        create.assert_called_once_with()
