"""Exercise both SDK upload requests with fake HTTP transports."""

from dataclasses import replace
from email.parser import BytesParser
from email.policy import default
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import tracemalloc
import unittest
from unittest.mock import patch

import httpx
import runwayml.resources.uploads as sdk_uploads

from core.storage import StorageError, publish_bytes, read_verified
from features.inference import runway as runway_module
from features.inference import upload as upload_module
from features.inference.runway import ProviderError
from features.inference.upload import upload_frozen_artifact


API_KEY = "fixture-provider-credential"
UPLOAD_HOST = "uploads.example.com"
UPLOAD_URL = "https://uploads.example.com/incoming"
URI = "runway://fixture-upload"
BODY = {"runwayUri": URI, "uploadUrl": UPLOAD_URL, "fields": {"key": "fixture-object", "policy": "fixture-policy"}}


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


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.evidence = self.root / "evidence"
        self.contents = bytes(range(256)) * 4
        self.source = publish_bytes(self.root, "frozen/guide_B.mp4", self.contents)
        self.api_requests = []
        self.upload_requests = []
        self.upload_clients = []
        self.api_response = lambda _: httpx.Response(200, json=BODY)
        self.upload_response = lambda _: httpx.Response(204)
        self.times = [1000.0, 1002.0]

        def upload_client(**arguments):
            client = httpx.Client(transport=httpx.MockTransport(self.handle_upload), **arguments)
            self.upload_clients.append(client)
            return client

        # The SDK creates a second HTTPX client internally. Replace only its
        # module reference, so both real SDK requests use fake HTTP transports.
        sdk_httpx = SimpleNamespace(**(vars(httpx) | {"Client": upload_client}))
        self.sdk_patch = patch.object(sdk_uploads, "httpx", sdk_httpx)
        self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)

    def handle_api(self, request):
        self.api_requests.append(request)
        return self.api_response(request)

    def handle_upload(self, request):
        self.upload_requests.append(request)
        return self.upload_response(request)

    def upload(self, source=None, **arguments):
        now = iter(self.times)
        return upload_frozen_artifact(self.root, source or self.source, API_KEY, self.evidence,
                                      allowed_upload_hosts=frozenset({UPLOAD_HOST}),
                                      transport=httpx.MockTransport(self.handle_api),
                                      clock=lambda: next(now), **arguments)

    def multipart(self):
        request = self.upload_requests[-1]
        body = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content
        message = BytesParser(policy=default).parsebytes(body)
        return {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}

    def test_sdk_uploads_exact_bytes_and_records_mapping_without_credentials(self):
        body = json.dumps({**BODY, "extra": "x" * (70 * 1024)}).encode()
        stream = TrackedStream(body[start:start + 37] for start in range(0, len(body), 37))
        self.api_response = lambda _: httpx.Response(200, stream=stream)
        result = self.upload()
        self.assertTrue(stream.closed)
        self.assertEqual((len(self.api_requests), len(self.upload_requests)), (1, 1))
        api = self.api_requests[0]
        self.assertEqual((api.method, str(api.url)), ("POST", "https://api.dev.runwayml.com/v1/uploads"))
        self.assertEqual(json.loads(api.content), {"filename": "guide_B.mp4", "type": "ephemeral"})
        self.assertEqual(api.headers["authorization"], "Bearer " + API_KEY)
        self.assertEqual(api.headers["accept-encoding"], "identity")
        request = self.upload_requests[0]
        self.assertEqual((request.method, str(request.url)), ("POST", UPLOAD_URL))
        self.assertNotIn("authorization", request.headers)
        self.assertEqual(request.extensions["timeout"], {"connect": 10.0, "read": 60.0, "write": 60.0, "pool": 60.0})
        parts = self.multipart()
        self.assertEqual(set(parts), {"key", "policy", "file"})
        self.assertEqual(parts["file"].get_filename(), "guide_B.mp4")
        self.assertEqual(parts["file"].get_payload(decode=True), self.contents)
        self.assertEqual(parts["policy"].get_payload(decode=True), b"fixture-policy")
        self.assertEqual(result.source, self.source)
        self.assertEqual((result.uri, result.started_unix, result.completed_unix, result.expires_unix),
                         (URI, 1000.0, 1002.0, 87400.0))
        body = read_verified(self.evidence, result.evidence)
        record = json.loads(body)
        self.assertEqual(record["source"]["sha256"], self.source.sha256)
        self.assertEqual(record["sdkVersion"], "5.24.0")
        self.assertEqual(record["uri"], URI)
        for secret in [API_KEY, UPLOAD_URL, "fixture-policy"]:
            self.assertNotIn(secret.encode(), body)
        self.assertTrue(all(client.is_closed for client in self.upload_clients))

    def test_file_change_after_verification_cannot_change_uploaded_bytes(self):
        def change_file(request):
            (self.root / self.source.path).write_bytes(b"changed" * 100)
            return httpx.Response(200, json=BODY)

        self.api_response = change_file
        result = self.upload()
        self.assertEqual(self.multipart()["file"].get_payload(decode=True), self.contents)
        self.assertEqual(result.source.sha256, self.source.sha256)

    def test_invalid_or_damaged_source_cannot_start_upload(self):
        cases = [replace(self.source, size=511), replace(self.source, size=200_000_001),
                 replace(self.source, size=True), replace(self.source, path="frozen/source.mov"),
                 replace(self.source, path="missing.mp4"), replace(self.source, sha256="0" * 64)]
        for source in cases:
            with self.subTest(source=source), self.assertRaises(ProviderError):
                self.upload(source)
        (self.root / self.source.path).write_bytes(b"x" * len(self.contents))
        with self.assertRaises(ProviderError):
            self.upload()
        self.assertEqual(self.api_requests, [])
        self.assertEqual(self.upload_requests, [])

    def test_api_errors_have_no_upload_or_retry(self):
        for status in [307, 401, 408, 429, 500, 503]:
            with self.subTest(status=status):
                self.api_requests.clear()
                self.api_response = lambda _, status=status: httpx.Response(status, content=API_KEY.encode(),
                                                                           headers={"Location": UPLOAD_URL})
                with self.assertRaises(ProviderError) as caught:
                    self.upload()
                self.assertEqual(len(self.api_requests), 1)
                self.assertEqual(self.upload_requests, [])
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertFalse(self.evidence.exists())

    def test_failed_multipart_upload_has_no_retry_or_success_record(self):
        for status in [307, 403, 408, 429, 500]:
            with self.subTest(status=status):
                self.api_requests.clear()
                self.upload_requests.clear()
                self.upload_response = lambda _, status=status: httpx.Response(status, content=API_KEY.encode(),
                                                                              headers={"Location": UPLOAD_URL})
                with self.assertRaises(ProviderError) as caught:
                    self.upload()
                self.assertEqual((len(self.api_requests), len(self.upload_requests)), (1, 1))
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assertFalse(self.evidence.exists())
        self.assertTrue(all(client.is_closed for client in self.upload_clients))

    def test_invalid_provider_uri_or_destination_is_rejected_before_multipart(self):
        invalid = [{**BODY, "runwayUri": "https://example.com/file"}, {**BODY, "runwayUri": None},
                   {**BODY, "runwayUri": "runway://" + API_KEY},
                   *[{**BODY, "uploadUrl": url} for url in ["http://uploads.example.com/file",
                       "https://other.example.com/file", "https://uploads.example.com:444/file",
                       "https://user:secret@uploads.example.com/file", UPLOAD_URL + "#fragment"]]]
        for data in invalid:
            with self.subTest(data=data):
                self.api_response = lambda _, data=data: httpx.Response(200, json=data)
                with self.assertRaises(ProviderError):
                    self.upload()
        duplicate = json.dumps(BODY).replace('"runwayUri":', '"runwayUri":"runway://other","runwayUri":')
        self.api_response = lambda _: httpx.Response(200, content=duplicate)
        with self.assertRaises(ProviderError):
            self.upload()
        self.assertEqual(self.upload_requests, [])
        self.assertFalse(self.evidence.exists())

    def test_explicit_host_set_is_required_before_network(self):
        for hosts in [UPLOAD_HOST, [UPLOAD_HOST], frozenset(), frozenset({None, UPLOAD_HOST})]:
            with self.subTest(hosts=hosts), self.assertRaises(ProviderError):
                upload_frozen_artifact(self.root, self.source, API_KEY, self.evidence,
                                       allowed_upload_hosts=hosts, transport=httpx.MockTransport(self.handle_api))
        self.assertEqual(self.api_requests, [])

    def test_large_metadata_stops_streaming_before_multipart_upload(self):
        body = json.dumps({**BODY, "extra": "x" * (2 * 1024 * 1024)}).encode()
        stream = TrackedStream(body[start:start + 64 * 1024] for start in range(0, len(body), 64 * 1024))
        self.api_response = lambda _: httpx.Response(200, stream=stream)
        with self.assertRaises(ProviderError):
            self.upload()
        self.assertLessEqual(stream.chunks_read, 17)
        self.assertTrue(stream.closed)
        self.assertEqual(self.upload_requests, [])
        self.assertFalse(self.evidence.exists())

    def test_encoded_metadata_is_rejected_before_reading_or_uploading(self):
        stream = TrackedStream([gzip.compress(json.dumps(BODY).encode())])
        self.api_response = lambda _: httpx.Response(200, stream=stream, headers={"Content-Encoding": "gzip"})
        with self.assertRaises(ProviderError):
            self.upload()
        self.assertEqual(stream.chunks_read, 0)
        self.assertTrue(stream.closed)
        self.assertEqual(self.upload_requests, [])
        self.assertFalse(self.evidence.exists())

    def test_fragmented_metadata_stays_below_memory_limit(self):
        body = json.dumps({**BODY, "extra": "x" * 100_000}).encode()
        stream = TrackedStream(body[index:index + 1] for index in range(len(body)))
        self.api_response = lambda _: httpx.Response(200, stream=stream)
        tracemalloc.start()
        try:
            result = self.upload()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result.uri, URI)
        self.assertTrue(stream.closed)
        # Buffering each one-byte chunk before a join used over 8 MiB here.
        self.assertLess(peak, 4 * 1024 * 1024)

    def test_invalid_clock_or_expired_uri_cannot_create_success_record(self):
        for started in [True, -1, float("nan"), float("inf")]:
            with self.subTest(started=started):
                self.times = [started]
                with self.assertRaises(ProviderError):
                    self.upload()
        self.assertEqual(self.api_requests, [])
        for completed in [True, 999.0, 87400.0, float("nan"), float("inf")]:
            with self.subTest(completed=completed):
                self.times = [1000.0, completed]
                with self.assertRaises(ProviderError):
                    self.upload()
                self.assertFalse(self.evidence.exists())

    def test_timeout_is_safe_and_does_not_repeat_either_request(self):
        def timeout(request):
            raise httpx.ReadTimeout(API_KEY, request=request)

        for stage in ["api", "multipart"]:
            with self.subTest(stage=stage):
                self.api_requests.clear()
                self.upload_requests.clear()
                self.api_response = timeout if stage == "api" else lambda _: httpx.Response(200, json=BODY)
                self.upload_response = timeout
                with self.assertRaises(ProviderError) as caught:
                    self.upload()
                self.assertEqual(str(caught.exception), "Upload or evidence recording failed; no generation was submitted.")
                self.assertEqual(len(self.api_requests), 1)
                self.assertEqual(len(self.upload_requests), 0 if stage == "api" else 1)
                self.assertFalse(self.evidence.exists())

    def test_evidence_write_failure_cannot_return_an_uploaded_record(self):
        with patch.object(upload_module, "publish_bytes", side_effect=StorageError("fixture write failed")):
            with self.assertRaises(ProviderError):
                self.upload()
        self.assertEqual((len(self.api_requests), len(self.upload_requests)), (1, 1))
        self.assertFalse(self.evidence.exists())

    def test_existing_evidence_is_verified_without_replacement(self):
        result = self.upload()
        path = self.evidence / result.evidence.path
        modified = path.stat().st_mtime_ns
        self.assertEqual(self.upload(), result)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        path.write_bytes(b"damaged fixture evidence")
        with self.assertRaises(ProviderError):
            self.upload()
        self.assertEqual(path.read_bytes(), b"damaged fixture evidence")

    def test_missing_key_or_wrong_sdk_version_cannot_reach_network(self):
        with patch.dict("os.environ", {"RUNWAYML_API_SECRET": API_KEY}):
            with self.assertRaises(ProviderError):
                upload_frozen_artifact(self.root, self.source, "", self.evidence,
                                       allowed_upload_hosts=frozenset({UPLOAD_HOST}),
                                       transport=httpx.MockTransport(self.handle_api))
        with patch.object(runway_module, "version", return_value="wrong-version"):
            with self.assertRaises(ProviderError):
                self.upload()
        self.assertEqual(self.api_requests, [])
        self.assertEqual(self.upload_requests, [])
