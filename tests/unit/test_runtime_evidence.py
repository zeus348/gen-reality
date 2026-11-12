"""Record local transport evidence without claiming genuine provider provenance."""

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import Mock, patch

import httpx

from app import submit as app_submit, resume as app_resume
from core.budget import BudgetLedger
from core.storage import StorageError, StoredArtifact, read_verified
from features.inference import runtime, runway, retrieve, frozen, send, submission
from features.output import fetch
import test_preflight as fixtures


TASK = "fixture-runtime-task"
HOST = "outputs.example.com"
URL = "https://outputs.example.com/raw.mp4?signature=private-runtime-token"


class RuntimeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PreflightTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.run = self.fixture.run
        self.checked = self.fixture.check()
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1001.0)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                               max_attempts=6, cost_cents=140,
                               deadline_unix_cap=self.fixture.started.deadline_unix)
        self.ledger.reserve("run-1", "runtime fixture", self.checked.frozen.payload_sha256,
                            attempt_id=fixtures.ATTEMPT)
        self.observed_unix = 1001.25
        self.requests = []

    def owner(self):
        owner = submission.SubmissionCoordinator(self.ledger)
        self.addCleanup(owner.close)
        return owner

    def recorder(self, **changes):
        return runtime.RuntimeEvidence(self.root, "run-1", fixtures.ATTEMPT,
                                       **({"clock": lambda: self.observed_unix} | changes))

    def records(self, recorder, kind=None):
        documents = [json.loads(read_verified(self.run, record)) for record in recorder.references]
        return [document for document in documents if kind is None or document["manifestType"] == kind]

    def send(self, owner, recorder=None, *, status=200, transport=None):
        def provider(request):
            self.requests.append(request)
            self.observed_unix = 1002.25
            return httpx.Response(status, json={"id": TASK} if status == 200 else {"error": "fixture rejection"})

        return send.submit_checked(owner, self.checked, self.root, fixtures.KEY,
            transport=transport or httpx.MockTransport(provider), runtime=recorder)

    def task_get(self, recorder=None, *, status="SUCCEEDED"):
        def provider(request):
            self.requests.append(request)
            self.observed_unix = 1003.25
            return httpx.Response(200, json={"id": TASK, "status": status, "output": [URL]})

        return retrieve.retrieve_task_once(TASK, fixtures.KEY, self.run,
            transport=httpx.MockTransport(provider), runtime=recorder)

    def assert_private_values_absent(self, recorder):
        for record in recorder.references:
            contents = read_verified(self.run, record)
            self.assertLessEqual(len(contents), 16 * 1024)
            for secret in (fixtures.KEY, "private-runtime-token", "runway://", "Authorization", URL):
                self.assertNotIn(secret.encode(), contents)
            document = json.loads(contents)
            for field in ("visualEffectDemonstrated", "automatedWorkflowComplete", "humanVerified", "liveVerified"):
                self.assertNotIn(field, document)

    def test_optional_recorder_keeps_service_return_types(self):
        owner = self.owner()
        created = self.send(owner)
        observed = self.task_get()
        payload = b"fixture output bytes"
        downloaded = fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
            allowed_hosts=frozenset({HOST}), transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload)))
        self.assertIsInstance(created, submission.Submission)
        self.assertEqual((created.state, created.provider_task_id), ("QUEUED", TASK))
        self.assertEqual((observed.task_id, observed.status), (TASK, "SUCCEEDED"))
        self.assertEqual(downloaded, payload)
        self.assertFalse((self.run / f"attempts/{fixtures.ATTEMPT}/runtime").exists())

    def test_wrong_recorder_identity_blocks_before_claim_or_transport(self):
        owner = self.owner()
        cases = [(self.root, "other-run", fixtures.ATTEMPT),
                 (self.root, "run-1", "att-" + "b" * 32),
                 (self.root / "other-root", "run-1", fixtures.ATTEMPT)]
        for output_root, run_id, attempt_id in cases:
            recorder = runtime.RuntimeEvidence(output_root, run_id, attempt_id, clock=lambda: 1001.25)
            with self.subTest(run_id=run_id, attempt_id=attempt_id, output_root=output_root):
                with patch.object(recorder, "bind_checked", wraps=recorder.bind_checked) as bind:
                    with self.assertRaises(ValueError):
                        self.send(owner, recorder)
                bind.assert_not_called()
                self.assertEqual(owner.get(fixtures.ATTEMPT).state, "RESERVED")
                self.assertEqual(recorder.summary()["receipts"], [])
        self.assertEqual(self.requests, [])

    def test_context_publication_failure_blocks_transport_and_keeps_cost_held(self):
        owner = self.owner()
        recorder = self.recorder()
        with patch.object(runtime, "publish_bytes", side_effect=StorageError("private recording error")):
            result = self.send(owner, recorder)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.requests, [])
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (1, 140, False))
            kinds = [row[0] for row in connection.execute("SELECT event_kind FROM submission_events")]
        self.assertEqual(kinds, ["CLAIMED", "SUBMISSION_UNKNOWN"])
        self.assertNotIn("private recording error", json.dumps(recorder.summary()))

    def test_result_publication_failure_does_not_lose_known_task(self):
        owner = self.owner()
        recorder = self.recorder()
        publish = runtime.publish_bytes

        def fail_result(root, path, contents):
            if json.loads(contents)["manifestType"] == "runtime_create_result":
                raise StorageError("private result error")
            return publish(root, path, contents)

        with patch.object(runtime, "publish_bytes", side_effect=fail_result):
            result = self.send(owner, recorder)
        self.assertEqual((result.state, result.provider_task_id), ("QUEUED", TASK))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.records(recorder, "runtime_create_context")), 1)
        self.assertEqual(self.records(recorder, "runtime_create_result"), [])
        self.assertTrue(recorder.summary()["issues"])
        self.assertNotIn("private result error", json.dumps(recorder.summary()))
        self.assertEqual(self.send(owner, recorder), result)
        self.assertEqual(len(self.requests), 1)

    def test_observation_clock_failure_does_not_lose_known_task(self):
        owner = self.owner()

        def clock():
            if self.observed_unix >= 1002.0:
                raise RuntimeError("private clock error")
            return self.observed_unix

        recorder = self.recorder(clock=clock)
        result = self.send(owner, recorder)
        self.assertEqual((result.state, result.provider_task_id), ("QUEUED", TASK))
        self.assertEqual(len(self.requests), 1)
        results = self.records(recorder, "runtime_create_result")
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["recordedAtUnix"])
        self.assertIsNone(results[0]["responseObservedAtUnix"])
        self.assertTrue(recorder.summary()["issues"])
        self.assertNotIn("private clock error", json.dumps(recorder.summary()))

    def test_clock_interruption_after_response_propagates_and_preserves_claim(self):
        owner = self.owner()

        def clock():
            if self.observed_unix >= 1002.0:
                raise KeyboardInterrupt("fixture interruption")
            return self.observed_unix

        recorder = self.recorder(clock=clock)
        with self.assertRaises(KeyboardInterrupt):
            self.send(owner, recorder)
        self.assertEqual(owner.get(fixtures.ATTEMPT).state, "SUBMITTING")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.records(recorder, "runtime_create_result"), [])

    def test_task_receipt_failure_preserves_saved_observation(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        with patch.object(runtime, "publish_bytes", side_effect=StorageError("private task receipt error")):
            observed = self.task_get(recorder)
        self.assertEqual((observed.task_id, observed.status), (TASK, "SUCCEEDED"))
        saved = self.run / f"provider_responses/{observed.evidence_sha256}.json"
        self.assertEqual(sha256(saved.read_bytes()).hexdigest(), observed.evidence_sha256)
        self.assertEqual(self.records(recorder, "runtime_task_get"), [])
        self.assertTrue(recorder.summary()["issues"])
        self.assertNotIn("private task receipt error", json.dumps(recorder.summary()))

    def test_get_clock_failure_preserves_task_observation_with_null_times(self):
        owner = self.owner()

        def clock():
            if self.observed_unix >= 1003.0:
                raise RuntimeError("private task clock error")
            return self.observed_unix

        recorder = self.recorder(clock=clock)
        self.send(owner, recorder)
        observed = self.task_get(recorder)
        self.assertEqual((observed.task_id, observed.status), (TASK, "SUCCEEDED"))
        receipt = self.records(recorder, "runtime_task_get")[0]
        self.assertEqual(receipt["requestStartedAtUnix"], 1002.25)
        self.assertIsNone(receipt["responseObservedAtUnix"])
        self.assertIsNone(receipt["bodyFinishedAtUnix"])
        self.assertEqual(receipt["response"]["sha256"], observed.evidence_sha256)
        self.assertEqual(recorder.summary()["issues"], ["CLOCK_UNAVAILABLE"])

    def test_get_http_error_receipt_has_status_but_no_success_evidence(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        with self.assertRaises(runway.ProviderError):
            retrieve.retrieve_task_once(TASK, fixtures.KEY, self.run, runtime=recorder,
                transport=httpx.MockTransport(lambda _: httpx.Response(400, json={"error": "fixture error"})))
        receipt = self.records(recorder, "runtime_task_get")[0]
        self.assertEqual(receipt["httpStatus"], 400)
        self.assertIsNone(receipt["response"])
        self.assertIsNone(receipt["providerStatus"])
        self.assertEqual(owner.get(fixtures.ATTEMPT).provider_task_id, TASK)

    def test_download_intent_failure_blocks_get(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        observed = self.task_get(recorder)
        provider = Mock(return_value=httpx.Response(200, content=b"fixture output"))
        with patch.object(runtime, "publish_bytes", side_effect=StorageError("private intent error")):
            with self.assertRaises(fetch.DownloadError):
                fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
                    allowed_hosts=frozenset({HOST}), transport=httpx.MockTransport(provider), runtime=recorder)
        provider.assert_not_called()
        self.assertEqual(owner.get(fixtures.ATTEMPT).state, "QUEUED")

    def test_download_result_failure_preserves_exact_output_bytes(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        observed = self.task_get(recorder)
        publish = runtime.publish_bytes
        payload = bytes(range(256)) * 7

        def fail_result(root, path, contents):
            if json.loads(contents)["manifestType"] == "runtime_download_result":
                raise StorageError("private output receipt error")
            return publish(root, path, contents)

        with patch.object(runtime, "publish_bytes", side_effect=fail_result):
            result = fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
                allowed_hosts=frozenset({HOST}), transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload)),
                runtime=recorder)
        self.assertEqual(result, payload)
        self.assertEqual(len(self.records(recorder, "runtime_download_intent")), 1)
        self.assertEqual(self.records(recorder, "runtime_download_result"), [])
        self.assertTrue(recorder.summary()["issues"])
        self.assertNotIn("private output receipt error", json.dumps(recorder.summary()))

    def test_download_clock_failure_preserves_bytes_with_null_observation_times(self):
        owner = self.owner()
        clock_failed = False

        def clock():
            if clock_failed:
                raise RuntimeError("private download clock error")
            return self.observed_unix

        recorder = self.recorder(clock=clock)
        self.send(owner, recorder)
        observed = self.task_get(recorder)
        payload = b"fixture body returned despite clock failure"

        def provider(request):
            nonlocal clock_failed
            clock_failed = True
            return httpx.Response(200, content=payload)

        result = fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
            allowed_hosts=frozenset({HOST}), transport=httpx.MockTransport(provider), runtime=recorder)
        self.assertEqual(result, payload)
        receipt = self.records(recorder, "runtime_download_result")[0]
        self.assertEqual(receipt["requestStartedAtUnix"], 1003.25)
        self.assertIsNone(receipt["responseObservedAtUnix"])
        self.assertIsNone(receipt["bodyFinishedAtUnix"])
        self.assertEqual(receipt["rawSha256"], sha256(payload).hexdigest())
        self.assertEqual(recorder.summary()["issues"], ["CLOCK_UNAVAILABLE"])

    def test_injected_constructed_sdk_client_cannot_claim_default_transport(self):
        owner = self.owner()
        recorder = self.recorder()
        make_client = runway.make_sdk_client
        provider = Mock(return_value=httpx.Response(200, json={"id": TASK}))

        def injected_client(key, **arguments):
            return make_client(key, **(arguments | {"transport": httpx.MockTransport(provider)}))

        with patch.object(runway, "make_sdk_client", side_effect=injected_client):
            result = send.submit_checked(owner, self.checked, self.root, fixtures.KEY, runtime=recorder)
        self.assertEqual(result.state, "QUEUED")
        provider.assert_called_once()
        self.assertEqual(self.records(recorder, "runtime_create_context")[0]["transportClass"], "INJECTED")

    def test_default_and_explicit_http_transports_are_classified_without_requests(self):
        with patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("No network allowed")) as boundary:
            with httpx.Client() as default:
                self.assertEqual(runtime._transport_class(None, default, URL), "DEFAULT_NETWORK")
            explicit = httpx.HTTPTransport()
            with httpx.Client(transport=explicit) as supplied:
                self.assertEqual(runtime._transport_class(explicit, supplied, URL), "INJECTED")
        boundary.assert_not_called()

    def test_create_context_precedes_gate_and_result_names_durable_events(self):
        owner = self.owner()
        recorder = self.recorder()
        gate = owner.require_send_allowed
        seen_at_gate = []

        def check_gate(*args, **kwargs):
            seen_at_gate.extend(self.records(recorder))
            self.assertEqual(owner.get(fixtures.ATTEMPT).state, "SUBMITTING")
            return gate(*args, **kwargs)

        with patch.object(owner, "require_send_allowed", side_effect=check_gate):
            result = self.send(owner, recorder)
        self.assertEqual(result.state, "QUEUED")
        context = self.records(recorder, "runtime_create_context")[0]
        finished = self.records(recorder, "runtime_create_result")[0]
        self.assertEqual(seen_at_gate, [context])
        self.assertEqual(context["runStart"], asdict(self.fixture.started.record))
        self.assertEqual(context["requestCheck"], asdict(self.checked.evidence))
        self.assertEqual(read_verified(self.run, StoredArtifact(**context["request"])), self.checked.frozen.request_body)
        self.assertEqual(read_verified(self.run, StoredArtifact(**context["frozenRequest"])), self.checked.frozen.record_bytes)
        self.assertEqual((context["apiOrigin"], context["sdkVersion"], context["transportClass"]),
                         (runway.API_URL, runway.SDK_VERSION, "INJECTED"))
        self.assertEqual(context["recordedAtUnix"], 1001.25)
        self.assertEqual(finished["createContext"], asdict(recorder.references[0]))
        self.assertEqual((finished["localState"], finished["providerTaskId"], finished["httpStatus"]),
                         ("QUEUED", TASK, 200))
        self.assertEqual((finished["responseObservedAtUnix"], finished["recordedAtUnix"]), (1002.25, 1002.25))
        with self.ledger._connection() as connection:
            events = {row["event_kind"]: row["event_id"] for row in connection.execute(
                "SELECT event_id, event_kind FROM submission_events WHERE attempt_id=?", (fixtures.ATTEMPT,))}
        self.assertEqual(context["claimEventId"], events["CLAIMED"])
        self.assertEqual((finished["claimEventId"], finished["sendGateEventId"], finished["resultEventId"]),
                         (events["CLAIMED"], events["SEND_GATE_PASSED"], events["TASK_RECORDED"]))
        self.assertEqual(recorder.summary(), {"receipts": [asdict(record) for record in recorder.references], "issues": []})
        self.assertIsInstance(recorder.references, tuple)
        self.assert_private_values_absent(recorder)

    def test_non_success_response_is_recorded_as_unknown_without_retry(self):
        owner = self.owner()
        recorder = self.recorder()
        result = self.send(owner, recorder, status=400)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        finished = self.records(recorder, "runtime_create_result")[0]
        self.assertEqual((finished["localState"], finished["providerTaskId"], finished["httpStatus"]),
                         ("SUBMISSION_UNKNOWN", None, 400))
        self.assertEqual(finished["responseObservedAtUnix"], 1002.25)
        before = {record.path: read_verified(self.run, record) for record in recorder.references}
        self.assertEqual(self.send(owner, recorder), result)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual({record.path: read_verified(self.run, record) for record in recorder.references}, before)
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (1, 140, False))

    def test_restart_restores_fixed_creation_links_without_republishing(self):
        owner = self.owner()
        original = self.recorder()
        self.send(owner, original)
        before = {record.path: read_verified(self.run, record) for record in original.references}
        restored = self.recorder(clock=Mock(side_effect=RuntimeError("no restoration clock")))
        with patch.object(runtime, "publish_bytes", side_effect=AssertionError("Restore cannot publish")):
            restored.bind_existing(owner)
        self.assertEqual(restored.references, original.references)
        self.assertEqual(restored.summary()["issues"], [])
        self.assertEqual({record.path: read_verified(self.run, record) for record in restored.references}, before)
        restored.clock.assert_not_called()

    def test_ambiguous_event_lookup_materializes_only_two_rows(self):
        owner = self.owner()
        recorder = self.recorder()
        recorder.bind_checked(owner, self.checked)
        owner._claim(fixtures.ATTEMPT)
        for _ in range(8):
            owner.require_send_allowed(fixtures.ATTEMPT)
        materialized = []
        connection = self.ledger._connection

        def count_event_rows(cursor, values):
            if tuple(column[0] for column in cursor.description) == ("event_id",):
                materialized.append(values[0])
            return sqlite3.Row(cursor, values)

        @contextmanager
        def counted_connection():
            with connection() as opened:
                opened.row_factory = count_event_rows
                yield opened

        with patch.object(self.ledger, "_connection", counted_connection):
            with self.assertRaisesRegex(runtime.RuntimeEvidenceError, "ambiguous"):
                recorder._event_id("SEND_GATE_PASSED")
        self.assertEqual(len(materialized), 2)
        self.assertEqual(len(set(materialized)), 2)

    def test_ambiguous_task_binding_materializes_two_rows_and_keeps_get_available(self):
        owner = self.owner()
        self.send(owner)
        with self.ledger._transaction() as opened:
            for _ in range(7):
                owner._record_event(opened, fixtures.ATTEMPT, "TASK_RECONCILED", 1002.0,
                                    provider_task_id=TASK, evidence_sha256="b" * 64)
        recorder = self.recorder()
        materialized = []
        connection = self.ledger._connection

        def count_event_rows(cursor, values):
            if tuple(column[0] for column in cursor.description) == ("event_id",):
                materialized.append(values[0])
            return sqlite3.Row(cursor, values)

        @contextmanager
        def counted_connection():
            with connection() as opened:
                opened.row_factory = count_event_rows
                yield opened

        with patch.object(self.ledger, "_connection", counted_connection):
            recorder.bind_existing(owner)
        self.assertEqual(len(materialized), 2)
        self.assertEqual(len(set(materialized)), 2)
        self.assertEqual(recorder.summary(), {
            "receipts": [], "issues": ["TASK_BINDING_MISSING", "CREATE_CONTEXT_MISSING"]})
        observed = self.task_get(recorder)
        self.assertEqual((observed.task_id, observed.status), (TASK, "SUCCEEDED"))
        self.assertIsNone(self.records(recorder, "runtime_task_get")[0]["taskBindingEventId"])
        self.assertEqual([request.method for request in self.requests], ["POST", "GET"])

    def test_missing_legacy_creation_context_allows_get_with_explicit_missing_links(self):
        owner = self.owner()
        self.send(owner)
        recorder = self.recorder()
        recorder.bind_existing(owner)
        self.assertEqual(recorder.summary()["issues"], ["CREATE_CONTEXT_MISSING"])
        observed = self.task_get(recorder)
        self.assertEqual(observed.status, "SUCCEEDED")
        saved = self.records(recorder, "runtime_task_get")[0]
        self.assertIsNone(saved["createContext"])
        self.assertIsNone(saved["createResult"])
        self.assertIsInstance(saved["taskBindingEventId"], int)
        self.assertEqual([request.method for request in self.requests], ["POST", "GET"])

    def test_corrupt_context_is_reported_without_inventing_creation_links(self):
        owner = self.owner()
        original = self.recorder()
        self.send(owner, original)
        context = self.run / original.references[0].path
        data = json.loads(context.read_bytes())
        data["payloadSha256"] = "0" * 64
        changed = json.dumps(data).encode()
        context.write_bytes(changed)
        recorder = self.recorder()
        recorder.bind_existing(owner)
        self.assertEqual(recorder.summary()["issues"], ["CREATE_CONTEXT_INVALID"])
        self.assertEqual(self.task_get(recorder).status, "SUCCEEDED")
        saved = self.records(recorder, "runtime_task_get")[0]
        self.assertIsNone(saved["createContext"])
        self.assertIsNone(saved["createResult"])
        self.assertEqual(context.read_bytes(), changed)

    def test_corrupt_result_cannot_link_another_provider_task(self):
        owner = self.owner()
        original = self.recorder()
        self.send(owner, original)
        path = self.run / original.references[1].path
        data = json.loads(path.read_bytes())
        data["providerTaskId"] = "another-task"
        path.write_bytes(json.dumps(data).encode())
        recorder = self.recorder()
        recorder.bind_existing(owner)
        self.assertEqual(recorder.summary()["issues"], ["CREATE_RESULT_INVALID"])
        self.assertEqual(self.task_get(recorder).status, "SUCCEEDED")
        saved = self.records(recorder, "runtime_task_get")[0]
        self.assertEqual(saved["createContext"], asdict(original.references[0]))
        self.assertIsNone(saved["createResult"])

    def test_get_uses_separate_client_and_records_exact_saved_response(self):
        owner = self.owner()
        recorder = self.recorder()
        clients = []
        make_client = runway.make_sdk_client

        def track_client(*args, **kwargs):
            client = make_client(*args, **kwargs)
            clients.append(client)
            return client

        with patch.object(runway, "make_sdk_client", side_effect=track_client), \
                patch.object(retrieve, "make_sdk_client", side_effect=track_client):
            self.send(owner, recorder)
            observed = self.task_get(recorder)
        self.assertEqual(len(clients), 2)
        self.assertIsNot(clients[0], clients[1])
        self.assertTrue(all(client.is_closed() for client in clients))
        saved = self.records(recorder, "runtime_task_get")[0]
        response = StoredArtifact(**saved["response"])
        self.assertEqual(response.sha256, observed.evidence_sha256)
        self.assertEqual(json.loads(read_verified(self.run, response)),
                         {"id": TASK, "status": "SUCCEEDED", "output": [URL]})
        self.assertEqual((saved["providerTaskId"], saved["providerStatus"], saved["httpStatus"]),
                         (TASK, "SUCCEEDED", 200))
        self.assertEqual((saved["requestStartedAtUnix"], saved["responseObservedAtUnix"], saved["bodyFinishedAtUnix"]),
                         (1002.25, 1003.25, 1003.25))
        self.assertEqual(saved["createContext"], asdict(recorder.references[0]))
        self.assertEqual(saved["createResult"], asdict(recorder.references[1]))
        self.assertEqual(saved["transportClass"], "INJECTED")
        self.assert_private_values_absent(recorder)

    def test_repeated_gets_have_distinct_immutable_operation_receipts(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        first = self.task_get(recorder)
        first_receipt = recorder.references[-1]
        first_bytes = read_verified(self.run, first_receipt)
        second = self.task_get(recorder)
        receipts = self.records(recorder, "runtime_task_get")
        self.assertEqual(first, second)
        self.assertEqual(len(receipts), 2)
        self.assertNotEqual(receipts[0]["operationId"], receipts[1]["operationId"])
        self.assertEqual(read_verified(self.run, first_receipt), first_bytes)
        self.assertEqual(receipts[0]["response"], receipts[1]["response"])

    def test_download_links_success_get_and_records_exact_body_interval(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        observed = self.task_get(recorder)
        success_get = recorder.references[-1]
        self.observed_unix = 1004.25
        payload = b"first chunk" + b"second chunk"
        publish = runtime.publish_bytes
        observed_at_send = []

        def publish_intent(root, path, contents):
            record = publish(root, path, contents)
            if json.loads(contents)["manifestType"] == "runtime_download_intent":
                self.observed_unix = 1005.25
            return record

        test = self

        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                test.observed_unix = 1007.25
                yield b"first chunk"
                test.observed_unix = 1008.25
                yield b"second chunk"
                test.observed_unix = 1009.25

        def provider(request):
            observed_at_send.extend(self.records(recorder, "runtime_download_intent"))
            self.observed_unix = 1006.25
            return httpx.Response(200, stream=Stream())

        with patch.object(runtime, "publish_bytes", side_effect=publish_intent):
            result = fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
                allowed_hosts=frozenset({HOST}), transport=httpx.MockTransport(provider), runtime=recorder)
        self.assertEqual(result, payload)
        intent = self.records(recorder, "runtime_download_intent")[0]
        finished = self.records(recorder, "runtime_download_result")[0]
        self.assertEqual(observed_at_send, [intent])
        self.assertEqual(intent["taskGet"], asdict(success_get))
        self.assertEqual((intent["providerTaskId"], intent["successResponseSha256"], intent["outputHost"]),
                         (TASK, observed.evidence_sha256, HOST))
        self.assertEqual(finished["operationId"], intent["operationId"])
        self.assertEqual(finished["intent"], asdict(recorder.references[-2]))
        self.assertEqual((finished["requestStartedAtUnix"], finished["responseObservedAtUnix"], finished["bodyFinishedAtUnix"]),
                         (1005.25, 1006.25, 1009.25))
        self.assertEqual((finished["readOutcome"], finished["rawSha256"], finished["rawSize"]),
                         ("BODY_RETURNED", sha256(payload).hexdigest(), len(payload)))
        self.assertEqual(finished["transportClass"], "INJECTED")
        self.assert_private_values_absent(recorder)

    def test_context_restore_rejects_invalid_fields_bounds_and_control_hashes(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        path = self.run / recorder.references[0].path
        original = json.loads(path.read_bytes())
        request = original["request"] | {"sha256": "0" * 64}
        cases = [original | {"runId": "other"}, original | {"attemptId": "att-" + "b" * 32},
                 original | {"claimEventId": 999999}, original | {"transportClass": "AUTHENTICATED"},
                 original | {"recordedAtUnix": True}, original | {"liveVerified": True},
                 original | {"request": request}]
        bodies = [json.dumps(data).encode() for data in cases]
        valid_bytes = path.read_bytes()
        bodies.extend([valid_bytes + b" " * (16 * 1024 + 1 - len(valid_bytes)), valid_bytes.replace(b'"runId": "run-1"',
                      b'"runId": "run-1", "runId": "run-1"')])
        for body in bodies:
            with self.subTest(body=body[:80]):
                path.write_bytes(body)
                restored = self.recorder()
                restored.bind_existing(owner)
                self.assertEqual(restored.summary(), {"receipts": [], "issues": ["CREATE_CONTEXT_INVALID"]})
                self.assertEqual(path.read_bytes(), body)
        self.assertEqual(owner.get(fixtures.ATTEMPT).provider_task_id, TASK)
        self.assertEqual(len(self.requests), 1)

    def test_task_recorder_rejects_wrong_task_before_get(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        provider = Mock(return_value=httpx.Response(200, json={"id": "another-task", "status": "SUCCEEDED"}))
        with self.assertRaises(ValueError):
            retrieve.retrieve_task_once("another-task", fixtures.KEY, self.run,
                transport=httpx.MockTransport(provider), runtime=recorder)
        provider.assert_not_called()
        self.assertEqual(self.records(recorder, "runtime_task_get"), [])

    def test_task_get_client_label_uses_constructed_transport(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        make_client = runway.make_sdk_client

        def injected_client(key, **arguments):
            transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"id": TASK, "status": "RUNNING"}))
            return make_client(key, **(arguments | {"transport": transport}))

        with patch.object(retrieve, "make_sdk_client", side_effect=injected_client):
            result = retrieve.retrieve_task_once(TASK, fixtures.KEY, self.run, runtime=recorder)
        self.assertEqual(result.status, "RUNNING")
        self.assertEqual(self.records(recorder, "runtime_task_get")[0]["transportClass"], "INJECTED")

    def test_download_client_label_uses_constructed_transport(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        observed = self.task_get(recorder)
        client = httpx.Client
        payload = b"fixture output"

        def injected_client(**arguments):
            transport = httpx.MockTransport(lambda _: httpx.Response(200, content=payload))
            return client(**(arguments | {"transport": transport}))

        with patch.object(fetch.httpx, "Client", side_effect=injected_client):
            result = fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
                allowed_hosts=frozenset({HOST}), runtime=recorder)
        self.assertEqual(result, payload)
        self.assertEqual(self.records(recorder, "runtime_download_intent")[0]["transportClass"], "INJECTED")
        self.assertEqual(self.records(recorder, "runtime_download_result")[0]["transportClass"], "INJECTED")

    def test_failed_download_and_retry_have_distinct_preserved_receipts(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        observed = self.task_get(recorder)
        with self.assertRaises(fetch.DownloadError):
            fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
                allowed_hosts=frozenset({HOST}), runtime=recorder,
                transport=httpx.MockTransport(lambda _: httpx.Response(503)))
        failed = self.records(recorder, "runtime_download_result")[0]
        self.assertEqual((failed["readOutcome"], failed["httpStatus"], failed["rawSha256"], failed["rawSize"]),
                         ("FAILED", 503, None, None))
        before = {record.path: read_verified(self.run, record) for record in recorder.references}
        payload = b"fixture retry output"
        result = fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
            allowed_hosts=frozenset({HOST}), runtime=recorder,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload)))
        self.assertEqual(result, payload)
        results = self.records(recorder, "runtime_download_result")
        intents = self.records(recorder, "runtime_download_intent")
        self.assertEqual((len(results), len(intents)), (2, 2))
        self.assertNotEqual(results[0]["operationId"], results[1]["operationId"])
        self.assertEqual(results[1]["readOutcome"], "BODY_RETURNED")
        for record in recorder.references:
            if record.path in before:
                self.assertEqual(read_verified(self.run, record), before[record.path])

    def test_saved_success_without_get_receipt_has_explicit_missing_link(self):
        owner = self.owner()
        recorder = self.recorder()
        self.send(owner, recorder)
        observed = self.task_get()
        payload = b"fixture output from saved success"
        result = fetch.download_task_output(TASK, observed.evidence_sha256, self.run,
            allowed_hosts=frozenset({HOST}), runtime=recorder,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload)))
        self.assertEqual(result, payload)
        intent = self.records(recorder, "runtime_download_intent")[0]
        self.assertIsNone(intent["taskGet"])
        self.assertEqual(intent["successResponseSha256"], observed.evidence_sha256)
        self.assertEqual(recorder.summary()["issues"], ["SUCCESS_GET_RECEIPT_MISSING"])

    def test_resume_restores_context_and_observes_get_without_another_create(self):
        owner = self.owner()
        original = self.recorder()
        self.send(owner, original)
        owner.close()
        recorder = self.recorder()
        gets = []

        def provider(request):
            gets.append(request)
            return httpx.Response(200, json={"id": TASK, "status": "RUNNING"})

        with patch.dict("os.environ", {"RUNTIME_TEST_KEY": fixtures.KEY}):
            result = app_resume.resume_attempt(self.root, self.ledger.path, run_id="run-1", attempt_id=fixtures.ATTEMPT,
                output_hosts=frozenset({HOST}), api_key_env="RUNTIME_TEST_KEY", clock=lambda: 2000.0,
                task_transport=httpx.MockTransport(provider), runtime=recorder)
        self.assertEqual((result.provider_task_id, result.provider_status), (TASK, "RUNNING"))
        self.assertEqual([request.method for request in gets], ["GET"])
        self.assertEqual(len(self.requests), 1)
        saved = self.records(recorder, "runtime_task_get")[0]
        self.assertEqual(saved["createContext"], asdict(original.references[0]))
        self.assertEqual(saved["createResult"], asdict(original.references[1]))




