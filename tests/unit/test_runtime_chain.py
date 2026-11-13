"""Check pinned runtime links with literal receipts and immutable local facts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import socket
import sqlite3
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from core import run_audit as ledger_audit
from core import run_snapshot as captured
from core.budget import BudgetLedger
from core.run_start import RunStart
from core.runtime_pins import RuntimePin, RuntimePinSnapshot
from core.storage import StoredArtifact
from features.inference.frozen import FrozenCreate as _FrozenCreate
from features.inference import preflight, runway
from features.output import fetch
from qa import runtime_chain as chain_audit
import test_run_audit as ledger_fixtures
import test_run_evidence as fact_fixtures


RUN, ATTEMPT = ledger_fixtures.RUN, ledger_fixtures.ATTEMPT_A
TASK = "task-a"
HOST = "outputs.example.com"
URL = "https://outputs.example.com/raw.mp4?signature=private-runtime-token"
GET_ID, DOWNLOAD_ID = "6" * 32, "7" * 32
RECEIPTS = ("create_context", "create_result", "task_get", "download_intent", "download_result")
PASS = ledger_audit.AuditAssessment("PASS", "COMPLETE")
BLOCKED = ledger_audit.AuditAssessment("BLOCKED", "INCOMPLETE")


def json_bytes(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def runtime_record(path, contents):
    digest = sha256(contents).hexdigest()
    selected = path(digest) if callable(path) else path
    return chain_audit.RuntimeRecord(StoredArtifact(selected, digest, len(contents)), contents)


def make_chain_fixture(*, mode="A", transport="DEFAULT_NETWORK", changes=None,
                       missing=(), reconciled=False, absent_result=False):
    """Build receipt bytes without calling a runtime writer or audit helper."""
    prefix = f"attempts/{ATTEMPT}/"
    original = ledger_fixtures.complete_snapshot()
    original_facts = fact_fixtures.archive_facts(original, 0, mode)
    preparation = original_facts.graph.review.preparation.preparation
    frozen = _FrozenCreate(mode, "runway://fixture-source", "Animate the creature.",
        preparation.sha256, "5" * 64, 140, "runway://fixture-keyframe" if mode == "A" else None)
    start = RunStart(RUN, 1000.0, 11800.0, StoredArtifact("run-start.json", "0" * 64, 200))
    records, documents = {}, {}

    def record(name, path, value):
        value = deepcopy(value)
        change = (changes or {}).get(name)
        if callable(change):
            change(value)
        elif change is not None:
            value = change
        contents = value if type(value) is bytes else json_bytes(value)
        records[name] = runtime_record(path, contents)
        documents[name] = value
        return records[name]

    request = record("request", prefix + "provider/request.json", frozen.request_body)
    frozen_request = record("frozen_request", prefix + "provider/frozen_request.json", frozen.record_bytes)
    check = record("request_check", lambda digest: prefix + f"provider_checks/{digest}.json", {
        "schemaVersion": "1.4", "manifestType": "checked_provider_request", "runId": RUN, "attemptId": ATTEMPT,
        "sourceKind": "synthetic_fixture", "checkedAtUnix": 1001.0, "validUntilUnix": 1100.0,
        "frozen": asdict(frozen), "payloadSha256": frozen.payload_sha256, "preparation": asdict(preparation),
        "videoUpload": asdict(StoredArtifact("uploads/video.json", "8" * 64, 100)),
        "keyframeUpload": asdict(StoredArtifact("uploads/keyframe.json", "9" * 64, 100)) if mode == "A" else None,
        "price": asdict(StoredArtifact("prices/quote.json", "5" * 64, 100)),
        "account": asdict(StoredArtifact("accounts/check.json", "a" * 64, 100)), "spendingAuthorized": False,
    })
    success = record("success_response", lambda digest: f"provider_responses/{digest}.json",
                     {"id": TASK, "status": "SUCCEEDED", "output": [URL]})
    accepted = None
    reconciliation_record = runtime_record("reconciliation/accepted.json", json_bytes({"taskId": TASK, "operator": "fixture"}))
    if reconciled:
        accepted = chain_audit.AcceptedReconciliation(RUN, ATTEMPT, frozen.payload_sha256, TASK, reconciliation_record)
    reservation = original.reservations[0]._replace(payload_sha256=frozen.payload_sha256)
    submission = original.submission_events
    if reconciled:
        submission = submission[:2] + (
            captured.SubmissionEventRow(3, ATTEMPT, "SUBMISSION_UNKNOWN", 1004.0, None, None),
            captured.SubmissionEventRow(4, ATTEMPT, "TASK_RECONCILED", 1004.1, TASK, reconciliation_record.record.sha256),
        )
    journal = (original.run_events[0], original.run_events[1]._replace(
        record_json=ledger_fixtures.record_json(ledger_fixtures.reservation_record(reservation)))) + tuple(
        captured.RunEventRow(index + 3, RUN, ATTEMPT, "SUBMISSION_EVENT", row.recorded_unix, row.event_id, None)
        for index, row in enumerate(submission))
    snapshot = ledger_fixtures.resign(original, reservations=(reservation,), submission_events=submission, run_events=journal,
        submission_jobs=(original.submission_jobs[0]._replace(evidence_sha256=reconciliation_record.record.sha256 if reconciled else None),),
        poll_events=(original.poll_events[0]._replace(evidence_sha256=success.record.sha256),),
        completed_outputs=(original.completed_outputs[0]._replace(provider_evidence_sha256=success.record.sha256),))
    attempt = replace(original_facts, payload_sha256=frozen.payload_sha256, request_body_sha256=request.record.sha256,
        sdk_version=runway.SDK_VERSION, request_check=check.record, request=request.record,
        frozen_request=frozen_request.record, provider_success=success.record)

    def common(kind):
        return {"schemaVersion": "1.4", "manifestType": "runtime_" + kind,
                "runId": RUN, "attemptId": ATTEMPT, "payloadSha256": frozen.payload_sha256}

    context = record("create_context", prefix + "runtime/create-context.json", common("create_context") | {
        "runStart": asdict(start.record), "requestCheck": asdict(check.record), "request": asdict(request.record),
        "frozenRequest": asdict(frozen_request.record), "claimEventId": 1, "apiOrigin": runway.API_URL,
        "sdkVersion": runway.SDK_VERSION, "transportClass": transport, "recordedAtUnix": 1002.5,
    })
    result = record("create_result", prefix + "runtime/create-result.json", common("create_result") | {
        "createContext": asdict(context.record), "claimEventId": 1, "sendGateEventId": 2, "resultEventId": 3,
        "localState": "SUBMISSION_UNKNOWN" if reconciled else "QUEUED",
        "providerTaskId": None if reconciled else TASK, "transportClass": transport,
        "httpStatus": None if reconciled else 200, "responseObservedAtUnix": None if reconciled else 1003.5,
        "recordedAtUnix": 1004.0,
    })
    task_get = record("task_get", prefix + f"runtime/task-gets/{GET_ID}.json", common("task_get") | {
        "operationId": GET_ID, "providerTaskId": TASK, "transportClass": transport, "httpStatus": 200,
        "requestStartedAtUnix": 1004.25, "bodyFinishedAtUnix": 1005.0, "responseObservedAtUnix": 1004.5,
        "apiOrigin": runway.API_URL, "sdkVersion": runway.SDK_VERSION,
        "createContext": asdict(context.record), "createResult": None if absent_result else asdict(result.record),
        "taskBindingEventId": 4 if reconciled else 3, "response": asdict(success.record), "providerStatus": "SUCCEEDED",
    })
    intent = record("download_intent", prefix + f"runtime/downloads/{DOWNLOAD_ID}/intent.json", common("download_intent") | {
        "operationId": DOWNLOAD_ID, "providerTaskId": TASK, "transportClass": transport, "outputHost": HOST,
        "successResponseSha256": success.record.sha256, "taskGet": asdict(task_get.record),
    })
    record("download_result", prefix + f"runtime/downloads/{DOWNLOAD_ID}/result.json", common("download_result") | {
        "operationId": DOWNLOAD_ID, "providerTaskId": TASK, "transportClass": transport, "httpStatus": 200,
        "requestStartedAtUnix": 1005.25, "bodyFinishedAtUnix": 1005.75, "responseObservedAtUnix": 1005.5,
        "intent": asdict(intent.record), "readOutcome": "BODY_RETURNED",
        "rawSha256": attempt.raw_output.sha256, "rawSize": attempt.raw_output.size,
    })
    absent = set(missing) | ({"create_result"} if absent_result else set())
    chain = chain_audit.RuntimeChain(**{name: None if name in absent else value for name, value in records.items()},
                                     send_gate_event_id=2)
    pins = tuple(RuntimePin(ATTEMPT, frozen.payload_sha256,
        None if name == "create_context" or name == "create_result" and reconciled else TASK,
        "runtime_" + name, records[name].record) for name in RECEIPTS if not (absent_result and name == "create_result"))
    pin_hash = sha256(json_bytes([asdict(pin) for pin in pins])).hexdigest()
    return SimpleNamespace(snapshot=snapshot, start=start, attempt=attempt, pins=RuntimePinSnapshot(RUN, pins, pin_hash),
        chain=chain, reconciliation=accepted, records=records, documents=documents, frozen=frozen)


class RuntimeChainTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_chain_fixture()

    def assess(self, fixture=None, **changes):
        fixture = self.fixture if fixture is None else fixture
        arguments = {"run_id": RUN, "attempt": fixture.attempt, "snapshot": fixture.snapshot,
                     "run_start": fixture.start, "pins": fixture.pins, "chain": fixture.chain,
                     "reconciliation": fixture.reconciliation, "output_hosts": frozenset({HOST})} | changes
        return chain_audit.audit_runtime_chain(**arguments)

    def changed(self, changes, **options):
        return make_chain_fixture(changes=changes, **options)

    def assert_bound(self, result):
        for name in ("pins", "creation", "task_get", "download", "observations"):
            self.assertEqual(getattr(result, name), PASS, (name, result.issues))

    def test_direct_modes_return_exact_frozen_facts_without_io_or_live_claims(self):
        for mode in ("A", "B"):
            fixture = make_chain_fixture(mode=mode)
            with self.subTest(mode=mode), \
                    patch.object(sqlite3, "connect", side_effect=AssertionError("No database")), \
                    patch.object(Path, "read_bytes", side_effect=AssertionError("No files")), \
                    patch.object(Path, "write_bytes", side_effect=AssertionError("No writes")), \
                    patch.object(httpx.Client, "send", side_effect=AssertionError("No HTTP")), \
                    patch.object(socket, "socket", side_effect=AssertionError("No network")), \
                    patch.object(time, "time", side_effect=AssertionError("No clock")), \
                    patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No writer")):
                result = self.assess(fixture)
            self.assertIsInstance(result, chain_audit.RuntimeChainAudit)
            self.assert_bound(result)
            self.assertEqual(result.default_network, PASS)
            self.assertEqual((result.run_id, result.attempt_id, result.payload_sha256, result.provider_task_id),
                             (RUN, ATTEMPT, fixture.attempt.payload_sha256, TASK))
            self.assertEqual((result.snapshot_sha256, result.pin_snapshot_sha256),
                             (fixture.snapshot.sha256, fixture.pins.sha256))
            self.assertEqual((result.creation_mode, result.create_response_observed, result.injected_receipt_kinds),
                             ("DIRECT", True, ()))
            self.assertEqual(set(result.references), {value.record for value in fixture.records.values()})
            self.assertIsNone(result.reconciliation)
            self.assertEqual(result.issues, ())
            with self.assertRaises(FrozenInstanceError):
                result.provider_task_id = "other"
            with self.assertRaises(FrozenInstanceError):
                result.creation.verdict = "FAIL"
            with self.assertRaises(FrozenInstanceError):
                result.references[0].sha256 = "f" * 64
            with self.assertRaises(FrozenInstanceError):
                fixture.chain.send_gate_event_id = 1
            with self.assertRaises(FrozenInstanceError):
                fixture.chain.create_context.contents = b"changed"
            for private in (URL, "private-runtime-token", "runway://fixture-source", "Animate the creature."):
                self.assertNotIn(private, repr(result))
            for field in ("liveVerified", "releaseClassification", "visualEffectDemonstrated", "automatedWorkflowComplete"):
                self.assertNotIn(field, asdict(result))

    def test_injected_receipts_keep_valid_bindings_but_fail_default_network(self):
        groups = (("create_context", "create_result"), ("task_get",),
                  ("download_intent", "download_result"), RECEIPTS)
        for roles in groups:
            fixture = self.changed({role: lambda data: data.update(transportClass="INJECTED") for role in roles})
            with self.subTest(roles=roles):
                result = self.assess(fixture)
                self.assert_bound(result)
                self.assertEqual(result.default_network.verdict, "FAIL")
                self.assertEqual(set(result.injected_receipt_kinds), {"runtime_" + role for role in roles})

    def test_absent_external_pins_cannot_establish_default_network(self):
        result = self.assess(pins=None)
        self.assertEqual(result.pins, BLOCKED)
        self.assertEqual(result.default_network, BLOCKED)
        self.assertIsNone(result.pin_snapshot_sha256)
        self.assertEqual((result.creation, result.task_get, result.download), (PASS, PASS, PASS))
        missing_one = replace(self.fixture.pins, pins=self.fixture.pins.pins[:-1])
        self.assertEqual(self.assess(pins=missing_one).pins, BLOCKED)

    def test_relabelled_mock_receipts_cannot_replace_retained_pins(self):
        injected = make_chain_fixture(transport="INJECTED")
        result = self.assess(self.fixture, pins=injected.pins)
        self.assertEqual(result.pins.verdict, "FAIL")
        self.assertEqual(result.default_network.verdict, "FAIL")
        self.assertEqual((result.creation, result.task_get, result.download), (PASS, PASS, PASS))
        self.assertEqual(result.injected_receipt_kinds, ())

    def test_pin_identity_duplicates_and_exact_record_fields_are_checked(self):
        pins = self.fixture.pins
        first, second = pins.pins[:2]
        changes = [replace(pins, run_id="other-run"), replace(pins, pins=pins.pins + (first,)),
            replace(pins, pins=(replace(first, payload_sha256="f" * 64), *pins.pins[1:])),
            replace(pins, pins=(first, replace(second, provider_task_id="other-task"), *pins.pins[2:])),
            replace(pins, pins=(replace(first, receipt=replace(first.receipt, sha256="f" * 64)), *pins.pins[1:])),
            replace(pins, pins=(replace(first, receipt=replace(first.receipt, size=first.receipt.size + 1)), *pins.pins[1:]))]
        for index, changed in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(pins=changed).pins.verdict, "FAIL")

    def test_record_hash_and_size_corruption_fail_the_affected_component(self):
        categories = {"create_context": "creation", "create_result": "creation", "task_get": "task_get",
            "download_intent": "download", "download_result": "download", "request_check": "creation",
            "request": "creation", "frozen_request": "creation", "success_response": "task_get"}
        for role, category in categories.items():
            original = getattr(self.fixture.chain, role)
            corruptions = [b"X" + original.contents[1:], original.contents[:-1]]
            if original.contents.endswith(b"\n"):
                corruptions.append(original.contents[:-1] + b" ")
            for contents in corruptions:
                changed = replace(original, contents=contents)
                with self.subTest(role=role, length=len(contents)):
                    result = self.assess(chain=replace(self.fixture.chain, **{role: changed}))
                    self.assertEqual(getattr(result, category).verdict, "FAIL")

    def test_receipt_json_and_fixed_identity_fields_are_strict(self):
        for role in RECEIPTS:
            body = json_bytes(self.fixture.documents[role])
            invalid = [b"not JSON", b"[]", b'{"runId":"other",' + body[1:],
                       body.replace(b'"schemaVersion":"1.4"', b'"schemaVersion":NaN'),
                       lambda data: data.update(unexpected="private-runtime-token"),
                       lambda data: data.pop("runId"),
                       lambda data: data.update(attemptId="att-" + "b" * 32),
                       lambda data: data.update(payloadSha256="f" * 64)]
            category = "creation" if role.startswith("create") else "task_get" if role == "task_get" else "download"
            for index, change in enumerate(invalid):
                with self.subTest(role=role, case=index):
                    result = self.assess(self.changed({role: change}))
                    self.assertEqual(getattr(result, category).verdict, "FAIL")
                    self.assertNotIn("private-runtime-token", repr(result))

    def test_missing_required_records_snapshot_and_start_remain_incomplete(self):
        categories = {"create_context": "creation", "create_result": "creation", "request_check": "creation",
            "request": "creation", "frozen_request": "creation", "task_get": "task_get",
            "success_response": "task_get", "download_intent": "download", "download_result": "download"}
        for role, category in categories.items():
            with self.subTest(role=role):
                result = self.assess(chain=replace(self.fixture.chain, **{role: None}))
                self.assertEqual(getattr(result, category).coverage, "INCOMPLETE")
        for changes in ({"snapshot": None}, {"run_start": None}, {"chain": chain_audit.RuntimeChain()}):
            with self.subTest(missing=tuple(changes)):
                result = self.assess(**changes)
                self.assertEqual(result.creation.coverage, "INCOMPLETE")
        self.assertEqual(self.assess(output_hosts=frozenset()).download, BLOCKED)

    def test_checked_request_and_private_bytes_match_the_canonical_send(self):
        changes = [
            {"request": b" " + self.fixture.records["request"].contents},
            {"frozen_request": b" " + self.fixture.records["frozen_request"].contents},
            {"request_check": lambda data: data["frozen"].update(prompt="Other prompt")},
            {"request_check": lambda data: data.update(sourceKind="clean_camera")},
            {"request_check": lambda data: data.update(validUntilUnix=1003.0)},
            {"request_check": lambda data: data.update(checkedAtUnix=1003.1)},
            {"request_check": lambda data: data["preparation"].update(sha256="f" * 64)},
            {"request_check": lambda data: data.update(spendingAuthorized=True)},
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(self.changed(change)).creation.verdict, "FAIL")
        valid = self.changed({"request_check": lambda data: data.update(checkedAtUnix=1003.0)})
        self.assertEqual(self.assess(valid).creation, PASS)
        root = Path("unused-private-run")
        record = self.fixture.chain.request_check.record
        with patch.object(preflight, "_read_record", return_value=deepcopy(self.fixture.documents["request_check"])) as reader:
            checked = preflight.read_checked_request(root, record, run_id=RUN, attempt_id=ATTEMPT)
            reader.assert_called_once_with(root, record)
        self.assertEqual((checked.frozen, checked.evidence), (self.fixture.frozen, record))
        invalid = deepcopy(self.fixture.documents["request_check"])
        invalid["sourceKind"] = "unknown"
        with patch.object(preflight, "_read_record", return_value=invalid), self.assertRaises(preflight.PreflightError):
            preflight.read_checked_request(root, record, run_id=RUN, attempt_id=ATTEMPT)
        with patch.object(preflight, "_read_record", side_effect=AssertionError("Invalid path must not be read")) as reader:
            with self.assertRaises(preflight.PreflightError):
                preflight.read_checked_request(root, replace(record, path="../private-runtime-token"),
                                               run_id=RUN, attempt_id=ATTEMPT)
            reader.assert_not_called()

    def test_create_context_binds_original_start_client_request_and_claim(self):
        changes = [lambda data: data.update(apiOrigin="https://other.example.com"),
            lambda data: data.update(sdkVersion="0.0.1"), lambda data: data.update(claimEventId=2),
            lambda data: data.update(claimEventId=True), lambda data: data["runStart"].update(sha256="f" * 64),
            lambda data: data["requestCheck"].update(sha256="f" * 64),
            lambda data: data["request"].update(size=data["request"]["size"] + 1),
            lambda data: data["frozenRequest"].update(sha256="f" * 64)]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(self.changed({"create_context": change})).creation.verdict, "FAIL")
        for start in (replace(self.fixture.start, deadline_unix=11801.0),
                      replace(self.fixture.start, run_id="other-run")):
            self.assertEqual(self.assess(run_start=start).creation.verdict, "FAIL")
        fixture = self.changed({"request_check": lambda data: data.update(validUntilUnix=11801.0)})
        gate = fixture.snapshot.submission_events[1]._replace(recorded_unix=11800.0)
        events = tuple(row._replace(recorded_unix=gate.recorded_unix) if row.submission_event_id == gate.event_id else row
                       for row in fixture.snapshot.run_events)
        value = ledger_fixtures.resign(fixture.snapshot,
            submission_events=(fixture.snapshot.submission_events[0], gate, fixture.snapshot.submission_events[2]),
            run_events=events)
        self.assertEqual(self.assess(fixture, snapshot=value).creation.verdict, "FAIL")

    def test_direct_result_requires_success_status_task_and_shared_event_order(self):
        self.assertEqual(self.assess(self.changed({"create_result": lambda data: data.update(httpStatus=201)})).creation, PASS)
        changes = [lambda data: data.update(httpStatus=500), lambda data: data.update(httpStatus=True),
            lambda data: data.update(localState="SUBMISSION_UNKNOWN"), lambda data: data.update(providerTaskId="other-task"),
            lambda data: data.update(sendGateEventId=1), lambda data: data.update(resultEventId=1),
            lambda data: data.update(claimEventId=2), lambda data: data["createContext"].update(sha256="f" * 64)]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(self.changed({"create_result": change})).creation.verdict, "FAIL")
        rows = tuple(row._replace(sequence=4 if row.sequence == 3 else 3 if row.sequence == 4 else row.sequence)
                     for row in self.fixture.snapshot.run_events)
        value = ledger_fixtures.resign(self.fixture.snapshot, run_events=rows)
        self.assertEqual(self.assess(snapshot=value).creation.verdict, "FAIL")
        for index, fields in ((0, {"provider_task_id": TASK}), (0, {"evidence_sha256": "f" * 64}),
                              (1, {"provider_task_id": TASK}), (1, {"evidence_sha256": "f" * 64}),
                              (2, {"evidence_sha256": "f" * 64})):
            rows = list(self.fixture.snapshot.submission_events)
            rows[index] = rows[index]._replace(**fields)
            value = ledger_fixtures.resign(self.fixture.snapshot, submission_events=tuple(rows))
            with self.subTest(event=index + 1, fields=fields):
                self.assertEqual(self.assess(snapshot=value).creation.verdict, "FAIL")
        unrelated = captured.RunEventRow(4, RUN, None, "CHARGE_RECORDED", None, None, "{}")
        value = ledger_fixtures.resign(self.fixture.snapshot, run_events=self.fixture.snapshot.run_events + (unrelated,))
        self.assertEqual(self.assess(snapshot=value).creation.verdict, "FAIL")
        reconciled = make_chain_fixture(reconciled=True)
        for fields in ({"provider_task_id": TASK}, {"evidence_sha256": "f" * 64}):
            rows = list(reconciled.snapshot.submission_events)
            rows[2] = rows[2]._replace(**fields)
            value = ledger_fixtures.resign(reconciled.snapshot, submission_events=tuple(rows))
            with self.subTest(unknown_fields=fields):
                self.assertEqual(self.assess(reconciled, snapshot=value).creation.verdict, "FAIL")
        rows = list(reconciled.snapshot.submission_events)
        rows[3] = rows[3]._replace(evidence_sha256="invalid-hash")
        value = ledger_fixtures.resign(reconciled.snapshot, submission_events=tuple(rows))
        self.assertEqual(self.assess(reconciled, snapshot=value).task_get.verdict, "FAIL")

    def test_task_get_requires_same_task_success_reference_and_task_binding(self):
        changes = [lambda data: data.update(providerTaskId="other-task"), lambda data: data.update(providerStatus="RUNNING"),
            lambda data: data.update(httpStatus=500), lambda data: data.update(taskBindingEventId=2),
            lambda data: data.update(apiOrigin="https://other.example.com"), lambda data: data.update(sdkVersion="0.0.1"),
            lambda data: data.update(operationId="8" * 32), lambda data: data["response"].update(sha256="f" * 64),
            lambda data: data["createContext"].update(sha256="f" * 64),
            lambda data: data["createResult"].update(sha256="f" * 64)]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(self.changed({"task_get": change})).task_get.verdict, "FAIL")
        rows = tuple(row._replace(attempt_id="att-" + "b" * 32) if row.submission_event_id == 3 else row
                     for row in self.fixture.snapshot.run_events)
        value = ledger_fixtures.resign(self.fixture.snapshot, run_events=rows)
        self.assertEqual(self.assess(snapshot=value).task_get.verdict, "FAIL")

    def test_success_response_uses_the_same_task_and_first_approved_https_output(self):
        for change in (lambda data: data.update(id="other-task"), lambda data: data.update(status="RUNNING")):
            result = self.assess(self.changed({"success_response": change}))
            self.assertEqual(result.task_get.verdict, "FAIL")
        urls = ["http://outputs.example.com/raw.mp4", "https://other.example.com/raw.mp4",
                "https://user:private-runtime-token@outputs.example.com/raw.mp4", "https://outputs.example.com:444/raw.mp4",
                "https://outputs.example.com/raw.mp4#private-runtime-token", "https://127.0.0.1/raw.mp4",
                "https://outputs.example.com/raw.mp4\n", "https://outputs.example.com/" + "x" * 8192]
        for url in urls:
            fixture = self.changed({"success_response": lambda data, url=url: data.update(output=[url, URL])})
            with self.subTest(url=url[:60]):
                result = self.assess(fixture)
                self.assertEqual(result.download.verdict, "FAIL")
                self.assertNotIn("private-runtime-token", repr(result))
        self.assertEqual(self.assess(output_hosts=frozenset({"other.example.com"})).download.verdict, "FAIL")
        record = self.fixture.chain.success_response
        root = Path("unused-private-run")
        with patch.object(Path, "lstat", return_value=SimpleNamespace(st_size=record.record.size)), \
                patch.object(fetch, "read_verified", return_value=record.contents) as reader:
            self.assertEqual(str(fetch._output_url(TASK, record.record.sha256, root, frozenset({HOST}))), URL)
            reader.assert_called_once_with(root, record.record)
        for task, digest, hosts in (("private task", record.record.sha256, frozenset({HOST})),
                                    (TASK, "private-runtime-token", frozenset({HOST})),
                                    (TASK, record.record.sha256, frozenset())):
            with self.subTest(wrapper_guard=(task, digest[:8], hosts)), \
                    patch.object(Path, "lstat", side_effect=AssertionError("Invalid input must not reach storage")) as lookup:
                with self.assertRaises(ValueError):
                    fetch._output_url(task, digest, root, hosts)
                lookup.assert_not_called()

    def test_download_links_exact_intent_task_success_and_returned_raw_bytes(self):
        changes = [
            {"download_intent": lambda data: data.update(outputHost="other.example.com")},
            {"download_intent": lambda data: data.update(successResponseSha256="f" * 64)},
            {"download_intent": lambda data: data["taskGet"].update(sha256="f" * 64)},
            {"download_result": lambda data: data.update(providerTaskId="other-task")},
            {"download_result": lambda data: data.update(operationId="8" * 32)},
            {"download_result": lambda data: data["intent"].update(sha256="f" * 64)},
            {"download_result": lambda data: data.update(httpStatus=206)},
            {"download_result": lambda data: data.update(readOutcome="FAILED", rawSha256=None, rawSize=None)},
            {"download_result": lambda data: data.update(rawSha256="f" * 64)},
            {"download_result": lambda data: data.update(rawSize=101)},
            {"download_result": lambda data: data.update(rawSize=True)},
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(self.changed(change)).download.verdict, "FAIL")

    def test_missing_observation_fields_are_incomplete_not_invented_times(self):
        fields_by_role = {"create_context": ("recordedAtUnix",),
            "create_result": ("recordedAtUnix", "responseObservedAtUnix"),
            "task_get": ("requestStartedAtUnix", "responseObservedAtUnix", "bodyFinishedAtUnix"),
            "download_result": ("requestStartedAtUnix", "responseObservedAtUnix", "bodyFinishedAtUnix")}
        for role, names in fields_by_role.items():
            for name in names:
                for absent in (False, True):
                    def change(data, name=name, absent=absent):
                        data.pop(name) if absent else data.update({name: None})
                    with self.subTest(role=role, field=name, absent=absent):
                        result = self.assess(self.changed({role: change}))
                        self.assertEqual(result.observations, BLOCKED)
                        self.assertEqual(result.default_network, PASS)
                        if role == "create_result" and name == "responseObservedAtUnix":
                            self.assertFalse(result.create_response_observed)

    def test_observation_order_is_exact_but_late_output_is_allowed(self):
        changes = [
            {"create_context": lambda data: data.update(recordedAtUnix=1003.25)},
            {"create_result": lambda data: data.update(responseObservedAtUnix=1002.75)},
            {"task_get": lambda data: data.update(responseObservedAtUnix=1004.0)},
            {"task_get": lambda data: data.update(bodyFinishedAtUnix=1004.4)},
            {"download_result": lambda data: data.update(requestStartedAtUnix=1004.9)},
            {"download_result": lambda data: data.update(bodyFinishedAtUnix=1005.2)},
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(self.changed(change)).observations.verdict, "FAIL")
        fixture = self.changed({
            "task_get": lambda data: data.update(requestStartedAtUnix=19000.0, responseObservedAtUnix=19001.0, bodyFinishedAtUnix=19002.0),
            "download_result": lambda data: data.update(requestStartedAtUnix=20000.0, responseObservedAtUnix=20001.0, bodyFinishedAtUnix=20002.0),
        })
        output = fixture.snapshot.completed_outputs[0]._replace(completed_unix=20003.0)
        snapshot = ledger_fixtures.resign(fixture.snapshot, completed_outputs=(output,),
            poll_events=(fixture.snapshot.poll_events[0]._replace(observed_unix=19002.0),))
        result = self.assess(fixture, snapshot=snapshot, attempt=replace(fixture.attempt, completed_unix=20003.0))
        self.assert_bound(result)
        self.assertEqual(result.default_network, PASS)

    def test_reconciled_unknown_or_absent_result_never_invents_create_success(self):
        for absent in (False, True):
            fixture = make_chain_fixture(reconciled=True, absent_result=absent)
            with self.subTest(result_absent=absent):
                result = self.assess(fixture)
                self.assert_bound(result)
                self.assertEqual(result.default_network, PASS)
                self.assertEqual(result.creation_mode, "RECONCILED")
                self.assertFalse(result.create_response_observed)
                self.assertEqual(result.reconciliation, fixture.reconciliation.evidence.record)

    def test_reconciliation_requires_the_explicit_matching_caller_anchor(self):
        fixture = make_chain_fixture(reconciled=True)
        self.assertEqual(self.assess(fixture, reconciliation=None).creation.coverage, "INCOMPLETE")
        accepted = fixture.reconciliation
        changes = [replace(accepted, run_id="other-run"), replace(accepted, attempt_id="att-" + "b" * 32),
                   replace(accepted, payload_sha256="f" * 64), replace(accepted, provider_task_id="other-task"),
                   replace(accepted, evidence=replace(accepted.evidence, contents=b"X" + accepted.evidence.contents[1:]))]
        for index, changed in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(fixture, reconciliation=changed).creation.verdict, "FAIL")
        for field in ("submission_events", "submission_jobs"):
            rows = getattr(fixture.snapshot, field)
            changed = tuple(row._replace(evidence_sha256="f" * 64)
                            if field == "submission_jobs" or row.event_kind == "TASK_RECONCILED" else row for row in rows)
            snapshot = ledger_fixtures.resign(fixture.snapshot, **{field: changed})
            with self.subTest(ledger=field):
                self.assertEqual(self.assess(fixture, snapshot=snapshot).creation.verdict, "FAIL")

    def test_reconciliation_still_needs_original_context_gate_and_linked_unknown(self):
        fixture = make_chain_fixture(reconciled=True, absent_result=True)
        retained_result = next(pin for pin in make_chain_fixture(reconciled=True).pins.pins
                               if pin.receipt_kind == "runtime_create_result")
        retained_pins = replace(fixture.pins, pins=fixture.pins.pins + (retained_result,))
        omitted_result = self.assess(fixture, pins=retained_pins)
        self.assertEqual(omitted_result.creation, PASS)
        self.assertEqual(omitted_result.pins, BLOCKED)
        self.assertEqual(omitted_result.default_network, BLOCKED)
        for chain in (replace(fixture.chain, create_context=None), replace(fixture.chain, send_gate_event_id=None)):
            result = self.assess(fixture, chain=chain)
            self.assertEqual(result.creation.coverage, "INCOMPLETE")
        for event_id in (2, 3, 4):
            events = tuple(row for row in fixture.snapshot.run_events if row.submission_event_id != event_id)
            snapshot = ledger_fixtures.resign(fixture.snapshot, run_events=events)
            with self.subTest(unlinked_event=event_id):
                result = self.assess(fixture, snapshot=snapshot)
                category = result.task_get if event_id == 4 else result.creation
                self.assertEqual(category.coverage, "INCOMPLETE")
                self.assertTrue(any(issue.missing for issue in result.issues))

    def test_malformed_api_and_fixed_byte_bounds_raise_safe_errors(self):
        record = self.fixture.chain.create_context
        cases = [{"run_id": "../private-runtime-token"}, {"attempt": {}}, {"snapshot": {}}, {"run_start": {}},
            {"pins": {}}, {"pins": replace(self.fixture.pins, pins=list(self.fixture.pins.pins))},
            {"chain": {}}, {"chain": replace(self.fixture.chain, send_gate_event_id=True)},
            {"chain": replace(self.fixture.chain, create_context=replace(record, contents=bytearray(record.contents)))},
            {"chain": replace(self.fixture.chain, create_context=replace(record, record=replace(record.record, size=True)))},
            {"output_hosts": [HOST]}]
        for index, changes in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(chain_audit.RuntimeChainError) as caught:
                    self.assess(**changes)
                self.assertNotIn("private-runtime-token", str(caught.exception))
        for role, limit in (("create_context", 16 * 1024), ("request_check", 64 * 1024),
                            ("success_response", 1024 * 1024)):
            original = getattr(self.fixture.chain, role)
            oversized = runtime_record(original.record.path, b"x" * (limit + 1))
            with self.subTest(role=role), self.assertRaises(chain_audit.RuntimeChainError):
                self.assess(chain=replace(self.fixture.chain, **{role: oversized}))
        fixture = make_chain_fixture(reconciled=True)
        oversized = runtime_record("reconciliation/accepted.json", b"x" * (64 * 1024 + 1))
        with self.assertRaises(chain_audit.RuntimeChainError):
            self.assess(fixture, reconciliation=replace(fixture.reconciliation, evidence=oversized))
        with patch.object(chain_audit, "MAX_PINS", 4), self.assertRaises(chain_audit.RuntimeChainError):
            self.assess()


if __name__ == "__main__":
    unittest.main()
