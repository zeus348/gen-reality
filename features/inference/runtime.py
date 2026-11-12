"""Record local client observations without claiming provider authentication."""

from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import time
from uuid import uuid4

import httpx

from core.budget import BudgetLedger
from core.json_data import unique_fields
from core.run_start import read_run_start
from core.runtime_pins import RuntimePin, RuntimePinError, RuntimePinStore
from core.storage import ArtifactExistsError, StoredArtifact, _parent_directory, publish_bytes, read_verified
from features.inference.runway import API_URL, SDK_VERSION
from features.inference.submission import SubmissionCoordinator


MAX_RECEIPT_BYTES = 16 * 1024
_COMMON = {"schemaVersion", "manifestType", "runId", "attemptId", "payloadSha256"}
_CONTEXT_FIELDS = _COMMON | {"runStart", "requestCheck", "request", "frozenRequest", "claimEventId",
                             "apiOrigin", "sdkVersion", "transportClass", "recordedAtUnix"}
_RESULT_FIELDS = _COMMON | {"createContext", "claimEventId", "sendGateEventId", "resultEventId",
                            "localState", "providerTaskId", "transportClass", "httpStatus",
                            "responseObservedAtUnix", "recordedAtUnix"}


class RuntimeEvidenceError(ValueError):
    """Runtime evidence does not match the trusted operation."""


def _transport_class(transport, client, url) -> str:
    """Inspect the selected client transport without replacing proxy mounts."""
    if transport is not None or type(client) is not httpx.Client:
        return "INJECTED"
    selected = client._transport_for_url(httpx.URL(url))
    return "DEFAULT_NETWORK" if type(selected) is httpx.HTTPTransport else "INJECTED"


def _valid_time(value) -> bool:
    return (type(value) in (int, float) and 0 <= value <= 253_402_300_799
            and math.isfinite(value))


def _reject_constant(value):
    raise ValueError("Receipt JSON must use finite standard numbers.")


class RuntimeEvidence:
    """One trusted caller owns the root, IDs, clock, and returned references.

    DEFAULT_NETWORK only means no transport was injected into that client.
    These local receipts are not authentication proofs. Keep selected references
    in a separate operator-owned record if QA must detect altered receipt files.
    """

    def __init__(self, output_root: Path, run_id: str, attempt_id: str, *, clock=time.time,
                 pin_store: RuntimePinStore | None = None):
        BudgetLedger._run_id(run_id)
        BudgetLedger._attempt_id(attempt_id)
        self.output_root = Path(output_root).absolute()
        self.run_id, self.attempt_id = run_id, attempt_id
        self.root = self.output_root / run_id
        self.prefix = f"attempts/{attempt_id}/runtime/"
        self.clock = clock
        if pin_store is not None:
            if type(pin_store) is not RuntimePinStore:
                raise RuntimeEvidenceError("Runtime pins need the fixed private store.")
            pin_store.require_owner(self.output_root, Path(pin_store._owner.ledger_path), run_id)
        self.pin_store = pin_store
        self._records = []
        self._issues = []
        self._owner = None
        self._payload = None
        self._task_id = None
        self._task_event_id = None
        self._checked = None
        self._started = None
        self._context = None
        self._context_data = None
        self._result = None
        self._create_status = None
        self._create_response_time = None
        self._task_get = None
        self._task_get_sha256 = None

    @property
    def references(self) -> tuple[StoredArtifact, ...]:
        return tuple(self._records)

    def summary(self) -> dict:
        return {"receipts": [asdict(record) for record in self._records], "issues": list(self._issues)}

    def require_identity(self, output_root: Path, run_id: str, attempt_id: str):
        if (Path(output_root).absolute(), run_id, attempt_id) != (
                self.output_root, self.run_id, self.attempt_id):
            raise RuntimeEvidenceError("Runtime recorder belongs to another root, run, or attempt.")

    def require_evidence_root(self, evidence_root: Path):
        if Path(evidence_root).absolute() != self.root:
            raise RuntimeEvidenceError("Runtime recorder belongs to another evidence root.")

    def require_ledger(self, ledger_path: Path):
        if self.pin_store is not None:
            self.pin_store.require_owner(self.output_root, ledger_path, self.run_id)

    def _issue(self, code):
        if code not in self._issues:
            self._issues.append(code)

    def _now(self):
        try:
            now = self.clock()
            if not _valid_time(now):
                raise ValueError("Invalid observation clock.")
            return now
        except Exception:
            self._issue("CLOCK_UNAVAILABLE")
            return None

    def _observe(self, action):
        """An ordinary recording failure must not discard a provider result."""
        try:
            return action()
        except Exception:
            self._issue("RECEIPT_RECORDING_FAILED")
            return None

    def _base(self, kind):
        if self._payload is None:
            raise RuntimeEvidenceError("Runtime recorder has no matching reservation.")
        return {"schemaVersion": "1.4", "manifestType": kind, "runId": self.run_id,
                "attemptId": self.attempt_id, "payloadSha256": self._payload}

    def _keep(self, record):
        if record not in self._records:
            self._records.append(record)
        return record

    def _pin(self, record, data):
        return RuntimePin(self.attempt_id, self._payload, data.get("providerTaskId"),
                          data["manifestType"], record)

    def _restore_pin(self, record, data):
        """Only compare old evidence. A new GET cannot backfill creation pins."""
        if self.pin_store is None:
            return
        try:
            if not self.pin_store.matches(self._pin(record, data)):
                self._issue("PIN_MISSING")
        except Exception:
            self._issue("PIN_MISMATCH")

    def _publish(self, name, data):
        contents = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(contents) > MAX_RECEIPT_BYTES:
            raise RuntimeEvidenceError("Runtime receipt exceeds its byte limit.")
        path = self.prefix + name
        expected = StoredArtifact(path, sha256(contents).hexdigest(), len(contents))
        fresh = True
        try:
            record = publish_bytes(self.root, path, contents)
        except ArtifactExistsError:
            read_verified(self.root, expected)
            record = expected
            fresh = False
        if self.pin_store is not None:
            try:
                pin = self._pin(record, data)
                if fresh:
                    self.pin_store.pin(pin)
                elif not self.pin_store.matches(pin):
                    self._issue("PIN_MISSING")
                    raise RuntimePinError("An existing receipt has no retained runtime pin.")
            except Exception:
                self._issue("PIN_RECORDING_FAILED")
                raise
        return self._keep(record)

    def _bind(self, owner):
        self.require_ledger(owner.ledger.path)
        owner._require_open()
        with owner.ledger._connection() as connection:
            row = connection.execute("SELECT * FROM reservations WHERE attempt_id=?", (self.attempt_id,)).fetchone()
        if row is None or row["run_id"] != self.run_id:
            raise RuntimeEvidenceError("Runtime recorder has no matching run reservation.")
        self._owner, self._payload = owner, row["payload_sha256"]
        self._task_id = owner.get(self.attempt_id).provider_task_id
        return row

    def bind_checked(self, owner, checked):
        self.require_identity(self.output_root, checked.run_id, checked.attempt_id)
        row = self._bind(owner)
        if (checked.frozen.payload_sha256 != row["payload_sha256"]
                or checked.frozen.estimated_cents != row["cost_cents"]):
            raise RuntimeEvidenceError("Runtime request differs from its reservation.")
        started = read_run_start(self.output_root, run_id=self.run_id)
        with owner.ledger._connection() as connection:
            approval = connection.execute("SELECT * FROM approved_runs WHERE run_id=?", (self.run_id,)).fetchone()
        if not (started.started_unix <= approval["started_unix"]
                < approval["deadline_unix"] <= started.deadline_unix):
            raise RuntimeEvidenceError("Runtime request needs the original experiment deadline.")
        self._checked, self._started = checked, started

    def _event(self, event_id, kind, task_id=None):
        if type(event_id) is not int or event_id <= 0:
            raise RuntimeEvidenceError("Runtime event ID is invalid.")
        with self._owner.ledger._connection() as connection:
            row = connection.execute("SELECT * FROM submission_events WHERE event_id=?", (event_id,)).fetchone()
        if (row is None or row["attempt_id"] != self.attempt_id or row["event_kind"] != kind
                or (task_id is not None and row["provider_task_id"] != task_id)):
            raise RuntimeEvidenceError("Runtime event does not match the attempt.")

    def _event_id(self, kind):
        with self._owner.ledger._connection() as connection:
            rows = connection.execute("SELECT event_id FROM submission_events WHERE attempt_id=? AND event_kind=? LIMIT 2",
                                      (self.attempt_id, kind)).fetchall()
        if len(rows) > 1:
            raise RuntimeEvidenceError("Runtime event is ambiguous.")
        return rows[0]["event_id"] if rows else None

    def prepare_create(self, request: StoredArtifact, frozen_request: StoredArtifact, transport, client):
        """Publish prerequisites before the existing final authority check."""
        if self._checked is None or self._started is None or self._owner.get(self.attempt_id).state != "SUBMITTING":
            raise RuntimeEvidenceError("Create observation needs a checked, claimed request.")
        if (request.path != f"attempts/{self.attempt_id}/provider/request.json"
                or frozen_request.path != f"attempts/{self.attempt_id}/provider/frozen_request.json"
                or read_verified(self.root, request) != self._checked.frozen.request_body
                or read_verified(self.root, frozen_request) != self._checked.frozen.record_bytes):
            raise RuntimeEvidenceError("Saved create request differs from its checked inputs.")
        claim = self._event_id("CLAIMED")
        self._event(claim, "CLAIMED")
        data = self._base("runtime_create_context") | {
            "runStart": asdict(self._started.record), "requestCheck": asdict(self._checked.evidence),
            "request": asdict(request), "frozenRequest": asdict(frozen_request), "claimEventId": claim,
            "apiOrigin": API_URL, "sdkVersion": SDK_VERSION,
            "transportClass": _transport_class(transport, client, API_URL + "/v1/video_to_video"),
            "recordedAtUnix": self._now(),
        }
        self._context = self._publish("create-context.json", data)
        self._context_data = data

    def observe_create_response(self, response):
        def capture():
            status = response.status_code
            if type(status) is not int or not 100 <= status <= 599:
                raise ValueError("Invalid HTTP status.")
            self._create_status = status
            self._create_response_time = self._now()
        self._observe(capture)

    def finish_create(self, result):
        """Observe only after the coordinator has saved its task or uncertainty."""
        self._task_id = result.provider_task_id
        if self._context is None:
            return
        def record():
            kind = "TASK_RECORDED" if result.state == "QUEUED" else "SUBMISSION_UNKNOWN"
            event = self._event_id(kind)
            self._event(event, kind, result.provider_task_id)
            if result.provider_task_id is not None:
                self._task_event_id = event
            data = self._base("runtime_create_result") | {
                "createContext": asdict(self._context), "claimEventId": self._context_data["claimEventId"],
                "sendGateEventId": self._event_id("SEND_GATE_PASSED"), "resultEventId": event,
                "localState": result.state, "providerTaskId": result.provider_task_id,
                "transportClass": self._context_data["transportClass"], "httpStatus": self._create_status,
                "responseObservedAtUnix": self._create_response_time, "recordedAtUnix": self._now(),
            }
            self._result = self._publish("create-result.json", data)
        self._observe(record)

    def _read_fixed(self, name, fields, kind):
        path = self.prefix + name
        parts = path.split("/")
        with _parent_directory(self.root, parts, create=False) as directory:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(descriptor, "rb") as file:
                info = os.fstat(file.fileno())
                if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_RECEIPT_BYTES:
                    raise RuntimeEvidenceError("Runtime receipt is not a bounded regular file.")
                contents = file.read(info.st_size + 1)
        if len(contents) != info.st_size:
            raise RuntimeEvidenceError("Runtime receipt size changed.")
        data = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields, parse_constant=_reject_constant)
        if type(data) is not dict or set(data) != fields:
            raise RuntimeEvidenceError("Runtime receipt fields are invalid.")
        if any(data[key] != value for key, value in self._base(kind).items()):
            raise RuntimeEvidenceError("Runtime receipt belongs to another request.")
        if data["transportClass"] not in ("DEFAULT_NETWORK", "INJECTED"):
            raise RuntimeEvidenceError("Runtime transport class is invalid.")
        for key in ("recordedAtUnix", "responseObservedAtUnix"):
            if key in data and data[key] is not None and not _valid_time(data[key]):
                raise RuntimeEvidenceError("Runtime observation timestamp is invalid.")
        return data, StoredArtifact(path, sha256(contents).hexdigest(), len(contents))

    def _control(self, value, expected_path=None):
        if type(value) is not dict or set(value) != {"path", "sha256", "size"}:
            raise RuntimeEvidenceError("Runtime control reference is invalid.")
        record = StoredArtifact(**value)
        if type(record.size) is not int or not 0 < record.size <= 64 * 1024:
            raise RuntimeEvidenceError("Runtime control exceeds its byte limit.")
        if expected_path is not None and record.path != expected_path:
            raise RuntimeEvidenceError("Runtime control path does not match the attempt.")
        return record, read_verified(self.root, record)

    def bind_existing(self, owner):
        """Restore fixed creation links; missing legacy evidence never blocks GET."""
        self._bind(owner)
        self._context = self._context_data = self._result = None
        self._task_event_id = None
        if self._task_id is not None:
            with owner.ledger._connection() as connection:
                rows = connection.execute(
                    "SELECT event_id FROM submission_events WHERE attempt_id=? AND provider_task_id=? "
                    "AND event_kind IN ('TASK_RECORDED', 'TASK_RECONCILED') LIMIT 2",
                    (self.attempt_id, self._task_id),
                ).fetchall()
            if len(rows) == 1:
                self._task_event_id = rows[0]["event_id"]
            else:
                self._issue("TASK_BINDING_MISSING")
        try:
            from features.inference.preflight import read_checked_request
            data, record = self._read_fixed("create-context.json", _CONTEXT_FIELDS, "runtime_create_context")
            if data["apiOrigin"] != API_URL or data["sdkVersion"] != SDK_VERSION:
                raise RuntimeEvidenceError("Runtime SDK configuration differs.")
            started = read_run_start(self.output_root, run_id=self.run_id)
            if data["runStart"] != asdict(started.record):
                raise RuntimeEvidenceError("Runtime start reference differs.")
            check, _ = self._control(data["requestCheck"])
            checked = read_checked_request(self.root, check, run_id=self.run_id, attempt_id=self.attempt_id)
            if checked.frozen.payload_sha256 != self._payload:
                raise RuntimeEvidenceError("Runtime checked payload differs.")
            _, request = self._control(data["request"], f"attempts/{self.attempt_id}/provider/request.json")
            _, frozen = self._control(data["frozenRequest"], f"attempts/{self.attempt_id}/provider/frozen_request.json")
            if request != checked.frozen.request_body or frozen != checked.frozen.record_bytes:
                raise RuntimeEvidenceError("Runtime saved request differs.")
            self._event(data["claimEventId"], "CLAIMED")
            self._restore_pin(record, data)
            self._context, self._context_data = self._keep(record), data
        except FileNotFoundError:
            self._issue("CREATE_CONTEXT_MISSING")
        except Exception:
            self._issue("CREATE_CONTEXT_INVALID")
        if self._context is None:
            return
        try:
            data, record = self._read_fixed("create-result.json", _RESULT_FIELDS, "runtime_create_result")
            if (data["createContext"] != asdict(self._context)
                    or data["claimEventId"] != self._context_data["claimEventId"]
                    or data["transportClass"] != self._context_data["transportClass"]):
                raise RuntimeEvidenceError("Runtime result does not match its context.")
            status = data["httpStatus"]
            if status is not None and (type(status) is not int or not 100 <= status <= 599):
                raise RuntimeEvidenceError("Runtime HTTP status is invalid.")
            if data["localState"] == "QUEUED":
                if data["providerTaskId"] != self._task_id or self._task_id is None:
                    raise RuntimeEvidenceError("Runtime result refers to another task.")
                self._event(data["resultEventId"], "TASK_RECORDED", self._task_id)
            elif data["localState"] == "SUBMISSION_UNKNOWN" and data["providerTaskId"] is None:
                self._event(data["resultEventId"], "SUBMISSION_UNKNOWN")
            else:
                raise RuntimeEvidenceError("Runtime result state is invalid.")
            if data["sendGateEventId"] is not None:
                self._event(data["sendGateEventId"], "SEND_GATE_PASSED")
            self._restore_pin(record, data)
            self._result = self._keep(record)
        except FileNotFoundError:
            self._issue("CREATE_RESULT_MISSING")
        except Exception:
            self._issue("CREATE_RESULT_INVALID")

    def _require_task(self, task_id):
        SubmissionCoordinator._task_id(task_id)
        if self._payload is None or task_id != self._task_id:
            raise RuntimeEvidenceError("Runtime task differs from the saved task.")

    def task_get(self, task_id, transport):
        self._require_task(task_id)
        self._task_get = self._task_get_sha256 = None
        return _TaskGet(self, task_id, transport)

    def download(self, task_id, success_sha256, host, transport, client, url):
        self._require_task(task_id)
        return _Download(self, task_id, success_sha256, host, transport, client, url)


class _Observation:
    def __init__(self, runtime, task_id, transport):
        self.runtime, self.task_id = runtime, task_id
        self.operation_id = uuid4().hex
        self.transport_class = None
        self.started = self.finished = self.status = self.response_time = None
        self.done = False

    def use_client(self, transport, client, url):
        self.transport_class = _transport_class(transport, client, url)

    def start(self):
        self.started = self.runtime._now()

    def body_finished(self):
        self.finished = self.runtime._now()
        if self.started is not None and self.finished is not None and self.finished < self.started:
            self.runtime._issue("CLOCK_MOVED_BACKWARDS")

    def observe_response(self, response):
        def capture():
            status = response.status_code
            if type(status) is not int or not 100 <= status <= 599:
                raise ValueError("Invalid HTTP status.")
            self.status, self.response_time = status, self.runtime._now()
        self.runtime._observe(capture)

    def _fields(self):
        return {"operationId": self.operation_id, "providerTaskId": self.task_id,
                "transportClass": self.transport_class, "httpStatus": self.status,
                "requestStartedAtUnix": self.started, "bodyFinishedAtUnix": self.finished,
                "responseObservedAtUnix": self.response_time}


class _TaskGet(_Observation):
    def complete(self, response=None, provider_status=None):
        if self.done:
            return
        self.done = True
        def record():
            runtime = self.runtime
            if response is not None:
                body = json.loads(read_verified(runtime.root, response).decode("utf-8"), object_pairs_hook=unique_fields)
                if body.get("id") != self.task_id or body.get("status") != provider_status:
                    raise RuntimeEvidenceError("Observed task response differs.")
            data = runtime._base("runtime_task_get") | self._fields() | {
                "apiOrigin": API_URL, "sdkVersion": SDK_VERSION,
                "createContext": asdict(runtime._context) if runtime._context else None,
                "createResult": asdict(runtime._result) if runtime._result else None,
                "taskBindingEventId": runtime._task_event_id,
                "response": asdict(response) if response else None, "providerStatus": provider_status,
            }
            receipt = runtime._publish(f"task-gets/{self.operation_id}.json", data)
            if response is not None and provider_status == "SUCCEEDED":
                runtime._task_get, runtime._task_get_sha256 = receipt, response.sha256
        self.runtime._observe(record)


class _Download(_Observation):
    def __init__(self, runtime, task_id, success_sha256, host, transport, client, url):
        super().__init__(runtime, task_id, transport)
        self.use_client(transport, client, url)
        selected = runtime._task_get if runtime._task_get_sha256 == success_sha256 else None
        if selected is None:
            runtime._issue("SUCCESS_GET_RECEIPT_MISSING")
        data = runtime._base("runtime_download_intent") | {
            "operationId": self.operation_id, "providerTaskId": task_id,
            "transportClass": self.transport_class, "outputHost": host,
            "successResponseSha256": success_sha256,
            "taskGet": asdict(selected) if selected else None,
        }
        self.intent = runtime._publish(f"downloads/{self.operation_id}/intent.json", data)

    def complete(self, contents=None):
        if self.done:
            return
        self.done = True
        def record():
            data = self.runtime._base("runtime_download_result") | self._fields() | {
                "intent": asdict(self.intent),
                "readOutcome": "BODY_RETURNED" if contents is not None else "FAILED",
                "rawSha256": sha256(contents).hexdigest() if contents is not None else None,
                "rawSize": len(contents) if contents is not None else None,
            }
            self.runtime._publish(f"downloads/{self.operation_id}/result.json", data)
        self.runtime._observe(record)
