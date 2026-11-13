"""Check one explicitly selected runtime chain against retained operator pins."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from core.json_data import unique_fields
from core.run_audit import AuditAssessment, AuditIssue, _check_input
from core.run_snapshot import RunSnapshot
from core.run_start import MAX_ELAPSED_SECONDS, MAX_RUN_START_BYTES, RunStart
from core.runtime_pins import MAX_CAPTURE_BYTES, MAX_PINS, RuntimePin, RuntimePinSnapshot, _validate_pin
from core.storage import StoredArtifact
from features.inference.preflight import _check_request_data, _match_fields
from features.inference.retrieve import MAX_TASK_RESPONSE_BYTES
from features.inference.runway import API_URL, SDK_VERSION
from features.inference.runtime import MAX_RECEIPT_BYTES, _COMMON, _CONTEXT_FIELDS, _RESULT_FIELDS
from features.output.fetch import _output_url_from_data
from features.review.attempt_records import ExportAttemptFacts
from qa.run_evidence import _archive_references, _assessment, _matches, _reference, _time


MAX_PRIVATE_BYTES = 64 * 1024
MAX_RECONCILIATION_BYTES = 64 * 1024
MAX_OUTPUT_HOSTS = 64
_RECEIPTS = ("create_context", "create_result", "task_get", "download_intent", "download_result")
_PRIVATE = ("request_check", "request", "frozen_request", "success_response")
_CATEGORIES = {"create_context": "creation", "create_result": "creation", "task_get": "task_get",
               "download_intent": "download", "download_result": "download", "request_check": "creation",
               "request": "creation", "frozen_request": "creation", "success_response": "task_get"}
_OBSERVED = {"transportClass", "httpStatus", "responseObservedAtUnix", "recordedAtUnix",
             "requestStartedAtUnix", "bodyFinishedAtUnix"}
_OPERATION_FIELDS = {"operationId", "providerTaskId", "transportClass", "httpStatus",
                     "requestStartedAtUnix", "bodyFinishedAtUnix", "responseObservedAtUnix"}
_FIELDS = {
    "create_context": _CONTEXT_FIELDS,
    "create_result": _RESULT_FIELDS,
    "task_get": _COMMON | _OPERATION_FIELDS | {"apiOrigin", "sdkVersion", "createContext", "createResult",
                                               "taskBindingEventId", "response", "providerStatus"},
    "download_intent": _COMMON | {"operationId", "providerTaskId", "transportClass", "outputHost",
                                   "successResponseSha256", "taskGet"},
    "download_result": _COMMON | _OPERATION_FIELDS | {"intent", "readOutcome", "rawSha256", "rawSize"},
}


class RuntimeChainError(ValueError):
    """The pure chain audit needs bounded values from explicitly selected inputs."""


@dataclass(frozen=True)
class RuntimeRecord:
    record: StoredArtifact
    contents: bytes


@dataclass(frozen=True)
class RuntimeChain:
    create_context: RuntimeRecord | None = None
    create_result: RuntimeRecord | None = None
    task_get: RuntimeRecord | None = None
    download_intent: RuntimeRecord | None = None
    download_result: RuntimeRecord | None = None
    request_check: RuntimeRecord | None = None
    request: RuntimeRecord | None = None
    frozen_request: RuntimeRecord | None = None
    success_response: RuntimeRecord | None = None
    send_gate_event_id: int | None = None


@dataclass(frozen=True)
class AcceptedReconciliation:
    """An operator-accepted identity and evidence, supplied outside the export."""

    run_id: str
    attempt_id: str
    payload_sha256: str
    provider_task_id: str
    evidence: RuntimeRecord


@dataclass(frozen=True)
class RuntimeChainAudit:
    run_id: str
    attempt_id: str
    payload_sha256: str
    provider_task_id: str
    snapshot_sha256: str | None
    pin_snapshot_sha256: str | None
    creation_mode: str
    create_response_observed: bool
    injected_receipt_kinds: tuple[str, ...]
    references: tuple[StoredArtifact, ...]
    reconciliation: StoredArtifact | None
    pins: AuditAssessment
    creation: AuditAssessment
    task_get: AuditAssessment
    download: AuditAssessment
    observations: AuditAssessment
    default_network: AuditAssessment
    issues: tuple[AuditIssue, ...]


def _note(issues, category, code, *, missing=False):
    issues.append(AuditIssue(category, code, missing))


def _reject_constant(value):
    raise ValueError("Runtime JSON numbers must be finite.")


def _record_shape(value, limit):
    if type(value) is not RuntimeRecord or type(value.contents) is not bytes or not 0 < len(value.contents) <= limit:
        raise RuntimeChainError("Runtime records must contain bounded immutable bytes.")
    _reference(value.record, limit=limit)


def _json(contents):
    value = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields, parse_constant=_reject_constant)
    if type(value) is not dict:
        raise ValueError("Runtime JSON must contain an object.")
    json.dumps(value, allow_nan=False)
    return value


def _expected_path(role, item, data, attempt_id):
    prefix = f"attempts/{attempt_id}/"
    fixed = {"create_context": prefix + "runtime/create-context.json",
             "create_result": prefix + "runtime/create-result.json",
             "request_check": prefix + f"provider_checks/{item.record.sha256}.json",
             "request": prefix + "provider/request.json", "frozen_request": prefix + "provider/frozen_request.json",
             "success_response": f"provider_responses/{item.record.sha256}.json"}
    if role in fixed:
        return fixed[role]
    operation = data.get("operationId")
    if not _matches(operation, r"[0-9a-f]{32}"):
        raise ValueError("Runtime operation ID is invalid.")
    if role == "task_get":
        return prefix + f"runtime/task-gets/{operation}.json"
    name = "intent" if role == "download_intent" else "result"
    return prefix + f"runtime/downloads/{operation}/{name}.json"


def _load_chain(chain, attempt, issues):
    data = {}
    for role in (*_RECEIPTS, *_PRIVATE):
        item, category = getattr(chain, role), _CATEGORIES[role]
        if item is None:
            if role != "create_result":
                _note(issues, category, role.upper() + "_MISSING", missing=True)
            continue
        limit = (MAX_RECEIPT_BYTES if role in _RECEIPTS else
                 MAX_TASK_RESPONSE_BYTES if role == "success_response" else MAX_PRIVATE_BYTES)
        _record_shape(item, limit)
        if len(item.contents) != item.record.size or sha256(item.contents).hexdigest() != item.record.sha256:
            _note(issues, category, role.upper() + "_HASH_MISMATCH")
            if role in _RECEIPTS:
                _note(issues, "pins", role.upper() + "_BYTES_MISMATCH")
            continue
        try:
            body = _json(item.contents)
            if role in _RECEIPTS:
                expected = _FIELDS[role]
                if set(body) - expected or expected - set(body) - _OBSERVED:
                    raise ValueError("Runtime receipt fields are invalid.")
                _match_fields(body, {"schemaVersion": "1.4", "manifestType": "runtime_" + role,
                                    "runId": attempt.run_id, "attemptId": attempt.attempt_id,
                                    "payloadSha256": attempt.payload_sha256})
                for name in expected & _OBSERVED:
                    body.setdefault(name, None)
            if item.record.path != _expected_path(role, item, body, attempt.attempt_id):
                raise ValueError("Runtime record path differs from its role.")
            data[role] = body
        except Exception:
            _note(issues, category, role.upper() + "_INVALID")
    return data


def _check_pins(pins, chain, data, attempt, issues):
    if pins is None:
        _note(issues, "pins", "RUNTIME_PINS_MISSING", missing=True)
        return
    if (type(pins) is not RuntimePinSnapshot or type(pins.pins) is not tuple or len(pins.pins) > MAX_PINS
            or not _matches(pins.sha256, r"[0-9a-f]{64}")):
        raise RuntimeChainError("Runtime pins must use the bounded snapshot contract.")
    if pins.run_id != attempt.run_id:
        _note(issues, "pins", "PIN_RUN_MISMATCH")
    saved, size = {}, 0
    for pin in pins.pins:
        _validate_pin(pin)
        size += len(json.dumps(asdict(pin), sort_keys=True).encode("utf-8"))
        if size > MAX_CAPTURE_BYTES:
            raise RuntimeChainError("Runtime pins exceed their capture limit.")
        if pin.receipt.path in saved:
            _note(issues, "pins", "PIN_PATH_DUPLICATE")
        saved[pin.receipt.path] = pin
    for role in _RECEIPTS:
        item = getattr(chain, role)
        if item is None:
            # An absent result is valid after a crash only when the index has
            # no retained result. Do not hide a known old result by omitting it.
            path = f"attempts/{attempt.attempt_id}/runtime/create-result.json"
            if role == "create_result" and path in saved:
                _note(issues, "pins", "CREATE_RESULT_SELECTION_MISSING", missing=True)
            continue
        body = data.get(role)
        if body is None:
            _note(issues, "pins", role.upper() + "_UNVERIFIED")
            continue
        expected = RuntimePin(attempt.attempt_id, attempt.payload_sha256, body.get("providerTaskId"),
                              "runtime_" + role, item.record)
        actual = saved.get(item.record.path)
        if actual is None:
            _note(issues, "pins", role.upper() + "_PIN_MISSING", missing=True)
        elif actual != expected:
            _note(issues, "pins", role.upper() + "_PIN_MISMATCH")


def _ref_matches(body, field, expected, category, issues, *, optional=False):
    value = body.get(field)
    if value is None:
        if not optional:
            _note(issues, category, field.upper() + "_REFERENCE_MISSING", missing=True)
        return
    try:
        if type(value) is not dict or set(value) != {"path", "sha256", "size"}:
            raise ValueError("Reference fields are invalid.")
        reference = StoredArtifact(**value)
        _reference(reference)
    except Exception:
        _note(issues, category, field.upper() + "_REFERENCE_INVALID")
        return
    if expected is None:
        _note(issues, category, field.upper() + "_SELECTION_MISSING", missing=True)
    elif reference != expected:
        _note(issues, category, field.upper() + "_REFERENCE_MISMATCH")


def _single(rows, category, code, issues):
    if len(rows) != 1:
        _note(issues, category, code + ("_MISSING" if not rows else "_DUPLICATE"), missing=not rows)
        return None
    return rows[0]


def _event(snapshot, event_id, attempt_id, kinds, category, issues):
    if event_id is None or snapshot is None:
        _note(issues, category, "EVENT_REFERENCE_MISSING", missing=True)
        return None, None
    if type(event_id) is not int or event_id <= 0:
        _note(issues, category, "EVENT_REFERENCE_INVALID")
        return None, None
    row = _single([row for row in snapshot.submission_events if row.event_id == event_id],
                  category, "SUBMISSION_EVENT", issues)
    if row is None:
        return None, None
    if row.attempt_id != attempt_id or row.event_kind not in kinds:
        _note(issues, category, "SUBMISSION_EVENT_MISMATCH")
        return None, None
    has_task = row.event_kind in ("TASK_RECORDED", "TASK_RECONCILED")
    if ((not has_task and row.provider_task_id is not None)
            or (row.event_kind != "TASK_RECONCILED" and row.evidence_sha256 is not None)
            or (row.event_kind == "TASK_RECONCILED" and not _matches(row.evidence_sha256, r"[0-9a-f]{64}"))):
        _note(issues, category, "SUBMISSION_EVENT_FIELDS_INVALID")
        return None, None
    linked = _single([link for link in snapshot.run_events
                      if link.event_kind == "SUBMISSION_EVENT" and link.submission_event_id == event_id],
                     category, "EVENT_SEQUENCE", issues)
    if linked is None:
        return row, None
    if (linked.run_id != snapshot.run_id or linked.attempt_id != attempt_id or linked.sequence <= 0
            or linked.recorded_unix != row.recorded_unix or linked.record_json is not None):
        _note(issues, category, "EVENT_SEQUENCE_MISMATCH")
        return row, None
    if sum(link.sequence == linked.sequence for link in snapshot.run_events) != 1:
        _note(issues, category, "EVENT_SEQUENCE_DUPLICATE")
        return row, None
    return row, linked.sequence


def _http(body, category, issues, *, exact=None):
    status = body.get("httpStatus")
    if status is None:
        _note(issues, category, "HTTP_STATUS_MISSING", missing=True)
    elif type(status) is not int or not (status == exact if exact is not None else 200 <= status < 300):
        _note(issues, category, "HTTP_STATUS_INVALID")


def _same(body, expected, category, code, issues):
    try:
        _match_fields(body, expected)
    except Exception:
        _note(issues, category, code)


def _selected(item):
    return item.record if item is not None else None


def _ledger_rows(snapshot, attempt, issues):
    if snapshot is None:
        for category in ("creation", "task_get", "download"):
            _note(issues, category, "LEDGER_MISSING", missing=True)
        return None, None, None, None
    actual_hash = _check_input(snapshot)
    if actual_hash != snapshot.sha256 or snapshot.run_id != attempt.run_id:
        _note(issues, "creation", "LEDGER_IDENTITY_MISMATCH")
    reservation = _single([row for row in snapshot.reservations if row.attempt_id == attempt.attempt_id],
                          "creation", "RESERVATION", issues)
    job = _single([row for row in snapshot.submission_jobs if row.attempt_id == attempt.attempt_id],
                  "task_get", "SUBMISSION_JOB", issues)
    output = _single([row for row in snapshot.completed_outputs if row.attempt_id == attempt.attempt_id],
                     "download", "COMPLETED_OUTPUT", issues)
    if reservation is not None and (reservation.run_id != attempt.run_id
            or reservation.payload_sha256 != attempt.payload_sha256 or reservation.cost_cents != attempt.reserved_cents):
        _note(issues, "creation", "RESERVATION_MISMATCH")
    if job is not None and (job.provider_task_id != attempt.provider_task_id or job.state != "SUCCEEDED"):
        _note(issues, "task_get", "SUBMISSION_JOB_MISMATCH")
    if output is not None:
        expected = (attempt.provider_task_id, attempt.provider_success.sha256, attempt.raw_output,
                    attempt.graph.review.raw_metadata, attempt.completed_unix)
        actual = (output.provider_task_id, output.provider_evidence_sha256,
                  StoredArtifact(output.raw_path, output.raw_sha256, output.raw_size),
                  StoredArtifact(output.metadata_path, output.metadata_sha256, output.metadata_size), output.completed_unix)
        if actual != expected:
            _note(issues, "download", "COMPLETED_OUTPUT_MISMATCH")
    return actual_hash, reservation, job, output


def _check_request(chain, data, attempt, gate, issues):
    body = data.get("request_check")
    if body is None:
        return None
    try:
        checked = _check_request_data(body, chain.request_check.record,
                                      run_id=attempt.run_id, attempt_id=attempt.attempt_id)
        _match_fields(body, {"preparation": asdict(attempt.graph.review.preparation.preparation)})
        if (checked.source_kind != attempt.source_kind_declared or checked.frozen.mode != attempt.mode
                or checked.frozen.preparation_sha256 != attempt.graph.review.preparation.preparation.sha256
                or checked.frozen.payload_sha256 != attempt.payload_sha256
                or checked.frozen.estimated_cents != attempt.reserved_cents):
            raise ValueError("Checked request does not match the archived attempt.")
        for role, expected, reference in (
            ("request", checked.frozen.request_body, attempt.request),
            ("frozen_request", checked.frozen.record_bytes, attempt.frozen_request),
        ):
            item = getattr(chain, role)
            if item is not None and (item.contents != expected or item.record != reference):
                _note(issues, "creation", role.upper() + "_BYTES_MISMATCH")
        if chain.request_check.record != attempt.request_check:
            _note(issues, "creation", "REQUEST_CHECK_REFERENCE_MISMATCH")
        if sha256(checked.frozen.request_body).hexdigest() != attempt.request_body_sha256:
            _note(issues, "creation", "REQUEST_BODY_HASH_MISMATCH")
        if gate is None or gate.recorded_unix is None:
            _note(issues, "creation", "REQUEST_GATE_TIME_MISSING", missing=True)
        else:
            checked.require_current(gate.recorded_unix)
        return checked
    except Exception:
        _note(issues, "creation", "CHECKED_REQUEST_INVALID")
        return None


def _check_start(started, snapshot, context, gate, attempt, issues):
    if started is None:
        _note(issues, "creation", "RUN_START_MISSING", missing=True)
        return
    if (type(started) is not RunStart or not _time(started.started_unix) or not _time(started.deadline_unix)):
        raise RuntimeChainError("Run start must use bounded immutable clock values.")
    _reference(started.record, limit=MAX_RUN_START_BYTES)
    if (started.run_id != attempt.run_id or started.record.path != "run-start.json"
            or started.deadline_unix != started.started_unix + MAX_ELAPSED_SECONDS):
        _note(issues, "creation", "RUN_START_INVALID")
    if context is not None:
        _ref_matches(context, "runStart", started.record, "creation", issues)
    approval = _single(list(snapshot.approved_runs), "creation", "APPROVAL", issues) if snapshot else None
    if approval is not None and (approval.run_id != attempt.run_id
            or not started.started_unix <= approval.started_unix < approval.deadline_unix <= started.deadline_unix):
        _note(issues, "creation", "APPROVAL_OUTSIDE_EXPERIMENT")
    if gate is not None:
        if gate.recorded_unix is None:
            _note(issues, "creation", "SEND_GATE_TIME_MISSING", missing=True)
        elif (not started.started_unix <= gate.recorded_unix < started.deadline_unix
              or approval is not None and not approval.started_unix <= gate.recorded_unix < approval.deadline_unix):
            _note(issues, "creation", "SEND_GATE_OUTSIDE_AUTHORITY")


def _check_reconciliation(accepted, binding, job, attempt, issues):
    if accepted is None:
        _note(issues, "creation", "ACCEPTED_RECONCILIATION_MISSING", missing=True)
        return
    if type(accepted) is not AcceptedReconciliation:
        raise RuntimeChainError("Reconciliation must be explicitly accepted by the trusted caller.")
    _record_shape(accepted.evidence, MAX_RECONCILIATION_BYTES)
    item = accepted.evidence
    if (accepted.run_id != attempt.run_id or accepted.attempt_id != attempt.attempt_id
            or accepted.payload_sha256 != attempt.payload_sha256 or accepted.provider_task_id != attempt.provider_task_id
            or len(item.contents) != item.record.size or sha256(item.contents).hexdigest() != item.record.sha256
            or binding.evidence_sha256 != item.record.sha256 or job is not None and job.evidence_sha256 != item.record.sha256):
        _note(issues, "creation", "ACCEPTED_RECONCILIATION_MISMATCH")


def _check_creation(chain, data, attempt, snapshot, started, job, accepted, issues):
    context, result, get = (data.get(role) for role in ("create_context", "create_result", "task_get"))
    claim, claim_sequence = _event(snapshot, context.get("claimEventId") if context else None,
        attempt.attempt_id, ("CLAIMED",), "creation", issues)
    gate, gate_sequence = _event(snapshot, chain.send_gate_event_id, attempt.attempt_id,
                                 ("SEND_GATE_PASSED",), "creation", issues)
    binding, binding_sequence = _event(snapshot, get.get("taskBindingEventId") if get else None,
        attempt.attempt_id, ("TASK_RECORDED", "TASK_RECONCILED"), "task_get", issues)
    if binding is not None and binding.provider_task_id != attempt.provider_task_id:
        _note(issues, "task_get", "TASK_BINDING_MISMATCH")
    if claim_sequence is not None and gate_sequence is not None and claim_sequence >= gate_sequence:
        _note(issues, "creation", "CLAIM_GATE_ORDER_INVALID")
    if gate_sequence is not None and binding_sequence is not None and gate_sequence >= binding_sequence:
        _note(issues, "creation", "GATE_TASK_ORDER_INVALID")
    _check_start(started, snapshot, context, gate, attempt, issues)
    checked = _check_request(chain, data, attempt, gate, issues)
    if context is not None:
        _same(context, {"apiOrigin": API_URL, "sdkVersion": SDK_VERSION}, "creation", "CREATE_CLIENT_MISMATCH", issues)
        if context["sdkVersion"] != attempt.sdk_version:
            _note(issues, "creation", "ARCHIVE_SDK_MISMATCH")
        for key, expected in (("requestCheck", attempt.request_check), ("request", attempt.request),
                              ("frozenRequest", attempt.frozen_request)):
            _ref_matches(context, key, expected, "creation", issues)

    mode = ("RECONCILED" if binding is not None and binding.event_kind == "TASK_RECONCILED" else
            "DIRECT" if binding is not None or result is not None and result.get("localState") == "QUEUED"
            else "UNESTABLISHED")
    result_event, result_sequence = None, None
    if result is not None:
        _ref_matches(result, "createContext", _selected(chain.create_context), "creation", issues)
        _same(result, {"claimEventId": context.get("claimEventId") if context else None,
                       "sendGateEventId": chain.send_gate_event_id}, "creation", "CREATE_EVENT_LINK_MISMATCH", issues)
        unknown = mode != "DIRECT"
        expected_state = "SUBMISSION_UNKNOWN" if unknown else "QUEUED"
        expected_task = None if unknown else attempt.provider_task_id
        _same(result, {"localState": expected_state, "providerTaskId": expected_task},
              "creation", "CREATE_RESULT_MISMATCH", issues)
        event_kind = "SUBMISSION_UNKNOWN" if unknown else "TASK_RECORDED"
        result_event, result_sequence = _event(snapshot, result.get("resultEventId"), attempt.attempt_id,
                                               (event_kind,), "creation", issues)
        if result_event is not None and result_event.provider_task_id != expected_task:
            _note(issues, "creation", "CREATE_RESULT_TASK_MISMATCH")
        if mode == "DIRECT":
            _http(result, "creation", issues)
            if binding is not None and result_event != binding:
                _note(issues, "creation", "CREATE_TASK_BINDING_MISMATCH")
    elif mode != "RECONCILED":
        _note(issues, "creation", "CREATE_RESULT_MISSING", missing=True)
    if mode == "RECONCILED":
        _check_reconciliation(accepted, binding, job, attempt, issues)
        if result_sequence is None and snapshot is not None:
            unknown = _single([row for row in snapshot.submission_events if row.attempt_id == attempt.attempt_id
                               and row.event_kind in ("SUBMISSION_UNKNOWN", "OWNER_RECOVERED_UNKNOWN")],
                              "creation", "UNKNOWN_HISTORY", issues)
            if unknown is not None:
                result_event, result_sequence = _event(snapshot, unknown.event_id, attempt.attempt_id,
                    ("SUBMISSION_UNKNOWN", "OWNER_RECOVERED_UNKNOWN"), "creation", issues)
        if (gate_sequence is not None and result_sequence is not None and binding_sequence is not None
                and not gate_sequence < result_sequence < binding_sequence):
            _note(issues, "creation", "RECONCILIATION_ORDER_INVALID")
    elif mode == "UNESTABLISHED":
        _note(issues, "creation", "CREATION_ROUTE_MISSING", missing=True)
    if accepted is not None and mode != "RECONCILED":
        _note(issues, "creation", "RECONCILIATION_NOT_APPLICABLE")
    return mode, claim, gate, binding, result_event, checked


def _check_get_download(chain, data, attempt, mode, output_hosts, issues):
    get, success = data.get("task_get"), data.get("success_response")
    intent, result = data.get("download_intent"), data.get("download_result")
    if get is not None:
        _same(get, {"providerTaskId": attempt.provider_task_id, "providerStatus": "SUCCEEDED",
                    "apiOrigin": API_URL, "sdkVersion": SDK_VERSION}, "task_get", "TASK_GET_IDENTITY_MISMATCH", issues)
        _http(get, "task_get", issues)
        _ref_matches(get, "createContext", _selected(chain.create_context), "task_get", issues)
        _ref_matches(get, "createResult", _selected(chain.create_result), "task_get", issues,
                     optional=mode == "RECONCILED" and chain.create_result is None)
        _ref_matches(get, "response", attempt.provider_success, "task_get", issues)
    if chain.success_response is not None and chain.success_response.record != attempt.provider_success:
        _note(issues, "task_get", "SUCCESS_RESPONSE_REFERENCE_MISMATCH")
    url = None
    if not output_hosts:
        _note(issues, "download", "OUTPUT_HOST_AUTHORITY_MISSING", missing=True)
    if success is not None:
        _same(success, {"id": attempt.provider_task_id, "status": "SUCCEEDED"},
              "task_get", "SUCCESS_RESPONSE_IDENTITY_MISMATCH", issues)
        if output_hosts:
            try:
                url = _output_url_from_data(success, attempt.provider_task_id, output_hosts)
            except Exception:
                _note(issues, "download", "OUTPUT_URL_INVALID")
    if intent is not None:
        _same(intent, {"providerTaskId": attempt.provider_task_id,
                       "successResponseSha256": attempt.provider_success.sha256},
              "download", "DOWNLOAD_INTENT_MISMATCH", issues)
        _ref_matches(intent, "taskGet", _selected(chain.task_get), "download", issues)
        if url is not None and intent["outputHost"] != url.host:
            _note(issues, "download", "OUTPUT_HOST_MISMATCH")
    if result is not None:
        _ref_matches(result, "intent", _selected(chain.download_intent), "download", issues)
        _same(result, {"providerTaskId": attempt.provider_task_id, "readOutcome": "BODY_RETURNED",
                       "rawSha256": attempt.raw_output.sha256, "rawSize": attempt.raw_output.size},
              "download", "DOWNLOAD_RESULT_MISMATCH", issues)
        _http(result, "download", issues, exact=200)
        if intent is not None and result["operationId"] != intent["operationId"]:
            _note(issues, "download", "DOWNLOAD_OPERATION_MISMATCH")


def _check_transports(data, mode, issues):
    injected = []
    for role in _RECEIPTS:
        if role == "create_result" and role not in data and mode == "RECONCILED":
            continue
        transport = data.get(role, {}).get("transportClass")
        if transport is None:
            _note(issues, "default_network", role.upper() + "_TRANSPORT_MISSING", missing=True)
        elif transport == "INJECTED":
            injected.append("runtime_" + role)
            _note(issues, "default_network", role.upper() + "_TRANSPORT_INJECTED")
        elif transport != "DEFAULT_NETWORK":
            _note(issues, "default_network", role.upper() + "_TRANSPORT_INVALID")
    for first, second, category in (("create_context", "create_result", "creation"),
                                     ("download_intent", "download_result", "download")):
        values = [data.get(role, {}).get("transportClass") for role in (first, second)]
        if all(value is not None for value in values) and values[0] != values[1]:
            _note(issues, category, "TRANSPORT_LINK_MISMATCH")
    return tuple(injected)


def _check_observations(data, mode, claim, gate, binding, result_event, checked, attempt, issues):
    times = {}

    def save(name, value, *, required=True):
        if value is None:
            if required:
                _note(issues, "observations", name.upper() + "_MISSING", missing=True)
            times[name] = None
        elif not _time(value):
            _note(issues, "observations", name.upper() + "_INVALID")
            times[name] = None
        else:
            times[name] = value

    def before(first, second):
        if times.get(first) is not None and times.get(second) is not None and times[first] > times[second]:
            _note(issues, "observations", first.upper() + "_AFTER_" + second.upper())

    context, result = data.get("create_context"), data.get("create_result")
    save("claim", claim.recorded_unix if claim else None)
    save("gate", gate.recorded_unix if gate else None)
    save("task_binding", binding.recorded_unix if binding else None)
    save("context", context.get("recordedAtUnix") if context else None)
    save("checked", checked.checked_unix if checked else None)
    save("completion", attempt.completed_unix)
    before("checked", "context")
    before("claim", "context")
    before("context", "gate")
    before("gate", "task_binding")
    if result is not None:
        save("create_recorded", result.get("recordedAtUnix"))
        save("create_response", result.get("responseObservedAtUnix"), required=mode == "DIRECT")
        save("create_event", result_event.recorded_unix if result_event else None)
        before("gate", "create_response")
        before("create_response", "create_event")
        before("create_event", "create_recorded")
    elif mode != "RECONCILED":
        _note(issues, "observations", "CREATE_OBSERVATIONS_MISSING", missing=True)
    for role in ("task_get", "download_result"):
        body = data.get(role, {})
        for name, field in (("start", "requestStartedAtUnix"), ("response", "responseObservedAtUnix"),
                             ("finish", "bodyFinishedAtUnix")):
            save(role + "_" + name, body.get(field))
        before(role + "_start", role + "_response")
        before(role + "_response", role + "_finish")
    before("task_binding", "task_get_start")
    before("create_recorded", "task_get_start")
    before("task_get_finish", "download_result_start")
    before("download_result_finish", "completion")
    status = result.get("httpStatus") if result else None
    if status is not None and (type(status) is not int or not 100 <= status <= 599):
        _note(issues, "observations", "CREATE_HTTP_OBSERVATION_INVALID")
    return (type(status) is int and 100 <= status <= 599
            and result is not None and _time(result.get("responseObservedAtUnix")))


def audit_runtime_chain(*, run_id: str, attempt: ExportAttemptFacts, snapshot: RunSnapshot | None,
                        run_start: RunStart | None, pins: RuntimePinSnapshot | None,
                        chain: RuntimeChain, output_hosts: frozenset[str],
                        reconciliation: AcceptedReconciliation | None = None) -> RuntimeChainAudit:
    """Check one chain supplied by the trusted caller, without external work.

    The operator must select the private ledger and external pin index using
    their existing read-only readers. A pin snapshot supplied by imported JSON
    is not trusted. Its digest includes private owner fields and is retained,
    not reconstructed here. Accepted reconciliation is a separate caller input.

    Consistent injected observations remain identifiable fixture evidence.
    Default transport alone does not authenticate an account or generation.
    No current clock, provider call, raw-video decode, or release decision occurs.
    """
    try:
        if not _matches(run_id, r"[A-Za-z0-9_-]{1,64}") or type(chain) is not RuntimeChain:
            raise RuntimeChainError("Runtime audit needs its fixed immutable inputs.")
        _archive_references(attempt)
        if (type(output_hosts) is not frozenset or len(output_hosts) > MAX_OUTPUT_HOSTS
                or any(type(host) is not str or not 1 <= len(host) <= 253 for host in output_hosts)
                or chain.send_gate_event_id is not None and (type(chain.send_gate_event_id) is not int
                                                            or not 0 < chain.send_gate_event_id < 2 ** 63)):
            raise RuntimeChainError("Runtime selection and host bounds are invalid.")
        if reconciliation is not None:
            if (type(reconciliation) is not AcceptedReconciliation
                    or not _matches(reconciliation.run_id, r"[A-Za-z0-9_-]{1,64}")
                    or not _matches(reconciliation.attempt_id, r"att-[0-9a-f]{32}")
                    or not _matches(reconciliation.payload_sha256, r"[0-9a-f]{64}")
                    or not _matches(reconciliation.provider_task_id, r"[A-Za-z0-9_-]{1,128}")):
                raise RuntimeChainError("Accepted reconciliation identity is invalid.")
            _record_shape(reconciliation.evidence, MAX_RECONCILIATION_BYTES)
        issues = []
        if attempt.run_id != run_id:
            _note(issues, "creation", "RUN_SELECTION_MISMATCH")
        data = _load_chain(chain, attempt, issues)
        _check_pins(pins, chain, data, attempt, issues)
        snapshot_hash, reservation, job, output = _ledger_rows(snapshot, attempt, issues)
        mode, claim, gate, binding, result_event, checked = _check_creation(
            chain, data, attempt, snapshot, run_start, job, reconciliation, issues)
        _check_get_download(chain, data, attempt, mode, output_hosts, issues)
        injected = _check_transports(data, mode, issues)
        response_observed = _check_observations(data, mode, claim, gate, binding, result_event, checked, attempt, issues)
        pin_assessment = _assessment([issue for issue in issues if issue.category == "pins"])
        assessments = tuple(_assessment([issue for issue in issues if issue.category == category])
                            for category in ("creation", "task_get", "download", "observations"))
        default_network = _assessment([issue for issue in issues if issue.category == "default_network"],
                                      shared=pin_assessment)
        return RuntimeChainAudit(
            run_id, attempt.attempt_id, attempt.payload_sha256, attempt.provider_task_id,
            snapshot_hash, pins.sha256 if pins else None, mode, response_observed, injected,
            tuple(item.record for role in (*_RECEIPTS, *_PRIVATE) if (item := getattr(chain, role)) is not None),
            reconciliation.evidence.record if reconciliation else None,
            pin_assessment, *assessments, default_network, tuple(issues),
        )
    except Exception:
        raise RuntimeChainError("Runtime chain could not be assessed from the supplied bounded values.") from None
