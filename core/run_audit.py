"""Audit captured local records and spending boundaries without performing I/O."""

from dataclasses import dataclass
import json
import math
import re
from typing import Literal

from core import run_snapshot as captured


Verdict = Literal["PASS", "FAIL", "BLOCKED", "NOT_RUN"]
Coverage = Literal["COMPLETE", "INCOMPLETE"]


class RunAuditError(ValueError):
    """The supplied value is not a bounded, typed ledger snapshot."""


@dataclass(frozen=True)
class AuditAssessment:
    verdict: Verdict
    coverage: Coverage


@dataclass(frozen=True)
class AuditIssue:
    category: str
    code: str
    missing: bool = False
    attempt_id: str | None = None
    sequence: int | None = None
    event_id: int | None = None


@dataclass(frozen=True)
class SpendingBoundary:
    sequence: int
    event_kind: str
    attempt_id: str | None
    attempt_count: int | None
    accounted_cents: int | None
    verdict: Verdict


@dataclass(frozen=True)
class RunAudit:
    run_id: str
    snapshot_sha256: str
    record_consistency: AuditAssessment
    current_limits: AuditAssessment
    spending_replay: AuditAssessment
    elapsed: AuditAssessment
    coverage: Coverage
    issues: tuple[AuditIssue, ...]
    reservation_count: int
    recorded_actual_cents: int | None
    held_reserve_cents: int | None
    actual_or_reserved_cents: int | None
    unknown_attempt_ids: tuple[str, ...]
    in_flight_attempt_ids: tuple[str, ...]
    unresolved_billing_attempt_ids: tuple[str, ...]
    overquote_attempt_ids: tuple[str, ...]
    limit_cents: int | None
    max_attempts: int | None
    approval_started_unix: float | None
    deadline_unix: float | None
    time_basis: str
    boundaries: tuple[SpendingBoundary, ...]


_CATEGORIES = ("records", "limits", "replay", "elapsed")
_JOB_STATES = {"SUBMITTING", "SUBMISSION_UNKNOWN", "QUEUED", "RUNNING", "FAILED", "CANCELED", "SUCCEEDED"}
_SUBMISSION_KINDS = {"CLAIMED", "SEND_GATE_PASSED", "TASK_RECORDED", "SUBMISSION_UNKNOWN",
                     "OWNER_RECOVERED_UNKNOWN", "TASK_RECONCILED"}
_PROVIDER_STATUSES = {"PENDING", "THROTTLED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}
_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def _note(issues, category, code, attempt=None, sequence=None, event=None, *, missing=False):
    issues.append(AuditIssue(category, code, missing, attempt if _attempt(attempt) else None, sequence, event))


def _assessment(issues, category):
    related = [issue for issue in issues if issue.category == category]
    incomplete = any(issue.missing for issue in related)
    verdict = "FAIL" if any(not issue.missing for issue in related) else "BLOCKED" if incomplete else "PASS"
    return AuditAssessment(verdict, "INCOMPLETE" if incomplete else "COMPLETE")


def _matches(value, pattern):
    return type(value) is str and re.fullmatch(pattern, value) is not None


def _attempt(value):
    return _matches(value, r"att-[0-9a-f]{32}")


def _hash(value):
    return _matches(value, r"[0-9a-f]{64}")


def _task(value):
    return _matches(value, r"[A-Za-z0-9_-]{1,128}")


def _time(value):
    return type(value) in (int, float) and 0 <= value <= 253_402_300_799 and math.isfinite(value)


def _cents(value, *, zero=False):
    return type(value) is int and (0 if zero else 1) <= value <= 2 ** 63 - 1


def _check_input(snapshot):
    if type(snapshot) is not captured.RunSnapshot or not _matches(snapshot.run_id, r"[A-Za-z0-9_-]{1,64}"):
        raise RunAuditError("Audit requires a typed run snapshot.")
    names = tuple(name for name, _ in captured._TABLES)
    for values, limit in ((snapshot.missing_tables, len(names)),
                          (snapshot.relationship_issues, len(captured._RELATIONS) + 2)):
        if (type(values) is not tuple or len(values) > limit
                or any(type(value) is not str or len(value) > 128 for value in values)):
            raise RunAuditError("Snapshot coverage fields must be bounded immutable strings.")
    allowed_issues = {f"{child}.{key}_without_{parent}" for child, key, parent, _ in captured._RELATIONS}
    allowed_issues.update(("run_events.cross_run_attempt", "run_events.submission_attempt_mismatch"))
    if (len(set(snapshot.missing_tables)) != len(snapshot.missing_tables)
            or not set(snapshot.missing_tables) <= set(names)
            or len(set(snapshot.relationship_issues)) != len(snapshot.relationship_issues)
            or not set(snapshot.relationship_issues) <= allowed_issues or not _hash(snapshot.sha256)):
        raise RunAuditError("Snapshot coverage or hash fields are invalid.")
    total_rows = total_bytes = 0
    types = {str: (str,), int: (int,), float: (int, float), str | None: (str, type(None)),
             int | None: (int, type(None)), float | None: (int, float, type(None))}
    tables = []
    for name, row_type in captured._TABLES:
        rows = getattr(snapshot, name)
        if type(rows) is not tuple or len(rows) > captured.MAX_TABLE_ROWS or (name in snapshot.missing_tables and rows):
            raise RunAuditError("Snapshot table rows are invalid or exceed their bounds.")
        total_rows += len(rows)
        if total_rows > captured.MAX_TOTAL_ROWS:
            raise RunAuditError("Snapshot exceeds its row bound.")
        for row in rows:
            if type(row) is not row_type:
                raise RunAuditError("Snapshot row has the wrong record type.")
            for field, kind in row_type.__annotations__.items():
                value = getattr(row, field)
                if type(value) not in types[kind] or (type(value) is float and not math.isfinite(value)):
                    raise RunAuditError("Snapshot cell has the wrong scalar type.")
                if type(value) is int and not -(2 ** 63) <= value < 2 ** 63:
                    raise RunAuditError("Snapshot integer exceeds SQLite's signed range.")
                if type(value) is str:
                    limit = 4096 if field == "record_json" else 1024
                    if len(value) > limit:
                        raise RunAuditError("Snapshot text exceeds its cell bound.")
                    try:
                        size = len(value.encode("utf-8"))
                    except UnicodeError:
                        raise RunAuditError("Snapshot text is not valid UTF-8.") from None
                    if size > limit:
                        raise RunAuditError("Snapshot text exceeds its cell bound.")
                else:
                    size = 0 if value is None else 8
                total_bytes += size
        tables.append(rows)
    if total_rows > captured.MAX_TOTAL_ROWS or total_bytes > captured.MAX_CAPTURE_BYTES:
        raise RunAuditError("Snapshot exceeds its aggregate bound.")
    return captured._logical_hash(snapshot.run_id, tuple(tables), snapshot.missing_tables,
                                  snapshot.relationship_issues)


def _index(rows, field, issues, code):
    result = {}
    for row in rows:
        key = getattr(row, field)
        if key in result:
            _note(issues, "records", code, getattr(row, "attempt_id", None))
        else:
            result[key] = row
    return result


def _check_current(snapshot, issues):
    for name in snapshot.missing_tables:
        _note(issues, "records", "TABLE_MISSING:" + name, missing=True)
    for flag in snapshot.relationship_issues:
        parent = flag.split("_without_")[-1]
        _note(issues, "records", "RELATIONSHIP:" + flag, missing=parent in snapshot.missing_tables)
    approvals = snapshot.approved_runs
    approval = approvals[0] if len(approvals) == 1 else None
    if not approvals:
        for category in _CATEGORIES:
            _note(issues, category, "APPROVAL_MISSING", missing=True)
    elif len(approvals) != 1:
        for category in _CATEGORIES:
            _note(issues, category, "APPROVAL_DUPLICATE")
    if approval is not None:
        if (approval.run_id != snapshot.run_id or not approval.approved_by.strip() or len(approval.approved_by) > 200
                or not 1 <= approval.limit_cents <= 1000 or not 1 <= approval.max_attempts <= 6
                or not 1 <= approval.cost_cents <= approval.limit_cents or approval.time_stopped not in (0, 1)):
            _note(issues, "records", "APPROVAL_INVALID")
            _note(issues, "limits", "APPROVAL_INVALID")
        if (not _time(approval.started_unix) or not math.isfinite(approval.deadline_unix)
                or not 0 < approval.deadline_unix - approval.started_unix <= 10800
                or not _time(approval.last_checked_unix)
                or not approval.started_unix <= approval.last_checked_unix < approval.deadline_unix):
            _note(issues, "records", "APPROVAL_TIME_INVALID")
            _note(issues, "elapsed", "APPROVAL_TIME_INVALID")
    reservations = _index(snapshot.reservations, "attempt_id", issues, "RESERVATION_DUPLICATE")
    charges = _index(snapshot.actual_charges, "attempt_id", issues, "CHARGE_DUPLICATE")
    jobs = _index(snapshot.submission_jobs, "attempt_id", issues, "JOB_DUPLICATE")
    completed = _index(snapshot.completed_outputs, "attempt_id", issues, "COMPLETION_DUPLICATE")
    keys, tasks = set(), set()
    for row in snapshot.reservations:
        if (row.run_id != snapshot.run_id or not _attempt(row.attempt_id) or not _hash(row.key_sha256)
                or not _hash(row.payload_sha256) or not _cents(row.cost_cents) or row.state != "RESERVED"
                or (approval is not None and row.cost_cents != approval.cost_cents)):
            _note(issues, "records", "RESERVATION_INVALID", row.attempt_id)
        if row.key_sha256 in keys:
            _note(issues, "records", "RESERVATION_KEY_DUPLICATE", row.attempt_id)
        keys.add(row.key_sha256)
    for row in snapshot.actual_charges:
        if not _attempt(row.attempt_id) or not _cents(row.actual_cents, zero=True) or not _hash(row.evidence_sha256):
            _note(issues, "records", "CHARGE_INVALID", row.attempt_id)
        if row.attempt_id not in reservations:
            _note(issues, "records", "CHARGE_ORPHAN", row.attempt_id)
    for row in snapshot.submission_jobs:
        unknown = row.state in ("SUBMITTING", "SUBMISSION_UNKNOWN")
        if (not _attempt(row.attempt_id) or row.state not in _JOB_STATES
                or (unknown and (row.provider_task_id is not None or row.evidence_sha256 is not None))
                or (not unknown and not _task(row.provider_task_id))
                or (row.evidence_sha256 is not None and not _hash(row.evidence_sha256))):
            _note(issues, "records", "JOB_INVALID", row.attempt_id)
        if row.attempt_id not in reservations:
            _note(issues, "records", "JOB_ORPHAN", row.attempt_id)
        if row.provider_task_id is not None:
            if row.provider_task_id in tasks:
                _note(issues, "records", "TASK_DUPLICATE", row.attempt_id)
            tasks.add(row.provider_task_id)
        if row.state == "SUCCEEDED" and row.attempt_id not in completed:
            _note(issues, "records", "COMPLETION_MISSING", row.attempt_id,
                  missing="completed_outputs" in snapshot.missing_tables)
    for row in snapshot.completed_outputs:
        prefix = f"attempts/{row.attempt_id}/output/"
        if (not _attempt(row.attempt_id) or not _task(row.provider_task_id) or not _hash(row.provider_evidence_sha256)
                or row.raw_path != prefix + "raw.mp4" or row.metadata_path != prefix + "metadata.json"
                or not _hash(row.raw_sha256) or not _hash(row.metadata_sha256)
                or not 1 <= row.raw_size <= 512 * 1024 * 1024
                or not 1 <= row.metadata_size <= 512 * 1024 * 1024 or not _time(row.completed_unix)):
            _note(issues, "records", "COMPLETION_INVALID", row.attempt_id)
        job = jobs.get(row.attempt_id)
        if job is None or job.state != "SUCCEEDED" or job.provider_task_id != row.provider_task_id:
            _note(issues, "records", "COMPLETION_JOB_MISMATCH", row.attempt_id,
                  missing=job is None and "submission_jobs" in snapshot.missing_tables)
    return approval, reservations, charges, jobs, completed


def _current_money(snapshot, approval, reservations, charges, issues):
    valid = (len(reservations) == len(snapshot.reservations) and len(charges) == len(snapshot.actual_charges)
             and all(_cents(row.cost_cents) for row in snapshot.reservations)
             and all(_cents(row.actual_cents, zero=True) and row.attempt_id in reservations
                     for row in snapshot.actual_charges))
    for name in ("approved_runs", "reservations", "actual_charges"):
        if name in snapshot.missing_tables:
            _note(issues, "limits", "TABLE_MISSING:" + name, missing=True)
    actual = sum(row.actual_cents for row in snapshot.actual_charges) if valid else None
    held = sum(row.cost_cents for row in snapshot.reservations if row.attempt_id not in charges) if valid else None
    total = actual + held if valid else None
    if not valid:
        _note(issues, "limits", "MONEY_RECORDS_INVALID")
    if len(snapshot.reservations) > 6 or (approval is not None and len(snapshot.reservations) > approval.max_attempts):
        _note(issues, "limits", "ATTEMPT_CAP_EXCEEDED")
    if total is not None and (total > 1000 or (approval is not None and total > approval.limit_cents)):
        _note(issues, "limits", "MONEY_CAP_EXCEEDED")
    overquotes = []
    for attempt, charge in charges.items():
        reservation = reservations.get(attempt)
        if (reservation is not None and _attempt(attempt) and _cents(reservation.cost_cents)
                and _cents(charge.actual_cents, zero=True) and charge.actual_cents > reservation.cost_cents):
            overquotes.append(attempt)
    return actual, held, total, tuple(sorted(overquotes))


def _check_polls(snapshot, jobs, completed, issues):
    _index(snapshot.poll_events, "event_id", issues, "POLL_EVENT_DUPLICATE")
    histories = {}
    for row in sorted(snapshot.poll_events, key=lambda row: row.event_id):
        histories.setdefault(row.attempt_id, []).append(row)
        if (row.event_id <= 0 or not _attempt(row.attempt_id) or not _time(row.observed_unix)
                or row.poll_status not in ("OBSERVED", "POLL_ERROR", "POLL_DEADLINE_REACHED")
                or row.provider_status not in _PROVIDER_STATUSES | {None}
                or (row.provider_status is None) != (row.evidence_sha256 is None)
                or (row.evidence_sha256 is not None and not _hash(row.evidence_sha256))):
            _note(issues, "records", "POLL_INVALID", row.attempt_id, event=row.event_id)
    for attempt, rows in histories.items():
        job = jobs.get(attempt)
        if job is None or not _task(job.provider_task_id):
            _note(issues, "records", "POLL_JOB_MISMATCH", attempt,
                  missing=job is None and "submission_jobs" in snapshot.missing_tables)
        status = evidence = None
        local_state = "QUEUED"
        observed_success = set()
        for row in rows:
            if local_state in ("FAILED", "CANCELED"):
                _note(issues, "records", "POLL_AFTER_TERMINAL", attempt, event=row.event_id)
            if row.poll_status == "OBSERVED":
                if row.provider_status is None or (status in _TERMINAL and row.provider_status != status):
                    _note(issues, "records", "POLL_OBSERVATION_CONFLICT", attempt, event=row.event_id)
                status, evidence = row.provider_status, row.evidence_sha256
                if status == "SUCCEEDED":
                    observed_success.add(evidence)
                local_state = {"RUNNING": "RUNNING", "FAILED": "FAILED", "CANCELLED": "CANCELED"}.get(status, local_state)
            elif (row.provider_status, row.evidence_sha256) != (status, evidence):
                _note(issues, "records", "POLL_RETAINED_EVIDENCE_MISMATCH", attempt, event=row.event_id)
        output = completed.get(attempt)
        if output is not None:
            if status != "SUCCEEDED" or evidence != output.provider_evidence_sha256 or evidence not in observed_success:
                _note(issues, "records", "COMPLETION_POLL_MISMATCH", attempt)
        if job is not None and job.state != ("SUCCEEDED" if output is not None else local_state):
            _note(issues, "records", "JOB_POLL_STATE_MISMATCH", attempt)
    for attempt, job in jobs.items():
        if attempt not in histories and (job.state in ("RUNNING", "FAILED", "CANCELED", "SUCCEEDED") or attempt in completed):
            _note(issues, "records", "POLL_HISTORY_MISSING", attempt, missing=True)


def _check_submission_history(snapshot, jobs, issues):
    events = _index(snapshot.submission_events, "event_id", issues, "SUBMISSION_EVENT_DUPLICATE")
    histories = {}
    for row in sorted(snapshot.submission_events, key=lambda row: row.event_id):
        histories.setdefault(row.attempt_id, []).append(row)
        task_event = row.event_kind in ("TASK_RECORDED", "TASK_RECONCILED")
        if (row.event_id <= 0 or not _attempt(row.attempt_id) or row.event_kind not in _SUBMISSION_KINDS
                or (row.recorded_unix is not None and not _time(row.recorded_unix))
                or (row.event_kind in ("CLAIMED", "SEND_GATE_PASSED") and row.recorded_unix is None)
                or (task_event and not _task(row.provider_task_id))
                or (not task_event and row.provider_task_id is not None)
                or (row.event_kind == "TASK_RECONCILED" and not _hash(row.evidence_sha256))
                or (row.event_kind != "TASK_RECONCILED" and row.evidence_sha256 is not None)):
            _note(issues, "records", "SUBMISSION_EVENT_INVALID", row.attempt_id, event=row.event_id)
    for attempt, rows in histories.items():
        state = task = evidence = None
        has_claim = has_gate = False
        for row in rows:
            kind = row.event_kind
            if state is None and kind != "CLAIMED":
                _note(issues, "records", "CLAIM_HISTORY_MISSING", attempt, missing=True)
                state = "SUBMISSION_UNKNOWN" if kind == "TASK_RECONCILED" else "SUBMITTING"
            expected = "SUBMISSION_UNKNOWN" if kind == "TASK_RECONCILED" else "SUBMITTING"
            if kind == "CLAIMED":
                if state is not None or has_claim:
                    _note(issues, "records", "SUBMISSION_TRANSITION_INVALID", attempt, event=row.event_id)
                state, has_claim = "SUBMITTING", True
            else:
                if state != expected:
                    _note(issues, "records", "SUBMISSION_TRANSITION_INVALID", attempt, event=row.event_id)
                if kind == "SEND_GATE_PASSED":
                    has_gate = True
                elif kind in ("SUBMISSION_UNKNOWN", "OWNER_RECOVERED_UNKNOWN"):
                    state = "SUBMISSION_UNKNOWN"
                elif kind in ("TASK_RECORDED", "TASK_RECONCILED"):
                    state, task, evidence = "QUEUED", row.provider_task_id, row.evidence_sha256
        if not has_claim:
            _note(issues, "replay", "CLAIM_HISTORY_MISSING", attempt, missing=True)
        if not has_gate and state != "SUBMITTING":
            _note(issues, "replay", "SEND_GATE_MISSING", attempt, missing=True)
            _note(issues, "elapsed", "SEND_GATE_MISSING", attempt, missing=True)
        job = jobs.get(attempt)
        if job is None:
            _note(issues, "records", "SUBMISSION_JOB_MISSING", attempt,
                  missing="submission_jobs" in snapshot.missing_tables)
        elif ((state == "QUEUED" and job.state not in ("QUEUED", "RUNNING", "FAILED", "CANCELED", "SUCCEEDED"))
              or (state != "QUEUED" and job.state != state)
              or job.provider_task_id != task or job.evidence_sha256 != evidence):
            _note(issues, "records", "SUBMISSION_JOB_MISMATCH", attempt)
    for attempt in jobs.keys() - histories.keys():
        for category in ("records", "replay", "elapsed"):
            _note(issues, category, "SUBMISSION_HISTORY_MISSING", attempt, missing=True)
    return events


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _replay(snapshot, approval, reservations, charges, events, issues):
    rows = sorted(snapshot.run_events, key=lambda row: row.sequence)
    sequences = _index(rows, "sequence", issues, "RUN_SEQUENCE_DUPLICATE")
    if len(sequences) != len(rows):
        _note(issues, "replay", "RUN_SEQUENCE_DUPLICATE")
    approval_fields = ("approved_by", "limit_cents", "max_attempts", "cost_cents", "started_unix", "deadline_unix")
    approval_record = {field: getattr(approval, field) for field in approval_fields} if approval else None
    seen_records, links, usable = {}, {}, []
    for row in rows:
        key = (row.event_kind, row.attempt_id)
        if (row.sequence <= 0 or row.run_id != snapshot.run_id
                or (row.recorded_unix is not None and not _time(row.recorded_unix))
                or (row.event_kind in ("APPROVAL_RECORDED", "RESERVATION_RECORDED") and row.recorded_unix is None)):
            _note(issues, "records", "RUN_EVENT_INVALID", row.attempt_id, row.sequence)
        if row.sequence <= 0 or row.run_id != snapshot.run_id:
            _note(issues, "replay", "RUN_EVENT_INVALID", row.attempt_id, row.sequence)
            continue
        if row.event_kind == "SUBMISSION_EVENT":
            linked = events.get(row.submission_event_id)
            if (row.record_json is not None or linked is None or linked.attempt_id != row.attempt_id
                    or linked.recorded_unix != row.recorded_unix):
                _note(issues, "records", "SUBMISSION_LINK_MISMATCH", row.attempt_id, row.sequence)
                _note(issues, "replay", "SUBMISSION_LINK_MISMATCH", row.attempt_id, row.sequence)
                continue
            links[row.submission_event_id] = links.get(row.submission_event_id, 0) + 1
            usable.append((row, linked))
            continue
        seen_records[key] = seen_records.get(key, 0) + 1
        record = None
        if row.event_kind == "APPROVAL_RECORDED" and row.attempt_id is None:
            record = approval_record
        elif row.event_kind == "RESERVATION_RECORDED" and row.attempt_id in reservations:
            reservation = reservations[row.attempt_id]
            record = {field: getattr(reservation, field) for field in ("key_sha256", "payload_sha256", "cost_cents")}
        elif row.event_kind == "CHARGE_RECORDED" and row.attempt_id in charges:
            charge = charges[row.attempt_id]
            record = {field: getattr(charge, field) for field in ("actual_cents", "evidence_sha256")}
        if record is None or row.submission_event_id is not None or row.record_json != _canonical(record):
            _note(issues, "records", "RUN_RECORD_MISMATCH", row.attempt_id, row.sequence)
            _note(issues, "replay", "RUN_RECORD_MISMATCH", row.attempt_id, row.sequence)
            continue
        usable.append((row, None))
    expected = [("APPROVAL_RECORDED", None)] + [("RESERVATION_RECORDED", key) for key in reservations]
    expected += [("CHARGE_RECORDED", key) for key in charges]
    for kind, attempt in expected:
        count = seen_records.get((kind, attempt), 0)
        if count != 1:
            _note(issues, "replay", "FINANCIAL_HISTORY_MISSING" if count == 0 else "FINANCIAL_HISTORY_DUPLICATE",
                  attempt, missing=count == 0)
            if kind != "CHARGE_RECORDED":
                _note(issues, "elapsed", "AUTHORITY_HISTORY_MISSING" if count == 0 else "AUTHORITY_HISTORY_DUPLICATE",
                      attempt, missing=count == 0)
    for event_id, event in events.items():
        count = links.get(event_id, 0)
        if count != 1:
            _note(issues, "replay", "SUBMISSION_LINK_MISSING" if count == 0 else "SUBMISSION_LINK_DUPLICATE",
                  event.attempt_id, event=event_id, missing=count == 0)
            if event.event_kind in ("CLAIMED", "SEND_GATE_PASSED"):
                _note(issues, "elapsed", "AUTHORITY_LINK_MISSING" if count == 0 else "AUTHORITY_LINK_DUPLICATE",
                      event.attempt_id, event=event_id, missing=count == 0)
    invalid_financial = {"APPROVAL_INVALID", "APPROVAL_DUPLICATE", "RESERVATION_DUPLICATE",
                         "RESERVATION_KEY_DUPLICATE", "RESERVATION_INVALID", "CHARGE_DUPLICATE",
                         "CHARGE_INVALID", "CHARGE_ORPHAN"}
    financial_invalid = any(issue.code in invalid_financial for issue in issues)
    if financial_invalid:
        _note(issues, "replay", "FINANCIAL_RECORDS_INVALID")
    missing_financial = set(snapshot.missing_tables) & {"approved_runs", "reservations", "actual_charges"}
    for name in sorted(missing_financial):
        _note(issues, "replay", "TABLE_MISSING:" + name, missing=True)
    complete = (approval is not None and all(seen_records.get(key) == 1 for key in expected)
                and len(sequences) == len(rows) and not missing_financial and not financial_invalid
                and not any(issue.category == "replay" and issue.code == "RUN_RECORD_MISMATCH" for issue in issues))
    balances, quotes, settled, boundaries = {}, {}, set(), []
    approved = False
    previous_time = None
    previous_submission_id = 0
    for row, linked in usable:
        kind = linked.event_kind if linked else row.event_kind
        if linked:
            if linked.event_id <= previous_submission_id:
                _note(issues, "replay", "SUBMISSION_ORDER_MISMATCH", row.attempt_id, row.sequence, linked.event_id)
            previous_submission_id = linked.event_id
        authority = kind in ("APPROVAL_RECORDED", "RESERVATION_RECORDED", "CLAIMED", "SEND_GATE_PASSED")
        if authority:
            timestamp = row.recorded_unix
            if timestamp is None:
                _note(issues, "elapsed", "AUTHORITY_TIME_MISSING", row.attempt_id, row.sequence, missing=True)
            elif not _time(timestamp):
                _note(issues, "elapsed", "AUTHORITY_TIME_INVALID", row.attempt_id, row.sequence)
            elif approval is not None:
                if (not approval.started_unix <= timestamp < approval.deadline_unix
                        or timestamp > approval.last_checked_unix
                        or (kind == "APPROVAL_RECORDED" and timestamp != approval.started_unix)
                        or (previous_time is not None and timestamp < previous_time)):
                    _note(issues, "elapsed", "AUTHORITY_TIME_INVALID", row.attempt_id, row.sequence)
                previous_time = timestamp
        if kind == "APPROVAL_RECORDED":
            approved = True
            continue
        if kind == "CHARGE_RECORDED":
            if row.attempt_id not in balances and seen_records.get(("RESERVATION_RECORDED", row.attempt_id)):
                _note(issues, "replay", "CHARGE_BEFORE_RESERVATION", row.attempt_id, row.sequence)
            if row.attempt_id in charges:
                balances[row.attempt_id] = charges[row.attempt_id].actual_cents
                settled.add(row.attempt_id)
            continue
        if kind not in ("RESERVATION_RECORDED", "CLAIMED", "SEND_GATE_PASSED"):
            continue
        prior_overquote = any(balances[key] > quote for key, quote in quotes.items() if key in settled)
        failed = False
        if not approved and seen_records.get(("APPROVAL_RECORDED", None)):
            _note(issues, "replay", "BOUNDARY_BEFORE_APPROVAL", row.attempt_id, row.sequence)
            failed = True
        if kind == "RESERVATION_RECORDED":
            quotes[row.attempt_id] = reservations[row.attempt_id].cost_cents
            balances[row.attempt_id] = quotes[row.attempt_id]
        elif row.attempt_id not in quotes and seen_records.get(("RESERVATION_RECORDED", row.attempt_id)):
            _note(issues, "replay", "BOUNDARY_BEFORE_RESERVATION", row.attempt_id, row.sequence)
            failed = True
        if row.attempt_id in settled:
            _note(issues, "replay", "SETTLED_SPENDING_BOUNDARY", row.attempt_id, row.sequence)
            failed = True
        if prior_overquote:
            _note(issues, "replay", "PRIOR_QUOTE_EXCEEDED", row.attempt_id, row.sequence)
            failed = True
        count, total = (len(quotes), sum(balances.values())) if complete else (None, None)
        attempt_cap = min(6, approval.max_attempts) if approval is not None else 6
        if len(quotes) > attempt_cap or (complete and total > min(1000, approval.limit_cents)):
            _note(issues, "replay", "BOUNDARY_CAP_EXCEEDED", row.attempt_id, row.sequence)
            failed = True
        boundaries.append(SpendingBoundary(row.sequence, kind, row.attempt_id if _attempt(row.attempt_id) else None, count, total,
                                            "FAIL" if failed else "PASS" if complete else "BLOCKED"))
    return tuple(boundaries)


def audit_run(snapshot: captured.RunSnapshot) -> RunAudit:
    """Assess local consistency; hashes and recorded amounts do not prove origin."""
    actual_hash = _check_input(snapshot)
    issues = []
    if actual_hash != snapshot.sha256:
        _note(issues, "records", "SNAPSHOT_HASH_MISMATCH")
    approval, reservations, charges, jobs, completed = _check_current(snapshot, issues)
    actual, held, total, overquotes = _current_money(snapshot, approval, reservations, charges, issues)
    _check_polls(snapshot, jobs, completed, issues)
    events = _check_submission_history(snapshot, jobs, issues)
    boundaries = _replay(snapshot, approval, reservations, charges, events, issues)
    assessments = tuple(_assessment(issues, category) for category in _CATEGORIES)
    return RunAudit(snapshot.run_id, actual_hash, *assessments,
                    "INCOMPLETE" if any(issue.missing for issue in issues) else "COMPLETE", tuple(issues),
                    len(snapshot.reservations), actual, held, total,
                    tuple(sorted(key for key, job in jobs.items() if _attempt(key) and job.state == "SUBMISSION_UNKNOWN")),
                    tuple(sorted(key for key, job in jobs.items() if _attempt(key) and job.state in ("SUBMITTING", "QUEUED", "RUNNING"))),
                    tuple(sorted(key for key in reservations.keys() - charges.keys() if _attempt(key))),
                    overquotes,
                    approval.limit_cents if approval else None, approval.max_attempts if approval else None,
                    approval.started_unix if approval else None, approval.deadline_unix if approval else None,
                    "APPROVAL_START", boundaries)
