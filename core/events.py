"""Order new budget records and submission events in one local journal."""

import json
from typing import Callable


MAX_RECORD_BYTES = 4096


class EventError(ValueError):
    """A journal entry cannot be recorded safely."""


def create_event_table(connection):
    """Create an empty journal without reconstructing existing history."""
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS run_events (
            sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
            run_id TEXT NOT NULL REFERENCES approved_runs(run_id),
            attempt_id TEXT REFERENCES reservations(attempt_id),
            event_kind TEXT NOT NULL CHECK (event_kind IN (
                'APPROVAL_RECORDED', 'RESERVATION_RECORDED',
                'CHARGE_RECORDED', 'SUBMISSION_EVENT'
            )),
            recorded_unix REAL CHECK (recorded_unix IS NULL OR (
                typeof(recorded_unix) IN ('integer', 'real')
                AND recorded_unix >= 0 AND recorded_unix <= 253402300799
            )),
            submission_event_id INTEGER UNIQUE CHECK (submission_event_id IS NULL OR (
                typeof(submission_event_id) = 'integer' AND submission_event_id > 0
            )),
            record_json TEXT CHECK (record_json IS NULL OR (
                typeof(record_json) = 'text'
                AND length(CAST(record_json AS BLOB)) BETWEEN 2 AND {MAX_RECORD_BYTES}
            )),
            CHECK ((event_kind = 'APPROVAL_RECORDED' AND attempt_id IS NULL)
                   OR (event_kind != 'APPROVAL_RECORDED' AND attempt_id IS NOT NULL)),
            CHECK (event_kind NOT IN ('APPROVAL_RECORDED', 'RESERVATION_RECORDED')
                   OR recorded_unix IS NOT NULL),
            CHECK ((event_kind = 'SUBMISSION_EVENT' AND submission_event_id IS NOT NULL
                    AND record_json IS NULL)
                   OR (event_kind != 'SUBMISSION_EVENT' AND submission_event_id IS NULL
                       AND record_json IS NOT NULL))
        )
    """)
    # The optional submission table does not exist in budget-only workflows.
    # record_submission verifies that target in the caller's transaction.
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS run_events_no_replace
        BEFORE INSERT ON run_events
        WHEN EXISTS (SELECT 1 FROM run_events
                     WHERE sequence = NEW.sequence
                        OR submission_event_id = NEW.submission_event_id)
        BEGIN
            SELECT RAISE(ABORT, 'Run events cannot be replaced.');
        END
    """)
    for operation in ("UPDATE", "DELETE"):
        connection.execute(f"""
            CREATE TRIGGER IF NOT EXISTS run_events_no_{operation.lower()}
            BEFORE {operation} ON run_events
            BEGIN
                SELECT RAISE(ABORT, 'Run events are append-only.');
            END
        """)


def observation_time(read_time: Callable[[], float]) -> float | None:
    """Keep an observed result when its validated local clock is unavailable."""
    try:
        return read_time()
    except Exception:
        return None


def _append(connection, run_id, attempt_id, kind, recorded_unix, *,
            record=None, submission_event_id=None):
    record_json = None
    if record is not None:
        record_json = json.dumps(record, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False)
        if len(record_json.encode("ascii")) > MAX_RECORD_BYTES:
            raise EventError("Journal record exceeds its byte limit.")
    connection.execute(
        "INSERT INTO run_events "
        "(run_id, attempt_id, event_kind, recorded_unix, submission_event_id, record_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, attempt_id, kind, recorded_unix, submission_event_id, record_json),
    )


def record_approval(connection, run_id: str, recorded_unix: float):
    """Copy the fixed approval fields in the approval's write transaction."""
    row = connection.execute(
        "SELECT approved_by, limit_cents, max_attempts, cost_cents, started_unix, deadline_unix "
        "FROM approved_runs WHERE run_id=?", (run_id,),
    ).fetchone()
    if row is None:
        raise EventError("Approval event has no matching approval.")
    _append(connection, run_id, None, "APPROVAL_RECORDED", recorded_unix, record=dict(row))


def record_reservation(connection, attempt_id: str, recorded_unix: float):
    """Copy the frozen payload, quoted cents, and hashed key after reservation."""
    row = connection.execute(
        "SELECT run_id, key_sha256, payload_sha256, cost_cents FROM reservations WHERE attempt_id=?",
        (attempt_id,),
    ).fetchone()
    if row is None:
        raise EventError("Reservation event has no matching reservation.")
    record = {key: row[key] for key in ("key_sha256", "payload_sha256", "cost_cents")}
    _append(connection, row["run_id"], attempt_id, "RESERVATION_RECORDED", recorded_unix, record=record)


def record_charge(connection, attempt_id: str, recorded_unix: float | None):
    """Copy the final amount and receipt hash without treating the hash as proof."""
    row = connection.execute(
        "SELECT r.run_id, c.actual_cents, c.evidence_sha256 "
        "FROM actual_charges c JOIN reservations r ON r.attempt_id=c.attempt_id "
        "WHERE c.attempt_id=?", (attempt_id,),
    ).fetchone()
    if row is None:
        raise EventError("Charge event has no matching charge and reservation.")
    record = {key: row[key] for key in ("actual_cents", "evidence_sha256")}
    _append(connection, row["run_id"], attempt_id, "CHARGE_RECORDED", recorded_unix, record=record)


def record_submission(connection, submission_event_id: int):
    """Link a saved event using its actual attempt, run, and local timestamp."""
    row = connection.execute(
        "SELECT e.attempt_id, e.recorded_unix, r.run_id "
        "FROM submission_events e JOIN reservations r ON r.attempt_id=e.attempt_id "
        "WHERE e.event_id=?", (submission_event_id,),
    ).fetchone()
    if row is None:
        raise EventError("Submission link has no matching event and reservation.")
    _append(connection, row["run_id"], row["attempt_id"], "SUBMISSION_EVENT", row["recorded_unix"],
            submission_event_id=submission_event_id)
