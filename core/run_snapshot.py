"""Capture bounded local ledger records without changing application state."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import stat
from typing import NamedTuple


MAX_DATABASE_BYTES = 64 * 1024 * 1024
MAX_SHM_BYTES = 4 * 1024 * 1024
MAX_TABLE_ROWS = 10_000
MAX_TOTAL_ROWS = 20_000
MAX_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_CANONICAL_BYTES = 64 * 1024 * 1024
MAX_QUERY_STEPS = 5_000_000
QUERY_STEP_INTERVAL = 1000
LOCK_TIMEOUT_SECONDS = 0.25
FORMAT_VERSION = "run-snapshot-1"


class RunSnapshotError(ValueError):
    """The fixed ledger snapshot cannot be read completely within its limits."""


class ApprovedRunRow(NamedTuple):
    run_id: str
    approved_by: str
    limit_cents: int
    max_attempts: int
    cost_cents: int
    started_unix: float
    deadline_unix: float
    last_checked_unix: float
    time_stopped: int


class ReservationRow(NamedTuple):
    run_id: str
    key_sha256: str
    attempt_id: str
    payload_sha256: str
    cost_cents: int
    state: str


class ActualChargeRow(NamedTuple):
    attempt_id: str
    actual_cents: int
    evidence_sha256: str


class SubmissionJobRow(NamedTuple):
    attempt_id: str
    state: str
    provider_task_id: str | None
    evidence_sha256: str | None


class SubmissionEventRow(NamedTuple):
    event_id: int
    attempt_id: str
    event_kind: str
    recorded_unix: float | None
    provider_task_id: str | None
    evidence_sha256: str | None


class RunEventRow(NamedTuple):
    sequence: int
    run_id: str
    attempt_id: str | None
    event_kind: str
    recorded_unix: float | None
    submission_event_id: int | None
    record_json: str | None


class PollEventRow(NamedTuple):
    event_id: int
    attempt_id: str
    observed_unix: float
    provider_status: str | None
    poll_status: str
    evidence_sha256: str | None


class CompletedOutputRow(NamedTuple):
    attempt_id: str
    provider_task_id: str
    provider_evidence_sha256: str
    raw_path: str
    raw_sha256: str
    raw_size: int
    metadata_path: str
    metadata_sha256: str
    metadata_size: int
    completed_unix: float


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    approved_runs: tuple[ApprovedRunRow, ...]
    reservations: tuple[ReservationRow, ...]
    actual_charges: tuple[ActualChargeRow, ...]
    submission_jobs: tuple[SubmissionJobRow, ...]
    submission_events: tuple[SubmissionEventRow, ...]
    run_events: tuple[RunEventRow, ...]
    poll_events: tuple[PollEventRow, ...]
    completed_outputs: tuple[CompletedOutputRow, ...]
    missing_tables: tuple[str, ...]
    relationship_issues: tuple[str, ...]
    sha256: str


# These are fixed application tables, not a caller-controlled SQL interface.
_TABLES = (
    ("approved_runs", ApprovedRunRow), ("reservations", ReservationRow),
    ("actual_charges", ActualChargeRow), ("submission_jobs", SubmissionJobRow),
    ("submission_events", SubmissionEventRow), ("run_events", RunEventRow),
    ("poll_events", PollEventRow), ("completed_outputs", CompletedOutputRow),
)
_KINDS = {
    str: ("TEXT", "'text'"), int: ("INTEGER", "'integer'"),
    float: ("REAL", "'integer','real'"),
    str | None: ("TEXT", "'null','text'"),
    int | None: ("INTEGER", "'null','integer'"),
    float | None: ("REAL", "'null','integer','real'"),
}
_RELATIONS = (
    ("reservations", "run_id", "approved_runs", "run_id"),
    ("actual_charges", "attempt_id", "reservations", "attempt_id"),
    ("submission_jobs", "attempt_id", "reservations", "attempt_id"),
    ("submission_events", "attempt_id", "reservations", "attempt_id"),
    ("run_events", "run_id", "approved_runs", "run_id"),
    ("run_events", "attempt_id", "reservations", "attempt_id"),
    ("run_events", "submission_event_id", "submission_events", "event_id"),
    ("poll_events", "attempt_id", "submission_jobs", "attempt_id"),
    ("completed_outputs", "attempt_id", "submission_jobs", "attempt_id"),
)


def _check_files(path: Path):
    """The operator owns the parent and ancestors; hostile replacement is excluded."""
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_mode & 0o077:
        raise RunSnapshotError("Ledger needs a private existing parent directory.")
    for suffix, limit in (("", MAX_DATABASE_BYTES), ("-wal", MAX_DATABASE_BYTES),
                          ("-shm", MAX_SHM_BYTES), ("-journal", MAX_DATABASE_BYTES)):
        try:
            info = Path(str(path) + suffix).lstat()
        except FileNotFoundError:
            if suffix:
                continue
            raise RunSnapshotError("Ledger database does not exist.") from None
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise RunSnapshotError("Ledger database or sidecar is not a bounded regular file.")


def _inspect_tables(connection) -> frozenset[str]:
    present = set()
    for name, row_type in _TABLES:
        tables = connection.execute(
            "SELECT type, ncol FROM pragma_table_list(?) WHERE schema='main' LIMIT 2", (name,),
        ).fetchall()
        if not tables:
            continue
        if len(tables) != 1 or tables[0] != ("table", len(row_type._fields)):
            raise RunSnapshotError("Ledger requires the fixed ordinary table schema.")
        columns = connection.execute(
            "SELECT name, type, hidden FROM pragma_table_xinfo(?) ORDER BY cid LIMIT ?",
            (name, len(row_type._fields) + 1),
        ).fetchall()
        expected = [(field, _KINDS[kind][0], 0)
                    for field, kind in row_type.__annotations__.items()]
        if [(field, kind.upper(), hidden) for field, kind, hidden in columns] != expected:
            raise RunSnapshotError("Ledger table columns differ from the fixed schema.")
        present.add(name)
    return frozenset(present)


def _scope(name, present):
    if name in ("approved_runs", "reservations", "run_events"):
        return "c.run_id COLLATE BINARY=?"
    if "reservations" not in present:
        return "0 AND ? IS NOT NULL"
    return ("EXISTS (SELECT 1 FROM reservations r "
            "WHERE r.attempt_id COLLATE BINARY=c.attempt_id "
            "AND r.run_id COLLATE BINARY=?)")


def _read_table(connection, table, run_id, present, budget):
    name, row_type = table
    if name not in present:
        return ()
    source = f'FROM "{name}" c WHERE {_scope(name, present)}'
    count = connection.execute(
        f"SELECT count(*) FROM (SELECT 1 {source} LIMIT ?)",
        (run_id, MAX_TABLE_ROWS + 1),
    ).fetchone()[0]
    if count > MAX_TABLE_ROWS or budget[0] + count > MAX_TOTAL_ROWS:
        raise RunSnapshotError("Ledger snapshot exceeds its row limit.")
    invalid, sizes = [], []
    for field, kind in row_type.__annotations__.items():
        column = f'c."{field}"'
        limit = 4096 if field == "record_json" else 1024
        invalid.append(f"typeof({column}) NOT IN ({_KINDS[kind][1]})")
        invalid.append(f"length(CAST({column} AS BLOB))>{limit}")
        sizes.append(f"CASE WHEN typeof({column})='text' THEN length(CAST({column} AS BLOB)) "
                     f"WHEN {column} IS NULL THEN 0 ELSE 8 END")
    if connection.execute(
        f"SELECT EXISTS(SELECT 1 {source} AND ({' OR '.join(invalid)}))", (run_id,),
    ).fetchone()[0]:
        raise RunSnapshotError("Ledger snapshot contains an invalid or oversized cell.")
    size = connection.execute(
        f"SELECT coalesce(sum({'+'.join(sizes)}),0) {source}", (run_id,),
    ).fetchone()[0]
    if budget[1] + size > MAX_CAPTURE_BYTES:
        raise RunSnapshotError("Ledger snapshot exceeds its aggregate byte limit.")
    budget[0] += count
    budget[1] += size
    columns = ",".join(f'c."{field}"' for field in row_type._fields)
    order = ",".join(f'c."{field}" COLLATE BINARY' for field in row_type._fields)
    rows = []
    for values in connection.execute(f"SELECT {columns} {source} ORDER BY {order}", (run_id,)):
        if any(type(value) is float and not math.isfinite(value) for value in values):
            raise RunSnapshotError("Ledger snapshot contains a nonfinite number.")
        rows.append(row_type(*values))
    return tuple(rows)


def _relationship_issues(connection, present) -> tuple[str, ...]:
    """Return global structural presence flags, never unrelated record contents."""
    issues = []
    for child, child_key, parent, parent_key in _RELATIONS:
        if child not in present:
            continue
        missing = "1"
        if parent in present:
            missing = (f'NOT EXISTS (SELECT 1 FROM "{parent}" p '
                       f'WHERE p."{parent_key}" COLLATE BINARY=c."{child_key}")')
        nullable = child == "run_events" and child_key != "run_id"
        present_key = f'c."{child_key}" IS NOT NULL AND ' if nullable else ""
        if connection.execute(
            f'SELECT EXISTS(SELECT 1 FROM "{child}" c '
            f'WHERE {present_key}{missing})',
        ).fetchone()[0]:
            issues.append(f"{child}.{child_key}_without_{parent}")
    if {"run_events", "reservations"} <= present and connection.execute(
        "SELECT EXISTS(SELECT 1 FROM run_events e JOIN reservations r "
        "ON r.attempt_id COLLATE BINARY=e.attempt_id "
        "WHERE e.run_id COLLATE BINARY IS NOT r.run_id)",
    ).fetchone()[0]:
        issues.append("run_events.cross_run_attempt")
    if {"run_events", "submission_events"} <= present and connection.execute(
        "SELECT EXISTS(SELECT 1 FROM run_events r JOIN submission_events s "
        "ON s.event_id=r.submission_event_id "
        "WHERE r.attempt_id COLLATE BINARY IS NOT s.attempt_id)",
    ).fetchone()[0]:
        issues.append("run_events.submission_attempt_mismatch")
    return tuple(issues)


def _logical_hash(run_id, captured, missing, issues):
    """Stream a versioned, type-preserving representation, bounded as encoded."""
    digest, total = sha256(), 0
    encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), allow_nan=False)

    def emit(value):
        nonlocal total
        for piece in encoder.iterencode(value):
            encoded = piece.encode("ascii")
            total += len(encoded)
            if total > MAX_CANONICAL_BYTES:
                raise RunSnapshotError("Ledger snapshot exceeds its canonical byte limit.")
            digest.update(encoded)
        # Each independently encoded item is one line; embedded newlines are escaped.
        total += 1
        if total > MAX_CANONICAL_BYTES:
            raise RunSnapshotError("Ledger snapshot exceeds its canonical byte limit.")
        digest.update(b"\n")

    emit([FORMAT_VERSION, run_id, missing, issues])
    for (name, row_type), rows in zip(_TABLES, captured):
        emit([name, row_type._fields, len(rows)])
        for row in rows:
            emit([("null", None) if value is None else
                  ("real", value.hex()) if type(value) is float else
                  ("integer", value) if type(value) is int else ("text", value)
                  for value in row])
    return digest.hexdigest()


def read_run_snapshot(path: Path, *, run_id: str) -> RunSnapshot:
    """Read one WAL-aware snapshot; no recovery, SQL writes, or acceptance verdict.

    Parent/ancestors must remain operator-owned and trusted during this call.
    SQLite may create/manage coordination sidecars even with mode=ro.
    Missing tables and global broken-link flags are evidence, not reconstructed history.
    """
    if type(run_id) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id) is None:
        raise RunSnapshotError("Run ID must contain 1 to 64 letters, digits, underscores, or hyphens.")
    connection = None
    try:
        path = Path(path).absolute()
        _check_files(path)
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True,
                                     timeout=LOCK_TIMEOUT_SECONDS, isolation_level=None)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 64 * 1024)
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16 * 1024)
        connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        steps = 0

        def progress():
            nonlocal steps
            steps += QUERY_STEP_INTERVAL
            return int(steps >= MAX_QUERY_STEPS)

        connection.set_progress_handler(progress, QUERY_STEP_INTERVAL)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("BEGIN")
        if connection.execute("PRAGMA encoding").fetchone() != ("UTF-8",):
            raise RunSnapshotError("Ledger snapshot requires UTF-8 database encoding.")
        present = _inspect_tables(connection)
        budget = [0, 0]
        captured = tuple(_read_table(connection, table, run_id, present, budget) for table in _TABLES)
        missing = tuple(name for name, _ in _TABLES if name not in present)
        issues = _relationship_issues(connection, present)
        # Catch ordinary concurrent growth beyond the supported file envelope too.
        _check_files(path)
        logical_hash = _logical_hash(run_id, captured, missing, issues)
        return RunSnapshot(run_id, *captured, missing, issues, logical_hash)
    except (OSError, sqlite3.Error, UnicodeError, OverflowError) as error:
        raise RunSnapshotError("Ledger snapshot could not be read within its fixed limits.") from error
    finally:
        if connection is not None:
            connection.close()
