"""Read real SQLite snapshots without constructing application state owners."""

from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.budget import BudgetLedger
from core import run_snapshot as snapshot
from features.inference.submission import SubmissionCoordinator
from features.inference.polling import TaskPoller
from features.output.complete import OutputCompleter


RUN = "snapshot-run"
ATTEMPT_A, ATTEMPT_B = "att-" + "a" * 32, "att-" + "b" * 32
CONNECT = sqlite3.connect
SCHEMAS = {
    "approved_runs": (("run_id", "TEXT"), ("approved_by", "TEXT"), ("limit_cents", "INTEGER"),
        ("max_attempts", "INTEGER"), ("cost_cents", "INTEGER"), ("started_unix", "REAL"),
        ("deadline_unix", "REAL"), ("last_checked_unix", "REAL"), ("time_stopped", "INTEGER")),
    "reservations": (("run_id", "TEXT"), ("key_sha256", "TEXT"), ("attempt_id", "TEXT"),
        ("payload_sha256", "TEXT"), ("cost_cents", "INTEGER"), ("state", "TEXT")),
    "actual_charges": (("attempt_id", "TEXT"), ("actual_cents", "INTEGER"), ("evidence_sha256", "TEXT")),
    "submission_jobs": (("attempt_id", "TEXT"), ("state", "TEXT"), ("provider_task_id", "TEXT"), ("evidence_sha256", "TEXT")),
    "submission_events": (("event_id", "INTEGER"), ("attempt_id", "TEXT"), ("event_kind", "TEXT"),
        ("recorded_unix", "REAL"), ("provider_task_id", "TEXT"), ("evidence_sha256", "TEXT")),
    "run_events": (("sequence", "INTEGER"), ("run_id", "TEXT"), ("attempt_id", "TEXT"),
        ("event_kind", "TEXT"), ("recorded_unix", "REAL"), ("submission_event_id", "INTEGER"), ("record_json", "TEXT")),
    "poll_events": (("event_id", "INTEGER"), ("attempt_id", "TEXT"), ("observed_unix", "REAL"),
        ("provider_status", "TEXT"), ("poll_status", "TEXT"), ("evidence_sha256", "TEXT")),
    "completed_outputs": (("attempt_id", "TEXT"), ("provider_task_id", "TEXT"), ("provider_evidence_sha256", "TEXT"),
        ("raw_path", "TEXT"), ("raw_sha256", "TEXT"), ("raw_size", "INTEGER"), ("metadata_path", "TEXT"),
        ("metadata_sha256", "TEXT"), ("metadata_size", "INTEGER"), ("completed_unix", "REAL")),
}
ROWS = {
    "approved_runs": [(RUN, "fixture operator", 1000, 6, 140, 1000.0, 11800.0, 1002.0, 0)],
    "reservations": [(RUN, "a" * 64, ATTEMPT_A, "c" * 64, 140, "RESERVED"),
                     (RUN, "b" * 64, ATTEMPT_B, "d" * 64, 140, "RESERVED")],
    "actual_charges": [(ATTEMPT_A, 139, "e" * 64)],
    "submission_jobs": [(ATTEMPT_A, "SUCCEEDED", "task-a", "f" * 64), (ATTEMPT_B, "SUBMITTING", None, None)],
    "submission_events": [(1, ATTEMPT_A, "CLAIMED", 1001.0, None, None),
                          (2, ATTEMPT_A, "TASK_RECORDED", 1001.5, "task-a", None)],
    "run_events": [(1, RUN, None, "APPROVAL_RECORDED", 1000.0, None, '{"fixture":"approval"}'),
                   (2, RUN, ATTEMPT_A, "RESERVATION_RECORDED", 1001.0, None, "{}"),
                   (3, RUN, ATTEMPT_A, "SUBMISSION_EVENT", 1001.5, 2, None)],
    "poll_events": [(1, ATTEMPT_A, 1002.0, "SUCCEEDED", "OBSERVED", "f" * 64),
                    (2, ATTEMPT_B, 1003.0, None, "NOT_POLLED", None)],
    "completed_outputs": [(ATTEMPT_A, "task-a", "f" * 64, f"attempts/{ATTEMPT_A}/output/raw.mp4", "1" * 64, 12,
                           f"attempts/{ATTEMPT_A}/output/metadata.json", "2" * 64, 34, 1003.0)],
}


class RunSnapshotTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.path = self.create_database("ledger.sqlite")

    def create_database(self, name, *, tables=None, rows=ROWS, nocase=False, encoding=None):
        path = self.root / name
        selected = tuple(SCHEMAS) if tables is None else tuple(tables)
        with CONNECT(path) as connection:
            if encoding is not None:
                connection.execute(f"PRAGMA encoding='{encoding}'")
            for table in selected:
                columns = ",".join(f'"{name}" {kind}' + (" COLLATE NOCASE" if nocase and kind == "TEXT" else "")
                                   for name, kind in SCHEMAS[table])
                connection.execute(f'CREATE TABLE "{table}" ({columns})')
                values = rows.get(table, ())
                if values:
                    marks = ",".join("?" for _ in SCHEMAS[table])
                    connection.executemany(f'INSERT INTO "{table}" VALUES ({marks})', values)
        connection.close()
        return path

    def execute(self, sql, parameters=(), *, path=None):
        with CONNECT(path or self.path) as connection:
            connection.execute(sql, parameters)
        connection.close()

    def read(self, *, path=None, run_id=RUN):
        return snapshot.read_run_snapshot(path or self.path, run_id=run_id)

    @staticmethod
    def canonical_bytes(result):
        lines = [["run-snapshot-1", result.run_id, result.missing_tables, result.relationship_issues]]
        for table, schema in SCHEMAS.items():
            rows = getattr(result, table)
            lines.append([table, tuple(name for name, _ in schema), len(rows)])
            for row in rows:
                values = []
                for value in row:
                    if value is None:
                        values.append(["null", None])
                    elif type(value) is float:
                        values.append(["real", value.hex()])
                    elif type(value) is int:
                        values.append(["integer", value])
                    else:
                        values.append(["text", value])
                lines.append(values)
        return b"".join(json.dumps(line, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                        for line in lines)

    def traced_connect(self, queries, calls=None):
        def connect(database, **arguments):
            if calls is not None:
                calls.append((database, arguments))
            connection = CONNECT(database, **arguments)
            connection.set_trace_callback(queries.append)
            return connection
        return connect

    def test_populated_snapshot_preserves_immutable_typed_rows_and_exact_logical_hash(self):
        result = self.read()
        self.assertEqual(result.run_id, RUN)
        self.assertEqual((result.missing_tables, result.relationship_issues), ((), ()))
        for table, schema in SCHEMAS.items():
            rows = getattr(result, table)
            self.assertIs(type(rows), tuple)
            self.assertEqual(rows, tuple(sorted(ROWS[table])))
            for row in rows:
                self.assertIsInstance(row, tuple)
                self.assertEqual(row._fields, tuple(name for name, _ in schema))
        self.assertIs(type(result.approved_runs[0].started_unix), float)
        self.assertIsNone(result.submission_jobs[1].provider_task_id)
        self.assertEqual(result.sha256, sha256(self.canonical_bytes(result)).hexdigest())
        with self.assertRaises(FrozenInstanceError):
            result.run_id = "other"
        with self.assertRaises(AttributeError):
            result.reservations[0].cost_cents = 0
        self.assertFalse(hasattr(result, "releaseClassification"))
        # Capture storage facts; domain validation belongs to the later auditor.
        self.execute("UPDATE reservations SET cost_cents=-7,state='UNASSESSED' WHERE attempt_id=?", (ATTEMPT_A,))
        changed = self.read()
        self.assertEqual((changed.reservations[0].cost_cents, changed.reservations[0].state), (-7, "UNASSESSED"))
        self.assertNotEqual(changed.sha256, result.sha256)

    def test_hash_ignores_insertion_order_and_distinguishes_null_from_empty_text(self):
        reverse = {table: list(reversed(rows)) for table, rows in ROWS.items()}
        other = self.create_database("reversed.sqlite", rows=reverse)
        first = self.read()
        self.assertEqual(self.read(path=other), first)
        self.execute("UPDATE submission_jobs SET evidence_sha256='' WHERE attempt_id=?", (ATTEMPT_B,), path=other)
        changed = self.read(path=other)
        self.assertEqual(changed.submission_jobs[1].evidence_sha256, "")
        self.assertNotEqual(changed.sha256, first.sha256)

    def test_committed_wal_changes_are_visible_without_checkpointing(self):
        writer = CONNECT(self.path, isolation_level=None)
        self.addCleanup(writer.close)
        self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone(), ("wal",))
        writer.execute("PRAGMA wal_autocheckpoint=0")
        main_before = self.path.read_bytes()
        writer.execute("UPDATE approved_runs SET cost_cents=141 WHERE run_id=?", (RUN,))
        self.assertEqual(self.path.read_bytes(), main_before)
        self.assertGreater(Path(str(self.path) + "-wal").stat().st_size, 0)
        result = self.read()
        self.assertEqual(result.approved_runs[0].cost_cents, 141)
        self.assertEqual(self.path.read_bytes(), main_before)

    def test_one_transaction_keeps_rows_and_relationship_flags_on_the_same_snapshot(self):
        writer = CONNECT(self.path, isolation_level=None)
        self.addCleanup(writer.close)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        before = self.read()
        original_read = snapshot._read_table
        commits = []

        def interleave(connection, table, run_id, present, budget):
            rows = original_read(connection, table, run_id, present, budget)
            if table[0] == "approved_runs" and not commits:
                writer.execute("BEGIN IMMEDIATE")
                writer.execute("UPDATE approved_runs SET cost_cents=150 WHERE run_id=?", (RUN,))
                writer.execute("UPDATE reservations SET cost_cents=150 WHERE run_id=?", (RUN,))
                writer.execute("INSERT INTO actual_charges VALUES (?,?,?)", (ATTEMPT_B, 150, "3" * 64))
                writer.execute("INSERT INTO actual_charges VALUES (?,?,?)", ("private-orphan", 1, "4" * 64))
                writer.execute("COMMIT")
                commits.append(True)
            return rows

        with patch.object(snapshot, "_read_table", side_effect=interleave):
            observed = self.read()
        self.assertEqual(commits, [True])
        self.assertEqual(observed, before)
        after = self.read()
        self.assertEqual(after.approved_runs[0].cost_cents, 150)
        self.assertEqual({row.cost_cents for row in after.reservations}, {150})
        self.assertEqual(len(after.actual_charges), 2)
        self.assertEqual(after.relationship_issues, ("actual_charges.attempt_id_without_reservations",))
        self.assertNotEqual(after.sha256, before.sha256)

    def test_missing_tables_are_recorded_without_fabricating_approval_or_history(self):
        empty = self.create_database("empty.sqlite", tables=(), rows={})
        result = self.read(path=empty)
        self.assertEqual(result.missing_tables, tuple(SCHEMAS))
        self.assertEqual(result.relationship_issues, ())
        for table in SCHEMAS:
            self.assertEqual(getattr(result, table), ())
        legacy = self.create_database("legacy.sqlite", tables=("reservations",))
        legacy_result = self.read(path=legacy)
        self.assertEqual(legacy_result.approved_runs, ())
        self.assertEqual(legacy_result.reservations, tuple(ROWS["reservations"]))
        self.assertEqual(legacy_result.missing_tables, tuple(name for name in SCHEMAS if name != "reservations"))
        self.assertEqual(legacy_result.relationship_issues, ("reservations.run_id_without_approved_runs",))
        self.assertNotEqual(legacy_result.sha256, result.sha256)

    def test_unrelated_run_rows_are_private_and_do_not_change_the_logical_hash(self):
        before = self.read()
        other_run, other_attempt = "private-other-run", "att-" + "c" * 32
        with CONNECT(self.path) as writer:
            writer.execute("INSERT INTO approved_runs VALUES (?,?,?,?,?,?,?,?,?)",
                (other_run, "private-operator-secret", 999, 5, 22, 0.0, 100.0, 2.0, 0))
            writer.execute("INSERT INTO reservations VALUES (?,?,?,?,?,?)",
                (other_run, "5" * 64, other_attempt, "6" * 64, 22, "RESERVED"))
            writer.execute("INSERT INTO submission_jobs VALUES (?,?,?,?)", (other_attempt, "QUEUED", "private-task", None))
            writer.execute("INSERT INTO actual_charges VALUES (?,?,?)", (other_attempt, 20, "7" * 64))
            writer.execute("INSERT INTO submission_events VALUES (?,?,?,?,?,?)", (30, other_attempt, "CLAIMED", 2.0, None, None))
            writer.execute("INSERT INTO run_events VALUES (?,?,?,?,?,?,?)", (30, other_run, other_attempt, "SUBMISSION_EVENT", 2.0, 30, None))
            writer.execute("INSERT INTO poll_events VALUES (?,?,?,?,?,?)", (30, other_attempt, 2.0, None, "POLL_ERROR", None))
            writer.execute("INSERT INTO completed_outputs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (other_attempt, "private-task", "8" * 64, "private/raw.mp4", "9" * 64, 20, "private/meta.json", "0" * 64, 30, 2.0))
        writer.close()
        after = self.read()
        self.assertEqual(after, before)
        self.assertNotIn("private-", repr(after))

    def test_relationship_flags_report_global_orphans_null_keys_and_cross_links_without_private_rows(self):
        with CONNECT(self.path) as writer:
            writer.execute("INSERT INTO reservations VALUES (?,?,?,?,?,?)", ("private-missing-run", "8" * 64, "orphan-reservation", "9" * 64, 1, "RESERVED"))
            writer.execute("INSERT INTO actual_charges VALUES (?,?,?)", (None, 1, "0" * 64))
            writer.execute("INSERT INTO submission_jobs VALUES (?,?,?,?)", ("private-job", "UNKNOWN", None, None))
            writer.execute("INSERT INTO submission_events VALUES (?,?,?,?,?,?)", (90, "private-event", "UNKNOWN", None, None, None))
            writer.execute("INSERT INTO run_events VALUES (?,?,?,?,?,?,?)", (90, "private-missing-run", "private-event", "UNKNOWN", None, 999, None))
            writer.execute("INSERT INTO poll_events VALUES (?,?,?,?,?,?)", (90, "private-poll", 0.0, None, "UNKNOWN", None))
            writer.execute("INSERT INTO completed_outputs VALUES (?,?,?,?,?,?,?,?,?,?)", ("private-complete", "task", "0" * 64, "raw", "0" * 64, 1, "meta", "0" * 64, 1, 0.0))
            writer.execute("INSERT INTO run_events VALUES (?,?,?,?,?,?,?)", (91, RUN, "orphan-reservation", "UNKNOWN", None, 2, None))
        writer.close()
        result = self.read()
        self.assertEqual(result.relationship_issues, (
            "reservations.run_id_without_approved_runs", "actual_charges.attempt_id_without_reservations",
            "submission_jobs.attempt_id_without_reservations", "submission_events.attempt_id_without_reservations",
            "run_events.run_id_without_approved_runs", "run_events.attempt_id_without_reservations",
            "run_events.submission_event_id_without_submission_events", "poll_events.attempt_id_without_submission_jobs",
            "completed_outputs.attempt_id_without_submission_jobs", "run_events.cross_run_attempt",
            "run_events.submission_attempt_mismatch"))
        self.assertEqual(result.actual_charges, tuple(ROWS["actual_charges"]))
        self.assertNotIn("private-job", repr(result))
        self.assertNotIn("private-complete", repr(result))

    def test_binary_run_and_attempt_identity_ignores_nocase_schema_collations(self):
        path = self.create_database("nocase.sqlite", nocase=True)
        with CONNECT(path) as writer:
            writer.execute("INSERT INTO approved_runs VALUES (?,?,?,?,?,?,?,?,?)", (RUN.upper(), "private-case", 900, 5, 100, 0.0, 100.0, 0.0, 0))
            writer.execute("INSERT INTO reservations VALUES (?,?,?,?,?,?)", (RUN.upper(), "f" * 64, "att-" + "c" * 32, "e" * 64, 100, "RESERVED"))
            writer.execute("INSERT INTO actual_charges VALUES (?,?,?)", (ATTEMPT_A.upper(), 1, "a" * 64))
            writer.execute("INSERT INTO run_events VALUES (?,?,?,?,?,?,?)", (90, RUN.upper(), ATTEMPT_A, "UNKNOWN", None, None, None))
        writer.close()
        result = self.read(path=path)
        self.assertEqual(result.approved_runs, tuple(ROWS["approved_runs"]))
        self.assertEqual(result.reservations, tuple(ROWS["reservations"]))
        self.assertEqual(result.actual_charges, tuple(ROWS["actual_charges"]))
        self.assertEqual(result.run_events, tuple(ROWS["run_events"]))
        self.assertEqual(result.relationship_issues, ("actual_charges.attempt_id_without_reservations", "run_events.cross_run_attempt"))

    def test_views_altered_columns_hidden_fields_and_non_utf8_schemas_are_rejected(self):
        statements = ["ALTER TABLE approved_runs ADD COLUMN extra TEXT",
            "ALTER TABLE approved_runs RENAME COLUMN approved_by TO wrong_name",
            "ALTER TABLE approved_runs RENAME TO approval_backing; CREATE VIEW approved_runs AS SELECT * FROM approval_backing",
            "DROP TABLE actual_charges; CREATE TABLE actual_charges (attempt_id TEXT, actual_cents REAL, evidence_sha256 TEXT)",
            "DROP TABLE actual_charges; CREATE TABLE actual_charges (attempt_id TEXT, actual_cents INTEGER, evidence_sha256 TEXT GENERATED ALWAYS AS ('x') VIRTUAL)"]
        for index, statement in enumerate(statements):
            path = self.create_database(f"schema-{index}.sqlite")
            with CONNECT(path) as writer:
                writer.executescript(statement)
            writer.close()
            with self.subTest(case=index), self.assertRaises(snapshot.RunSnapshotError):
                self.read(path=path)
        path = self.create_database("utf16.sqlite", encoding="UTF-16")
        with self.assertRaises(snapshot.RunSnapshotError):
            self.read(path=path)

    def test_storage_scalar_types_and_nonfinite_reals_are_rejected(self):
        cases = [("UPDATE approved_runs SET limit_cents=?", (1.5,)),
            ("UPDATE approved_runs SET approved_by=?", (sqlite3.Binary(b"private-blob"),)),
            ("UPDATE approved_runs SET approved_by=NULL", ()),
            ("UPDATE approved_runs SET deadline_unix=?", (float("inf"),)),
            ("UPDATE submission_events SET recorded_unix=?", (-float("inf"),))]
        for index, (sql, arguments) in enumerate(cases):
            path = self.create_database(f"scalar-{index}.sqlite")
            self.execute(sql, arguments, path=path)
            with self.subTest(case=index), self.assertRaises(snapshot.RunSnapshotError):
                self.read(path=path)

    def test_cell_byte_limits_are_checked_in_sql_before_fetching_body_values(self):
        cases = [("approved_runs", "approved_by", "é" * 512, "é" * 513),
                 ("run_events", "record_json", "x" * 4096, "x" * 4097)]
        for index, (table, column, exact, oversized) in enumerate(cases):
            path = self.create_database(f"cell-{index}.sqlite")
            self.execute(f'UPDATE "{table}" SET "{column}"=?', (exact,), path=path)
            self.read(path=path)
            self.execute(f'UPDATE "{table}" SET "{column}"=?', (oversized,), path=path)
            queries = []
            with self.subTest(table=table), patch.object(snapshot.sqlite3, "connect", side_effect=self.traced_connect(queries)):
                with self.assertRaises(snapshot.RunSnapshotError):
                    self.read(path=path)
            self.assertFalse(any(query.startswith('SELECT c."') and f'FROM "{table}" c' in query for query in queries))

    def test_table_total_row_and_aggregate_capture_limits_reject_before_the_affected_fetch(self):
        original = self.read()
        total_rows = sum(len(rows) for rows in ROWS.values())
        total_bytes = sum(len(value.encode("utf-8")) if type(value) is str else 0 if value is None else 8
                          for rows in ROWS.values() for row in rows for value in row)
        with patch.object(snapshot, "MAX_TABLE_ROWS", 3), patch.object(snapshot, "MAX_TOTAL_ROWS", total_rows), \
                patch.object(snapshot, "MAX_CAPTURE_BYTES", total_bytes):
            self.assertEqual(self.read(), original)
        limits = [("MAX_TABLE_ROWS", 1, "reservations"),
            ("MAX_TOTAL_ROWS", total_rows - 1, "completed_outputs"),
            ("MAX_CAPTURE_BYTES", total_bytes - 1, "completed_outputs")]
        for name, limit, table in limits:
            queries = []
            with self.subTest(limit=name), patch.object(snapshot, name, limit), \
                    patch.object(snapshot.sqlite3, "connect", side_effect=self.traced_connect(queries)):
                with self.assertRaises(snapshot.RunSnapshotError):
                    self.read()
            self.assertFalse(any(query.startswith('SELECT c."') and f'FROM "{table}" c' in query for query in queries))

    def test_query_and_canonical_output_budgets_abort_instead_of_returning_partial_data(self):
        result = self.read()
        size = len(self.canonical_bytes(result))
        with patch.object(snapshot, "MAX_CANONICAL_BYTES", size):
            self.assertEqual(self.read(), result)
        with patch.object(snapshot, "MAX_CANONICAL_BYTES", size - 1), self.assertRaises(snapshot.RunSnapshotError):
            self.read()
        with patch.object(snapshot, "MAX_QUERY_STEPS", 0), patch.object(snapshot, "QUERY_STEP_INTERVAL", 1), \
                self.assertRaises(snapshot.RunSnapshotError):
            self.read()

    def test_database_parent_and_sidecars_are_bounded_regular_nonsymlink_paths(self):
        missing = self.root / "missing.sqlite"
        paths = [missing]
        linked = self.root / "linked.sqlite"
        linked.symlink_to(self.path)
        paths.append(linked)
        directory = self.root / "directory.sqlite"
        directory.mkdir()
        paths.append(directory)
        fifo = self.root / "fifo.sqlite"
        os.mkfifo(fifo)
        paths.append(fifo)
        parent_link = self.root / "parent-link"
        parent_link.symlink_to(self.root, target_is_directory=True)
        paths.append(parent_link / self.path.name)
        for path in paths:
            with self.subTest(path=path.name), patch.object(snapshot.sqlite3, "connect") as connector:
                with self.assertRaises(snapshot.RunSnapshotError):
                    self.read(path=path)
                connector.assert_not_called()
        self.assertFalse(missing.exists())
        self.root.chmod(0o755)
        try:
            with self.assertRaises(snapshot.RunSnapshotError):
                self.read()
        finally:
            self.root.chmod(0o700)
        for index, suffix in enumerate(("-wal", "-shm", "-journal")):
            path = self.create_database(f"sidecar-{index}.sqlite")
            sidecar = Path(str(path) + suffix)
            sidecar.symlink_to(self.path)
            with self.subTest(sidecar=suffix), patch.object(snapshot.sqlite3, "connect") as connector:
                with self.assertRaises(snapshot.RunSnapshotError):
                    self.read(path=path)
                connector.assert_not_called()
            sidecar.unlink()
            limit = snapshot.MAX_SHM_BYTES if suffix == "-shm" else snapshot.MAX_DATABASE_BYTES
            with sidecar.open("wb") as file:
                file.truncate(limit + 1)
            with self.subTest(oversized=suffix), patch.object(snapshot.sqlite3, "connect") as connector:
                with self.assertRaises(snapshot.RunSnapshotError):
                    self.read(path=path)
                connector.assert_not_called()
        with patch.object(snapshot, "MAX_DATABASE_BYTES", self.path.stat().st_size - 1), \
                patch.object(snapshot.sqlite3, "connect") as connector:
            with self.assertRaises(snapshot.RunSnapshotError):
                self.read()
            connector.assert_not_called()

    def test_read_only_connection_configuration_no_constructors_or_sql_state_changes(self):
        queries, calls = [], []
        before = self.path.read_bytes()
        original_inspect = snapshot._inspect_tables

        def inspect_policy(connection):
            self.assertEqual(connection.execute("PRAGMA query_only").fetchone(), (1,))
            self.assertEqual(connection.execute("PRAGMA trusted_schema").fetchone(), (0,))
            self.assertTrue(connection.in_transaction)
            return original_inspect(connection)

        with patch.object(snapshot.sqlite3, "connect", side_effect=self.traced_connect(queries, calls)), \
                patch.object(snapshot, "_inspect_tables", side_effect=inspect_policy), \
                patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No ledger constructor")), \
                patch.object(SubmissionCoordinator, "__init__", side_effect=AssertionError("No recovery")), \
                patch.object(TaskPoller, "__init__", side_effect=AssertionError("No poll schema")), \
                patch.object(OutputCompleter, "__init__", side_effect=AssertionError("No output schema")):
            result = self.read()
        self.assertEqual(len(calls), 1)
        uri, arguments = calls[0]
        self.assertTrue(uri.endswith("?mode=ro"))
        self.assertNotIn("immutable", uri)
        self.assertEqual(arguments, {"uri": True, "timeout": 0.25, "isolation_level": None})
        self.assertIn("PRAGMA query_only=ON", queries)
        self.assertIn("PRAGMA trusted_schema=OFF", queries)
        self.assertEqual(queries.count("BEGIN"), 1)
        self.assertFalse(any(query.split()[0].upper() in {"INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"} for query in queries))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(result.submission_jobs[1].state, "SUBMITTING")
        for run_id in ("../private-secret", "", True, "x" * 65):
            with self.subTest(run_id=run_id), patch.object(snapshot.sqlite3, "connect") as connector:
                with self.assertRaises(snapshot.RunSnapshotError):
                    self.read(run_id=run_id)
                connector.assert_not_called()

    def test_current_application_schemas_and_submission_journals_are_captured(self):
        path = self.root / "application.sqlite"
        ledger = BudgetLedger(path, clock=lambda: 1001.0)
        ledger.approve_run(RUN, approved_by="fixture operator", limit_cents=1000,
                           max_attempts=6, cost_cents=140)
        ledger.reserve(RUN, "fixture-key", "c" * 64, attempt_id=ATTEMPT_A)
        with SubmissionCoordinator(ledger) as owner:
            TaskPoller(owner)
            OutputCompleter(owner, self.root / "outputs")

            def fake_create():
                owner.require_send_allowed(ATTEMPT_A)
                return "fixture-task"

            submitted = owner.submit_once(ATTEMPT_A, fake_create)
            result = self.read(path=path)
        self.assertEqual(submitted.state, "QUEUED")
        self.assertEqual((result.missing_tables, result.relationship_issues), ((), ()))
        self.assertIsInstance(result.approved_runs[0], snapshot.ApprovedRunRow)
        self.assertEqual(result.approved_runs[0].limit_cents, 1000)
        self.assertEqual(result.reservations, (snapshot.ReservationRow(RUN,
            sha256(b"fixture-key").hexdigest(), ATTEMPT_A, "c" * 64, 140, "RESERVED"),))
        self.assertEqual(result.submission_jobs, (snapshot.SubmissionJobRow(ATTEMPT_A, "QUEUED", "fixture-task", None),))
        self.assertEqual(tuple(row.event_kind for row in result.submission_events),
                         ("CLAIMED", "SEND_GATE_PASSED", "TASK_RECORDED"))
        self.assertEqual(tuple(row.event_kind for row in result.run_events),
                         ("APPROVAL_RECORDED", "RESERVATION_RECORDED", "SUBMISSION_EVENT", "SUBMISSION_EVENT", "SUBMISSION_EVENT"))
        self.assertEqual(tuple(row.submission_event_id for row in result.run_events), (None, None, 1, 2, 3))
        self.assertEqual((result.actual_charges, result.poll_events, result.completed_outputs), ((), (), ()))
        self.assertEqual(result.sha256, sha256(self.canonical_bytes(result)).hexdigest())


if __name__ == "__main__":
    unittest.main()
