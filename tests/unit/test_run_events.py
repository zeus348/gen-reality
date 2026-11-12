"""Check one committed order for local budget and submission records."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from core import budget, events
from features.inference import submission


PAYLOAD = "a" * 64
EVIDENCE = "b" * 64
TASK = "fixture-task"
ATTEMPT = "att-" + "c" * 32


class RunEventTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "budget.sqlite"

    def ledger(self, clock=None):
        return budget.BudgetLedger(self.path, clock=clock or (lambda: 1000.0))

    def approve(self, ledger, *, run="run-1", limit=1000, name="fixture operator"):
        ledger.approve_run(run, approved_by=name, limit_cents=limit, max_attempts=6,
                           cost_cents=140, max_elapsed_minutes=1)

    def prepared(self, *, clock=None, limit=1000):
        ledger = self.ledger(clock)
        self.approve(ledger, limit=limit)
        reserved = ledger.reserve("run-1", "first", PAYLOAD, attempt_id=ATTEMPT)
        return ledger, reserved

    def owner(self, ledger):
        owner = submission.SubmissionCoordinator(ledger)
        self.addCleanup(owner.close)
        return owner

    def rows(self, table="run_events"):
        order = "sequence" if table == "run_events" else "event_id"
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}")]

    def count(self, table):
        with sqlite3.connect(self.path) as connection:
            return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def block(self, kind, *, submission_kind=None):
        condition = f"NEW.event_kind='{kind}'"
        if submission_kind is not None:
            condition += (" AND NEW.submission_event_id IN (SELECT event_id FROM submission_events "
                          f"WHERE event_kind='{submission_kind}')")
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TRIGGER fixture_run_event_failure BEFORE INSERT ON run_events "
                               f"WHEN {condition} BEGIN SELECT RAISE(ABORT, 'fixture event failure'); END")

    def unblock(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TRIGGER fixture_run_event_failure")

    def record(self, row):
        return json.loads(row["record_json"])

    def test_budget_only_records_exact_snapshots_with_existing_clock_reads(self):
        clock = Mock(side_effect=[1000.0, 1001.0, 1002.0])
        ledger = self.ledger(clock)
        clock.assert_not_called()
        self.approve(ledger)
        secret = "private idempotency key"
        reserved = ledger.reserve("run-1", secret, PAYLOAD, attempt_id=ATTEMPT)
        ledger.record_final_charge(reserved.attempt_id, 85, EVIDENCE)
        rows = self.rows()
        self.assertEqual([row["sequence"] for row in rows], [1, 2, 3])
        self.assertEqual([row["event_kind"] for row in rows],
                         ["APPROVAL_RECORDED", "RESERVATION_RECORDED", "CHARGE_RECORDED"])
        self.assertEqual([row["recorded_unix"] for row in rows], [1000.0, 1001.0, 1002.0])
        self.assertEqual([row["attempt_id"] for row in rows], [None, ATTEMPT, ATTEMPT])
        self.assertTrue(all(row["run_id"] == "run-1" and row["submission_event_id"] is None for row in rows))
        expected = [
            {"approved_by": "fixture operator", "limit_cents": 1000, "max_attempts": 6,
             "cost_cents": 140, "started_unix": 1000.0, "deadline_unix": 1060.0},
            {"key_sha256": sha256(secret.encode()).hexdigest(), "payload_sha256": PAYLOAD, "cost_cents": 140},
            {"actual_cents": 85, "evidence_sha256": EVIDENCE},
        ]
        self.assertEqual([self.record(row) for row in rows], expected)
        self.assertEqual([row["record_json"] for row in rows],
                         [json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                          for record in expected])
        self.assertEqual(clock.call_count, 3)
        with sqlite3.connect(self.path) as connection:
            self.assertIsNone(connection.execute(
                "SELECT name FROM sqlite_master WHERE name='submission_events'").fetchone())
            self.assertNotIn(secret, "\n".join(connection.iterdump()))

    def test_valid_long_unicode_approver_fits_the_bounded_snapshot(self):
        ledger = self.ledger()
        name = "\U0001f600" * 200
        self.approve(ledger, name=name)
        row = self.rows()[0]
        self.assertEqual(self.record(row)["approved_by"], name)
        encoded = row["record_json"].encode("ascii")
        self.assertGreater(len(encoded), 2400)
        self.assertLessEqual(len(encoded), 4096)

    def test_shared_order_reconstructs_held_cost_at_equal_time_send_gates(self):
        ledger, first = self.prepared()
        second = ledger.reserve("run-1", "second", PAYLOAD)
        owner = self.owner(ledger)

        def send(attempt, task):
            def create():
                owner.require_send_allowed(attempt)
                return task
            return owner.submit_once(attempt, create)

        self.assertEqual(send(first.attempt_id, TASK).state, "QUEUED")
        ledger.record_final_charge(first.attempt_id, 0, EVIDENCE)
        self.assertEqual(send(second.attempt_id, "fixture-task-2").state, "QUEUED")
        rows = self.rows()
        source_events = {row["event_id"]: row for row in self.rows("submission_events")}
        held, gate_totals = {}, []
        for row in rows:
            self.assertEqual(row["recorded_unix"], 1000.0)
            if row["event_kind"] == "RESERVATION_RECORDED":
                held[row["attempt_id"]] = self.record(row)["cost_cents"]
            elif row["event_kind"] == "CHARGE_RECORDED":
                held[row["attempt_id"]] = self.record(row)["actual_cents"]
            elif row["event_kind"] == "SUBMISSION_EVENT":
                source = source_events[row["submission_event_id"]]
                self.assertEqual((row["attempt_id"], row["recorded_unix"]),
                                 (source["attempt_id"], source["recorded_unix"]))
                self.assertIsNone(row["record_json"])
                if source["event_kind"] == "SEND_GATE_PASSED":
                    gate_totals.append(sum(held.values()))
        self.assertEqual(gate_totals, [280, 140])
        self.assertEqual(len([row for row in rows if row["event_kind"] == "SUBMISSION_EVENT"]), 6)
        self.assertEqual([row["sequence"] for row in rows], list(range(1, 11)))

    def test_submission_links_copy_source_times_without_extra_clock_reads(self):
        ledger, reserved = self.prepared()
        owner = self.owner(ledger)
        ledger.clock = Mock(side_effect=[1001.0, 1002.0, 1003.0, RuntimeError("clock unavailable")])
        checked = Mock()

        def create():
            owner.require_send_allowed(reserved.attempt_id, check_current=checked)
            return TASK

        result = owner.submit_once(reserved.attempt_id, create)
        self.assertEqual((result.state, result.provider_task_id), ("QUEUED", TASK))
        links = self.rows()[2:]
        self.assertEqual([row["recorded_unix"] for row in links], [1001.0, 1003.0, None])
        self.assertEqual([row["submission_event_id"] for row in links], [1, 2, 3])
        self.assertTrue(all(row["record_json"] is None for row in links))
        self.assertEqual(ledger.clock.call_count, 4)
        checked.assert_called_once_with(1003.0)

    def test_repeated_reservation_charge_and_submission_add_no_records(self):
        ledger, reserved = self.prepared()
        owner = self.owner(ledger)
        result = owner.submit_once(reserved.attempt_id, lambda: TASK)
        charge = ledger.record_final_charge(reserved.attempt_id, 100, EVIDENCE)
        before = self.rows()
        ledger.clock = Mock(side_effect=RuntimeError("clock unavailable"))
        self.assertEqual(ledger.reserve("run-1", "first", PAYLOAD, attempt_id=ATTEMPT), reserved)
        self.assertEqual(ledger.record_final_charge(ATTEMPT, 100, EVIDENCE), charge)
        create = Mock()
        self.assertEqual(owner.submit_once(ATTEMPT, create), result)
        owner.close()
        self.assertEqual(self.owner(ledger).submit_once(ATTEMPT, create), result)
        self.assertEqual(self.rows(), before)
        create.assert_not_called()
        ledger.clock.assert_not_called()

    def test_zero_charge_releases_money_but_keeps_attempt_count(self):
        ledger, reserved = self.prepared(limit=140)
        ledger.record_final_charge(reserved.attempt_id, 0, EVIDENCE)
        second = ledger.reserve("run-1", "second", PAYLOAD)
        with ledger._connection() as connection:
            self.assertEqual(ledger._usage(connection, "run-1"), (2, 140, False))
        rows = self.rows()
        self.assertEqual(self.record(rows[2]), {"actual_cents": 0, "evidence_sha256": EVIDENCE})
        self.assertEqual(rows[3]["attempt_id"], second.attempt_id)

    def test_maximum_charge_is_exact_and_blocks_spending_even_without_clock(self):
        ledger, reserved = self.prepared()
        ledger.clock = Mock(side_effect=RuntimeError("clock unavailable"))
        maximum = 2 ** 63 - 1
        recorded = ledger.record_final_charge(reserved.attempt_id, maximum, EVIDENCE)
        self.assertEqual(recorded.actual_cents, maximum)
        row = self.rows()[-1]
        self.assertEqual(self.record(row), {"actual_cents": maximum, "evidence_sha256": EVIDENCE})
        self.assertIsNone(row["recorded_unix"])
        ledger.clock.assert_called_once_with()
        ledger.clock = lambda: 1000.0
        with self.assertRaises(budget.BudgetExceeded):
            ledger.reserve("run-1", "second", PAYLOAD)
        with ledger._connection() as connection:
            self.assertEqual(ledger._usage(connection, "run-1"), (1, maximum, True))

    def test_invalid_settlement_clock_keeps_the_charge_with_null_time(self):
        ledger = self.ledger()
        self.approve(ledger)
        for index, invalid in enumerate((None, float("nan"), float("inf"), True, -1)):
            with self.subTest(clock=invalid):
                ledger.clock = lambda: 1000.0
                reserved = ledger.reserve("run-1", f"attempt-{index}", PAYLOAD)
                ledger.clock = Mock(return_value=invalid)
                ledger.record_final_charge(reserved.attempt_id, 100, EVIDENCE)
                row = self.rows()[-1]
                self.assertEqual((row["event_kind"], row["attempt_id"], row["recorded_unix"]),
                                 ("CHARGE_RECORDED", reserved.attempt_id, None))
                self.assertEqual(self.record(row)["actual_cents"], 100)
                ledger.clock.assert_called_once_with()

    def test_settlement_clock_interruption_rolls_back_charge_and_journal(self):
        ledger, reserved = self.prepared()
        before = self.rows()
        interruption = KeyboardInterrupt("fixture interruption")
        ledger.clock = Mock(side_effect=interruption)
        with self.assertRaises(KeyboardInterrupt) as caught:
            ledger.record_final_charge(reserved.attempt_id, 100, EVIDENCE)
        self.assertIs(caught.exception, interruption)
        self.assertEqual(self.count("actual_charges"), 0)
        self.assertEqual(self.rows(), before)

    def test_concurrent_reservations_have_one_ordered_record_each(self):
        ledger = self.ledger()
        self.approve(ledger)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda index: ledger.reserve("run-1", f"key-{index}", PAYLOAD), range(6)))
        rows = self.rows()
        self.assertEqual([row["sequence"] for row in rows], list(range(1, 8)))
        self.assertEqual({row["attempt_id"] for row in rows[1:]}, {row.attempt_id for row in results})
        self.assertTrue(all(row["event_kind"] == "RESERVATION_RECORDED" for row in rows[1:]))
        self.assertEqual(self.count("reservations"), 6)

    def test_approval_journal_failure_rolls_back_approval(self):
        ledger = self.ledger()
        self.block("APPROVAL_RECORDED")
        with self.assertRaises(sqlite3.IntegrityError):
            self.approve(ledger)
        self.assertEqual(self.count("approved_runs"), 0)
        self.assertEqual(self.rows(), [])

    def test_reservation_journal_failure_rolls_back_reserve_and_high_water(self):
        ledger = self.ledger()
        self.approve(ledger)
        before = self.rows()
        self.block("RESERVATION_RECORDED")
        ledger.clock = lambda: 1001.0
        with self.assertRaises(sqlite3.IntegrityError):
            ledger.reserve("run-1", "first", PAYLOAD)
        self.assertEqual(self.count("reservations"), 0)
        self.assertEqual(self.rows(), before)
        with ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT last_checked_unix FROM approved_runs").fetchone()[0], 1000.0)

    def test_charge_journal_failure_keeps_the_full_reservation(self):
        ledger, reserved = self.prepared()
        before = self.rows()
        self.block("CHARGE_RECORDED")
        with self.assertRaises(sqlite3.IntegrityError):
            ledger.record_final_charge(reserved.attempt_id, 0, EVIDENCE)
        self.assertEqual(self.count("actual_charges"), 0)
        self.assertEqual(self.rows(), before)
        with ledger._connection() as connection:
            self.assertEqual(ledger._usage(connection, "run-1"), (1, 140, False))

    def test_claim_link_failure_rolls_back_source_event_and_prevents_create(self):
        ledger, reserved = self.prepared()
        owner = self.owner(ledger)
        self.block("SUBMISSION_EVENT", submission_kind="CLAIMED")
        before = self.rows()
        create = Mock()
        with self.assertRaises(sqlite3.IntegrityError):
            owner.submit_once(reserved.attempt_id, create)
        self.assertEqual(owner.get(reserved.attempt_id).state, "RESERVED")
        self.assertEqual(self.rows("submission_events"), [])
        self.assertEqual(self.rows(), before)
        create.assert_not_called()

    def test_gate_link_failure_rolls_back_gate_and_blocks_post(self):
        ledger, reserved = self.prepared()
        owner = self.owner(ledger)
        self.block("SUBMISSION_EVENT", submission_kind="SEND_GATE_PASSED")
        ledger.clock = Mock(side_effect=[1001.0, 1002.0, 1003.0, 1004.0])
        post = Mock()

        def create():
            owner.require_send_allowed(reserved.attempt_id, check_current=lambda now: None)
            post()
            return TASK

        self.assertEqual(owner.submit_once(reserved.attempt_id, create).state, "SUBMISSION_UNKNOWN")
        post.assert_not_called()
        self.assertEqual([row["event_kind"] for row in self.rows("submission_events")],
                         ["CLAIMED", "SUBMISSION_UNKNOWN"])
        self.assertEqual([row["submission_event_id"] for row in self.rows()[2:]], [1, 2])
        with ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT last_checked_unix FROM approved_runs").fetchone()[0], 1002.0)

    def test_task_link_failure_keeps_unknown_and_prevents_replay(self):
        ledger, reserved = self.prepared()
        owner = self.owner(ledger)
        self.block("SUBMISSION_EVENT", submission_kind="TASK_RECORDED")
        create = Mock(return_value=TASK)
        result = owner.submit_once(reserved.attempt_id, create)
        self.assertEqual((result.state, result.provider_task_id), ("SUBMISSION_UNKNOWN", None))
        self.assertEqual([row["event_kind"] for row in self.rows("submission_events")],
                         ["CLAIMED", "SUBMISSION_UNKNOWN"])
        self.assertEqual([row["submission_event_id"] for row in self.rows()[2:]], [1, 2])
        self.assertEqual(owner.submit_once(reserved.attempt_id, create), result)
        create.assert_called_once_with()
        with ledger._connection() as connection:
            self.assertEqual(ledger._usage(connection, "run-1"), (1, 140, False))

    def test_unknown_link_failure_leaves_claim_for_later_recovery(self):
        ledger, reserved = self.prepared()
        owner = self.owner(ledger)
        self.block("SUBMISSION_EVENT", submission_kind="SUBMISSION_UNKNOWN")
        create = Mock(side_effect=TimeoutError)
        with self.assertRaises(sqlite3.IntegrityError):
            owner.submit_once(reserved.attempt_id, create)
        self.assertEqual(owner.get(reserved.attempt_id).state, "SUBMITTING")
        self.assertEqual([row["event_kind"] for row in self.rows("submission_events")], ["CLAIMED"])
        self.assertEqual(len(self.rows()), 3)
        owner.close()
        restarted = self.owner(ledger)
        self.assertEqual(restarted.submit_once(reserved.attempt_id, create).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.rows("submission_events")[-1]["event_kind"], "OWNER_RECOVERED_UNKNOWN")
        self.assertEqual(self.rows()[-1]["submission_event_id"], 2)
        create.assert_called_once_with()

    def test_recovery_link_failure_rolls_back_recovery_and_releases_lock(self):
        ledger, reserved = self.prepared()
        owner = self.owner(ledger)
        with self.assertRaises(KeyboardInterrupt):
            owner.submit_once(reserved.attempt_id, Mock(side_effect=KeyboardInterrupt))
        self.block("SUBMISSION_EVENT", submission_kind="OWNER_RECOVERED_UNKNOWN")
        before = self.rows()
        owner.close()
        with self.assertRaises(sqlite3.IntegrityError):
            self.owner(ledger)
        self.assertEqual(self.rows(), before)
        self.assertEqual([row["event_kind"] for row in self.rows("submission_events")], ["CLAIMED"])
        with ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT state FROM submission_jobs").fetchone()[0], "SUBMITTING")
        self.unblock()
        self.assertEqual(self.owner(ledger).get(reserved.attempt_id).state, "SUBMISSION_UNKNOWN")

    def test_reconciliation_link_failure_preserves_unknown_and_evidence(self):
        ledger, reserved = self.prepared()
        owner = self.owner(ledger)
        owner.submit_once(reserved.attempt_id, Mock(side_effect=TimeoutError))
        before, source_before = self.rows(), self.rows("submission_events")
        self.block("SUBMISSION_EVENT", submission_kind="TASK_RECONCILED")
        with self.assertRaises(sqlite3.IntegrityError):
            owner.record_verified_task(reserved.attempt_id, TASK, EVIDENCE)
        result = owner.get(reserved.attempt_id)
        self.assertEqual((result.state, result.provider_task_id, result.evidence_sha256),
                         ("SUBMISSION_UNKNOWN", None, None))
        self.assertEqual(self.rows(), before)
        self.assertEqual(self.rows("submission_events"), source_before)

    def test_existing_database_gains_no_backfilled_financial_or_submission_links(self):
        with sqlite3.connect(self.path) as connection:
            connection.executescript("""
                CREATE TABLE approved_runs (run_id TEXT PRIMARY KEY, approved_by TEXT, limit_cents INTEGER,
                    max_attempts INTEGER, cost_cents INTEGER, started_unix REAL, deadline_unix REAL,
                    last_checked_unix REAL, time_stopped INTEGER);
                CREATE TABLE reservations (run_id TEXT, key_sha256 TEXT, attempt_id TEXT UNIQUE,
                    payload_sha256 TEXT, cost_cents INTEGER, state TEXT, PRIMARY KEY(run_id, key_sha256));
                CREATE TABLE actual_charges (attempt_id TEXT PRIMARY KEY, actual_cents INTEGER,
                    evidence_sha256 TEXT);
                CREATE TABLE submission_jobs (attempt_id TEXT PRIMARY KEY, state TEXT,
                    provider_task_id TEXT UNIQUE, evidence_sha256 TEXT);
                CREATE TABLE submission_events (event_id INTEGER PRIMARY KEY, attempt_id TEXT,
                    event_kind TEXT, recorded_unix REAL, provider_task_id TEXT, evidence_sha256 TEXT);
            """)
            connection.execute("INSERT INTO approved_runs VALUES ('run-1', 'old operator', 1000, 6, 140, "
                               "1000, 1060, 1000, 0)")
            connection.execute("INSERT INTO reservations VALUES ('run-1', ?, ?, ?, 140, 'RESERVED')",
                               (sha256(b"old-key").hexdigest(), ATTEMPT, PAYLOAD))
            connection.execute("INSERT INTO actual_charges VALUES (?, 100, ?)", (ATTEMPT, EVIDENCE))
            connection.execute("INSERT INTO submission_jobs VALUES (?, 'QUEUED', ?, NULL)", (ATTEMPT, TASK))
            connection.execute("INSERT INTO submission_events VALUES (7, ?, 'TASK_RECORDED', 1000, ?, NULL)",
                               (ATTEMPT, TASK))
        old_source = self.rows("submission_events")
        clock = Mock(side_effect=RuntimeError("clock unavailable"))
        ledger = self.ledger(clock)
        owner = self.owner(ledger)
        ledger.record_final_charge(ATTEMPT, 100, EVIDENCE)
        self.assertEqual(owner.get(ATTEMPT).state, "QUEUED")
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.rows("submission_events"), old_source)
        clock.assert_not_called()
        ledger.clock = lambda: 1001.0
        fresh = ledger.reserve("run-1", "new-key", PAYLOAD)
        owner.submit_once(fresh.attempt_id, lambda: "new-task")
        rows = self.rows()
        self.assertEqual([row["event_kind"] for row in rows],
                         ["RESERVATION_RECORDED", "SUBMISSION_EVENT", "SUBMISSION_EVENT"])
        self.assertEqual([row["submission_event_id"] for row in rows], [None, 8, 9])
        self.assertEqual(self.rows("submission_events")[0], old_source[0])

    def test_journal_rows_cannot_be_updated_or_deleted(self):
        ledger, reserved = self.prepared()
        before = self.rows()
        for statement in ("UPDATE run_events SET recorded_unix=1005", "DELETE FROM run_events"):
            with self.subTest(statement=statement), sqlite3.connect(self.path) as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(statement)
            self.assertEqual(self.rows(), before)

    def test_replace_cannot_collide_with_sequence_or_submission_link(self):
        ledger, reserved = self.prepared()
        self.owner(ledger).submit_once(reserved.attempt_id, lambda: TASK)
        before = self.rows()
        collisions = [("sequence", before[0], before[0]["sequence"]),
                      ("submission link", before[-1], before[-1]["sequence"] + 100)]
        for kind, source, sequence in collisions:
            with self.subTest(collision=kind), sqlite3.connect(self.path) as connection:
                self.assertEqual(connection.execute("PRAGMA recursive_triggers").fetchone()[0], 0)
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT OR REPLACE INTO run_events (sequence, run_id, attempt_id, event_kind, "
                        "recorded_unix, submission_event_id, record_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (sequence, source["run_id"], source["attempt_id"], source["event_kind"],
                         1005.0, source["submission_event_id"], source["record_json"]))
            self.assertEqual(self.rows(), before)

    def test_missing_submission_source_cannot_create_a_link(self):
        ledger, reserved = self.prepared()
        self.owner(ledger)
        before = self.rows()
        with ledger._transaction() as connection:
            with self.assertRaises(ValueError):
                events.record_submission(connection, 999)
        self.assertEqual(self.rows(), before)

    def test_schema_rejects_invalid_event_shapes_and_bounds(self):
        ledger, reserved = self.prepared()
        before = self.rows()
        base = {"sequence": 50, "run_id": "run-1", "attempt_id": ATTEMPT,
                "event_kind": "CHARGE_RECORDED", "recorded_unix": 1000.0,
                "submission_event_id": None, "record_json": "{}"}
        invalid = [
            {"sequence": 0}, {"sequence": -1}, {"event_kind": "REMOTE_RECEIPT"},
            {"recorded_unix": -1.0}, {"recorded_unix": float("inf")},
            {"record_json": "x" * 4097}, {"record_json": None},
            {"event_kind": "APPROVAL_RECORDED"}, {"attempt_id": None},
            {"event_kind": "RESERVATION_RECORDED", "recorded_unix": None},
            {"submission_event_id": 1},
            {"event_kind": "SUBMISSION_EVENT", "record_json": None},
            {"event_kind": "SUBMISSION_EVENT", "submission_event_id": 0, "record_json": None},
        ]
        for changed in invalid:
            row = base | changed
            with self.subTest(changed=changed), ledger._connection() as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT INTO run_events (sequence, run_id, attempt_id, event_kind, recorded_unix, "
                        "submission_event_id, record_json) VALUES (:sequence, :run_id, :attempt_id, :event_kind, "
                        ":recorded_unix, :submission_event_id, :record_json)", row)
        self.assertEqual(self.rows(), before)
