"""Keep one-run operator approval atomic without changing legacy ledger use."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
import unittest
from unittest.mock import Mock, patch

from core import budget
from features.inference import submission


SCOPE_ERROR = "A different run already has approval in this ledger."
DUPLICATE_ERROR = "This run already has a server-side approval."
PAYLOAD = "a" * 64
EVIDENCE = "b" * 64


class ApprovalScopeTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / "budget.sqlite"

    def approve(self, ledger, run_id="run-1", **changes):
        arguments = {"approved_by": "fixture operator", "limit_cents": 1000,
                     "max_attempts": 6, "cost_cents": 140, "max_elapsed_minutes": 180}
        arguments.update(changes)
        return ledger.approve_run(run_id, **arguments)

    def snapshot(self, path=None):
        with sqlite3.connect(path or self.path) as connection:
            return tuple(connection.iterdump())

    def rows(self, table, path=None):
        with sqlite3.connect(path or self.path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]

    def assert_scope_error(self, operation):
        with self.assertRaises(budget.BudgetError) as caught:
            operation()
        self.assertIs(type(caught.exception), budget.BudgetError)
        self.assertEqual(str(caught.exception), SCOPE_ERROR)

    def test_first_guarded_approval_keeps_exact_terms_journal_and_one_clock_read(self):
        clock = Mock(return_value=1000.25)
        ledger = budget.BudgetLedger(self.path, clock=clock)
        clock.assert_not_called()
        self.assertIsNone(self.approve(ledger, only_run=True, deadline_unix_cap=2500.0))
        clock.assert_called_once_with()
        self.assertEqual(self.rows("approved_runs"), [{
            "run_id": "run-1", "approved_by": "fixture operator", "limit_cents": 1000,
            "max_attempts": 6, "cost_cents": 140, "started_unix": 1000.25,
            "deadline_unix": 2500.0, "last_checked_unix": 1000.25, "time_stopped": 0,
        }])
        expected_record = {"approved_by": "fixture operator", "limit_cents": 1000,
            "max_attempts": 6, "cost_cents": 140, "started_unix": 1000.25, "deadline_unix": 2500.0}
        self.assertEqual(self.rows("run_events"), [{
            "sequence": 1, "run_id": "run-1", "attempt_id": None, "event_kind": "APPROVAL_RECORDED",
            "recorded_unix": 1000.25, "submission_event_id": None,
            "record_json": json.dumps(expected_record, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        }])
        self.assertEqual(self.rows("reservations"), [])
        self.assertEqual(self.rows("actual_charges"), [])

    def test_non_boolean_scope_fails_before_clock_or_database_work(self):
        clock = Mock(side_effect=budget.BudgetError("Fixture clock must not be read."))
        ledger = budget.BudgetLedger(self.path, clock=clock)
        before = self.snapshot()
        for value in (None, 0, 1, 0.0, 1.0, "false", "true", [], {}, object()):
            with self.subTest(value=repr(value)):
                with patch.object(ledger, "_transaction", side_effect=budget.BudgetError(
                        "Fixture approval transaction must not open.")) as transaction, \
                        patch.object(ledger, "_connection", side_effect=budget.BudgetError(
                            "Fixture approval connection must not open.")) as connection:
                    with self.assertRaises(budget.BudgetError) as caught:
                        self.approve(ledger, only_run=value)
                    self.assertEqual(str(caught.exception), "Approval scope must be a boolean.")
                    transaction.assert_not_called()
                    connection.assert_not_called()
                    clock.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_other_run_history_blocks_active_expired_stopped_settled_and_null_rows(self):
        for kind in ("active", "expired", "stopped", "zero_charge", "settled", "null_legacy"):
            with self.subTest(history=kind):
                path = self.root / (kind + ".sqlite")
                clock = Mock(return_value=1000.0)
                ledger = budget.BudgetLedger(path, clock=clock)
                if kind == "null_legacy":
                    # SQLite permits NULL in this historical TEXT PRIMARY KEY.
                    # The guard must fail closed without inventing a journal row.
                    with sqlite3.connect(path) as connection:
                        connection.execute("INSERT INTO approved_runs VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
                            ("legacy operator", 1000, 6, 140, 1000.0, 1060.0, 1000.0, 0))
                else:
                    self.approve(ledger, "run-old", max_elapsed_minutes=1)
                if kind in ("expired", "stopped"):
                    clock.return_value = 1060.0
                if kind == "stopped":
                    with self.assertRaises(budget.BudgetExceeded):
                        ledger.reserve("run-old", "expired reservation", PAYLOAD)
                    self.assertEqual(self.rows("approved_runs", path)[0]["time_stopped"], 1)
                if kind in ("zero_charge", "settled"):
                    reserved = ledger.reserve("run-old", "first reservation", PAYLOAD)
                    ledger.record_final_charge(reserved.attempt_id, 0 if kind == "zero_charge" else 140, EVIDENCE)
                before = self.snapshot(path)
                self.assert_scope_error(lambda: self.approve(ledger, "run-new", only_run=True))
                self.assertEqual(self.snapshot(path), before,
                                 "A rejected scope change altered previous authority or its journal.")

    def test_same_run_duplicate_preserves_original_terms_deadline_and_journal(self):
        clock = Mock(return_value=1000.0)
        ledger = budget.BudgetLedger(self.path, clock=clock)
        terms = {"limit_cents": 200, "max_attempts": 1, "cost_cents": 100,
                 "max_elapsed_minutes": 1, "deadline_unix_cap": 1050.0, "only_run": True}
        self.approve(ledger, **terms)
        before = self.snapshot()
        for changes in (terms, {"approved_by": "replacement operator", "deadline_unix_cap": 9000.0,
                               "only_run": True}):
            with self.subTest(replacement=changes):
                clock.return_value = 1010.0
                with self.assertRaises(budget.BudgetError) as caught:
                    self.approve(ledger, **changes)
                self.assertIs(type(caught.exception), budget.BudgetError)
                self.assertEqual(str(caught.exception), DUPLICATE_ERROR)
                self.assertEqual(self.snapshot(), before,
                                 "Duplicate approval changed retained terms, time or journal history.")

    def test_default_and_explicit_false_preserve_legacy_multi_run_approval(self):
        ledger = budget.BudgetLedger(self.path, clock=lambda: 1000.0)
        self.approve(ledger, "run-1")
        self.approve(ledger, "run-2")
        self.approve(ledger, "run-3", only_run=False)
        self.assertEqual([row["run_id"] for row in self.rows("approved_runs")], ["run-1", "run-2", "run-3"])
        self.assertEqual([(row["run_id"], row["event_kind"]) for row in self.rows("run_events")],
                         [("run-1", "APPROVAL_RECORDED"), ("run-2", "APPROVAL_RECORDED"),
                          ("run-3", "APPROVAL_RECORDED")])
        before = self.snapshot()
        self.assert_scope_error(lambda: self.approve(ledger, "run-4", only_run=True))
        self.assertEqual(self.snapshot(), before, "The guarded request changed a legacy multi-run ledger.")

    def test_two_guarded_callers_commit_exactly_one_approval(self):
        barrier = Barrier(2, timeout=3)

        def clock():
            barrier.wait(timeout=3)
            return 1000.0

        clocks = [Mock(side_effect=clock), Mock(side_effect=clock)]
        ledgers = [budget.BudgetLedger(self.path, clock=value) for value in clocks]
        outcomes = []

        def approve(index):
            run_id = "run-" + str(index + 1)
            try:
                self.approve(ledgers[index], run_id, only_run=True)
            except BaseException as error:
                outcomes.append((run_id, error))
            else:
                outcomes.append((run_id, None))

        threads = [Thread(target=approve, args=(index,), name=f"approval-scope-{index}") for index in range(2)]
        started = []
        try:
            for thread in threads:
                thread.start()
                started.append(thread)
            for thread in started:
                thread.join(timeout=5)
        finally:
            barrier.abort()
            for thread in started:
                thread.join(timeout=5)
        self.assertTrue(all(not thread.is_alive() for thread in started), "The bounded approval race did not finish.")
        self.assertEqual(len(outcomes), 2)
        winners = [run_id for run_id, error in outcomes if error is None]
        losers = [error for run_id, error in outcomes if error is not None]
        self.assertEqual(len(winners), 1, "Two guarded callers did not commit exactly one approval.")
        self.assertEqual(len(losers), 1)
        self.assertIs(type(losers[0]), budget.BudgetError, "The losing approval failed for an unrelated reason.")
        self.assertEqual(str(losers[0]), SCOPE_ERROR)
        for read_clock in clocks:
            read_clock.assert_called_once_with()
        self.assertEqual([row["run_id"] for row in self.rows("approved_runs")], winners)
        self.assertEqual([(row["run_id"], row["event_kind"]) for row in self.rows("run_events")],
                         [(winners[0], "APPROVAL_RECORDED")])

    def test_guard_lookup_already_holds_a_real_sqlite_writer_lock(self):
        ledger = budget.BudgetLedger(self.path, clock=lambda: 1000.0)
        self.approve(ledger, "run-old")
        before = self.snapshot()
        original_connection = ledger._connection
        probes = []

        def observe_read(action, table, column, database, trigger):
            if action != sqlite3.SQLITE_READ or table != "approved_runs" or probes:
                return sqlite3.SQLITE_OK
            # A separate real connection cannot write during the guard lookup.
            # Store observations: SQLite callbacks must not carry assertions.
            connection = sqlite3.connect(self.path, timeout=0.05, isolation_level=None)
            try:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as error:
                    probes.append(error.sqlite_errorcode == sqlite3.SQLITE_BUSY)
                else:
                    probes.append(False)
            finally:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                connection.close()
            return sqlite3.SQLITE_OK

        @contextmanager
        def observed_connection():
            with original_connection() as connection:
                connection.set_authorizer(observe_read)
                try:
                    yield connection
                finally:
                    connection.set_authorizer(None)

        with patch.object(ledger, "_connection", observed_connection):
            self.assert_scope_error(lambda: self.approve(ledger, "run-new", only_run=True))
        self.assertEqual(probes, [True], "The approval guard read without owning the SQLite write transaction.")
        self.assertEqual(self.snapshot(), before)

    def test_guarded_approval_uses_existing_live_owner_without_recovery_or_lock_release(self):
        ledger = budget.BudgetLedger(self.path, clock=lambda: 1000.0)
        owner = submission.SubmissionCoordinator(ledger)
        try:
            owner_lock = owner._lock
            writer = budget.BudgetLedger(self.path, clock=lambda: 1000.0)
            self.approve(writer, "run-live", only_run=True)
            reserved = writer.reserve("run-live", "held reservation", PAYLOAD)
            self.assertTrue(owner._claim(reserved.attempt_id))
            self.assertEqual(owner.get(reserved.attempt_id).state, "SUBMITTING")
            before = self.snapshot()
            self.assert_scope_error(lambda: self.approve(writer, "run-other", only_run=True))
            self.assertEqual(self.snapshot(), before, "Guarded approval recovered or changed the live submission.")
            self.assertEqual(owner.get(reserved.attempt_id).state, "SUBMITTING")
            self.assertEqual(writer.reserve("run-live", "held reservation", PAYLOAD), reserved)
            self.assertEqual(self.rows("actual_charges"), [])
            self.assertEqual(owner._lock, owner_lock)
            descriptor = os.open(str(self.path) + ".submission.lock", os.O_RDWR | os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    held = True
                else:
                    held = False
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            self.assertTrue(held, "Approval released the existing coordinator's process lock.")
        finally:
            owner.close()


if __name__ == "__main__":
    unittest.main()
