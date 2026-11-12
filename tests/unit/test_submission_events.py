"""Check local submission events without claiming remote receipt or billing."""

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from core.budget import BudgetError, BudgetExceeded, BudgetLedger
from features.inference import submission


PAYLOAD = "a" * 64
EVIDENCE = "b" * 64
TASK = "fixture-task-1"


class SubmissionEventTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "budget.sqlite"
        self.ledger = BudgetLedger(self.path, clock=lambda: 1000.0)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                                max_attempts=6, cost_cents=140, max_elapsed_minutes=1)
        self.attempt = self.ledger.reserve("run-1", "first", PAYLOAD).attempt_id

    def owner(self):
        owner = submission.SubmissionCoordinator(self.ledger)
        self.addCleanup(owner.close)
        return owner

    def events(self):
        with self.ledger._connection() as connection:
            rows = connection.execute(
                "SELECT attempt_id, event_kind, recorded_unix, provider_task_id, evidence_sha256 "
                "FROM submission_events ORDER BY event_id").fetchall()
        return [dict(row) for row in rows]

    def event(self, kind, time, task=None, evidence=None):
        return {"attempt_id": self.attempt, "event_kind": kind, "recorded_unix": time,
                "provider_task_id": task, "evidence_sha256": evidence}

    def authority(self):
        with self.ledger._connection() as connection:
            row = connection.execute(
                "SELECT last_checked_unix, time_stopped FROM approved_runs WHERE run_id='run-1'").fetchone()
        return tuple(row)

    def assert_cost_held(self):
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (1, 140, False))
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM actual_charges").fetchone()[0], 0)

    def block_event(self, kind):
        # SQLite rejects the event inside the real transaction. No product method
        # is replaced, so the test can observe whether its state change commits.
        with self.ledger._connection() as connection:
            connection.execute(
                "CREATE TRIGGER fail_fixture_event BEFORE INSERT ON submission_events "
                f"WHEN NEW.event_kind='{kind}' BEGIN SELECT RAISE(ABORT, 'fixture event failure'); END")

    def allow_events(self):
        with self.ledger._connection() as connection:
            connection.execute("DROP TRIGGER fail_fixture_event")

    def test_success_records_committed_boundaries_with_one_clock_read_each(self):
        self.ledger.clock = Mock(side_effect=[1001.0, 1002.0, 1003.0, 1004.0])
        owner = self.owner()
        self.ledger.clock.assert_not_called()
        checked_times = []

        def create():
            self.assertEqual(owner.get(self.attempt).state, "SUBMITTING")
            self.assertEqual(self.events(), [self.event("CLAIMED", 1001.0)])
            owner.require_send_allowed(self.attempt, check_current=checked_times.append)
            self.assertEqual(self.events(), [self.event("CLAIMED", 1001.0),
                                            self.event("SEND_GATE_PASSED", 1003.0)])
            # The adapter runs after the gate transaction has committed.
            with sqlite3.connect(self.path, timeout=0.1) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.rollback()
            return TASK

        result = owner.submit_once(self.attempt, create)
        self.assertEqual((result.state, result.provider_task_id), ("QUEUED", TASK))
        self.assertEqual(self.events(), [self.event("CLAIMED", 1001.0),
                                        self.event("SEND_GATE_PASSED", 1003.0),
                                        self.event("TASK_RECORDED", 1004.0, TASK)])
        self.assertEqual(checked_times, [1003.0])
        self.assertEqual(self.ledger.clock.call_count, 4)
        self.assertEqual(self.authority(), (1003.0, 0))
        self.assert_cost_held()

    def test_gate_without_freshness_keeps_its_single_clock_read(self):
        self.ledger.clock = Mock(side_effect=[1001.0, 1002.0, 1003.0])
        owner = self.owner()

        def create():
            owner.require_send_allowed(self.attempt)
            return TASK

        self.assertEqual(owner.submit_once(self.attempt, create).state, "QUEUED")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1001.0),
                                        self.event("SEND_GATE_PASSED", 1002.0),
                                        self.event("TASK_RECORDED", 1003.0, TASK)])
        self.assertEqual(self.ledger.clock.call_count, 3)

    def test_bare_adapter_does_not_invent_a_send_gate(self):
        owner = self.owner()
        owner.submit_once(self.attempt, lambda: TASK)
        self.assertEqual(self.events(), [self.event("CLAIMED", 1000.0),
                                        self.event("TASK_RECORDED", 1000.0, TASK)])

    def test_known_retry_and_owner_restart_add_no_events_or_clock_reads(self):
        owner = self.owner()
        result = owner.submit_once(self.attempt, lambda: TASK)
        before = self.events()
        self.ledger.clock = Mock(side_effect=RuntimeError("clock unavailable"))
        create = Mock()
        self.assertEqual(owner.submit_once(self.attempt, create), result)
        owner.close()
        restarted = self.owner()
        self.assertEqual(restarted.submit_once(self.attempt, create), result)
        self.assertEqual(self.events(), before)
        create.assert_not_called()
        self.ledger.clock.assert_not_called()

    def test_uncertain_retry_keeps_cost_and_records_no_exception_text(self):
        owner = self.owner()
        create = Mock(side_effect=TimeoutError("private provider token"))
        result = owner.submit_once(self.attempt, create)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        expected = [self.event("CLAIMED", 1000.0), self.event("SUBMISSION_UNKNOWN", 1000.0)]
        self.assertEqual(self.events(), expected)
        self.ledger.clock = Mock(side_effect=RuntimeError("clock unavailable"))
        self.assertEqual(owner.submit_once(self.attempt, create), result)
        owner.close()
        self.assertEqual(self.owner().submit_once(self.attempt, create), result)
        self.assertEqual(self.events(), expected)
        create.assert_called_once_with()
        self.ledger.clock.assert_not_called()
        self.assert_cost_held()
        with sqlite3.connect(self.path) as connection:
            self.assertNotIn("private provider token", "\n".join(connection.iterdump()))

    def test_existing_jobs_gain_only_a_current_recovery_event(self):
        second = self.ledger.reserve("run-1", "second", PAYLOAD).attempt_id
        third = self.ledger.reserve("run-1", "third", PAYLOAD).attempt_id
        with self.ledger._transaction() as connection:
            connection.execute("CREATE TABLE submission_jobs (attempt_id TEXT PRIMARY KEY "
                               "REFERENCES reservations(attempt_id), state TEXT NOT NULL, "
                               "provider_task_id TEXT UNIQUE, evidence_sha256 TEXT)")
            connection.executemany("INSERT INTO submission_jobs VALUES (?, ?, ?, ?)", [
                (self.attempt, "SUBMITTING", None, None),
                (second, "QUEUED", TASK, None),
                (third, "SUBMISSION_UNKNOWN", None, None)])
        self.ledger.clock = Mock(return_value=1200.0)
        owner = self.owner()
        self.assertEqual(owner.get(self.attempt).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(owner.get(second).provider_task_id, TASK)
        self.assertEqual(owner.get(third).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events(), [self.event("OWNER_RECOVERED_UNKNOWN", 1200.0)])
        self.ledger.clock.assert_called_once_with()
        self.assertEqual(self.authority(), (1000.0, 0))

    def test_observation_clock_failure_does_not_lose_a_returned_task(self):
        for invalid in (RuntimeError("clock unavailable"), float("nan"), float("inf"), True, -1):
            with self.subTest(clock=invalid):
                # One independent attempt per clock failure avoids retry shortcuts.
                attempt = self.ledger.reserve("run-1", f"clock-{str(invalid)}", PAYLOAD).attempt_id
                owner = self.owner()

                def create():
                    self.ledger.clock = (Mock(side_effect=invalid) if isinstance(invalid, Exception)
                                         else Mock(return_value=invalid))
                    return "task-" + attempt

                result = owner.submit_once(attempt, create)
                self.assertEqual((result.state, result.provider_task_id), ("QUEUED", "task-" + attempt))
                self.assertEqual(self.events()[-1], {"attempt_id": attempt, "event_kind": "TASK_RECORDED",
                                                     "recorded_unix": None,
                                                     "provider_task_id": "task-" + attempt,
                                                     "evidence_sha256": None})
                self.ledger.clock.assert_called_once_with()
                owner.close()
                self.ledger.clock = lambda: 1000.0

    def test_unknown_and_reconciliation_allow_missing_clock(self):
        owner = self.owner()

        def uncertain():
            self.ledger.clock = Mock(side_effect=RuntimeError("clock unavailable"))
            raise TimeoutError("create did not return")

        self.assertEqual(owner.submit_once(self.attempt, uncertain).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events()[-1], self.event("SUBMISSION_UNKNOWN", None))
        result = owner.record_verified_task(self.attempt, TASK, EVIDENCE)
        self.assertEqual((result.state, result.provider_task_id, result.evidence_sha256),
                         ("QUEUED", TASK, EVIDENCE))
        self.assertEqual(self.events()[-1], self.event("TASK_RECONCILED", None, TASK, EVIDENCE))
        self.assert_cost_held()

    def test_observation_clock_interruption_propagates_with_claim_still_durable(self):
        owner = self.owner()
        interruption = KeyboardInterrupt("fixture clock interruption")

        def create():
            self.ledger.clock = Mock(side_effect=interruption)
            return TASK

        with self.assertRaises(KeyboardInterrupt) as caught:
            owner.submit_once(self.attempt, create)
        self.assertIs(caught.exception, interruption)
        self.assertEqual(owner.get(self.attempt).state, "SUBMITTING")
        self.assertIsNone(owner.get(self.attempt).provider_task_id)
        self.assertEqual(self.events(), [self.event("CLAIMED", 1000.0)])
        owner.close()
        self.ledger.clock = lambda: 1001.0
        restarted = self.owner()
        create_again = Mock()
        self.assertEqual(restarted.submit_once(self.attempt, create_again).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events()[-1], self.event("OWNER_RECOVERED_UNKNOWN", 1001.0))
        create_again.assert_not_called()
        self.assert_cost_held()

    def test_abandoned_claim_recovers_with_null_time_and_cannot_replay(self):
        owner = self.owner()
        with self.assertRaises(KeyboardInterrupt):
            owner.submit_once(self.attempt, Mock(side_effect=KeyboardInterrupt))
        owner.close()
        self.ledger.clock = Mock(side_effect=RuntimeError("clock unavailable"))
        restarted = self.owner()
        create = Mock()
        self.assertEqual(restarted.submit_once(self.attempt, create).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1000.0),
                                        self.event("OWNER_RECOVERED_UNKNOWN", None)])
        create.assert_not_called()
        self.ledger.clock.assert_called_once_with()
        self.assert_cost_held()

    def test_reconciliation_records_the_task_evidence_and_observation_time(self):
        owner = self.owner()
        owner.submit_once(self.attempt, Mock(side_effect=TimeoutError))
        self.ledger.clock = Mock(return_value=1200.0)
        result = owner.record_verified_task(self.attempt, TASK, EVIDENCE)
        self.assertEqual((result.state, result.provider_task_id, result.evidence_sha256),
                         ("QUEUED", TASK, EVIDENCE))
        self.assertEqual(self.events()[-1], self.event("TASK_RECONCILED", 1200.0, TASK, EVIDENCE))
        self.ledger.clock.assert_called_once_with()
        self.assertEqual(self.authority(), (1000.0, 0))

    def test_claim_requires_a_valid_time_before_any_transition(self):
        owner = self.owner()
        create = Mock()
        for invalid in (None, float("nan"), float("inf"), True, -1):
            with self.subTest(clock=invalid):
                self.ledger.clock = Mock(return_value=invalid)
                with self.assertRaises(BudgetError):
                    owner.submit_once(self.attempt, create)
                self.assertEqual(owner.get(self.attempt).state, "RESERVED")
                self.assertEqual(self.events(), [])
                self.ledger.clock.assert_called_once_with()
        create.assert_not_called()

    def test_gate_requires_a_valid_time_and_does_not_record_a_pass(self):
        owner = self.owner()
        checked, post = Mock(), Mock()

        def create():
            self.ledger.clock = Mock(return_value=float("nan"))
            owner.require_send_allowed(self.attempt, check_current=checked)
            post()
            return TASK

        result = owner.submit_once(self.attempt, create)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1000.0),
                                        self.event("SUBMISSION_UNKNOWN", None)])
        checked.assert_not_called()
        post.assert_not_called()
        self.assertEqual(self.authority(), (1000.0, 0))

    def test_freshness_error_keeps_final_high_water_without_gate_pass(self):
        self.ledger.clock = Mock(side_effect=[1001.0, 1002.0, 1003.0, 1004.0])
        owner = self.owner()
        checked = Mock(side_effect=ValueError("fixture check expired"))
        post = Mock()

        def create():
            owner.require_send_allowed(self.attempt, check_current=checked)
            post()
            return TASK

        self.assertEqual(owner.submit_once(self.attempt, create).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1001.0),
                                        self.event("SUBMISSION_UNKNOWN", 1004.0)])
        self.assertEqual(self.authority(), (1003.0, 0))
        checked.assert_called_once_with(1003.0)
        post.assert_not_called()
        self.assertEqual(self.ledger.clock.call_count, 4)
        self.assert_cost_held()

    def test_freshness_interruption_keeps_final_high_water_and_claim(self):
        self.ledger.clock = Mock(side_effect=[1001.0, 1002.0, 1003.0])
        owner = self.owner()
        interruption = KeyboardInterrupt("fixture interruption")
        checked = Mock(side_effect=interruption)
        post = Mock()

        def create():
            owner.require_send_allowed(self.attempt, check_current=checked)
            post()
            return TASK

        with self.assertRaises(KeyboardInterrupt) as caught:
            owner.submit_once(self.attempt, create)
        self.assertIs(caught.exception, interruption)
        self.assertEqual(owner.get(self.attempt).state, "SUBMITTING")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1001.0)])
        self.assertEqual(self.authority(), (1003.0, 0))
        checked.assert_called_once_with(1003.0)
        post.assert_not_called()
        self.assertEqual(self.ledger.clock.call_count, 3)

    def test_final_deadline_stays_stopped_without_gate_or_freshness_call(self):
        self.ledger.clock = Mock(side_effect=[1001.0, 1059.0, 1060.0, 1061.0])
        owner = self.owner()
        checked, post = Mock(), Mock()

        def create():
            owner.require_send_allowed(self.attempt, check_current=checked)
            post()
            return TASK

        self.assertEqual(owner.submit_once(self.attempt, create).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1001.0),
                                        self.event("SUBMISSION_UNKNOWN", 1061.0)])
        self.assertEqual(self.authority(), (1059.0, 1))
        checked.assert_not_called()
        post.assert_not_called()
        self.ledger.clock = lambda: 1050.0
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "late", PAYLOAD)
        self.assert_cost_held()

    def test_claim_event_failure_rolls_back_claim_and_budget_high_water(self):
        owner = self.owner()
        self.block_event("CLAIMED")
        self.ledger.clock = lambda: 1001.0
        create = Mock()
        with self.assertRaises(sqlite3.IntegrityError):
            owner.submit_once(self.attempt, create)
        self.assertEqual(owner.get(self.attempt).state, "RESERVED")
        self.assertEqual(self.events(), [])
        self.assertEqual(self.authority(), (1000.0, 0))
        create.assert_not_called()

    def test_gate_event_failure_rolls_back_final_high_water_and_blocks_send(self):
        owner = self.owner()
        self.block_event("SEND_GATE_PASSED")
        self.ledger.clock = Mock(side_effect=[1001.0, 1002.0, 1003.0, 1004.0])
        checked, post = Mock(), Mock()

        def create():
            owner.require_send_allowed(self.attempt, check_current=checked)
            post()
            return TASK

        self.assertEqual(owner.submit_once(self.attempt, create).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1001.0),
                                        self.event("SUBMISSION_UNKNOWN", 1004.0)])
        self.assertEqual(self.authority(), (1002.0, 0))
        checked.assert_called_once_with(1003.0)
        post.assert_not_called()
        self.assert_cost_held()

    def test_task_event_failure_rolls_back_task_and_keeps_unknown(self):
        owner = self.owner()
        self.block_event("TASK_RECORDED")
        create = Mock(return_value=TASK)
        result = owner.submit_once(self.attempt, create)
        self.assertEqual((result.state, result.provider_task_id), ("SUBMISSION_UNKNOWN", None))
        self.assertEqual(self.events(), [self.event("CLAIMED", 1000.0),
                                        self.event("SUBMISSION_UNKNOWN", 1000.0)])
        self.assertEqual(owner.submit_once(self.attempt, create), result)
        create.assert_called_once_with()
        self.assert_cost_held()

    def test_unknown_event_failure_leaves_durable_claim_for_owner_recovery(self):
        owner = self.owner()
        self.block_event("SUBMISSION_UNKNOWN")
        create = Mock(side_effect=TimeoutError)
        with self.assertRaises(sqlite3.IntegrityError):
            owner.submit_once(self.attempt, create)
        self.assertEqual(owner.get(self.attempt).state, "SUBMITTING")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1000.0)])
        owner.close()
        self.ledger.clock = lambda: 1001.0
        restarted = self.owner()
        self.assertEqual(restarted.submit_once(self.attempt, create).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1000.0),
                                        self.event("OWNER_RECOVERED_UNKNOWN", 1001.0)])
        create.assert_called_once_with()
        self.assert_cost_held()

    def test_recovery_event_failure_rolls_back_recovery_and_releases_owner_lock(self):
        owner = self.owner()
        with self.assertRaises(KeyboardInterrupt):
            owner.submit_once(self.attempt, Mock(side_effect=KeyboardInterrupt))
        self.block_event("OWNER_RECOVERED_UNKNOWN")
        owner.close()
        with self.assertRaises(sqlite3.IntegrityError):
            self.owner()
        with self.ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT state FROM submission_jobs").fetchone()[0], "SUBMITTING")
        self.assertEqual(self.events(), [self.event("CLAIMED", 1000.0)])
        self.allow_events()
        restarted = self.owner()
        self.assertEqual(restarted.get(self.attempt).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.events()[-1], self.event("OWNER_RECOVERED_UNKNOWN", 1000.0))

    def test_reconciliation_event_failure_preserves_unknown_and_no_task(self):
        owner = self.owner()
        owner.submit_once(self.attempt, Mock(side_effect=TimeoutError))
        before = self.events()
        self.block_event("TASK_RECONCILED")
        with self.assertRaises(sqlite3.IntegrityError):
            owner.record_verified_task(self.attempt, TASK, EVIDENCE)
        result = owner.get(self.attempt)
        self.assertEqual((result.state, result.provider_task_id, result.evidence_sha256),
                         ("SUBMISSION_UNKNOWN", None, None))
        self.assertEqual(self.events(), before)
        self.assert_cost_held()

    def test_event_rows_cannot_be_updated_or_deleted(self):
        owner = self.owner()
        owner.submit_once(self.attempt, lambda: TASK)
        before = self.events()
        for statement in ("UPDATE submission_events SET recorded_unix=1005", "DELETE FROM submission_events"):
            with self.subTest(statement=statement), self.ledger._connection() as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(statement)
            self.assertEqual(self.events(), before)

    def test_insert_or_replace_cannot_change_an_existing_event(self):
        owner = self.owner()
        owner.submit_once(self.attempt, lambda: TASK)
        before = self.events()
        # A normal SQLite client does not run DELETE triggers during REPLACE.
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA recursive_triggers").fetchone()[0], 0)
            event_id = connection.execute("SELECT MIN(event_id) FROM submission_events").fetchone()[0]
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT OR REPLACE INTO submission_events "
                    "(event_id, attempt_id, event_kind, recorded_unix) VALUES (?, ?, 'CLAIMED', 1005)",
                    (event_id, self.attempt))
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM submission_events").fetchone()[0], 2)
        self.assertEqual(self.events(), before)

    def test_event_schema_rejects_invalid_kinds_times_and_evidence_shapes(self):
        self.owner()
        invalid = [
            ("REMOTE_ACCEPTED", 1000.0, None, None),
            ("CLAIMED", None, None, None),
            ("SEND_GATE_PASSED", None, None, None),
            ("CLAIMED", -1.0, None, None),
            ("CLAIMED", float("inf"), None, None),
            ("CLAIMED", 1000.0, TASK, None),
            ("SUBMISSION_UNKNOWN", 1000.0, None, EVIDENCE),
            ("TASK_RECORDED", 1000.0, None, None),
            ("TASK_RECORDED", 1000.0, TASK, EVIDENCE),
            ("TASK_RECONCILED", 1000.0, TASK, None),
        ]
        for values in invalid:
            with self.subTest(values=values), self.ledger._connection() as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT INTO submission_events "
                        "(attempt_id, event_kind, recorded_unix, provider_task_id, evidence_sha256) "
                        "VALUES (?, ?, ?, ?, ?)", (self.attempt, *values))
        for event_id in (0, -1):
            with self.subTest(event_id=event_id), self.ledger._connection() as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT INTO submission_events (event_id, attempt_id, event_kind, recorded_unix) "
                        "VALUES (?, ?, 'CLAIMED', 1000)", (event_id, self.attempt))
        self.assertEqual(self.events(), [])
