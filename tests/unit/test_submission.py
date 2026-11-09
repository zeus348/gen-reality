"""Check the durable boundary before each paid create call."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock

from core.budget import BudgetError, BudgetExceeded, BudgetLedger
from features.inference.submission import SubmissionCoordinator, SubmissionError


PAYLOAD = sha256(b"frozen inputs").hexdigest()
EVIDENCE = sha256(b"verified provider task record").hexdigest()
TASK_ID = "c26d7283-d514-4492-8589-760dfe6f0289"


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "budget.sqlite"
        self.now = 1000.0
        self.ledger = BudgetLedger(self.path, clock=lambda: self.now)

    def reserve(self, limit=1000):
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=limit,
                                max_attempts=6, cost_cents=140, max_elapsed_minutes=1)
        return self.ledger.reserve("run-1", "first", PAYLOAD).attempt_id

    def coordinator(self):
        owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(owner.close)
        return owner

    def test_submitting_is_committed_before_the_create_call(self):
        attempt = self.reserve()
        owner = self.coordinator()
        self.assertEqual(owner.get(attempt).state, "RESERVED")

        def create():
            self.assertEqual(owner.get(attempt).state, "SUBMITTING")
            # A second writer can start: no database transaction spans the call.
            with sqlite3.connect(self.path, timeout=0.1) as connection:
                connection.execute("BEGIN IMMEDIATE")
                state = connection.execute("SELECT state FROM submission_jobs").fetchone()[0]
                self.assertEqual(state, "SUBMITTING")
                connection.rollback()
            return TASK_ID

        result = owner.submit_once(attempt, create)
        self.assertEqual((result.state, result.provider_task_id), ("QUEUED", TASK_ID))

    def test_concurrent_duplicates_and_restart_make_one_create_call(self):
        attempt = self.reserve()
        owner = self.coordinator()
        create = Mock(return_value=TASK_ID)
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(lambda _: owner.submit_once(attempt, create), range(20)))
        create.assert_called_once_with()
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(results[0].state, "QUEUED")
        owner.close()
        restarted = self.coordinator()
        self.assertEqual(restarted.submit_once(attempt, create), results[0])
        create.assert_called_once_with()

    def test_timeout_keeps_money_and_prevents_replay(self):
        attempt = self.reserve(limit=140)
        owner = self.coordinator()
        create = Mock(side_effect=TimeoutError("private credential must not be stored"))
        result = owner.submit_once(attempt, create)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertIsNone(result.provider_task_id)
        self.assertEqual(owner.submit_once(attempt, create), result)
        create.assert_called_once_with()
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "next", PAYLOAD)
        with sqlite3.connect(self.path) as connection:
            self.assertNotIn("private credential", "\n".join(connection.iterdump()))

    def test_malformed_provider_ids_are_unknown(self):
        self.reserve()
        owner = self.coordinator()
        for index, invalid in enumerate([None, "", "with spaces", "a" * 129, "../task"]):
            with self.subTest(task_id=invalid):
                attempt = self.ledger.reserve("run-1", f"invalid-{index}", PAYLOAD).attempt_id
                create = Mock(return_value=invalid)
                result = owner.submit_once(attempt, create)
                self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
                self.assertIsNone(result.provider_task_id)
                owner.submit_once(attempt, create)
                create.assert_called_once_with()

    def test_process_exit_recovers_unknown_without_replay(self):
        attempt = self.reserve(limit=140)
        code = """
import os
from pathlib import Path
import sys
from core.budget import BudgetLedger
from features.inference.submission import SubmissionCoordinator
ledger = BudgetLedger(Path(sys.argv[1]), clock=lambda: 1000.0)
with SubmissionCoordinator(ledger) as owner:
    owner.submit_once(sys.argv[2], lambda: os._exit(17))
"""
        child = subprocess.run([sys.executable, "-c", code, str(self.path), attempt],
                               capture_output=True, timeout=10)
        self.assertEqual(child.returncode, 17, child.stderr.decode())
        owner = self.coordinator()
        create = Mock(return_value=TASK_ID)
        self.assertEqual(owner.submit_once(attempt, create).state, "SUBMISSION_UNKNOWN")
        create.assert_not_called()
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "next", PAYLOAD)

    def test_a_second_owner_cannot_recover_an_active_call(self):
        attempt = self.reserve()
        owner = self.coordinator()
        entered, finish = Event(), Event()

        def create():
            entered.set()
            if not finish.wait(5):
                raise TimeoutError("Test did not release the create call.")
            return TASK_ID

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(owner.submit_once, attempt, create)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(SubmissionError, "already owns"):
                    self.coordinator()
                self.assertEqual(owner.get(attempt).state, "SUBMITTING")
            finally:
                finish.set()
            self.assertEqual(pending.result(timeout=5).state, "QUEUED")

    def test_expired_approval_stays_stopped_after_clock_rollback(self):
        attempt = self.reserve()
        owner = self.coordinator()
        create = Mock(return_value=TASK_ID)
        self.now = 1060.0
        with self.assertRaises(BudgetExceeded):
            owner.submit_once(attempt, create)
        owner.close()
        self.now = 1050.0
        restarted = self.coordinator()
        with self.assertRaises(BudgetExceeded):
            restarted.submit_once(attempt, create)
        create.assert_not_called()
        self.assertEqual(restarted.get(attempt).state, "RESERVED")

    def test_clock_rollback_cannot_start_a_submission(self):
        attempt = self.reserve()
        owner = self.coordinator()
        self.now = 999.0
        create = Mock(return_value=TASK_ID)
        with self.assertRaisesRegex(BudgetError, "clock moved backwards"):
            owner.submit_once(attempt, create)
        create.assert_not_called()

    def test_charge_above_quote_blocks_an_existing_reservation(self):
        first = self.reserve()
        second = self.ledger.reserve("run-1", "second", PAYLOAD).attempt_id
        self.ledger.record_final_charge(first, 141, EVIDENCE)
        owner = self.coordinator()
        create = Mock(return_value=TASK_ID)
        with self.assertRaises(BudgetExceeded):
            owner.submit_once(second, create)
        create.assert_not_called()

    def test_settled_reservation_cannot_start_a_create_call(self):
        attempt = self.reserve()
        self.ledger.record_final_charge(attempt, 0, EVIDENCE)
        owner = self.coordinator()
        create = Mock(return_value=TASK_ID)
        with self.assertRaisesRegex(SubmissionError, "settled reservation"):
            owner.submit_once(attempt, create)
        create.assert_not_called()

    def test_unknown_task_requires_evidence_for_reconciliation(self):
        attempt = self.reserve()
        owner = self.coordinator()
        owner.submit_once(attempt, Mock(side_effect=TimeoutError))
        for task_id, evidence in [(TASK_ID, "bad"), ("../task", EVIDENCE)]:
            with self.subTest(task_id=task_id, evidence=evidence):
                with self.assertRaises(SubmissionError):
                    owner.record_verified_task(attempt, task_id, evidence)
                self.assertEqual(owner.get(attempt).state, "SUBMISSION_UNKNOWN")
        reconciled = owner.record_verified_task(attempt, TASK_ID, EVIDENCE)
        self.assertEqual((reconciled.state, reconciled.provider_task_id, reconciled.evidence_sha256),
                         ("QUEUED", TASK_ID, EVIDENCE))
        owner.close()
        restarted = self.coordinator()
        create = Mock()
        self.assertEqual(restarted.submit_once(attempt, create), reconciled)
        create.assert_not_called()

    def test_reconciliation_cannot_replace_a_known_task(self):
        attempt = self.reserve()
        owner = self.coordinator()
        for state in ("RESERVED", "QUEUED"):
            with self.subTest(state=state):
                self.assertEqual(owner.get(attempt).state, state)
                with self.assertRaisesRegex(SubmissionError, "Only an unknown"):
                    owner.record_verified_task(attempt, "another-task", EVIDENCE)
            if state == "RESERVED":
                owner.submit_once(attempt, Mock(return_value=TASK_ID))
        self.assertEqual(owner.get(attempt).provider_task_id, TASK_ID)

    def test_duplicate_provider_id_keeps_second_submission_unknown(self):
        first = self.reserve()
        second = self.ledger.reserve("run-1", "second", PAYLOAD).attempt_id
        owner = self.coordinator()
        create = Mock(return_value=TASK_ID)
        self.assertEqual(owner.submit_once(first, create).state, "QUEUED")
        self.assertEqual(owner.submit_once(second, create).state, "SUBMISSION_UNKNOWN")
        self.assertIsNone(owner.get(second).provider_task_id)
        self.assertEqual(owner.get(first).provider_task_id, TASK_ID)
        owner.submit_once(second, create)
        self.assertEqual(create.call_count, 2)

    def test_closed_coordinator_cannot_start_a_create_call(self):
        attempt = self.reserve()
        owner = self.coordinator()
        owner.close()
        create = Mock(return_value=TASK_ID)
        with self.assertRaisesRegex(SubmissionError, "closed"):
            owner.submit_once(attempt, create)
        create.assert_not_called()

    def test_missing_reservation_cannot_start_a_create_call(self):
        owner = self.coordinator()
        create = Mock(return_value=TASK_ID)
        attempt = "att-" + "f" * 32
        with self.assertRaisesRegex(SubmissionError, "no matching reservation"):
            owner.submit_once(attempt, create)
        with self.assertRaisesRegex(SubmissionError, "no matching reservation"):
            owner.get(attempt)
        create.assert_not_called()
