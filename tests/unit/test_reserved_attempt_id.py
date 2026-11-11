"""Keep prepared attempt identity through reservation, submission, and billing."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from core.budget import BudgetError, BudgetExceeded, BudgetLedger, IdempotencyConflict
from features.inference.submission import SubmissionCoordinator


ATTEMPT_ID = "att-" + "a" * 32
OTHER_ID = "att-" + "b" * 32
PAYLOAD = "1" * 64


class ReservedAttemptIdTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "budget.sqlite"
        self.now = 1000.0
        self.ledger = BudgetLedger(self.path, clock=lambda: self.now)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                               max_attempts=6, cost_cents=140, max_elapsed_minutes=1)

    def reserve(self, key="first", attempt_id=ATTEMPT_ID):
        return self.ledger.reserve("run-1", key, PAYLOAD, attempt_id=attempt_id)

    def test_prepared_id_survives_submission_billing_and_restart(self):
        first = self.reserve()
        self.assertEqual(first.attempt_id, ATTEMPT_ID)
        with SubmissionCoordinator(self.ledger) as owner:
            create = Mock(return_value="fixture-task")
            result = owner.submit_once(ATTEMPT_ID, create)
            self.assertEqual((result.attempt_id, result.state), (ATTEMPT_ID, "QUEUED"))
            create.assert_called_once_with()
        charge = self.ledger.record_final_charge(ATTEMPT_ID, 140, "2" * 64)
        self.assertEqual(charge.attempt_id, ATTEMPT_ID)
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.reserve("run-1", "first", PAYLOAD, attempt_id=ATTEMPT_ID), first)
        self.assertEqual(restarted.reserve("run-1", "first", PAYLOAD), first)
        with restarted._connection() as connection:
            self.assertEqual(restarted._usage(connection, "run-1"), (1, 140, False))

    def test_same_key_cannot_switch_attempt_id(self):
        self.reserve()
        with self.assertRaises(IdempotencyConflict):
            self.reserve(attempt_id=OTHER_ID)
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (1, 140, False))

    def test_attempt_id_cannot_be_claimed_by_another_key_or_run(self):
        self.reserve()
        self.ledger.approve_run("run-2", approved_by="fixture operator", limit_cents=1000,
                               max_attempts=6, cost_cents=140)
        with self.assertRaises(IdempotencyConflict):
            self.reserve(key="second")
        with self.assertRaises(IdempotencyConflict):
            self.ledger.reserve("run-2", "first", PAYLOAD, attempt_id=ATTEMPT_ID)
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (1, 140, False))
            self.assertEqual(self.ledger._usage(connection, "run-2"), (0, 0, False))
        self.assertEqual(self.reserve(key="second", attempt_id=OTHER_ID).attempt_id, OTHER_ID)

    def test_invalid_prepared_ids_do_not_consume_money_or_attempts(self):
        for attempt in [True, 12, "", "att-short", "att-" + "A" * 32, "att-" + "a" * 33, "../attempt"]:
            with self.subTest(attempt=attempt), self.assertRaises(BudgetError):
                self.reserve(attempt_id=attempt)
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (0, 0, False))

    def test_concurrent_keys_cannot_share_a_prepared_attempt(self):
        def reserve(index):
            try:
                return self.reserve(key=f"key-{index}")
            except (IdempotencyConflict, BudgetExceeded):
                return None

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(reserve, range(20)))
        successes = [result for result in results if result is not None]
        self.assertEqual(len(successes), 1)
        self.assertEqual(successes[0].attempt_id, ATTEMPT_ID)
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (1, 140, False))

    def test_prepared_id_keeps_deadline_and_cap_checks(self):
        first = self.reserve()
        self.now = 1060.0
        with self.assertRaises(BudgetExceeded):
            self.reserve(key="late", attempt_id=OTHER_ID)
        self.assertEqual(self.reserve(), first)
        self.now = 1000.0
        self.ledger.approve_run("capped", approved_by="fixture operator", limit_cents=140,
                               max_attempts=1, cost_cents=140)
        self.ledger.reserve("capped", "first", PAYLOAD, attempt_id=OTHER_ID)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("capped", "second", PAYLOAD, attempt_id="att-" + "c" * 32)
