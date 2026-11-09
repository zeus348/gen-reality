"""Check final billing without dropping unknown costs or changing authority."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from core.budget import BudgetError, BudgetExceeded, BudgetLedger


PAYLOAD = sha256(b"frozen inputs").hexdigest()
RECEIPT = sha256(b"verified fixture receipt").hexdigest()


class ChargeTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "budget.sqlite"
        self.now = 1000.0
        self.ledger = BudgetLedger(self.path, clock=lambda: self.now)

    def approve(self, limit=240, attempts=6):
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=limit,
                                max_attempts=attempts, cost_cents=140, max_elapsed_minutes=1)
        return self.ledger.reserve("run-1", "first", PAYLOAD)

    def test_final_charge_replaces_reservation_instead_of_adding_to_it(self):
        first = self.approve()
        self.ledger.record_final_charge(first.attempt_id, 100, RECEIPT)
        second = self.ledger.reserve("run-1", "second", PAYLOAD)
        self.assertEqual(second.cost_cents, 140)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "third", PAYLOAD)

    def test_unknown_charge_keeps_its_reservation(self):
        first = self.approve()
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        with self.assertRaises(BudgetExceeded):
            restarted.reserve("run-1", "second", PAYLOAD)
        self.assertEqual(restarted.reserve("run-1", "first", PAYLOAD), first)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM actual_charges").fetchone()[0], 0)

    def test_confirmed_zero_releases_money_but_not_attempts(self):
        first = self.approve(limit=140, attempts=2)
        self.ledger.record_final_charge(first.attempt_id, 0, RECEIPT)
        second = self.ledger.reserve("run-1", "second", PAYLOAD)
        self.ledger.record_final_charge(second.attempt_id, 0, RECEIPT)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "third", PAYLOAD)

    def test_over_budget_billing_is_retained_and_stops_new_reservations(self):
        first = self.approve(limit=1000)
        charge = self.ledger.record_final_charge(first.attempt_id, 1500, RECEIPT)
        self.assertEqual(charge.actual_cents, 1500)
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.record_final_charge(first.attempt_id, 1500, RECEIPT), charge)
        with self.assertRaises(BudgetExceeded):
            restarted.reserve("run-1", "second", PAYLOAD)

    def test_large_final_charges_do_not_overflow_the_budget_total(self):
        first = self.approve(limit=1000)
        second = self.ledger.reserve("run-1", "second", PAYLOAD)
        for reservation in (first, second):
            self.ledger.record_final_charge(reservation.attempt_id, 2 ** 63 - 1, RECEIPT)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "third", PAYLOAD)

    def test_charge_above_quote_blocks_work_even_with_money_remaining(self):
        first = self.approve(limit=1000)
        self.ledger.record_final_charge(first.attempt_id, 141, RECEIPT)
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        with self.assertRaisesRegex(BudgetExceeded, "exceeded its quote"):
            restarted.reserve("run-1", "second", PAYLOAD)

    def test_charge_is_recorded_after_deadline_without_reopening_the_run(self):
        first = self.approve(limit=140)
        self.now = 1060.0
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "second", PAYLOAD)
        charge = self.ledger.record_final_charge(first.attempt_id, 0, RECEIPT)
        self.assertEqual(charge.actual_cents, 0)
        self.now = 1050.0
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        with self.assertRaises(BudgetExceeded):
            restarted.reserve("run-1", "third", PAYLOAD)

    def test_final_charge_and_receipt_are_immutable_and_survive_restart(self):
        first = self.approve()
        charge = self.ledger.record_final_charge(first.attempt_id, 100, RECEIPT)
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.record_final_charge(first.attempt_id, 100, RECEIPT), charge)
        for cents, receipt in [(101, RECEIPT), (100, "0" * 64)]:
            with self.subTest(cents=cents, receipt=receipt):
                with self.assertRaisesRegex(BudgetError, "cannot be replaced"):
                    restarted.record_final_charge(first.attempt_id, cents, receipt)
        self.assertEqual(restarted.record_final_charge(first.attempt_id, 100, RECEIPT), charge)

    def test_invalid_billing_cannot_create_or_change_a_charge(self):
        first = self.approve()
        for cents, receipt in [(-1, RECEIPT), (True, RECEIPT), (1.4, RECEIPT),
                               (2 ** 63, RECEIPT), (0, "invalid")]:
            with self.subTest(cents=cents, receipt=receipt):
                with self.assertRaises(BudgetError):
                    self.ledger.record_final_charge(first.attempt_id, cents, receipt)
        with self.assertRaisesRegex(BudgetError, "no matching reservation"):
            self.ledger.record_final_charge("att-" + "f" * 32, 0, RECEIPT)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM actual_charges").fetchone()[0], 0)

    def test_existing_reservation_database_keeps_its_held_cost(self):
        first = self.approve()
        # Removing this new table restores the exact earlier reservation schema.
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE actual_charges")
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.reserve("run-1", "first", PAYLOAD), first)
        with self.assertRaises(BudgetExceeded):
            restarted.reserve("run-1", "second", PAYLOAD)
        restarted.record_final_charge(first.attempt_id, 100, RECEIPT)
        restarted.reserve("run-1", "second", PAYLOAD)

    def test_concurrent_reconciliation_cannot_double_spend_released_money(self):
        first = self.approve()

        def settle_and_reserve(index):
            self.ledger.record_final_charge(first.attempt_id, 100, RECEIPT)
            try:
                return self.ledger.reserve("run-1", f"new-{index}", PAYLOAD)
            except BudgetExceeded:
                return None

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(settle_and_reserve, range(20)))
        self.assertEqual(sum(result is not None for result in results), 1)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM actual_charges").fetchone()[0], 1)
