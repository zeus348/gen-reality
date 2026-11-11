"""Check durable operator authority, atomic caps, and reservation idempotency."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from core.budget import BudgetError, BudgetExceeded, BudgetLedger, IdempotencyConflict


PAYLOAD = sha256(b"frozen inputs").hexdigest()


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "budget.sqlite"
        self.now = 1000.0
        self.ledger = BudgetLedger(self.path, clock=lambda: self.now)

    def approve(self, **changes):
        settings = {"approved_by": "fixture operator", "limit_cents": 1000,
                    "max_attempts": 6, "cost_cents": 140, "max_elapsed_minutes": 180}
        settings.update(changes)
        self.ledger.approve_run("run-1", **settings)

    def test_unapproved_run_cannot_reserve(self):
        with self.assertRaisesRegex(BudgetError, "no server-side"):
            self.ledger.reserve("run-1", "request-1", PAYLOAD)

    def test_reservation_uses_approved_quote_and_persists_across_restart(self):
        self.approve()
        first = self.ledger.reserve("run-1", "request-1", PAYLOAD)
        self.assertEqual(first.cost_cents, 140)
        self.assertEqual(first.state, "RESERVED")
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.reserve("run-1", "request-1", PAYLOAD), first)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*), sum(cost_cents) FROM reservations").fetchone(),
                             (1, 140))

    def test_same_key_with_changed_payload_conflicts(self):
        self.approve()
        self.ledger.reserve("run-1", "request-1", PAYLOAD)
        with self.assertRaises(IdempotencyConflict):
            self.ledger.reserve("run-1", "request-1", sha256(b"changed").hexdigest())

    def test_twenty_concurrent_requests_reserve_only_one_available_charge(self):
        self.approve(limit_cents=140)

        def reserve(index):
            try:
                return self.ledger.reserve("run-1", f"request-{index}", PAYLOAD)
            except BudgetExceeded:
                return None

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(reserve, range(20)))
        self.assertEqual(sum(result is not None for result in results), 1)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*), sum(cost_cents) FROM reservations").fetchone(),
                             (1, 140))

    def test_money_and_attempt_caps_are_independent(self):
        for settings in [{"limit_cents": 200}, {"max_attempts": 1}]:
            with self.subTest(settings=settings):
                path = Path(self.folder.name) / ("money.sqlite" if "limit_cents" in settings else "attempts.sqlite")
                ledger = BudgetLedger(path, clock=lambda: self.now)
                approval = {"limit_cents": 1000, "max_attempts": 6, "cost_cents": 140}
                approval.update(settings)
                ledger.approve_run("run-1", approved_by="fixture operator", **approval)
                ledger.reserve("run-1", "first", PAYLOAD)
                with self.assertRaises(BudgetExceeded):
                    ledger.reserve("run-1", "second", PAYLOAD)

    def test_deadline_blocks_new_work_but_preserves_idempotent_lookup(self):
        self.approve(max_elapsed_minutes=1)
        first = self.ledger.reserve("run-1", "first", PAYLOAD)
        self.now = 1060.0
        with self.assertRaisesRegex(BudgetExceeded, "elapsed-time"):
            self.ledger.reserve("run-1", "second", PAYLOAD)
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.reserve("run-1", "first", PAYLOAD), first)

    def test_clock_anomalies_block_new_reservations(self):
        self.approve()
        self.now = 1010.0
        self.ledger.reserve("run-1", "first", PAYLOAD)
        for now in [1005.0, float("nan"), float("inf"), -1, True, 10 ** 1000]:
            with self.subTest(now=str(now)[:20]):
                self.now = now
                with self.assertRaises(BudgetError):
                    self.ledger.reserve("run-1", "second", PAYLOAD)

    def test_deadline_stop_survives_restart_and_clock_rollback(self):
        self.approve(max_elapsed_minutes=1)
        first = self.ledger.reserve("run-1", "first", PAYLOAD)
        self.now = 1060.0
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "second", PAYLOAD)
        self.now = 1050.0
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        with self.assertRaises(BudgetExceeded):
            restarted.reserve("run-1", "third", PAYLOAD)
        self.assertEqual(restarted.reserve("run-1", "first", PAYLOAD), first)

    def test_approval_limits_and_operator_identity_are_required(self):
        for changes in [{"limit_cents": 1001}, {"limit_cents": True}, {"limit_cents": 0},
                        {"max_attempts": 7}, {"cost_cents": 1001}, {"cost_cents": 1.4},
                        {"max_elapsed_minutes": 181}, {"approved_by": ""}]:
            with self.subTest(changes=changes):
                with self.assertRaises(BudgetError):
                    self.approve(**changes)

    def test_approval_cannot_be_replaced_to_increase_budget(self):
        self.approve(limit_cents=140)
        with self.assertRaisesRegex(BudgetError, "already has"):
            self.approve(limit_cents=1000)
        self.ledger.reserve("run-1", "first", PAYLOAD)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "second", PAYLOAD)

    def test_raw_keys_are_not_stored_and_bad_inputs_are_rejected(self):
        self.approve()
        secret = "dummy-secret-idempotency-key"
        self.ledger.reserve("run-1", secret, PAYLOAD)
        with sqlite3.connect(self.path) as connection:
            data = "\n".join(connection.iterdump())
        self.assertNotIn(secret, data)
        self.assertIn(sha256(secret.encode()).hexdigest(), data)
        for key, payload in [("", PAYLOAD), ("x" * 513, PAYLOAD), ("é" * 257, PAYLOAD),
                             ("\ud800", PAYLOAD), (None, PAYLOAD), ("key", "not-a-hash")]:
            with self.subTest(size=len(key) if isinstance(key, str) else None):
                with self.assertRaises(BudgetError):
                    self.ledger.reserve("run-1", key, payload)

    def test_database_parent_must_be_private_and_existing(self):
        with self.assertRaises(BudgetError):
            BudgetLedger(Path(self.folder.name) / "missing/budget.sqlite")
        public = Path(self.folder.name) / "public"
        public.mkdir(mode=0o755)
        public.chmod(0o755)
        with self.assertRaises(BudgetError):
            BudgetLedger(public / "budget.sqlite")
        link = Path(self.folder.name) / "link.sqlite"
        link.symlink_to(self.path)
        with self.assertRaises(BudgetError):
            BudgetLedger(link)
