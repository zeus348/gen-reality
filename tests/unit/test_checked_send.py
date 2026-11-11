"""Check the final SDK send gate after durable request publication."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import httpx

import test_preflight as fixtures
from core.budget import BudgetExceeded, BudgetLedger
from features.inference import frozen as frozen_module
from features.inference.preflight import PreflightError
from features.inference.send import submit_checked
from features.inference.submission import SubmissionCoordinator, SubmissionError


class CheckedSendTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PreflightTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.checked = self.fixture.check()
        self.now = 1001.0
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: self.now)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                               max_attempts=6, cost_cents=140, max_elapsed_minutes=1)
        self.ledger.reserve("run-1", "first", self.checked.frozen.payload_sha256, attempt_id=fixtures.ATTEMPT)
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.requests = []

    def send(self, checked=None, key=fixtures.KEY):
        def create(request):
            self.requests.append(request)
            self.assertEqual(self.owner.get(fixtures.ATTEMPT).state, "SUBMITTING")
            self.assertTrue((self.fixture.run / f"attempts/{fixtures.ATTEMPT}/provider/request.json").is_file())
            return httpx.Response(200, json={"id": "fixture-provider-task"})
        return submit_checked(self.owner, checked or self.checked, self.root, key,
                               transport=httpx.MockTransport(create))

    def after_publication(self, action):
        original = frozen_module.publish_bytes

        def publish(root, path, contents):
            result = original(root, path, contents)
            if path.endswith("/request.json"):
                action()
            return result
        return patch.object(frozen_module, "publish_bytes", side_effect=publish)

    def assert_unknown_without_post(self):
        self.assertEqual(self.owner.get(fixtures.ATTEMPT).state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.requests, [])
        with self.ledger._connection() as connection:
            row = connection.execute("SELECT cost_cents FROM reservations WHERE attempt_id=?", (fixtures.ATTEMPT,)).fetchone()
            self.assertEqual(row["cost_cents"], 140)

    def test_final_gate_runs_after_publication_and_duplicate_call_does_not_send(self):
        original = self.owner.require_send_allowed

        def gate(attempt, **arguments):
            folder = self.fixture.run / f"attempts/{attempt}/provider"
            self.assertTrue((folder / "request.json").is_file())
            self.assertTrue((folder / "frozen_request.json").is_file())
            original(attempt, **arguments)

        with patch.object(self.owner, "require_send_allowed", side_effect=gate) as check:
            result = self.send()
            self.assertEqual(result.state, "QUEUED")
            self.now = 2000.0
            self.assertEqual(self.send(key=""), result)
        check.assert_called_once_with(fixtures.ATTEMPT, check_current=self.checked.require_current)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].content, self.checked.frozen.request_body)

    def test_stale_check_is_rejected_before_claim_or_evidence_publication(self):
        self.now = 1060.0
        with self.assertRaises(PreflightError):
            self.send()
        self.assertEqual(self.owner.get(fixtures.ATTEMPT).state, "RESERVED")
        self.assertFalse((self.fixture.run / f"attempts/{fixtures.ATTEMPT}/provider").exists())
        self.assertEqual(self.requests, [])

    def test_evidence_expiring_during_publication_prevents_post_and_replay(self):
        with self.after_publication(lambda: setattr(self, "now", 1060.0)):
            result = self.send()
        self.assert_unknown_without_post()
        self.assertEqual(self.send(key=""), result)
        self.assertEqual(self.requests, [])

    def test_check_record_changed_during_publication_prevents_post(self):
        path = self.fixture.run / self.checked.evidence.path
        with self.after_publication(lambda: path.write_bytes(b"damaged fixture evidence")):
            self.send()
        self.assert_unknown_without_post()
        self.assertEqual(path.read_bytes(), b"damaged fixture evidence")

    def test_in_memory_check_cannot_extend_its_recorded_lifetime(self):
        forged = replace(self.checked, valid_until_unix=999999.0)
        with self.assertRaises(PreflightError):
            self.send(checked=forged)
        self.assertEqual(self.owner.get(fixtures.ATTEMPT).state, "RESERVED")
        self.assertEqual(self.requests, [])

    def test_run_deadline_reached_during_publication_stays_stopped(self):
        self.checked = self.fixture.check(now=1006.0, account=self.fixture.make_account(now=1005.0))
        self.now = 1006.0
        with self.after_publication(lambda: setattr(self, "now", 1061.0)):
            self.send()
        self.assert_unknown_without_post()
        self.owner.close()
        self.now = 1006.0
        restarted = BudgetLedger(self.ledger.path, clock=lambda: self.now)
        with self.assertRaises(BudgetExceeded):
            restarted.reserve("run-1", "late", "b" * 64)

    def test_charge_over_quote_before_send_blocks_new_spending(self):
        other = self.ledger.reserve("run-1", "other", "b" * 64)
        with self.after_publication(lambda: self.ledger.record_final_charge(other.attempt_id, 141, "c" * 64)):
            self.send()
        self.assert_unknown_without_post()

    def test_settled_reservation_cannot_send_after_publication(self):
        with self.after_publication(lambda: self.ledger.record_final_charge(fixtures.ATTEMPT, 0, "c" * 64)):
            self.send()
        self.assert_unknown_without_post()

    def test_budget_clock_rollback_before_send_is_rejected(self):
        self.now = 1002.0
        with self.after_publication(lambda: setattr(self, "now", 1001.5)):
            self.send()
        self.assert_unknown_without_post()

    def test_wrong_run_or_attempt_cannot_use_a_reservation(self):
        for changed in [replace(self.checked, run_id="other"),
                        replace(self.checked, attempt_id="att-" + "b" * 32)]:
            with self.subTest(checked=changed), self.assertRaises(SubmissionError):
                self.send(checked=changed)
        self.assertEqual(self.owner.get(fixtures.ATTEMPT).state, "RESERVED")
        self.assertEqual(self.requests, [])

    def test_final_authority_check_requires_the_claimed_state(self):
        with self.assertRaises(SubmissionError):
            self.owner.require_send_allowed(fixtures.ATTEMPT)
        self.send()
        with self.assertRaises(SubmissionError):
            self.owner.require_send_allowed(fixtures.ATTEMPT)

    def test_final_timestamp_cannot_cross_the_run_deadline(self):
        self.checked = self.fixture.check(now=1006.0, account=self.fixture.make_account(now=1005.0))

        def advance_clock():
            times = iter([1060.5, 1061.5])
            self.ledger.clock = lambda: next(times, 1061.5)

        self.now = 1006.0
        with self.after_publication(advance_clock):
            self.send()
        self.assert_unknown_without_post()
        with self.ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT time_stopped FROM approved_runs").fetchone()[0], 1)

    def test_final_timestamp_cannot_roll_back_from_the_durable_high_water_mark(self):
        def roll_back_clock():
            times = iter([1010.0, 1009.0])
            self.ledger.clock = lambda: next(times, 1009.0)

        with self.after_publication(roll_back_clock):
            self.send()
        self.assert_unknown_without_post()
        with self.ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT last_checked_unix FROM approved_runs").fetchone()[0], 1010.0)
