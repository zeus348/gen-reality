"""Exercise saved-request submission and restart behavior without live calls."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import httpx

from app import submit as submit_module
from core.budget import BudgetError, BudgetLedger, IdempotencyConflict
from core.storage import publish_bytes
from features.inference.preflight import PreflightError, read_checked_request
import test_preflight as fixtures


class SubmitCommandTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PreflightTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.checked = self.fixture.check()
        self.now = 1001.0
        self.ledger_path = self.root / "budget.sqlite"
        self.ledger = BudgetLedger(self.ledger_path, clock=lambda: self.now)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                               max_attempts=6, cost_cents=140,
                               deadline_unix_cap=self.fixture.started.deadline_unix)
        self.requests = []
        self.uncertain = False
        self.response_task_id = "fixture-task"
        self.addCleanup(patch.stopall)
        patch.dict("os.environ", {"TEST_RUNWAY_KEY": fixtures.KEY}).start()

    def provider(self, request):
        self.requests.append(request)
        if self.uncertain:
            raise httpx.ReadTimeout("fixture failure with a secret: " + fixtures.KEY)
        return httpx.Response(200, json={"id": self.response_task_id})

    def send(self, evidence=None):
        return submit_module.submit_saved_request(self.root, self.ledger_path,
            evidence or self.checked.evidence, run_id="run-1", attempt_id=fixtures.ATTEMPT,
            api_key_env="TEST_RUNWAY_KEY", clock=lambda: self.now,
            transport=httpx.MockTransport(self.provider))

    def reservations(self):
        with self.ledger._connection() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM reservations")]

    def changed_record(self, changes):
        data = json.loads((self.fixture.run / self.checked.evidence.path).read_bytes())
        data.update(changes)
        body = json.dumps(data, sort_keys=True).encode()
        return publish_bytes(self.fixture.run,
            f"attempts/{fixtures.ATTEMPT}/provider_checks/{sha256(body).hexdigest()}.json", body)

    def arguments(self):
        record = self.checked.evidence
        return ["--output-root", str(self.root), "--ledger", str(self.ledger_path),
                "--run-id", "run-1", "--attempt-id", fixtures.ATTEMPT,
                "--check-path", record.path, "--check-sha256", record.sha256,
                "--check-size", str(record.size), "--api-key-env", "TEST_RUNWAY_KEY"]

    def test_restores_exact_request_and_repeated_command_needs_no_key(self):
        restored = read_checked_request(self.fixture.run, self.checked.evidence,
                                       run_id="run-1", attempt_id=fixtures.ATTEMPT)
        self.assertEqual(restored, self.checked)
        first = self.send()
        self.assertEqual(first.state, "QUEUED")
        self.assertEqual(self.requests[0].content, self.checked.frozen.request_body)
        self.now = 2000.0
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.send(), first)
        self.assertEqual(len(self.requests), 1)
        rows = self.reservations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempt_id"], fixtures.ATTEMPT)
        key = f"saved-request:run-1:{fixtures.ATTEMPT}"
        self.assertEqual(rows[0]["key_sha256"], sha256(key.encode()).hexdigest())

    def test_unknown_submission_survives_restart_without_another_post(self):
        self.uncertain = True
        first = self.send()
        self.assertEqual(first.state, "SUBMISSION_UNKNOWN")
        self.now = 2000.0
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.send(), first)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reservations()), 1)

    def test_existing_reservation_keeps_its_original_idempotency_key(self):
        self.ledger.reserve("run-1", "earlier operator key", self.checked.frozen.payload_sha256,
                            attempt_id=fixtures.ATTEMPT)
        before = self.reservations()
        self.assertEqual(self.send().state, "QUEUED")
        self.assertEqual(self.reservations(), before)
        self.assertEqual(len(self.requests), 1)

    def test_damaged_or_mismatched_file_reference_cannot_reserve(self):
        record = self.checked.evidence
        invalid = [replace(record, size=record.size + 1), replace(record, sha256="0" * 64),
                   replace(record, path=record.path.replace(fixtures.ATTEMPT, "att-" + "b" * 32))]
        for changed in invalid:
            with self.subTest(record=changed), self.assertRaises(PreflightError):
                self.send(changed)
        (self.fixture.run / record.path).write_bytes(b"damaged saved request")
        with self.assertRaises(PreflightError):
            self.send()
        self.assertEqual(self.reservations(), [])
        self.assertEqual(self.requests, [])

    def test_saved_record_rejects_wrong_identity_payload_and_times(self):
        changes = [{"schemaVersion": "other"}, {"runId": "other"},
                   {"attemptId": "att-" + "b" * 32}, {"spendingAuthorized": True},
                   {"payloadSha256": "0" * 64}, {"sourceKind": "unknown"},
                   {"checkedAtUnix": True}, {"validUntilUnix": 1001.0},
                   {"validUntilUnix": float("inf")}, {"frozen": {"mode": "A"}}]
        for fields in changes:
            record = self.changed_record(fields)
            with self.subTest(fields=fields), self.assertRaises(PreflightError):
                self.send(record)
        self.assertEqual(self.reservations(), [])
        self.assertEqual(self.requests, [])

    def test_missing_database_approval_or_matching_quote_cannot_reserve(self):
        for setup in ("missing_database", "missing_approval", "different_quote"):
            with self.subTest(setup=setup):
                self.ledger_path = self.root / (setup + ".sqlite")
                if setup != "missing_database":
                    other = BudgetLedger(self.ledger_path, clock=lambda: self.now)
                    if setup == "different_quote":
                        other.approve_run("run-1", approved_by="fixture", limit_cents=1000,
                                          max_attempts=6, cost_cents=141)
                with self.assertRaises(BudgetError):
                    self.send()
                if setup == "missing_database":
                    self.assertFalse(self.ledger_path.exists())
                else:
                    with other._connection() as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 0)
        self.assertEqual(self.requests, [])

    def test_conflicting_existing_payload_cannot_send(self):
        self.ledger.reserve("run-1", "other payload", "b" * 64, attempt_id=fixtures.ATTEMPT)
        before = self.reservations()
        with self.assertRaises(IdempotencyConflict):
            self.send()
        self.assertEqual(self.reservations(), before)
        self.assertEqual(self.requests, [])

    def test_stale_check_does_not_reserve_or_claim_existing_work(self):
        self.now = 1060.0
        with self.assertRaises(PreflightError):
            self.send()
        self.assertEqual(self.reservations(), [])
        self.ledger.reserve("run-1", "earlier key", self.checked.frozen.payload_sha256,
                            attempt_id=fixtures.ATTEMPT)
        before = self.reservations()
        with self.assertRaises(PreflightError):
            self.send()
        self.assertEqual(self.reservations(), before)
        with self.ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM submission_jobs").fetchone()[0], 0)
        self.assertEqual(self.requests, [])

    def test_missing_key_does_not_create_a_reservation(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            self.send()
        self.assertEqual(self.reservations(), [])
        self.assertEqual(self.requests, [])

    def test_cli_outputs_safe_json_and_reports_uncertainty(self):
        self.uncertain = True
        original = submit_module.submit_saved_request

        def call(*args, **kwargs):
            return original(*args, **kwargs, clock=lambda: self.now,
                            transport=httpx.MockTransport(self.provider))

        output, errors = io.StringIO(), io.StringIO()
        with patch.object(submit_module, "submit_saved_request", side_effect=call), \
                redirect_stdout(output), redirect_stderr(errors):
            status = submit_module.main(self.arguments())
        self.assertEqual(status, 2)
        response = json.loads(output.getvalue())
        runtime_summary = response.pop("runtime")
        self.assertEqual(response, {
            "attempt_id": fixtures.ATTEMPT, "state": "SUBMISSION_UNKNOWN",
            "provider_task_id": None, "evidence_sha256": None})
        self.assertEqual(runtime_summary["issues"], [])
        self.assertEqual([record["path"] for record in runtime_summary["receipts"]], [
            f"attempts/{fixtures.ATTEMPT}/runtime/create-context.json",
            f"attempts/{fixtures.ATTEMPT}/runtime/create-result.json"])
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn(fixtures.KEY, output.getvalue())
        self.assertEqual(len(self.requests), 1)

    def test_cli_does_not_print_sensitive_exception_text(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(submit_module, "submit_saved_request", side_effect=ValueError(fixtures.KEY)), \
                redirect_stdout(output), redirect_stderr(errors):
            status = submit_module.main(self.arguments())
        self.assertEqual(status, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn(fixtures.KEY, errors.getvalue())
        self.assertIn("Submission could not proceed.", errors.getvalue())

    def test_cli_rejects_a_credential_in_the_task_id_before_saving_or_printing(self):
        self.response_task_id = "task-" + fixtures.KEY
        original = submit_module.submit_saved_request

        def call(*args, **kwargs):
            return original(*args, **kwargs, clock=lambda: self.now,
                            transport=httpx.MockTransport(self.provider))

        output, errors = io.StringIO(), io.StringIO()
        with patch.object(submit_module, "submit_saved_request", side_effect=call), \
                redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(submit_module.main(self.arguments()), 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["state"], "SUBMISSION_UNKNOWN")
        self.assertIsNone(result["provider_task_id"])
        self.assertNotIn(fixtures.KEY, output.getvalue() + errors.getvalue())
        with self.ledger._connection() as connection:
            self.assertNotIn(fixtures.KEY, "\n".join(connection.iterdump()))
        self.now = 2000.0
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.send().state, "SUBMISSION_UNKNOWN")
        self.assertEqual(len(self.requests), 1)

    def test_module_command_restores_uncertain_work_without_credentials(self):
        self.uncertain = True
        self.assertEqual(self.send().state, "SUBMISSION_UNKNOWN")
        before = self.reservations()
        environment = dict(os.environ)
        environment.pop("TEST_RUNWAY_KEY", None)
        result = subprocess.run([sys.executable, "-m", "app.submit", *self.arguments()],
            cwd=Path(__file__).resolve().parents[2], env=environment,
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["state"], "SUBMISSION_UNKNOWN")
        self.assertEqual(result.stderr, "")
        self.assertNotIn(fixtures.KEY, result.stdout)
        self.assertEqual(self.reservations(), before)
        self.assertEqual(len(self.requests), 1)
