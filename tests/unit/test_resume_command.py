"""Run the known-task recovery path with fake HTTP and real video validation."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
from runwayml._utils import _logs as sdk_logs

from app import resume as resume_module
from core.budget import BudgetError, BudgetLedger
from core.storage import StorageError, read_verified
from features.inference.polling import TaskObservation, TaskPoller
from features.inference.submission import SubmissionCoordinator, SubmissionError
from features.output.complete import OutputError
import test_output_completion as video_fixtures


KEY = "fixture-resume-secret"
HOST = "outputs.example.net"
TASK = "fixture-provider-task"


class ResumeCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        video_fixtures.OutputCompletionTests.setUpClass()
        cls.video = video_fixtures.OutputCompletionTests.contents

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.output = self.root / "runs"
        self.now = 1000.0
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: self.now)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                               max_attempts=6, cost_cents=140, max_elapsed_minutes=1)
        self.attempt = self.ledger.reserve("run-1", "first", "a" * 64).attempt_id
        with SubmissionCoordinator(self.ledger) as owner:
            owner.submit_once(self.attempt, lambda: TASK)
        self.requests = []
        self.task_timeout = False
        self.download_status = 200
        self.output_url = "https://" + HOST + "/first.mp4?signature=fixture"
        self.addCleanup(patch.stopall)
        patch.dict("os.environ", {"TEST_RESUME_KEY": KEY}).start()

    def task_response(self, request):
        self.requests.append(request)
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.url.path, "/v1/tasks/" + TASK)
        self.assertEqual(request.headers["Authorization"], "Bearer " + KEY)
        if self.task_timeout:
            raise httpx.ReadTimeout("fixture timeout with " + KEY)
        return httpx.Response(200, json={"id": TASK, "status": "SUCCEEDED", "output": [self.output_url]})

    def output_response(self, request):
        self.requests.append(request)
        self.assertEqual(request.method, "GET")
        self.assertNotIn("Authorization", request.headers)
        self.assertNotIn("Cookie", request.headers)
        return httpx.Response(self.download_status, content=self.video)

    def resume(self, **changes):
        arguments = {"run_id": "run-1", "attempt_id": self.attempt, "output_hosts": frozenset({HOST}),
                     "api_key_env": "TEST_RESUME_KEY", "clock": lambda: self.now,
                     "task_transport": httpx.MockTransport(self.task_response),
                     "output_transport": httpx.MockTransport(self.output_response)}
        arguments.update(changes)
        return resume_module.resume_attempt(self.output, self.ledger.path, **arguments)

    def reservations(self):
        with self.ledger._connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM actual_charges").fetchone()[0], 0)
            return [dict(row) for row in connection.execute("SELECT * FROM reservations")]

    def arguments(self):
        return ["--output-root", str(self.output), "--ledger", str(self.ledger.path),
                "--run-id", "run-1", "--attempt-id", self.attempt, "--api-key-env", "TEST_RESUME_KEY"]

    def test_known_task_gets_validated_output_and_restarts_without_network(self):
        before = self.reservations()
        result = self.resume()
        self.assertEqual(result.state, "SUCCEEDED")
        self.assertEqual(result.provider_task_id, TASK)
        self.assertEqual(result.provider_status, "SUCCEEDED")
        self.assertEqual(read_verified(self.output / "run-1", result.output.raw), self.video)
        metadata = json.loads(read_verified(self.output / "run-1", result.output.metadata))
        self.assertEqual(metadata["visualEffectDemonstrated"], "NOT_RUN")
        self.assertEqual(metadata["automatedWorkflowComplete"], "NOT_RUN")
        self.now = 2000.0
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.resume(), result)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.reservations(), before)

    def test_timeout_resumes_the_same_task_after_the_spending_deadline(self):
        before = self.reservations()
        self.task_timeout = True
        first = self.resume()
        self.assertEqual(first.state, "QUEUED")
        self.assertEqual(first.poll_status, "POLL_DEADLINE_REACHED")
        self.assertIsNone(first.output)
        self.task_timeout = False
        self.now = 2000.0
        self.assertEqual(self.resume().state, "SUCCEEDED")
        self.assertEqual([r.url.path for r in self.requests[:2]], ["/v1/tasks/" + TASK] * 2)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.reservations(), before)

    def test_reserved_unknown_and_terminal_attempts_need_no_credentials(self):
        for state in ("RESERVED", "SUBMISSION_UNKNOWN", "FAILED", "CANCELED"):
            with self.subTest(state=state):
                attempt = self.ledger.reserve("run-1", state, "b" * 64).attempt_id
                with SubmissionCoordinator(self.ledger) as owner:
                    if state == "SUBMISSION_UNKNOWN":
                        owner._claim(attempt)  # Simulate an abandoned durable claim.
                    elif state != "RESERVED":
                        task = "task-" + state
                        owner.submit_once(attempt, lambda: task)
                        provider_status = "CANCELLED" if state == "CANCELED" else "FAILED"
                        TaskPoller(owner).poll_once(attempt, lambda _: TaskObservation(task, provider_status, "c" * 64),
                                                   deadline_unix=1001.0)
                with patch.dict("os.environ", {}, clear=True):
                    result = self.resume(attempt_id=attempt)
                self.assertEqual(result.state, state)
                self.assertIsNone(result.output)
        self.assertEqual(self.requests, [])

    def test_status_only_and_module_command_make_no_provider_calls(self):
        before = self.reservations()
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.resume(status_only=True).state, "QUEUED")
        environment = dict(os.environ)
        environment.pop("TEST_RESUME_KEY", None)
        result = subprocess.run([sys.executable, "-m", "app.resume", *self.arguments(), "--status-only"],
            cwd=Path(__file__).resolve().parents[2], env=environment,
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["state"], "QUEUED")
        self.assertNotIn(KEY, result.stdout + result.stderr)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.reservations(), before)

    def test_wrong_run_attempt_or_missing_database_cannot_reach_network(self):
        for arguments in [{"run_id": "other"}, {"attempt_id": "att-" + "f" * 32}]:
            with self.subTest(arguments=arguments), self.assertRaises(SubmissionError):
                self.resume(**arguments)
        missing = self.root / "missing.sqlite"
        with self.assertRaises(BudgetError):
            resume_module.resume_attempt(self.output, missing, run_id="run-1", attempt_id=self.attempt,
                                          output_hosts=frozenset({HOST}))
        self.assertFalse(missing.exists())
        self.assertEqual(self.requests, [])

    def test_missing_key_or_hosts_preserves_recoverable_work(self):
        before = self.reservations()
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            self.resume()
        self.assertEqual(self.requests, [])
        with self.assertRaises(OutputError):
            self.resume(output_hosts=frozenset())
        self.assertEqual(len(self.requests), 1)
        saved = self.resume(status_only=True)
        self.assertEqual(saved.state, "QUEUED")
        self.assertEqual(saved.provider_status, "SUCCEEDED")
        self.assertIsNone(saved.output)
        self.assertEqual(self.resume().state, "SUCCEEDED")
        self.assertEqual(self.reservations(), before)

    def test_failed_download_uses_fresh_task_evidence_on_next_resume(self):
        before = self.reservations()
        self.download_status = 503
        with self.assertRaises(OutputError):
            self.resume()
        self.assertEqual(self.resume(status_only=True).provider_status, "SUCCEEDED")
        self.download_status = 200
        self.output_url = "https://" + HOST + "/refreshed.mp4?signature=new"
        result = self.resume()
        self.assertEqual(result.state, "SUCCEEDED")
        self.assertEqual([r.url.path for r in self.requests], ["/v1/tasks/" + TASK, "/first.mp4",
                                                            "/v1/tasks/" + TASK, "/refreshed.mp4"])
        self.assertEqual(self.reservations(), before)

    def test_completed_file_corruption_cannot_be_reported_as_success(self):
        result = self.resume()
        (self.output / "run-1" / result.output.raw.path).write_bytes(b"damaged output fixture")
        with self.assertRaises(StorageError):
            self.resume(status_only=True)
        self.assertEqual(len(self.requests), 2)

    def test_success_state_without_output_record_is_rejected(self):
        with self.ledger._transaction() as connection:
            connection.execute("UPDATE submission_jobs SET state='SUCCEEDED' WHERE attempt_id=?", (self.attempt,))
        with self.assertRaises(OutputError):
            self.resume(status_only=True)
        self.assertEqual(self.requests, [])

    def test_cli_does_not_print_credentials_or_signed_urls_in_errors(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(resume_module, "resume_attempt", side_effect=ValueError(KEY + self.output_url)), \
                redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(resume_module.main(self.arguments()), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn(KEY, errors.getvalue())
        self.assertNotIn(self.output_url, errors.getvalue())
        self.assertIn("Resume could not proceed.", errors.getvalue())

    def test_info_logging_omits_signed_output_urls_and_credentials(self):
        logger = logging.getLogger("httpx")
        sdk_logger = logging.getLogger("runwayml")
        root_logger = logging.getLogger()
        previous_handlers = list(root_logger.handlers)
        self.addCleanup(setattr, root_logger, "handlers", previous_handlers)
        self.addCleanup(sdk_logger.setLevel, sdk_logger.level)
        with patch.dict("os.environ", {"RUNWAYML_LOG": "info"}), \
                patch.object(logger, "filters", []), self.assertLogs("httpx", level="INFO") as captured:
            sdk_logs.setup_logging()
            self.assertEqual(self.resume().state, "SUCCEEDED")
        messages = "\n".join(captured.output)
        self.assertIn("HTTP Request: GET", messages)
        self.assertIn(HOST, messages)
        self.assertIn("200 OK", messages)
        self.assertNotIn(self.output_url, messages)
        self.assertNotIn("/first.mp4", messages)
        self.assertNotIn("signature=fixture", messages)
        self.assertNotIn(KEY, messages)

    def test_module_command_reports_uncertain_work_with_exit_code_two(self):
        attempt = self.ledger.reserve("run-1", "abandoned", "b" * 64).attempt_id
        with SubmissionCoordinator(self.ledger) as owner:
            owner._claim(attempt)
        self.attempt = attempt
        before = self.reservations()
        environment = dict(os.environ)
        environment.pop("TEST_RESUME_KEY", None)
        result = subprocess.run([sys.executable, "-m", "app.resume", *self.arguments()],
            cwd=Path(__file__).resolve().parents[2], env=environment,
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["state"], "SUBMISSION_UNKNOWN")
        self.assertEqual(result.stderr, "")
        self.assertNotIn(KEY, result.stdout)
        self.assertEqual(self.reservations(), before)
        self.assertEqual(self.requests, [])
