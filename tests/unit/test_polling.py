"""Keep local polling limits separate from provider terminal states."""

from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from core.budget import BudgetExceeded, BudgetLedger
from features.inference.polling import TaskObservation, TaskPoller
from features.inference.submission import SubmissionCoordinator, SubmissionError


TASK_ID = "c26d7283-d514-4492-8589-760dfe6f0289"
EVIDENCE = sha256(b"saved provider response").hexdigest()


class PollingTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "budget.sqlite"
        self.now = 1000.0
        self.ledger = BudgetLedger(self.path, clock=lambda: self.now)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=280,
                               max_attempts=2, cost_cents=140)
        self.attempt = self.ledger.reserve("run-1", "first", "1" * 64).attempt_id
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.create = Mock(return_value=TASK_ID)
        self.owner.submit_once(self.attempt, self.create)
        self.poller = TaskPoller(self.owner)

    def observe(self, status, attempt=None, task_id=TASK_ID):
        retrieve = Mock(return_value=TaskObservation(task_id, status, EVIDENCE))
        result = self.poller.poll_once(attempt or self.attempt, retrieve, deadline_unix=1100.0)
        retrieve.assert_called_once_with(task_id)
        return result

    def test_known_task_moves_to_running_and_keeps_its_response_evidence(self):
        self.assertEqual(self.poller.get(self.attempt).poll_status, "NOT_POLLED")
        result = self.observe("RUNNING")
        self.assertEqual((result.state, result.provider_status, result.poll_status, result.evidence_sha256),
                         ("RUNNING", "RUNNING", "OBSERVED", EVIDENCE))
        self.assertEqual(self.poller.get(self.attempt), result)
        self.create.assert_called_once_with()

    def test_get_timeout_preserves_state_and_restart_resumes_the_same_id(self):
        self.observe("RUNNING")
        timeout = Mock(side_effect=TimeoutError("private response detail"))
        result = self.poller.poll_once(self.attempt, timeout, deadline_unix=1100.0)
        timeout.assert_called_once_with(TASK_ID)
        self.assertEqual((result.state, result.poll_status), ("RUNNING", "POLL_DEADLINE_REACHED"))
        self.assertEqual((result.provider_status, result.evidence_sha256), ("RUNNING", EVIDENCE))
        self.owner.close()
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.poller = TaskPoller(self.owner)
        self.assertEqual(self.poller.get(self.attempt), result)
        resumed = self.observe("RUNNING")
        self.assertEqual(resumed.poll_status, "OBSERVED")
        self.create.assert_called_once_with()
        self.assertEqual(self.owner.get(self.attempt).provider_task_id, TASK_ID)

    def test_deadline_prevents_get_but_allows_later_resume(self):
        retrieve = Mock(return_value=TaskObservation(TASK_ID, "RUNNING", EVIDENCE))
        self.now = 1100.0
        result = self.poller.poll_once(self.attempt, retrieve, deadline_unix=1100.0)
        self.assertEqual((result.state, result.poll_status), ("QUEUED", "POLL_DEADLINE_REACHED"))
        retrieve.assert_not_called()
        result = self.poller.poll_once(self.attempt, retrieve, deadline_unix=1200.0)
        self.assertEqual(result.state, "RUNNING")
        retrieve.assert_called_once_with(TASK_ID)

    def test_provider_success_does_not_claim_validated_local_success(self):
        for previous in ["QUEUED", "RUNNING"]:
            with self.subTest(previous=previous):
                # Use a new task for the running-to-success path.
                if previous == "RUNNING":
                    self.attempt = self.ledger.reserve("run-1", "second", "2" * 64).attempt_id
                    self.owner.submit_once(self.attempt, Mock(return_value="second-task"))
                    task_id = "second-task"
                    self.observe("RUNNING", task_id=task_id)
                else:
                    task_id = TASK_ID
                result = self.observe("SUCCEEDED", task_id=task_id)
                self.assertEqual((result.state, result.provider_status), (previous, "SUCCEEDED"))
                self.assertNotEqual(self.owner.get(self.attempt).state, "SUCCEEDED")

    def test_failed_and_canceled_tasks_stay_terminal_without_releasing_money(self):
        for index, (provider, local) in enumerate([("FAILED", "FAILED"), ("CANCELLED", "CANCELED")]):
            if index:
                attempt = self.ledger.reserve("run-1", "second", "2" * 64).attempt_id
                self.owner.submit_once(attempt, Mock(return_value="second-task"))
                task_id = "second-task"
            else:
                attempt, task_id = self.attempt, TASK_ID
            result = self.observe(provider, attempt=attempt, task_id=task_id)
            self.assertEqual(result.state, local)
            retrieve = Mock()
            self.assertEqual(self.poller.poll_once(attempt, retrieve, deadline_unix=1100.0), result)
            retrieve.assert_not_called()
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (2, 280, False))
            self.assertEqual(connection.execute("SELECT count(*) FROM actual_charges").fetchone()[0], 0)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "third", "3" * 64)

    def test_queued_observation_cannot_move_running_task_backwards(self):
        self.observe("RUNNING")
        self.assertEqual(self.observe("PENDING").state, "RUNNING")
        with sqlite3.connect(self.path) as connection:
            statuses = connection.execute("SELECT provider_status FROM poll_events ORDER BY event_id").fetchall()
        self.assertEqual(statuses, [("RUNNING",), ("PENDING",)])

    def test_conflicting_terminal_observation_is_a_poll_error(self):
        self.observe("SUCCEEDED")
        result = self.observe("RUNNING")
        self.assertEqual((result.state, result.provider_status, result.poll_status),
                         ("QUEUED", "SUCCEEDED", "POLL_ERROR"))

    def test_invalid_provider_observation_cannot_change_the_job(self):
        invalid = [None, {"status": "RUNNING"}, TaskObservation("wrong-task", "RUNNING", EVIDENCE),
                   TaskObservation(TASK_ID, "UNKNOWN_STATUS", EVIDENCE),
                   TaskObservation(TASK_ID, "RUNNING", "invalid")]
        for observation in invalid:
            with self.subTest(observation=observation):
                retrieve = Mock(return_value=observation)
                result = self.poller.poll_once(self.attempt, retrieve, deadline_unix=1100.0)
                self.assertEqual((result.state, result.provider_status, result.poll_status),
                                 ("QUEUED", None, "POLL_ERROR"))

    def test_get_error_keeps_evidence_private_and_allows_resume(self):
        retrieve = Mock(side_effect=ValueError("private API credential"))
        result = self.poller.poll_once(self.attempt, retrieve, deadline_unix=1100.0)
        self.assertEqual((result.state, result.poll_status), ("QUEUED", "POLL_ERROR"))
        with sqlite3.connect(self.path) as connection:
            self.assertNotIn("private API credential", "\n".join(connection.iterdump()))
        self.assertEqual(self.observe("RUNNING").state, "RUNNING")

    def test_unknown_submission_cannot_start_polling(self):
        attempt = self.ledger.reserve("run-1", "second", "2" * 64).attempt_id
        self.owner.submit_once(attempt, Mock(side_effect=TimeoutError))
        retrieve = Mock()
        with self.assertRaisesRegex(SubmissionError, "known queued or running"):
            self.poller.poll_once(attempt, retrieve, deadline_unix=1100.0)
        retrieve.assert_not_called()

    def test_invalid_deadline_cannot_start_a_get(self):
        retrieve = Mock()
        for deadline in [None, True, -1, float("nan"), float("inf"), 10 ** 400]:
            with self.subTest(deadline=deadline):
                with self.assertRaises(ValueError):
                    self.poller.poll_once(self.attempt, retrieve, deadline_unix=deadline)
        retrieve.assert_not_called()
        self.assertEqual(self.poller.get(self.attempt).poll_status, "NOT_POLLED")

    def test_closed_coordinator_cannot_poll(self):
        self.owner.close()
        retrieve = Mock()
        with self.assertRaisesRegex(SubmissionError, "closed"):
            self.poller.poll_once(self.attempt, retrieve, deadline_unix=1100.0)
        retrieve.assert_not_called()

    def test_poll_event_and_state_change_commit_together(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TRIGGER fail_poll_event BEFORE INSERT ON poll_events "
                               "BEGIN SELECT RAISE(FAIL, 'fixture persistence failure'); END")
        retrieve = Mock(return_value=TaskObservation(TASK_ID, "RUNNING", EVIDENCE))
        with self.assertRaises(sqlite3.IntegrityError):
            self.poller.poll_once(self.attempt, retrieve, deadline_unix=1100.0)
        self.assertEqual(self.owner.get(self.attempt).state, "QUEUED")
        self.assertEqual(self.poller.get(self.attempt).poll_status, "NOT_POLLED")
