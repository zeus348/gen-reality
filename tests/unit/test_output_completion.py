"""Commit success only after validated, immutable output is available."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from core.budget import BudgetExceeded, BudgetLedger
from core.storage import StorageError, read_verified
from features.inference.polling import TaskObservation, TaskPoller
from features.inference.submission import SubmissionCoordinator
from features.media.ingest import MediaError
from features.output import complete as completion_module
from features.output.complete import OutputCompleter, OutputError


TASK_ID = "c26d7283-d514-4492-8589-760dfe6f0289"
EVIDENCE = "a" * 64


class OutputCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                            "color=green:s=64x36:r=30:d=5", "-c:v", "libx264", "-threads", "2",
                            "-pix_fmt", "yuv420p", str(path)], capture_output=True, check=True, timeout=30)
            cls.contents = path.read_bytes()

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.output = self.root / "runs"
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1000.0)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=140,
                               max_attempts=6, cost_cents=140)
        self.attempt = self.ledger.reserve("run-1", "first", "1" * 64).attempt_id
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.create = Mock(return_value=TASK_ID)
        self.owner.submit_once(self.attempt, self.create)
        self.poller = TaskPoller(self.owner)
        self.completer = OutputCompleter(self.owner, self.output)
        self.download = Mock(return_value=self.contents)

    def observe(self, status="SUCCEEDED", evidence=EVIDENCE):
        return self.poller.poll_once(self.attempt, Mock(return_value=TaskObservation(TASK_ID, status, evidence)),
                                    deadline_unix=1100.0)

    def test_artifacts_precede_success_and_fixture_content_never_claims_workflow_pass(self):
        self.observe()
        original = completion_module._publish_or_verify
        publications = []

        def publish(root, path, contents):
            self.assertEqual(self.owner.get(self.attempt).state, "QUEUED")
            self.assertIsNone(self.completer.get(self.attempt))
            result = original(root, path, contents)
            publications.append(result)
            return result

        with patch.object(completion_module, "_publish_or_verify", side_effect=publish):
            result = self.completer.complete(self.attempt, self.download)
        self.download.assert_called_once_with(TASK_ID, EVIDENCE)
        self.assertEqual(self.owner.get(self.attempt).state, "SUCCEEDED")
        self.assertEqual(publications, [result.raw, result.metadata])
        root = self.output / result.run_id
        self.assertEqual(read_verified(root, result.raw), self.contents)
        metadata = json.loads(read_verified(root, result.metadata))
        self.assertEqual(metadata["rawContentValidation"], "PASS")
        self.assertEqual(metadata["visualEffectDemonstrated"], "NOT_RUN")
        self.assertEqual(metadata["automatedWorkflowComplete"], "NOT_RUN")
        self.assertEqual(metadata["providerTaskId"], TASK_ID)
        self.assertEqual(len(metadata["frameSeconds"]), 150)
        self.assertEqual(self.completer.get(self.attempt), result)

    def test_download_requires_provider_success_for_a_known_active_task(self):
        for status in [None, "RUNNING", "FAILED"]:
            with self.subTest(status=status):
                if status:
                    self.observe(status)
                with self.assertRaises(OutputError):
                    self.completer.complete(self.attempt, self.download)
        with self.assertRaises(ValueError):
            self.completer.complete("att-" + "f" * 32, self.download)
        self.download.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_corrupt_download_cannot_create_artifacts_or_local_success(self):
        self.observe()
        self.download.return_value = b"private fixture detail"
        with self.assertRaises(MediaError) as caught:
            self.completer.complete(self.attempt, self.download)
        self.assertNotIn("private fixture detail", str(caught.exception))
        self.assertEqual(self.owner.get(self.attempt).state, "QUEUED")
        self.assertIsNone(self.completer.get(self.attempt))
        self.assertFalse(self.output.exists())

    def test_download_failure_is_safe_and_can_resume_the_same_task(self):
        self.observe()
        self.download.side_effect = RuntimeError("private provider credential")
        with self.assertRaises(OutputError) as caught:
            self.completer.complete(self.attempt, self.download)
        self.assertNotIn("private provider credential", str(caught.exception))
        self.assertEqual(self.owner.get(self.attempt).state, "QUEUED")
        self.assertFalse(self.output.exists())
        self.download.side_effect = None
        self.completer.complete(self.attempt, self.download)
        self.assertEqual(self.download.call_count, 2)
        self.create.assert_called_once_with()

    def test_partial_publication_can_resume_without_replacing_raw_bytes(self):
        self.observe()
        original = completion_module.publish_bytes
        for failed_name in ["raw.mp4", "metadata.json"]:
            def fail(root, path, contents):
                if path.endswith(failed_name):
                    raise StorageError("fixture disk failure")
                return original(root, path, contents)

            with self.subTest(file=failed_name), patch.object(completion_module, "publish_bytes", side_effect=fail):
                with self.assertRaises(StorageError):
                    self.completer.complete(self.attempt, self.download)
                self.assertEqual(self.owner.get(self.attempt).state, "QUEUED")
                self.assertIsNone(self.completer.get(self.attempt))
        path = self.output / "run-1" / "attempts" / self.attempt / "output" / "raw.mp4"
        before = path.stat().st_mtime_ns
        result = self.completer.complete(self.attempt, self.download)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(read_verified(self.output / "run-1", result.raw), self.contents)

    def test_artifact_record_and_success_commit_together_and_resume_after_restart(self):
        self.observe()
        with sqlite3.connect(self.ledger.path) as connection:
            connection.execute("CREATE TRIGGER fail_success BEFORE UPDATE OF state ON submission_jobs "
                               "WHEN NEW.state='SUCCEEDED' BEGIN SELECT RAISE(FAIL, 'fixture commit failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.completer.complete(self.attempt, self.download)
        self.assertEqual(self.owner.get(self.attempt).state, "QUEUED")
        self.assertIsNone(self.completer.get(self.attempt))
        folder = self.output / "run-1" / "attempts" / self.attempt / "output"
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in folder.iterdir()}
        self.assertEqual(set(before), {"raw.mp4", "metadata.json"})
        with sqlite3.connect(self.ledger.path) as connection:
            connection.execute("DROP TRIGGER fail_success")
        self.owner.close()
        restarted = SubmissionCoordinator(self.ledger)
        self.addCleanup(restarted.close)
        result = OutputCompleter(restarted, self.output).complete(self.attempt, self.download)
        self.assertEqual(result.provider_task_id, TASK_ID)
        self.assertEqual(restarted.get(self.attempt).state, "SUCCEEDED")
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in folder.iterdir()}, before)
        self.create.assert_called_once_with()

    def test_completed_output_is_verified_without_download_or_repair(self):
        self.observe()
        result = self.completer.complete(self.attempt, self.download)
        self.owner.close()
        restarted = SubmissionCoordinator(self.ledger)
        self.addCleanup(restarted.close)
        completer = OutputCompleter(restarted, self.output)
        download = Mock(side_effect=AssertionError("Completed task must not download again."))
        self.assertEqual(completer.complete(self.attempt, download), result)
        path = self.output / "run-1" / result.raw.path
        path.write_bytes(b"damaged fixture output")
        with self.assertRaises(StorageError):
            completer.complete(self.attempt, download)
        self.assertEqual(path.read_bytes(), b"damaged fixture output")
        download.assert_not_called()

    def test_local_success_does_not_release_the_reserved_charge(self):
        self.observe()
        self.completer.complete(self.attempt, self.download)
        with self.ledger._connection() as connection:
            self.assertEqual(self.ledger._usage(connection, "run-1"), (1, 140, False))
            self.assertEqual(connection.execute("SELECT count(*) FROM actual_charges").fetchone()[0], 0)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve("run-1", "second", "2" * 64)

    def test_changed_success_evidence_is_checked_again_before_commit(self):
        self.observe()

        def refreshed_during_download(task, evidence):
            self.observe(evidence="b" * 64)
            return self.contents

        with self.assertRaisesRegex(OutputError, "evidence changed"):
            self.completer.complete(self.attempt, refreshed_during_download)
        self.assertEqual(self.owner.get(self.attempt).state, "QUEUED")
        self.assertIsNone(self.completer.get(self.attempt))
        result = self.completer.complete(self.attempt, self.download)
        self.assertEqual(result.provider_evidence_sha256, "b" * 64)
        self.download.assert_called_once_with(TASK_ID, "b" * 64)

    def test_concurrent_completion_calls_download_and_commit_once(self):
        self.observe()
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(lambda _: self.completer.complete(self.attempt, self.download), range(20)))
        self.assertTrue(all(result == results[0] for result in results))
        self.download.assert_called_once_with(TASK_ID, EVIDENCE)
        self.assertEqual(self.owner.get(self.attempt).state, "SUCCEEDED")
        self.create.assert_called_once_with()
