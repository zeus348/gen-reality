"""Exercise the scorecard command with real saved media and local processes."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from app import scorecard as command
from core.budget import BudgetLedger
from core.storage import StoredArtifact, read_verified
from features.inference.submission import SubmissionCoordinator
import test_scorecard_binding as fixtures


RUN, ATTEMPT = fixtures.RUN, fixtures.ATTEMPT_A
ABANDONED = "att-" + "c" * 32


class ScorecardCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = type("CommandBindingFixture", (fixtures.ScorecardBindingTests,), {})
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.base = cls.fixture.base
        cls.manifest = cls.fixture.manifests[ATTEMPT]

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)
        self.run_root = self.root / RUN
        self.ledger_path = self.root / "budget.sqlite"
        self.ledger = BudgetLedger(self.ledger_path, clock=lambda: 5000.0)
        self.card_path = self.root / "entered scorecard.json"
        self.contents = self.fixture.card()
        self.card_path.write_bytes(self.contents)

    def arguments(self, **changes):
        values = {"output-root": self.root, "ledger": self.ledger_path, "run-id": RUN, "attempt-id": ATTEMPT,
                  "review-path": self.manifest.path, "review-sha256": self.manifest.sha256,
                  "review-size": self.manifest.size, "scorecard-file": self.card_path} | changes
        return [item for name, value in values.items() for item in ("--" + name, str(value))]

    def run_command(self, arguments=None):
        environment = dict(os.environ)
        environment.pop("RUNWAY_API_KEY", None)
        return subprocess.run([sys.executable, "-m", "app.scorecard",
                               *(self.arguments() if arguments is None else arguments)],
                              cwd=Path(__file__).resolve().parents[2], env=environment,
                              capture_output=True, text=True, timeout=10)

    def bind_file(self, **changes):
        arguments = {"output_root": self.root, "ledger_path": self.ledger_path, "review_manifest": self.manifest,
                     "scorecard_path": self.card_path, "run_id": RUN, "attempt_id": ATTEMPT,
                     "clock": lambda: 5000.0} | changes
        return command.bind_scorecard_file(**arguments)

    def data(self, record):
        return json.loads(read_verified(self.run_root, record))

    def ledger_rows(self):
        with self.ledger._connection() as connection:
            return {table: [tuple(row) for row in connection.execute("SELECT * FROM " + table + " ORDER BY 1")]
                    for table in ("approved_runs", "reservations", "actual_charges", "submission_jobs")}

    def abandon_claim(self):
        self.ledger.reserve(RUN, "abandoned-command-fixture", "c" * 64, attempt_id=ABANDONED)
        with SubmissionCoordinator(self.ledger) as owner:
            self.assertTrue(owner._claim(ABANDONED))

    def abandoned_state(self):
        with self.ledger._connection() as connection:
            return connection.execute("SELECT state FROM submission_jobs WHERE attempt_id=?", (ABANDONED,)).fetchone()[0]

    def assert_no_binding(self):
        folder = self.run_root / f"attempts/{ATTEMPT}/review"
        self.assertFalse((folder / "scorecards").exists())
        self.assertFalse((folder / "scorecard-bindings").exists())

    def assert_safe_failure(self, result):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "Scorecard could not be verified or bound to this attempt.\n")

    def test_subprocess_returns_explicit_binding_and_repeats_without_credentials(self):
        before = self.ledger_rows()
        first = self.run_command()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stderr, "")
        result = json.loads(first.stdout)
        self.assertEqual(set(result), {"runId", "attemptId", "binding"})
        self.assertEqual((result["runId"], result["attemptId"]), (RUN, ATTEMPT))
        record = StoredArtifact(**result["binding"])
        receipt = self.data(record)
        self.assertEqual(receipt["reviewManifest"], asdict(self.manifest))
        self.assertEqual(read_verified(self.run_root, StoredArtifact(**receipt["scorecard"])), self.contents)
        self.assertEqual(receipt["rubricVerdict"], "PASS")
        self.assertEqual(receipt["visualEffectDemonstrated"], "NOT_RUN")
        self.assertEqual(receipt["automatedWorkflowComplete"], "NOT_RUN")
        repeated = self.run_command()
        self.assertEqual((repeated.returncode, repeated.stdout, repeated.stderr), (0, first.stdout, ""))
        self.assertEqual(self.ledger_rows(), before)
        self.assertEqual(self.card_path.read_bytes(), self.contents)

    def test_subprocess_correction_preserves_both_cards_and_returns_rubric_failure_successfully(self):
        first = self.run_command()
        self.assertEqual(first.returncode, 0, first.stderr)
        old = StoredArtifact(**json.loads(first.stdout)["binding"])
        old_bytes = read_verified(self.run_root, old)
        corrected = self.fixture.card(scores={"identity": 2, "motion": 3, "scene": 3, "placement": 3})
        self.card_path.write_bytes(corrected)
        second = self.run_command()
        self.assertEqual(second.returncode, 0, second.stderr)
        new = StoredArtifact(**json.loads(second.stdout)["binding"])
        self.assertNotEqual(new, old)
        receipt = self.data(new)
        self.assertEqual(receipt["rubricVerdict"], "FAIL")
        self.assertEqual(read_verified(self.run_root, StoredArtifact(**receipt["scorecard"])), corrected)
        self.assertEqual(read_verified(self.run_root, old), old_bytes)
        self.assertEqual(read_verified(self.run_root, StoredArtifact(**self.data(old)["scorecard"])), self.contents)

    def test_subprocess_failure_is_safe_and_creates_no_missing_ledger(self):
        missing = self.root / "private-secret-missing.sqlite"
        cases = [self.arguments(**{"ledger": missing}), self.arguments(**{"review-sha256": "f" * 64}),
                 self.arguments(**{"run-id": "wrong-run"}), self.arguments(**{"attempt-id": fixtures.ATTEMPT_B}),
                 self.arguments(**{"scorecard-file": self.root / "private-secret-missing.json"})]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assert_safe_failure(self.run_command(arguments))
                self.assert_no_binding()
        self.assertFalse(missing.exists())

    def test_complete_card_and_expected_ids_are_checked_before_recovery(self):
        self.abandon_claim()
        before = self.ledger_rows()
        cases = [self.fixture.card(signature=""), self.fixture.card(reviewerName=""),
                 self.fixture.card(reviewedAtUtc=""), self.fixture.card(runId="other-run"),
                 self.fixture.card(attempt=fixtures.ATTEMPT_B), b"{invalid", self.contents + b"\xff"]
        for contents in cases:
            self.card_path.write_bytes(contents)
            with self.subTest(contents=contents[:60]), self.assertRaises(ValueError):
                self.bind_file()
            self.assertEqual(self.abandoned_state(), "SUBMITTING")
            self.assertEqual(self.ledger_rows(), before)
            self.assert_no_binding()
        self.card_path.write_bytes(self.contents)
        with self.assertRaises(ValueError):
            self.bind_file(run_id="../invalid")
        with self.assertRaises(ValueError):
            self.bind_file(attempt_id="invalid")
        self.assertEqual(self.ledger_rows(), before)

    def test_coordinator_recovers_abandoned_claim_but_keeps_reservations_and_charges(self):
        self.abandon_claim()
        before = self.ledger_rows()
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(httpx.Client, "send", side_effect=AssertionError("No provider work allowed")), \
                patch.object(BudgetLedger, "approve_run", side_effect=AssertionError("No approval allowed")), \
                patch.object(BudgetLedger, "reserve", side_effect=AssertionError("No reservation allowed")):
            record = self.bind_file()
        self.assertEqual(self.data(record)["rubricVerdict"], "PASS")
        self.assertEqual(self.abandoned_state(), "SUBMISSION_UNKNOWN")
        after = self.ledger_rows()
        for table in ("approved_runs", "reservations", "actual_charges"):
            self.assertEqual(after[table], before[table])

    def test_exact_maximum_file_size_preserves_bytes_and_over_limit_or_empty_is_rejected(self):
        self.abandon_claim()
        for contents in (b"", self.contents + b" " * (65537 - len(self.contents))):
            self.card_path.write_bytes(contents)
            with self.subTest(size=len(contents)), self.assertRaises(ValueError):
                self.bind_file()
            self.assertEqual(self.abandoned_state(), "SUBMITTING")
            self.assert_no_binding()
        maximum = self.contents + b" " * (65536 - len(self.contents))
        self.card_path.write_bytes(maximum)
        record = self.bind_file()
        self.assertEqual(read_verified(self.run_root, StoredArtifact(**self.data(record)["scorecard"])), maximum)

    def test_symlink_file_and_parent_directory_are_rejected(self):
        self.abandon_claim()
        link = self.root / "card-link.json"
        link.symlink_to(self.card_path)
        parent_link = self.root / "parent-link"
        parent_link.symlink_to(self.root, target_is_directory=True)
        for path in (link, parent_link / self.card_path.name):
            with self.subTest(path=path), self.assertRaises(OSError):
                self.bind_file(scorecard_path=path)
            self.assertEqual(self.abandoned_state(), "SUBMITTING")
            self.assert_no_binding()

    def test_fifo_device_and_directory_fail_in_a_bounded_subprocess(self):
        fifo = self.root / "scorecard.pipe"
        os.mkfifo(fifo)
        for path in (fifo, Path("/dev/null"), self.root):
            with self.subTest(path=path):
                self.assert_safe_failure(self.run_command(self.arguments(**{"scorecard-file": path})))
                self.assert_no_binding()

    def test_file_growth_or_shrink_during_read_is_rejected_before_recovery(self):
        self.abandon_claim()
        real_fstat = command.os.fstat
        for changed in (self.contents + b" ", self.contents[:-1]):
            self.card_path.write_bytes(self.contents)

            def change_after_stat(descriptor):
                result = real_fstat(descriptor)
                self.card_path.write_bytes(changed)
                return result

            with self.subTest(size=len(changed)), patch.object(command.os, "fstat", side_effect=change_after_stat):
                with self.assertRaises(ValueError):
                    self.bind_file()
            self.assertEqual(self.abandoned_state(), "SUBMITTING")
            self.assert_no_binding()

    def test_ledger_symlink_and_nonregular_paths_are_rejected(self):
        link = self.root / "ledger-link.sqlite"
        link.symlink_to(self.ledger_path)
        for path in (link, self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.bind_file(ledger_path=path)
            self.assert_no_binding()

    def test_active_coordinator_blocks_command_then_releases_for_retry(self):
        with SubmissionCoordinator(self.ledger):
            self.assert_safe_failure(self.run_command())
            self.assert_no_binding()
        result = self.run_command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.data(StoredArtifact(**json.loads(result.stdout)["binding"]))["rubricVerdict"], "PASS")

    def test_helper_forwards_trusted_clock_without_fabricating_review_time(self):
        with self.assertRaises(ValueError):
            self.bind_file(clock=lambda: 1799.999999)
        self.assert_no_binding()
        record = self.bind_file(clock=lambda: 1800.0)
        receipt = self.data(record)
        self.assertEqual(receipt["reviewedAtUtc"], "1970-01-01T00:30:00Z")
        self.assertEqual(read_verified(self.run_root, StoredArtifact(**receipt["scorecard"])), self.contents)

    def test_argument_errors_have_safe_exit_one(self):
        cases = [[], self.arguments(**{"review-size": "private-secret-invalid-size"}),
                 [*self.arguments(), "--private-secret-unknown-option"]]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assert_safe_failure(self.run_command(arguments))
        self.assert_no_binding()

    def test_main_hides_sensitive_exception_details(self):
        output, errors = StringIO(), StringIO()
        with patch.object(command, "bind_scorecard_file", side_effect=OSError("private-secret-provider-url")), \
                redirect_stdout(output), redirect_stderr(errors):
            result = command.main(self.arguments())
        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(errors.getvalue(), "Scorecard could not be verified or bound to this attempt.\n")


if __name__ == "__main__":
    unittest.main()
