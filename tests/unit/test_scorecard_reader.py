"""Read explicit historical bindings without publication or a current clock."""

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.budget import BudgetLedger
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference.submission import SubmissionCoordinator
from features.review import bind
import test_scorecard_binding as fixtures
import test_scorecard as rubric_fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B


class ScorecardReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = type("ReaderBindingFixture", (fixtures.ScorecardBindingTests,), {})
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.base = cls.fixture.base
        cls.manifests = cls.fixture.manifests
        cls.contents = {attempt: cls.fixture.card(attempt) for attempt in (ATTEMPT_A, ATTEMPT_B)}
        cls.bindings = {}
        ledger = BudgetLedger(cls.base / "budget.sqlite", clock=lambda: 5000.0)
        with SubmissionCoordinator(ledger) as owner:
            for attempt in (ATTEMPT_A, ATTEMPT_B):
                cls.bindings[attempt] = bind.bind_scorecard(owner, cls.base, cls.manifests[attempt],
                    cls.contents[attempt], run_id=RUN, attempt_id=attempt)
            cls.corrected_contents = cls.fixture.card(
                scores={"identity": 2, "motion": 3, "scene": 3, "placement": 3})
            cls.corrected = bind.bind_scorecard(owner, cls.base, cls.manifests[ATTEMPT_A],
                cls.corrected_contents, run_id=RUN, attempt_id=ATTEMPT_A)

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)
        self.run_root = self.root / RUN
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 5000.0)
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.original = self.data(self.bindings[ATTEMPT_A])

    def data(self, record):
        return json.loads(read_verified(self.run_root, record))

    def read(self, record=None, *, attempt=ATTEMPT_A, **changes):
        arguments = {"run_id": RUN, "attempt_id": attempt} | changes
        return bind.read_scorecard_binding(self.owner, self.root, record or self.bindings[attempt], **arguments)

    def save_receipt(self, data):
        contents = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
        path = f"attempts/{ATTEMPT_A}/review/scorecard-bindings/{sha256(contents).hexdigest()}.json"
        return publish_bytes(self.run_root, path, contents)

    def receipt_for_card(self, contents, **changes):
        path = f"attempts/{ATTEMPT_A}/review/scorecards/{sha256(contents).hexdigest()}.json"
        card = publish_bytes(self.run_root, path, contents)
        return self.save_receipt(self.original | {"scorecard": asdict(card)} | changes)

    def file_hashes(self):
        return {str(path.relative_to(self.run_root)): sha256(path.read_bytes()).hexdigest()
                for path in self.run_root.rglob("*") if path.is_file()}

    def ledger_dump(self):
        with self.ledger._connection() as connection:
            return list(connection.iterdump())

    def test_writer_retains_legacy_receipt_bytes_and_reader_returns_frozen_references(self):
        for attempt, mode in ((ATTEMPT_A, "A"), (ATTEMPT_B, "B")):
            with self.subTest(mode=mode):
                contents = self.contents[attempt]
                digest = sha256(contents).hexdigest()
                card = StoredArtifact(f"attempts/{attempt}/review/scorecards/{digest}.json", digest, len(contents))
                completed = self.fixture.fixture.completed[attempt]
                # Spell out the pre-reader format independently of the shared builder.
                legacy = {
                    "schemaVersion": "1.4", "manifestType": "bound_review_scorecard",
                    "runId": RUN, "attemptId": attempt, "mode": mode,
                    "providerTaskId": "fixture-task-" + mode.lower(),
                    "reviewManifest": asdict(self.manifests[attempt]), "scorecard": asdict(card),
                    "rawOutput": asdict(completed.raw), "rawMetadata": asdict(completed.metadata),
                    "providerSuccess": self.data(self.manifests[attempt])["providerSuccess"],
                    "completedUnix": 1001.0, "reviewedAtUtc": "1970-01-01T00:30:00Z",
                    "rubricVerdict": "PASS", "qualifyingFailureIndexes": [],
                    "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
                }
                expected = (json.dumps(legacy, sort_keys=True, indent=2) + "\n").encode()
                record = self.bindings[attempt]
                self.assertEqual(read_verified(self.run_root, record), expected)
                self.assertEqual(record.sha256, sha256(expected).hexdigest())
                result = self.read(attempt=attempt)
                self.assertEqual((result.record, result.scorecard, result.review_manifest),
                                 (record, card, self.manifests[attempt]))
                self.assertEqual((result.assessment.rubric_verdict, result.assessment.qualifying_failure_indexes),
                                 ("PASS", ()))
                with self.assertRaises(FrozenInstanceError):
                    result.scorecard = card

    def test_original_and_corrected_bindings_are_selected_explicitly(self):
        original = self.read()
        corrected = self.read(self.corrected)
        self.assertEqual(original.assessment.rubric_verdict, "PASS")
        self.assertEqual(corrected.assessment.rubric_verdict, "FAIL")
        self.assertNotEqual(original.scorecard, corrected.scorecard)
        self.assertEqual(read_verified(self.run_root, original.scorecard), self.contents[ATTEMPT_A])
        self.assertEqual(read_verified(self.run_root, corrected.scorecard), self.corrected_contents)
        self.assertEqual(self.read(), original)

    def test_historical_read_never_consults_current_clock_or_budget_deadline(self):
        expected = self.read()
        for value in (0, -1, True, float("nan"), float("inf"), 9999999999):
            self.ledger.clock = lambda value=value: value
            with self.subTest(clock=value):
                self.assertEqual(self.read(), expected)
        with patch.object(self.ledger, "_now", side_effect=RuntimeError("private broken clock")):
            self.assertEqual(self.read(), expected)

    def test_reader_issues_no_ddl_or_dml_and_publishes_no_artifacts(self):
        before_files, before_rows = self.file_hashes(), self.ledger_dump()
        real_connection = self.ledger._connection
        denied, statements = [], []
        write_actions = {getattr(sqlite3, name) for name in (
            "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TABLE",
            "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TABLE", "SQLITE_CREATE_TEMP_TRIGGER",
            "SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW", "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE", "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE", "SQLITE_DROP_TEMP_TRIGGER",
            "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW", "SQLITE_ALTER_TABLE",
            "SQLITE_REINDEX", "SQLITE_ANALYZE", "SQLITE_ATTACH", "SQLITE_DETACH", "SQLITE_TRANSACTION")}

        def authorize(action, first, second, database, trigger):
            if action in write_actions:
                denied.append((action, first))
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        @contextmanager
        def guarded_connection():
            with real_connection() as connection:
                connection.set_authorizer(authorize)
                connection.set_trace_callback(statements.append)
                yield connection

        with patch.object(self.ledger, "_connection", guarded_connection), \
                patch.object(bind, "_publish_or_verify", side_effect=AssertionError("No publication")), \
                patch.object(bind, "bind_scorecard", side_effect=AssertionError("No writer")), \
                patch.object(bind.OutputCompleter, "__init__", side_effect=AssertionError("No constructor")), \
                patch.object(SubmissionCoordinator, "__init__", side_effect=AssertionError("No recovery")):
            result = self.read()
        self.assertEqual(result.record, self.bindings[ATTEMPT_A])
        self.assertEqual(denied, [])
        self.assertTrue(statements)
        self.assertTrue(all(statement.lstrip().upper().startswith("SELECT") for statement in statements))
        self.assertEqual(self.file_hashes(), before_files)
        self.assertEqual(self.ledger_dump(), before_rows)

    def test_full_receipt_identity_outcomes_and_extra_claims_are_rejected(self):
        changes = {"schemaVersion": "1.3", "manifestType": "other", "runId": "other-run",
                   "attemptId": ATTEMPT_B, "mode": "B", "providerTaskId": "other-task",
                   "completedUnix": 1000.0, "reviewedAtUtc": "1970-01-01T00:30:01Z",
                   "visualEffectDemonstrated": "PASS", "automatedWorkflowComplete": "PASS",
                   "humanAuthenticated": True, "liveProviderProvenance": True, "extra": "private unknown value"}
        for field, value in changes.items():
            record = self.save_receipt(self.original | {field: value})
            with self.subTest(field=field), self.assertRaises(bind.ScorecardBindingError):
                self.read(record)

    def test_rubric_and_qualifying_failure_indexes_are_recomputed(self):
        changed_verdict = self.save_receipt(self.data(self.corrected) | {"rubricVerdict": "PASS"})
        with self.assertRaises(bind.ScorecardBindingError):
            self.read(changed_verdict)
        failure = rubric_fixtures.event(start=0.033333, end=0.333333)
        contents = self.fixture.card(hardFailures=[failure])
        record = self.receipt_for_card(contents, rubricVerdict="FAIL", qualifyingFailureIndexes=[])
        with self.assertRaises(bind.ScorecardBindingError):
            self.read(record)

    def test_unsigned_card_and_intrinsically_early_review_are_rejected(self):
        unsigned = self.receipt_for_card(self.fixture.card(signature=""))
        early = self.receipt_for_card(self.fixture.card(reviewedAtUtc="1970-01-01T00:16:40Z"),
                                      reviewedAtUtc="1970-01-01T00:16:40Z")
        for record in (unsigned, early):
            with self.subTest(record=record), self.assertRaises(bind.ScorecardBindingError):
                self.read(record)

    def test_failure_endpoints_still_require_exact_native_frame_times(self):
        failure = rubric_fixtures.event(start=0.033333001, end=0.333333)
        record = self.receipt_for_card(self.fixture.card(hardFailures=[failure]),
                                      rubricVerdict="FAIL", qualifyingFailureIndexes=[0])
        with self.assertRaises(bind.ScorecardBindingError):
            self.read(record)

    def test_foreign_review_card_output_and_success_references_are_rejected(self):
        other = self.data(self.bindings[ATTEMPT_B])
        for name in ("reviewManifest", "scorecard", "rawOutput", "rawMetadata", "providerSuccess"):
            record = self.save_receipt(self.original | {name: other[name]})
            with self.subTest(field=name), self.assertRaises(bind.ScorecardBindingError):
                self.read(record)

    def test_binding_and_card_require_canonical_paths_hashes_and_bounded_sizes(self):
        original_ref = self.bindings[ATTEMPT_A]
        copied = publish_bytes(self.run_root, f"attempts/{ATTEMPT_A}/review/copied-binding.json",
                               read_verified(self.run_root, original_ref))
        bad_refs = [copied, replace(original_ref, sha256="f" * 64), replace(original_ref, size=True),
                    replace(original_ref, size=65537), replace(original_ref, path="../binding.json")]
        for record in bad_refs:
            with self.subTest(record=record), self.assertRaises(bind.ScorecardBindingError):
                self.read(record)
        card = StoredArtifact(**self.original["scorecard"])
        copied_card = publish_bytes(self.run_root, f"attempts/{ATTEMPT_A}/review/copied-card.json",
                                    read_verified(self.run_root, card))
        bad_cards = [copied_card, replace(card, size=True), replace(card, size=65537),
                     replace(card, sha256="f" * 64)]
        for changed in bad_cards:
            record = self.save_receipt(self.original | {"scorecard": asdict(changed)})
            with self.subTest(card=changed), self.assertRaises(bind.ScorecardBindingError):
                self.read(record)

    def test_noncanonical_receipt_encoding_is_not_a_writer_receipt(self):
        contents = json.dumps(self.original, separators=(",", ":")).encode()
        record = publish_bytes(self.run_root,
            f"attempts/{ATTEMPT_A}/review/scorecard-bindings/{sha256(contents).hexdigest()}.json", contents)
        with self.assertRaises(bind.ScorecardBindingError):
            self.read(record)

    def test_missing_or_changed_source_files_fail_without_repair(self):
        review = self.data(self.manifests[ATTEMPT_A])
        records = [self.original["scorecard"], self.original["reviewManifest"], self.original["rawOutput"],
                   self.original["rawMetadata"], self.original["providerSuccess"], review["sourceOriginal"],
                   review["guide"]["artifact"], review["samples"][4]["raw"]["image"]]
        for record in records:
            path = self.run_root / record["path"]
            original = path.read_bytes()
            path.unlink()
            with self.subTest(path=record["path"]), self.assertRaises(bind.ScorecardBindingError):
                self.read()
            self.assertFalse(path.exists())
            path.write_bytes(original)
        path = self.run_root / self.bindings[ATTEMPT_A].path
        path.write_bytes(b"private changed receipt")
        with self.assertRaises(bind.ScorecardBindingError):
            self.read()
        self.assertEqual(path.read_bytes(), b"private changed receipt")

    def test_current_task_and_completed_record_are_still_required(self):
        with self.ledger._connection() as connection:
            connection.execute("UPDATE submission_jobs SET state='RUNNING' WHERE attempt_id=?", (ATTEMPT_A,))
        with self.assertRaises(bind.ScorecardBindingError):
            self.read()
        with self.ledger._connection() as connection:
            connection.execute("UPDATE submission_jobs SET state='SUCCEEDED', provider_task_id='other-task' "
                               "WHERE attempt_id=?", (ATTEMPT_A,))
        with self.assertRaises(bind.ScorecardBindingError):
            self.read()
        with self.ledger._connection() as connection:
            connection.execute("DELETE FROM completed_outputs WHERE attempt_id=?", (ATTEMPT_A,))
        with self.assertRaises(bind.ScorecardBindingError):
            self.read()

    def test_missing_completion_table_is_not_recreated(self):
        with self.ledger._connection() as connection:
            connection.execute("DROP TABLE completed_outputs")
        with self.assertRaises(bind.ScorecardBindingError):
            self.read()
        with self.ledger._connection() as connection:
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='completed_outputs'").fetchone())

    def test_invalid_identity_closed_owner_and_private_errors_are_safe(self):
        for changes in ({"run_id": "../private-path"}, {"attempt_id": ATTEMPT_B}):
            with self.subTest(changes=changes), self.assertRaises(bind.ScorecardBindingError):
                self.read(**changes)
        with patch.object(bind, "read_completed_output", side_effect=OSError("private signed-url-secret")):
            with self.assertRaises(bind.ScorecardBindingError) as raised:
                self.read()
        self.assertEqual(str(raised.exception), "Saved scorecard binding could not be verified for this attempt.")
        self.owner.close()
        with self.assertRaises(bind.ScorecardBindingError):
            self.read()


if __name__ == "__main__":
    unittest.main()
