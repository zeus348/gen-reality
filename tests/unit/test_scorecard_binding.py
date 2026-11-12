"""Bind scorecards to real media with mocked provider transport only."""

from dataclasses import asdict, replace
from decimal import DefaultContext, Inexact, Rounded, localcontext
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.budget import BudgetLedger
from core.storage import StoredArtifact, read_verified
from features.inference.submission import SubmissionCoordinator
from features.output import review
from features.review import bind
import test_review_evidence as media_fixtures
import test_scorecard as rubric_fixtures


RUN = media_fixtures.RUN
ATTEMPT_A = media_fixtures.ATTEMPT_A
ATTEMPT_B = media_fixtures.ATTEMPT_B


class ScorecardBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Cache actual preparation, checked SDK submission, retrieval, completion,
        # and PNG extraction. The local subclass is not another discovered suite.
        cls.fixture = type("BindingMediaFixture", (media_fixtures.ReviewEvidenceTests,), {})
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.base = cls.fixture.base
        cls.manifests = {}
        ledger = BudgetLedger(cls.base / "budget.sqlite", clock=lambda: 5000.0)
        with SubmissionCoordinator(ledger) as owner:
            for attempt in (ATTEMPT_A, ATTEMPT_B):
                cls.manifests[attempt] = review.build_review_evidence(owner, cls.base,
                    cls.fixture.manifests[attempt], cls.fixture.checks[attempt].evidence,
                    run_id=RUN, attempt_id=attempt)

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)
        self.run_root = self.root / RUN
        self.ledger = BudgetLedger(self.root / "budget.sqlite", clock=lambda: 5000.0)
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.original_manifest = self.data(self.manifests[ATTEMPT_A])
        self.original_metadata = self.data(self.fixture.completed[ATTEMPT_A].metadata)

    def data(self, record):
        return json.loads(read_verified(self.run_root, record))

    @staticmethod
    def card(attempt=ATTEMPT_A, **changes):
        data = rubric_fixtures.card(**({"attemptId": attempt, "reviewedAtUtc": "1970-01-01T00:30:00Z"} | changes))
        return (json.dumps(data, indent=2) + "\n").encode()

    def build(self, contents=None, *, attempt=ATTEMPT_A, manifest=None, **changes):
        arguments = {"run_id": RUN, "attempt_id": attempt} | changes
        return bind.bind_scorecard(self.owner, self.root, manifest or self.manifests[attempt],
                                   self.card(attempt) if contents is None else contents, **arguments)

    def assert_no_scorecards(self):
        for attempt in (ATTEMPT_A, ATTEMPT_B):
            folder = self.run_root / f"attempts/{attempt}/review"
            self.assertFalse((folder / "scorecards").exists())
            self.assertFalse((folder / "scorecard-bindings").exists())

    def replace_manifest(self, data):
        old = self.manifests[ATTEMPT_A]
        contents = json.dumps(data, sort_keys=True).encode()
        (self.run_root / old.path).write_bytes(contents)
        return StoredArtifact(old.path, sha256(contents).hexdigest(), len(contents))

    def replace_metadata(self, data):
        old = self.fixture.completed[ATTEMPT_A].metadata
        contents = json.dumps(data, sort_keys=True).encode()
        (self.run_root / old.path).write_bytes(contents)
        record = StoredArtifact(old.path, sha256(contents).hexdigest(), len(contents))
        with self.ledger._connection() as connection:
            connection.execute("UPDATE completed_outputs SET metadata_sha256=?, metadata_size=? WHERE attempt_id=?",
                               (record.sha256, record.size, ATTEMPT_A))
        return self.replace_manifest(self.original_manifest | {"rawMetadata": asdict(record)})

    def test_real_mode_a_and_b_bind_exact_bytes_and_keep_global_outcomes_not_run(self):
        for attempt, mode in ((ATTEMPT_A, "A"), (ATTEMPT_B, "B")):
            with self.subTest(mode=mode):
                contents = self.card(attempt)
                manifest_before = read_verified(self.run_root, self.manifests[attempt])
                result = self.build(contents, attempt=attempt)
                receipt = self.data(result)
                card_record = StoredArtifact(**receipt["scorecard"])
                completed = self.fixture.completed[attempt]
                self.assertEqual(read_verified(self.run_root, card_record), contents)
                self.assertEqual(card_record.path,
                    f"attempts/{attempt}/review/scorecards/{sha256(contents).hexdigest()}.json")
                self.assertEqual(result.path,
                    f"attempts/{attempt}/review/scorecard-bindings/{result.sha256}.json")
                self.assertEqual((receipt["schemaVersion"], receipt["manifestType"]), ("1.4", "bound_review_scorecard"))
                self.assertEqual((receipt["runId"], receipt["attemptId"], receipt["mode"]), (RUN, attempt, mode))
                self.assertEqual(receipt["reviewManifest"], asdict(self.manifests[attempt]))
                self.assertEqual(receipt["providerTaskId"], completed.provider_task_id)
                self.assertEqual(receipt["rawOutput"], asdict(completed.raw))
                self.assertEqual(receipt["rawMetadata"], asdict(completed.metadata))
                self.assertEqual(receipt["providerSuccess"], self.data(self.manifests[attempt])["providerSuccess"])
                self.assertEqual((receipt["completedUnix"], receipt["reviewedAtUtc"]),
                                 (1001.0, "1970-01-01T00:30:00Z"))
                self.assertEqual((receipt["rubricVerdict"], receipt["qualifyingFailureIndexes"]), ("PASS", []))
                self.assertEqual(receipt["visualEffectDemonstrated"], "NOT_RUN")
                self.assertEqual(receipt["automatedWorkflowComplete"], "NOT_RUN")
                self.assertEqual(read_verified(self.run_root, self.manifests[attempt]), manifest_before)
                self.assertNotIn("signature=", json.dumps(receipt))

    def test_native_decimal_endpoints_bind_and_preserve_a_failing_rubric_result(self):
        failure = rubric_fixtures.event(start=0.033333, end=0.333333)
        contents = self.card(hardFailures=[failure])
        with localcontext() as context:
            context.prec = 1
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            receipt = self.data(self.build(contents))
        self.assertEqual((receipt["rubricVerdict"], receipt["qualifyingFailureIndexes"]), ("FAIL", [0]))
        self.assertEqual(read_verified(self.run_root, StoredArtifact(**receipt["scorecard"])), contents)
        self.assertEqual(receipt["visualEffectDemonstrated"], "NOT_RUN")

    def test_non_native_endpoints_and_inferred_terminal_end_are_rejected(self):
        cases = [(0.033333001, 0.333333), (0.033333, 0.333333001), (4.966667, 5.0)]
        self.assertEqual(self.original_metadata["durationSeconds"], 5.0)
        self.assertLess(self.original_metadata["frameSeconds"][-1], 5.0)
        for start, end in cases:
            with self.subTest(start=start, end=end), self.assertRaises(bind.ScorecardBindingError):
                self.build(self.card(hardFailures=[rubric_fixtures.event(start=start, end=end)]))
            self.assert_no_scorecards()

    def test_unsigned_malformed_and_foreign_scorecards_publish_nothing(self):
        cards = [self.card(reviewerName=""), self.card(signature=""), self.card(reviewedAtUtc=""),
                 self.card(attempt=ATTEMPT_B), self.card(runId="other-run"), b"{malformed"]
        for contents in cards:
            with self.subTest(contents=contents[:80]), self.assertRaises(bind.ScorecardBindingError):
                self.build(contents)
            self.assert_no_scorecards()
        with self.assertRaises(bind.ScorecardBindingError):
            bind.bind_scorecard(self.owner, self.root, self.manifests[ATTEMPT_A], None,
                                run_id=RUN, attempt_id=ATTEMPT_A)
        self.assert_no_scorecards()

    def test_completion_and_current_time_are_inclusive_boundaries(self):
        for timestamp in ("1970-01-01T00:16:41Z", "1970-01-01T01:23:20+00:00"):
            with self.subTest(timestamp=timestamp):
                self.assertEqual(self.data(self.build(self.card(reviewedAtUtc=timestamp)))["rubricVerdict"], "PASS")

    def test_review_before_completion_or_after_now_publishes_nothing(self):
        for timestamp in ("1970-01-01T00:16:40.999999Z", "1970-01-01T01:23:20.000001Z"):
            with self.subTest(timestamp=timestamp), self.assertRaises(bind.ScorecardBindingError):
                self.build(self.card(reviewedAtUtc=timestamp))
            self.assert_no_scorecards()

    def test_fractional_chronology_ignores_ambient_decimal_settings(self):
        with self.ledger._connection() as connection:
            connection.execute("UPDATE completed_outputs SET completed_unix=? WHERE attempt_id=?",
                               (1001.000001, ATTEMPT_A))
        self.ledger.clock = lambda: 1001.000001
        original = DefaultContext.copy()
        try:
            DefaultContext.Emin, DefaultContext.Emax, DefaultContext.clamp = -2, 2, 1
            with localcontext() as context:
                context.prec = 1
                context.traps[Inexact] = True
                context.traps[Rounded] = True
                receipt = self.data(self.build(self.card(reviewedAtUtc="1970-01-01T00:16:41.000001Z")))
                self.assertEqual(receipt["reviewedAtUtc"], "1970-01-01T00:16:41.000001Z")
                with self.assertRaises(bind.ScorecardBindingError):
                    self.build(self.card(reviewedAtUtc="1970-01-01T00:16:41.000002Z"))
        finally:
            DefaultContext.Emin, DefaultContext.Emax, DefaultContext.clamp = original.Emin, original.Emax, original.clamp

    def test_invalid_clock_and_completion_time_publish_nothing(self):
        for value in (True, float("nan"), float("inf"), -1, 253402300800):
            self.ledger.clock = lambda value=value: value
            with self.subTest(clock=value), self.assertRaises(bind.ScorecardBindingError):
                self.build()
            self.assert_no_scorecards()
        self.ledger.clock = lambda: 5000.0
        for value in (-1, float("inf"), 253402300800, "invalid"):
            with self.ledger._connection() as connection:
                connection.execute("UPDATE completed_outputs SET completed_unix=? WHERE attempt_id=?", (value, ATTEMPT_A))
            with self.subTest(completed=value), self.assertRaises(bind.ScorecardBindingError):
                self.build()
            self.assert_no_scorecards()

    def test_mixed_manifest_reference_identity_and_path_publish_nothing(self):
        bad_refs = [self.manifests[ATTEMPT_B], replace(self.manifests[ATTEMPT_A], sha256="f" * 64),
                    replace(self.manifests[ATTEMPT_A], path="../manifest.json")]
        for record in bad_refs:
            with self.subTest(record=record), self.assertRaises(bind.ScorecardBindingError):
                self.build(manifest=record)
            self.assert_no_scorecards()
        for name, value in {"runId": "other-run", "attemptId": ATTEMPT_B, "mode": "C",
                            "sourceKind": "unknown", "visualEffectDemonstrated": "PASS"}.items():
            record = self.replace_manifest(self.original_manifest | {name: value})
            with self.subTest(field=name), self.assertRaises(bind.ScorecardBindingError):
                self.build(manifest=record)
            self.assert_no_scorecards()

    def test_manifest_must_match_current_task_raw_and_metadata_records(self):
        other = self.fixture.completed[ATTEMPT_B]
        changes = [{"providerTaskId": other.provider_task_id}, {"rawOutput": asdict(other.raw)},
                   {"rawMetadata": asdict(other.metadata)},
                   {"providerSuccess": self.data(self.manifests[ATTEMPT_B])["providerSuccess"]}]
        for change in changes:
            record = self.replace_manifest(self.original_manifest | change)
            with self.subTest(change=change), self.assertRaises(bind.ScorecardBindingError):
                self.build(manifest=record)
            self.assert_no_scorecards()

    def test_all_displayed_files_are_verified_before_publication(self):
        manifest = self.original_manifest
        records = [manifest["sourceOriginal"], manifest["sourceNormalized"], manifest["guide"]["artifact"],
                   manifest["rawOutput"], manifest["samples"][3]["source"]["image"],
                   manifest["samples"][4]["raw"]["image"]]
        for data in records:
            path = self.run_root / data["path"]
            original = path.read_bytes()
            path.write_bytes(b"changed visible artifact")
            with self.subTest(path=data["path"]), self.assertRaises(bind.ScorecardBindingError):
                self.build()
            self.assert_no_scorecards()
            path.write_bytes(original)

    def test_missing_completion_changed_state_and_success_evidence_publish_nothing(self):
        success = self.original_manifest["providerSuccess"]
        path = self.run_root / success["path"]
        original = path.read_bytes()
        path.write_bytes(b"changed private provider evidence")
        with self.assertRaises(bind.ScorecardBindingError):
            self.build()
        self.assert_no_scorecards()
        path.write_bytes(original)
        with self.ledger._connection() as connection:
            connection.execute("UPDATE submission_jobs SET state='RUNNING' WHERE attempt_id=?", (ATTEMPT_A,))
        with self.assertRaises(bind.ScorecardBindingError):
            self.build()
        self.assert_no_scorecards()
        with self.ledger._connection() as connection:
            connection.execute("DELETE FROM completed_outputs WHERE attempt_id=?", (ATTEMPT_A,))
        with self.assertRaises(bind.ScorecardBindingError):
            self.build()
        self.assert_no_scorecards()

    def test_invalid_native_metadata_is_rejected_even_with_matching_hashes(self):
        changes = [{"durationSeconds": value} for value in (True, 0, 21)]
        changes += [{"frameSeconds": value} for value in ([], [0, 0], [0.1, 0], [-0.1, 0],
                                                           [0, True], [0, 5.1], [0] * 2401)]
        changes += [{"runId": "other-run"}, {"providerTaskId": "different-task"}, {"width": True}]
        for change in changes:
            record = self.replace_metadata(self.original_metadata | change)
            with self.subTest(change=repr(change)[:100]), self.assertRaises(bind.ScorecardBindingError):
                self.build(manifest=record)
            self.assert_no_scorecards()

    def test_repeat_is_stable_and_a_correction_preserves_both_explicit_bindings(self):
        original_card = self.card()
        first = self.build(original_card)
        first_bytes = read_verified(self.run_root, first)
        self.ledger.clock = lambda: 9000.0
        self.assertEqual(self.build(original_card), first)
        corrected_card = self.card(scores={"identity": 2, "motion": 3, "scene": 3, "placement": 3},
                                   comment="Corrected fixture score; no live visual acceptance.")
        second = self.build(corrected_card)
        self.assertNotEqual(first, second)
        self.assertEqual(self.data(second)["rubricVerdict"], "FAIL")
        self.assertEqual(read_verified(self.run_root, first), first_bytes)
        self.assertEqual(read_verified(self.run_root, StoredArtifact(**self.data(first)["scorecard"])), original_card)
        self.assertEqual(read_verified(self.run_root, StoredArtifact(**self.data(second)["scorecard"])), corrected_card)
        folder = self.run_root / f"attempts/{ATTEMPT_A}/review"
        self.assertEqual(sorted(path.name for path in folder.iterdir() if path.name.startswith("scorecard")),
                         ["scorecard-bindings", "scorecards"])
        self.assertEqual(len(list((folder / "scorecard-bindings").iterdir())), 2)

    def test_interrupted_receipt_publication_resumes_without_replacing_the_card(self):
        contents = self.card()
        publish = bind._publish_or_verify

        def stop_before_receipt(root, path, data):
            if "/scorecard-bindings/" in path:
                raise OSError("private fixture interruption")
            return publish(root, path, data)

        with patch.object(bind, "_publish_or_verify", side_effect=stop_before_receipt):
            with self.assertRaises(bind.ScorecardBindingError) as raised:
                self.build(contents)
        self.assertNotIn("private fixture", str(raised.exception))
        folder = self.run_root / f"attempts/{ATTEMPT_A}/review"
        card_path = folder / f"scorecards/{sha256(contents).hexdigest()}.json"
        self.assertEqual(card_path.read_bytes(), contents)
        self.assertFalse((folder / "scorecard-bindings").exists())
        before = card_path.stat().st_ino
        result = self.build(contents)
        self.assertEqual(card_path.stat().st_ino, before)
        self.assertEqual(self.build(contents), result)

    def test_changed_published_card_or_receipt_is_never_overwritten(self):
        result = self.build()
        card = StoredArtifact(**self.data(result)["scorecard"])
        for record in (card, result):
            path = self.run_root / record.path
            original = path.read_bytes()
            path.write_bytes(b"changed immutable record")
            with self.subTest(path=record.path), self.assertRaises(bind.ScorecardBindingError):
                self.build()
            self.assertEqual(path.read_bytes(), b"changed immutable record")
            path.write_bytes(original)

    def test_closed_coordinator_cannot_publish_a_binding(self):
        self.owner.close()
        with self.assertRaises(bind.ScorecardBindingError):
            self.build()
        self.assert_no_scorecards()


if __name__ == "__main__":
    unittest.main()
