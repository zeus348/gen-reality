"""Verify selected original scorecards and independent archived receipt fixtures."""

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, asdict, replace
from decimal import DefaultContext, Inexact, Rounded, localcontext
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import httpx

from core.budget import BudgetLedger
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference.submission import SubmissionCoordinator
from features.review import bind, integrity, preparation, review_records, score, scorecard_records
import test_archive_integrity as archive_fixtures
import test_archive_review as fixtures
import test_scorecard as rubric_fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B
_VERIFY_ARCHIVE = integrity.verify_export_archive
_VERIFY_REVIEW = review_records.verify_export_review
_ASSESS = score.assess_scorecard


class ArchiveScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        review_fixture = type("ScorecardReviewFixture", (fixtures.ArchiveReviewTests,), {})
        review_fixture.setUpClass()
        cls.addClassCleanup(review_fixture.doClassCleanups)
        cls.fixture = review_fixture
        cls.archive_bytes, cls.inventories = review_fixture.archive_bytes, review_fixture.inventories
        cls.saved_members = review_fixture.saved_members
        preparation_fixture = review_fixture.fixture
        integrity_fixture = preparation_fixture.fixture
        archive_fixture = integrity_fixture.fixture
        cls.base = archive_fixture.base

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.run_root = self.root / RUN
        self.contents = self.archive_bytes[ATTEMPT_A, True]
        self.archive = self.save_archive(self.contents)
        self.inventory = self.inventories[ATTEMPT_A, True]
        self.members = self.saved_members[ATTEMPT_A, True]
        self.original = json.loads(self.members["inventory.json"])
        self.review = StoredArtifact(**self.original["selection"]["reviewManifest"])
        self.review_data = json.loads(self.members[self.review.path])
        self.card = StoredArtifact(**self.original["selection"]["scorecard"])
        self.binding = StoredArtifact(**self.original["selection"]["scorecardBinding"])
        self.card_data = json.loads(self.members[self.card.path])

    def save_archive(self, contents, *, attempt=ATTEMPT_A):
        digest = sha256(contents).hexdigest()
        path = f"attempts/{attempt}/export/archives/{digest}.zip"
        if (self.run_root / path).exists():
            return StoredArtifact(path, digest, len(contents))
        return publish_bytes(self.run_root, path, contents)

    def verify(self, candidate=None, *, attempt=ATTEMPT_A):
        archive = self.archive if candidate is None else candidate.archive
        inventory = self.inventory if candidate is None else candidate.inventory
        return scorecard_records.verify_export_scorecard(self.root, archive, inventory,
            run_id=RUN, attempt_id=attempt)

    @staticmethod
    def receipt_bytes(data):
        return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()

    def candidate(self, *, change_card=None, card_contents=None, assessment=None, change_binding=None,
                  binding_contents=None, completion=None, change_review=None, metadata_contents=None,
                  check_review=True):
        members, inventory = dict(self.members), deepcopy(self.original)
        review = deepcopy(self.review_data)

        def replace_member(old_path, path, contents):
            record = StoredArtifact(path, sha256(contents).hexdigest(), len(contents))
            members.pop(old_path)
            members[path] = contents
            next(entry for entry in inventory["files"] if entry["path"] == old_path).update(asdict(record))
            return record

        review_ref = self.review
        if metadata_contents is not None:
            metadata_path = review["rawMetadata"]["path"]
            metadata_ref = replace_member(metadata_path, metadata_path, metadata_contents)
            review["rawMetadata"] = asdict(metadata_ref)
        if change_review is not None:
            change_review(review)
        if change_review is not None or metadata_contents is not None:
            review_ref = replace_member(self.review.path, self.review.path, json.dumps(review, sort_keys=True).encode())
            inventory["selection"]["reviewManifest"] = asdict(review_ref)
        if completion is not None:
            inventory["provider"]["completedUnix"] = completion
        card = deepcopy(self.card_data)
        if change_card is not None:
            change_card(card)
        contents = self.members[self.card.path] if change_card is None else json.dumps(card, sort_keys=True).encode()
        if card_contents is not None:
            contents = card_contents
            try:
                parsed = json.loads(contents)
                card = parsed if type(parsed) is dict else {}
            except (ValueError, UnicodeError):
                card = {}
        assessment = _ASSESS(contents, run_id=RUN, attempt_id=ATTEMPT_A) if assessment is None else assessment
        digest = sha256(contents).hexdigest()
        card_ref = replace_member(self.card.path, f"attempts/{ATTEMPT_A}/review/scorecards/{digest}.json", contents)
        # This fixture describes the receipt independently of the shared writer helper.
        receipt = {
            "schemaVersion": "1.4", "manifestType": "bound_review_scorecard",
            "runId": card.get("runId", RUN), "attemptId": card.get("attemptId", ATTEMPT_A), "mode": review["mode"],
            "providerTaskId": review["providerTaskId"], "reviewManifest": asdict(review_ref), "scorecard": asdict(card_ref),
            "rawOutput": review["rawOutput"], "rawMetadata": review["rawMetadata"], "providerSuccess": review["providerSuccess"],
            "completedUnix": inventory["provider"]["completedUnix"],
            "reviewedAtUtc": card.get("reviewedAtUtc", self.card_data["reviewedAtUtc"]),
            "rubricVerdict": assessment.rubric_verdict, "qualifyingFailureIndexes": list(assessment.qualifying_failure_indexes),
            "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
        }
        if change_binding is not None:
            change_binding(receipt)
        body = self.receipt_bytes(receipt) if binding_contents is None else binding_contents(receipt)
        digest = sha256(body).hexdigest()
        binding_ref = replace_member(self.binding.path, f"attempts/{ATTEMPT_A}/review/scorecard-bindings/{digest}.json", body)
        inventory["selection"].update(scorecard=asdict(card_ref), scorecardBinding=asdict(binding_ref))
        inventory["files"].sort(key=lambda entry: entry["path"])
        inventory["totalFileBytes"] = sum(entry["size"] for entry in inventory["files"])
        inventory_bytes = json.dumps(inventory, sort_keys=True).encode()
        digest = sha256(inventory_bytes).hexdigest()
        inventory_ref = StoredArtifact(f"attempts/{ATTEMPT_A}/export/inventories/{digest}.json", digest, len(inventory_bytes))
        members["inventory.json"] = inventory_bytes
        ordered = [("inventory.json", inventory_bytes), *[(path, members[path]) for path in sorted(members)
                                                        if path != "inventory.json"]]
        archive_ref = self.save_archive(archive_fixtures.ArchiveIntegrityTests.pack(ordered))
        self.assertEqual(_VERIFY_ARCHIVE(self.root, archive_ref, inventory_ref,
            run_id=RUN, attempt_id=ATTEMPT_A).archive, archive_ref)
        if check_review:
            self.assertEqual(_VERIFY_REVIEW(self.root, archive_ref, inventory_ref,
                run_id=RUN, attempt_id=ATTEMPT_A).review, review_ref)
        return SimpleNamespace(archive=archive_ref, inventory=inventory_ref, members=members,
            review=review_ref, binding=binding_ref, card=card_ref, receipt=receipt, assessment=assessment)

    def reject(self, candidate):
        with self.assertRaises(scorecard_records.ExportScorecardError):
            self.verify(candidate)

    def test_real_modes_and_selections_return_exact_frozen_results_without_external_work(self):
        original_open = os.open

        def readonly_open(path, flags, *args, **kwargs):
            self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            return original_open(path, flags, *args, **kwargs)

        for key, contents in self.archive_bytes.items():
            attempt, selected = key
            with self.subTest(attempt=attempt, selected=selected):
                record = self.save_archive(contents, attempt=attempt)
                inventory = self.inventories[key]
                members = self.saved_members[key]
                data = json.loads(members["inventory.json"])
                selection = data["selection"]
                review_ref = StoredArtifact(**selection["reviewManifest"])
                review = json.loads(members[review_ref.path])
                archive_result = integrity.ExportArchiveIntegrity(record, inventory, len(data["files"]), data["totalFileBytes"])
                prepared = preparation.ExportPreparationIntegrity(archive_result,
                    StoredArtifact(**selection["preparationManifest"]), "synthetic_fixture")
                reviewed = review_records.ExportReviewIntegrity(prepared, review_ref, StoredArtifact(**review["rawMetadata"]))
                with patch.object(os, "open", side_effect=readonly_open), \
                        patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No ledger")), \
                        patch.object(SubmissionCoordinator, "__init__", side_effect=AssertionError("No coordinator")), \
                        patch.object(httpx.Client, "send", side_effect=AssertionError("No HTTP")), \
                        patch.object(socket, "socket", side_effect=AssertionError("No network")), \
                        patch.object(time, "time", side_effect=AssertionError("No clock")), \
                        patch.object(subprocess, "run", side_effect=AssertionError("No decoding")), \
                        patch.object(ZipFile, "extractall", side_effect=AssertionError("No extraction")), \
                        patch.object(ZipFile, "extract", side_effect=AssertionError("No extraction")), \
                        patch.object(Path, "write_bytes", side_effect=AssertionError("No writes")), \
                        patch.object(Path, "glob", side_effect=AssertionError("No latest selection")), \
                        patch.object(Path, "rglob", side_effect=AssertionError("No scan")):
                    result = self.verify(SimpleNamespace(archive=record, inventory=inventory), attempt=attempt)
                expected = scorecard_records.ExportScorecardIntegrity(reviewed,
                    "SELECTED" if selected else "NOT_SELECTED",
                    StoredArtifact(**selection["scorecardBinding"]) if selected else None,
                    StoredArtifact(**selection["scorecard"]) if selected else None,
                    score.ScorecardAssessment("PASS", ()) if selected else None)
                self.assertEqual(result, expected)
                with self.assertRaises(FrozenInstanceError):
                    result.selection_status = "SELECTED"
                self.assertFalse(hasattr(result, "visualEffectDemonstrated"))
                self.assertFalse(hasattr(result, "automatedWorkflowComplete"))

    def test_declared_rubric_failure_is_valid_selected_evidence(self):
        low_score = self.candidate(change_card=lambda data: data["scores"].update(identity=2))
        self.assertEqual(self.verify(low_score).assessment, score.ScorecardAssessment("FAIL", ()))
        failure = rubric_fixtures.event(start=0.033333, end=0.333333)
        failed_interval = self.candidate(change_card=lambda data: data.update(hardFailures=[failure]))
        result = self.verify(failed_interval)
        self.assertEqual(result.assessment, score.ScorecardAssessment("FAIL", (0,)))
        self.assertEqual(result.selection_status, "SELECTED")

    def test_no_selection_does_not_assess_or_choose_an_available_card(self):
        record = self.save_archive(self.archive_bytes[ATTEMPT_A, False])
        inventory = self.inventories[ATTEMPT_A, False]
        publish_bytes(self.run_root, self.card.path, self.members[self.card.path])
        publish_bytes(self.run_root, self.binding.path, self.members[self.binding.path])
        with patch.object(scorecard_records, "_assess_bound_card") as assessor:
            result = self.verify(SimpleNamespace(archive=record, inventory=inventory))
            assessor.assert_not_called()
        self.assertEqual((result.selection_status, result.binding, result.scorecard, result.assessment),
                         ("NOT_SELECTED", None, None, None))

    def test_original_noncanonical_card_bytes_are_preserved_and_bound_exactly(self):
        body = b" \n" + json.dumps(self.card_data, separators=(",", ":")).encode() + b"\n\t"
        candidate = self.candidate(card_contents=body)
        result = self.verify(candidate)
        self.assertEqual(result.scorecard, candidate.card)
        self.assertEqual(result.scorecard.sha256, sha256(body).hexdigest())
        with ZipFile(BytesIO(read_verified(self.run_root, candidate.archive))) as saved:
            self.assertEqual(saved.read(candidate.card.path), body)
        changed = self.candidate(card_contents=body,
            change_binding=lambda receipt: receipt.update(scorecard=asdict(self.card)))
        self.reject(changed)

    def test_selected_cards_require_valid_identity_schema_and_complete_signoff(self):
        for field in ("signature", "reviewerName", "reviewedAtUtc"):
            with self.subTest(unsigned=field):
                candidate = self.candidate(change_card=lambda data: data.update({field: ""}))
                self.assertEqual(candidate.receipt["rubricVerdict"], "NOT_RUN")
                self.reject(candidate)
        missing_signature = self.candidate(change_card=lambda data: data.pop("signature"))
        self.assertEqual(missing_signature.receipt["rubricVerdict"], "NOT_RUN")
        self.reject(missing_signature)
        assumed = score.ScorecardAssessment("PASS", ())
        changes = [lambda data: data.update(runId="other-run"), lambda data: data.update(attemptId=ATTEMPT_B),
            lambda data: data.update(normalSpeedPlays=2.0), lambda data: data["scores"].update(identity=True),
            lambda data: data.update(visualEffectDemonstrated="PASS")]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.reject(self.candidate(change_card=change, assessment=assumed))
        body = self.members[self.card.path]
        for contents in (b"not JSON", b"[]", b"\xff", b'{"schemaVersion":"1.4",' + body[1:]):
            with self.subTest(json=contents[:30]):
                self.reject(self.candidate(card_contents=contents, assessment=assumed))

    def test_receipt_identity_links_chronology_fields_and_global_flags_are_exact(self):
        fields = {"schemaVersion": "other", "manifestType": "other", "runId": "other-run", "attemptId": ATTEMPT_B,
            "mode": "B", "providerTaskId": "other-task", "completedUnix": 1000.0,
            "reviewedAtUtc": "1970-01-01T00:31:00Z", "visualEffectDemonstrated": "PASS", "automatedWorkflowComplete": "PASS"}
        for field, value in fields.items():
            with self.subTest(field=field):
                self.reject(self.candidate(change_binding=lambda data: data.update({field: value})))
        for field in ("reviewManifest", "scorecard", "rawOutput", "rawMetadata", "providerSuccess"):
            with self.subTest(reference=field):
                self.reject(self.candidate(change_binding=lambda data: data[field].update(sha256="f" * 64)))
        self.reject(self.candidate(change_binding=lambda data: data.update(unexpected="private-token")))

    def test_receipt_rubric_and_qualifying_indexes_are_recomputed(self):
        self.reject(self.candidate(change_binding=lambda data: data.update(rubricVerdict="FAIL")))
        self.reject(self.candidate(change_binding=lambda data: data.update(qualifyingFailureIndexes=[0])))
        failure = rubric_fixtures.event(start=0.033333, end=0.333333)
        for change in (lambda data: data.update(rubricVerdict="PASS"),
                       lambda data: data.update(qualifyingFailureIndexes=[]),
                       lambda data: data.update(qualifyingFailureIndexes=[0.0])):
            self.reject(self.candidate(change_card=lambda data: data.update(hardFailures=[failure]), change_binding=change))

    def test_receipt_requires_complete_canonical_json_bytes(self):
        formats = [lambda data: json.dumps(data, separators=(",", ":")).encode(),
            lambda data: b" " + self.receipt_bytes(data), lambda data: self.receipt_bytes(data).rstrip(b"\n"),
            lambda data: b'{"schemaVersion":"1.4",' + self.receipt_bytes(data)[1:],
            lambda data: b"not JSON"]
        for index, formatter in enumerate(formats):
            with self.subTest(case=index):
                self.reject(self.candidate(binding_contents=formatter))

    def test_native_endpoints_use_exact_decimal_tokens_without_terminal_inference(self):
        failure = rubric_fixtures.event(start=0.033333, end=0.333333)
        candidate = self.candidate(change_card=lambda data: data.update(hardFailures=[failure]))
        self.assertEqual(self.verify(candidate).assessment, score.ScorecardAssessment("FAIL", (0,)))
        for start, end in ((0.033333001, 0.333333), (0.033333, 0.333333001), (4.966667, 5.0)):
            failure = rubric_fixtures.event(start=start, end=end)
            with self.subTest(start=start, end=end):
                self.reject(self.candidate(change_card=lambda data: data.update(hardFailures=[failure])))
        original = self.members[self.review_data["rawMetadata"]["path"]]
        different_decimal = original.replace(b"0.3,", b"0.3000000000000000000001,", 1)
        self.assertNotEqual(original, different_decimal)
        self.assertEqual(json.loads(original), json.loads(different_decimal))
        failure = rubric_fixtures.event(start=0.3, end=0.6)
        self.reject(self.candidate(metadata_contents=different_decimal,
            change_card=lambda data: data.update(hardFailures=[failure])))

    def test_chronology_uses_exact_completion_boundary_under_hostile_decimal_contexts(self):
        exact = self.candidate(completion=1001.000001,
            change_card=lambda data: data.update(reviewedAtUtc="1970-01-01T00:16:41.000001Z"))
        early = self.candidate(completion=1001.000001,
            change_card=lambda data: data.update(reviewedAtUtc="1970-01-01T00:16:41.000000Z"))
        future = self.candidate(change_card=lambda data: data.update(reviewedAtUtc="2099-01-01T00:00:00Z"))
        saved_default = DefaultContext.copy()
        try:
            DefaultContext.prec, DefaultContext.Emin, DefaultContext.Emax = 1, -2, 2
            DefaultContext.traps[Inexact] = DefaultContext.traps[Rounded] = True
            with localcontext() as context:
                context.prec, context.Emin, context.Emax = 1, -2, 2
                context.traps[Inexact] = context.traps[Rounded] = True
                self.assertEqual(self.verify(exact).scorecard, exact.card)
                self.reject(early)
                self.assertEqual(self.verify(future).assessment, score.ScorecardAssessment("PASS", ()))
        finally:
            DefaultContext.prec, DefaultContext.Emin, DefaultContext.Emax = saved_default.prec, saved_default.Emin, saved_default.Emax
            DefaultContext.traps.update(saved_default.traps)

    def test_selected_card_and_binding_limits_precede_retained_reads(self):
        body = self.members[self.card.path]
        exact_card = body + b" " * (65536 - len(body))
        candidate = self.candidate(card_contents=exact_card)
        self.assertEqual(self.verify(candidate).scorecard, candidate.card)
        oversized_card = self.candidate(card_contents=exact_card + b" ", assessment=score.ScorecardAssessment("PASS", ()))
        exact_binding = self.candidate(binding_contents=lambda data: self.receipt_bytes(data) +
            b" " * (65536 - len(self.receipt_bytes(data))))
        oversized_binding = self.candidate(binding_contents=lambda data: self.receipt_bytes(data) +
            b" " * (65537 - len(self.receipt_bytes(data))))
        for candidate, path, expected_retained in ((oversized_card, oversized_card.card.path, False),
                (exact_binding, exact_binding.binding.path, True), (oversized_binding, oversized_binding.binding.path, False)):
            retained = []
            original_member = integrity._read_member

            def track_member(saved, member, record, *, retain=False):
                if retain:
                    retained.append(member.filename)
                return original_member(saved, member, record, retain=retain)

            with self.subTest(path=path, retained=expected_retained), patch.object(integrity, "_read_member", side_effect=track_member):
                self.reject(candidate)
            self.assertEqual(path in retained, expected_retained)

    def test_one_verified_snapshot_retains_only_the_explicitly_selected_controls(self):
        for selected in (False, True):
            contents = self.archive_bytes[ATTEMPT_A, selected]
            record = self.save_archive(contents)
            inventory = self.inventories[ATTEMPT_A, selected]
            reads, retained = [], []
            original_read, original_member = integrity.read_verified, integrity._read_member

            def change_after_read(root, artifact):
                reads.append(artifact)
                body = original_read(root, artifact)
                (Path(root) / artifact.path).write_bytes(b"changed after the verified snapshot")
                return body

            def track_member(saved, member, artifact, *, retain=False):
                if retain:
                    retained.append(member.filename)
                return original_member(saved, member, artifact, retain=retain)

            with self.subTest(selected=selected), patch.object(integrity, "read_verified", side_effect=change_after_read), \
                    patch.object(integrity, "_read_member", side_effect=track_member):
                result = self.verify(SimpleNamespace(archive=record, inventory=inventory))
            expected = {"inventory.json", *[f"attempts/{ATTEMPT_A}/" + name for name in
                ("manifest.json", "frozen/config.json", "frozen/placement.json", "frozen/inputs.json", "frozen/prompt.txt",
                 "review/manifest.json", "output/metadata.json")]}
            if selected:
                expected.update((self.card.path, self.binding.path))
            self.assertEqual(set(retained), expected)
            self.assertEqual(len(retained), 10 if selected else 8)
            self.assertEqual(reads, [record])
            self.assertEqual(result.selection_status, "SELECTED" if selected else "NOT_SELECTED")

    def test_complete_archive_and_review_checks_precede_selected_card_semantics(self):
        members = dict(self.members)
        path = sorted(name for name in members if "/review/raw/" in name)[-1]
        contents = members[path]
        members[path] = bytes([contents[0] ^ 1]) + contents[1:]
        record = self.save_archive(archive_fixtures.ArchiveIntegrityTests.pack(list(members.items())))
        with patch.object(scorecard_records, "_assess_bound_card") as assessor:
            self.reject(SimpleNamespace(archive=record, inventory=self.inventory))
            assessor.assert_not_called()
        candidate = self.candidate(change_review=lambda data: data.update(visualEffectDemonstrated="PASS"), check_review=False)
        with patch.object(scorecard_records, "_assess_bound_card") as assessor:
            self.reject(candidate)
            assessor.assert_not_called()

    def test_existing_filesystem_binding_reader_and_early_guards_are_preserved(self):
        candidate = self.candidate(change_card=lambda data: data["scores"].update(identity=2))
        with TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(self.base, root, dirs_exist_ok=True)
            for path in (candidate.card.path, candidate.binding.path):
                publish_bytes(root / RUN, path, candidate.members[path])
            ledger = BudgetLedger(root / "budget.sqlite", clock=lambda: 5000.0)
            with SubmissionCoordinator(ledger) as owner:
                bound = bind.read_scorecard_binding(owner, root, candidate.binding, run_id=RUN, attempt_id=ATTEMPT_A)
                self.assertEqual((bound.record, bound.scorecard, bound.review_manifest, bound.assessment),
                    (candidate.binding, candidate.card, self.review, score.ScorecardAssessment("FAIL", ())))
                foreign = replace(candidate.binding, path=f"attempts/{ATTEMPT_B}/review/scorecard-bindings/{candidate.binding.sha256}.json")
                with patch.object(bind, "_read_record") as reader:
                    with self.assertRaises(bind.ScorecardBindingError):
                        bind.read_scorecard_binding(owner, root, foreign, run_id=RUN, attempt_id=ATTEMPT_A)
                    reader.assert_not_called()
        unsigned = json.dumps(self.card_data | {"signature": ""}).encode()
        with patch.object(bind, "_verify_review") as reader:
            with self.assertRaises(bind.ScorecardBindingError):
                bind._binding_receipt(self.run_root, self.review, unsigned, None, run_id=RUN, attempt_id=ATTEMPT_A)
            reader.assert_not_called()
        for contents in (unsigned, b"not JSON"):
            with self.subTest(card=contents[:30]), patch.object(bind, "OutputCompleter") as completer:
                owner = SimpleNamespace(_mutex=nullcontext(), _require_open=lambda: None)
                with self.assertRaises(bind.ScorecardBindingError):
                    bind.bind_scorecard(owner, self.root, self.review, contents, run_id=RUN, attempt_id=ATTEMPT_A)
                completer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
