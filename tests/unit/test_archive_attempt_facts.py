"""Read exact attempt declarations from one complete export ZIP snapshot."""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import httpx
from PIL import Image

from core.budget import BudgetLedger
from core import storage
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference.submission import SubmissionCoordinator
from features.review import attempt_records, integrity, preparation, review_records, score, scorecard_records
import test_archive_integrity as archive_fixtures
import test_archive_review as review_fixtures
import test_archive_scorecard as fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B


class ArchiveAttemptFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse fixture builders without adding their test methods to this suite.
        cls.fixture_type = type("AttemptFactsFixture", (fixtures.ArchiveScorecardTests,), {})
        cls.fixture_type.setUpClass()
        cls.addClassCleanup(cls.fixture_type.doClassCleanups)

    def setUp(self):
        self.case = self.fixture_type("runTest")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def read(self, candidate=None, *, attempt=ATTEMPT_A, **changes):
        archive = self.case.archive if candidate is None else candidate.archive
        inventory = self.case.inventory if candidate is None else candidate.inventory
        arguments = {"run_id": RUN, "attempt_id": attempt} | changes
        return attempt_records.read_export_attempt_facts(self.case.root, archive, inventory, **arguments)

    def review_candidate(self, **changes):
        return review_fixtures.ArchiveReviewTests.candidate(self.case, **changes)

    def reject(self, candidate=None, **changes):
        with self.assertRaises(attempt_records.ExportAttemptFactsError) as caught:
            self.read(candidate, **changes)
        self.assertNotIn(str(self.case.root), str(caught.exception))
        self.assertNotIn("private-token", str(caught.exception))

    @staticmethod
    def expected_graph(archive, inventory, members):
        data = json.loads(members["inventory.json"])
        selection = data["selection"]
        review_ref = StoredArtifact(**selection["reviewManifest"])
        review = json.loads(members[review_ref.path])
        archived = integrity.ExportArchiveIntegrity(archive, inventory, len(data["files"]), data["totalFileBytes"])
        prepared = preparation.ExportPreparationIntegrity(archived,
            StoredArtifact(**selection["preparationManifest"]), data["sourceKindDeclared"])
        reviewed = review_records.ExportReviewIntegrity(prepared, review_ref, StoredArtifact(**review["rawMetadata"]))
        selected = selection["scorecardBinding"] is not None
        return scorecard_records.ExportScorecardIntegrity(reviewed, "SELECTED" if selected else "NOT_SELECTED",
            StoredArtifact(**selection["scorecardBinding"]) if selected else None,
            StoredArtifact(**selection["scorecard"]) if selected else None,
            score.ScorecardAssessment("PASS", ()) if selected else None)

    def test_exact_facts_for_both_modes_and_selections_are_frozen(self):
        for key, contents in self.case.archive_bytes.items():
            attempt, selected = key
            with self.subTest(attempt=attempt, selected=selected):
                archive = self.case.save_archive(contents, attempt=attempt)
                inventory = self.case.inventories[key]
                members = self.case.saved_members[key]
                data = json.loads(members["inventory.json"])
                review = json.loads(members[data["selection"]["reviewManifest"]["path"]])
                provider = data["provider"]
                graph = self.expected_graph(archive, inventory, members)
                result = self.read(SimpleNamespace(archive=archive, inventory=inventory), attempt=attempt)
                self.assertIsInstance(result, attempt_records.ExportAttemptFacts)
                expected = {
                    "run_id": RUN, "attempt_id": attempt, "mode": data["mode"],
                    "source_kind_declared": data["sourceKindDeclared"], "sdk_version": provider["sdkVersion"],
                    "provider_task_id": provider["taskId"], "payload_sha256": provider["payloadSha256"],
                    "request_body_sha256": provider["requestBodySha256"],
                    "reserved_cents": provider["reservedCents"], "completed_unix": provider["completedUnix"],
                    "raw_output": review["rawOutput"], "request_check": review["requestCheck"],
                    "request": review["request"], "frozen_request": review["frozenRequest"],
                    "provider_success": review["providerSuccess"], "graph": asdict(graph),
                }
                self.assertEqual(asdict(result), expected)
                self.assertEqual(scorecard_records.verify_export_scorecard(self.case.root, archive, inventory,
                    run_id=RUN, attempt_id=attempt), graph)
                with self.assertRaises(FrozenInstanceError):
                    result.provider_task_id = "changed"
                for name in ("raw_output", "request_check", "request", "frozen_request", "provider_success"):
                    record = getattr(result, name)
                    self.assertIsInstance(record, StoredArtifact)
                    with self.assertRaises(FrozenInstanceError):
                        record.sha256 = "f" * 64
                with self.assertRaises(FrozenInstanceError):
                    result.graph.selection_status = "NOT_SELECTED"
                with self.assertRaises(FrozenInstanceError):
                    result.graph.review.review.path = "changed"
        low_score = self.case.candidate(change_card=lambda data: data["scores"].update(identity=2))
        result = self.read(low_score)
        self.assertEqual(result.graph.assessment, score.ScorecardAssessment("FAIL", ()))
        self.assertEqual(result.graph.assessment, low_score.assessment)
        self.assertEqual(result.graph.selection_status, "SELECTED")
        for field in ("visualEffectDemonstrated", "automatedWorkflowComplete", "releaseClassification"):
            self.assertNotIn(field, asdict(result))

    def test_absent_selection_ignores_available_external_scorecards(self):
        archive = self.case.save_archive(self.case.archive_bytes[ATTEMPT_A, False])
        inventory = self.case.inventories[ATTEMPT_A, False]
        publish_bytes(self.case.run_root, self.case.card.path, self.case.members[self.case.card.path])
        publish_bytes(self.case.run_root, self.case.binding.path, self.case.members[self.case.binding.path])
        with patch.object(scorecard_records, "_assess_bound_card", side_effect=AssertionError("No selected card")) as assessor:
            result = self.read(SimpleNamespace(archive=archive, inventory=inventory))
            assessor.assert_not_called()
        self.assertEqual((result.graph.selection_status, result.graph.binding,
                          result.graph.scorecard, result.graph.assessment), ("NOT_SELECTED", None, None, None))
        self.assertEqual(result.provider_task_id, self.case.original["provider"]["taskId"])

    def test_selected_unsigned_cards_and_invalid_bindings_are_rejected(self):
        candidates = [self.case.candidate(change_card=lambda data: data.update(signature="")),
                      self.case.candidate(change_card=lambda data: data.pop("signature")),
                      self.case.candidate(change_binding=lambda data: data.update(providerTaskId="other-task")),
                      self.case.candidate(change_binding=lambda data: data["rawOutput"].update(sha256="f" * 64))]
        for candidate in candidates:
            with self.subTest(card=candidate.card, binding=candidate.binding):
                self.reject(candidate)
                with self.assertRaises(scorecard_records.ExportScorecardError):
                    scorecard_records.verify_export_scorecard(self.case.root, candidate.archive, candidate.inventory,
                        run_id=RUN, attempt_id=ATTEMPT_A)

    def test_invalid_references_missing_and_corrupt_payloads_are_rejected_safely(self):
        for archive, inventory in [
                (replace(self.case.archive, size=True), self.case.inventory),
                (replace(self.case.archive, path="../private-token.zip"), self.case.inventory),
                (self.case.archive, replace(self.case.inventory, sha256="f" * 64))]:
            with self.subTest(archive=archive, inventory=inventory):
                self.reject(SimpleNamespace(archive=archive, inventory=inventory))
        self.reject(run_id="../private-token")
        self.reject(attempt=ATTEMPT_B)
        path = self.case.run_root / self.case.archive.path
        path.unlink()
        self.reject()
        path.write_bytes(self.case.contents[:-1] + b"X")
        self.reject()
        path.write_bytes(self.case.contents)
        raw_path = self.case.review_data["rawOutput"]["path"]
        for missing in (False, True):
            members = dict(self.case.members)
            original = members.pop(raw_path)
            if not missing:
                members[raw_path] = bytes([original[0] ^ 1]) + original[1:]
            ordered = [("inventory.json", members.pop("inventory.json")), *sorted(members.items())]
            archive = self.case.save_archive(archive_fixtures.ArchiveIntegrityTests.pack(ordered))
            with self.subTest(missing=missing):
                self.reject(SimpleNamespace(archive=archive, inventory=self.case.inventory))

    def test_task_raw_and_private_references_must_match_the_inventory(self):
        changes = [lambda data: data.update(providerTaskId="other-task"),
                   lambda data: data["rawOutput"].update(sha256="f" * 64)]
        changes.extend(lambda data, name=name: data[name].update(sha256="f" * 64)
                       for name in ("requestCheck", "request", "frozenRequest", "providerSuccess"))
        changes.append(lambda data: data["requestCheck"].update(path="private-token.json"))
        for index, change in enumerate(changes):
            candidate = self.review_candidate(change_review=change)
            with self.subTest(reference=index):
                self.reject(candidate)
        for field in ("payloadSha256", "requestBodySha256"):
            candidate = self.review_candidate(change_inventory=lambda data: data["provider"].update({field: "f" * 64}))
            with self.subTest(provider=field):
                self.reject(candidate)

    def test_preparation_and_review_checks_remain_required(self):
        candidates = [self.review_candidate(change_preparation=lambda data: data.update(state="SUCCEEDED"),
                                             check_preparation=False),
                      self.review_candidate(change_review=lambda data: data.update(mode="B")),
                      self.review_candidate(change_metadata=lambda data: data.update(width=True)),
                      self.review_candidate(change_metadata=lambda data: data.update(visualEffectDemonstrated="PASS"))]
        for index, candidate in enumerate(candidates):
            with self.subTest(case=index):
                self.reject(candidate)

    def test_zip_only_read_has_no_private_reads_writes_or_external_work(self):
        reads = []
        original_read, original_open = integrity.read_verified, os.open

        def track_read(root, record):
            reads.append(record)
            return original_read(root, record)

        def readonly_open(path, flags, *arguments, **options):
            self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            return original_open(path, flags, *arguments, **options)

        before = sorted(str(path.relative_to(self.case.root)) for path in self.case.root.rglob("*"))
        self.assertFalse((self.case.run_root / self.case.inventory.path).exists())
        for name in ("requestCheck", "request", "frozenRequest", "providerSuccess"):
            self.assertFalse((self.case.run_root / self.case.review_data[name]["path"]).exists())
        with patch.object(integrity, "read_verified", side_effect=track_read), \
                patch.object(os, "open", side_effect=readonly_open), \
                patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No ledger")), \
                patch.object(SubmissionCoordinator, "__init__", side_effect=AssertionError("No coordinator")), \
                patch.object(httpx.Client, "send", side_effect=AssertionError("No HTTP")), \
                patch.object(socket, "socket", side_effect=AssertionError("No network")), \
                patch.object(time, "time", side_effect=AssertionError("No clock")), \
                patch.object(subprocess, "run", side_effect=AssertionError("No decode")), \
                patch.object(Image, "open", side_effect=AssertionError("No image decode")), \
                patch.object(ZipFile, "extract", side_effect=AssertionError("No extraction")), \
                patch.object(ZipFile, "extractall", side_effect=AssertionError("No extraction")), \
                patch.object(storage, "publish_bytes", side_effect=AssertionError("No publication")), \
                patch.object(Path, "write_bytes", side_effect=AssertionError("No writes")), \
                patch.object(Path, "glob", side_effect=AssertionError("No scan")), \
                patch.object(Path, "rglob", side_effect=AssertionError("No scan")):
            result = self.read()
        self.assertEqual(result.raw_output, StoredArtifact(**self.case.review_data["rawOutput"]))
        self.assertEqual(reads, [self.case.archive])
        self.assertEqual(sorted(str(path.relative_to(self.case.root)) for path in self.case.root.rglob("*")), before)
        self.assertEqual(read_verified(self.case.run_root, self.case.archive), self.case.contents)

    def test_graph_and_projection_share_one_verified_snapshot(self):
        reads = []
        original_read = integrity.read_verified
        original_snapshot = attempt_records._read_verified_archive

        def change_after_read(root, record):
            reads.append(record)
            contents = original_read(root, record)
            (Path(root) / record.path).write_bytes(b"changed after the verified read")
            return contents

        with patch.object(integrity, "read_verified", side_effect=change_after_read), \
                patch.object(attempt_records, "_read_verified_archive", wraps=original_snapshot) as snapshot:
            result = self.read()
        snapshot.assert_called_once_with(self.case.root, self.case.archive, self.case.inventory,
            run_id=RUN, attempt_id=ATTEMPT_A, include_scorecard=True)
        self.assertEqual(reads, [self.case.archive])
        self.assertEqual(result.graph, self.expected_graph(self.case.archive, self.case.inventory, self.case.members))
        self.assertEqual(result.provider_task_id, self.case.original["provider"]["taskId"])
        self.assertEqual(result.raw_output, StoredArtifact(**self.case.review_data["rawOutput"]))
        self.assertEqual((self.case.run_root / self.case.archive.path).read_bytes(), b"changed after the verified read")


if __name__ == "__main__":
    unittest.main()
