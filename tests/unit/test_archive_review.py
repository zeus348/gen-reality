"""Check linked review declarations inside verified, independently repacked ZIPs."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import httpx
from PIL import Image

from core.budget import BudgetLedger
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference.submission import SubmissionCoordinator
from features.output import review as output_review, view
from features.output.complete import CompletedOutput
from features.review import integrity, preparation, review_records
import test_archive_integrity as archive_fixtures
import test_archive_preparation as fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B
# Corrupt review fixtures must pass the earlier, unchanged verification gates.
_VERIFY_ARCHIVE = integrity.verify_export_archive
_VERIFY_PREPARATION = preparation.verify_export_preparation


class ArchiveReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        preparation_fixture = type("ReviewPreparationFixture", (fixtures.ArchivePreparationTests,), {})
        preparation_fixture.setUpClass()
        cls.addClassCleanup(preparation_fixture.doClassCleanups)
        cls.fixture = preparation_fixture
        cls.archive_bytes = preparation_fixture.archive_bytes
        cls.inventories = preparation_fixture.inventories
        cls.saved_members = {}
        for key, contents in cls.archive_bytes.items():
            with ZipFile(BytesIO(contents)) as saved:
                cls.saved_members[key] = {info.filename: saved.read(info) for info in saved.infolist()}

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.run_root = self.root / RUN
        self.contents = self.archive_bytes[ATTEMPT_A, False]
        self.archive = self.save_archive(self.contents)
        self.inventory = self.inventories[ATTEMPT_A, False]
        self.prefix = f"attempts/{ATTEMPT_A}/"
        self.members = self.saved_members[ATTEMPT_A, False]
        self.original = json.loads(self.members["inventory.json"])
        self.review = StoredArtifact(**self.original["selection"]["reviewManifest"])
        self.review_data = json.loads(self.members[self.review.path])
        self.metadata = StoredArtifact(**self.review_data["rawMetadata"])
        self.metadata_data = json.loads(self.members[self.metadata.path])

    def save_archive(self, contents, *, attempt=ATTEMPT_A):
        digest = sha256(contents).hexdigest()
        path = f"attempts/{attempt}/export/archives/{digest}.zip"
        if (self.run_root / path).exists():
            return StoredArtifact(path, digest, len(contents))
        return publish_bytes(self.run_root, path, contents)

    def verify(self, candidate=None, *, attempt=ATTEMPT_A):
        archive = self.archive if candidate is None else candidate.archive
        inventory = self.inventory if candidate is None else candidate.inventory
        return review_records.verify_export_review(self.root, archive, inventory,
            run_id=RUN, attempt_id=attempt)

    def candidate(self, *, attempt=ATTEMPT_A, selected=False, change_review=None, change_metadata=None,
                  review_contents=None, metadata_contents=None, change_inventory=None,
                  rename_members=None, change_preparation=None, check_preparation=True):
        members = dict(self.saved_members[attempt, selected])
        inventory = json.loads(members["inventory.json"])
        review_path = inventory["selection"]["reviewManifest"]["path"]
        review = json.loads(members[review_path])
        metadata_path = review["rawMetadata"]["path"]
        metadata = json.loads(members[metadata_path])

        def replace_member(path, body):
            record = StoredArtifact(path, sha256(body).hexdigest(), len(body))
            members[path] = body
            next(entry for entry in inventory["files"] if entry["path"] == path).update(asdict(record))
            return record

        def rename_references(value, old, new):
            if isinstance(value, dict):
                if value.get("path") == old:
                    value["path"] = new
                for child in value.values():
                    rename_references(child, old, new)
            elif isinstance(value, list):
                for child in value:
                    rename_references(child, old, new)

        for old, new in (rename_members or {}).items():
            members[new] = members.pop(old)
            next(entry for entry in inventory["files"] if entry["path"] == old)["path"] = new
            rename_references(review, old, new)
        if change_preparation is not None:
            path = inventory["selection"]["preparationManifest"]["path"]
            data = json.loads(members[path])
            change_preparation(data)
            record = replace_member(path, json.dumps(data, sort_keys=True).encode())
            inventory["selection"]["preparationManifest"] = asdict(record)
            review["preparation"] = asdict(record)
        if change_metadata is not None:
            change_metadata(metadata)
        if change_metadata is not None or metadata_contents is not None:
            body = json.dumps(metadata, sort_keys=True).encode() if metadata_contents is None else metadata_contents
            review["rawMetadata"] = asdict(replace_member(metadata_path, body))
        if change_review is not None:
            change_review(review)
        body = json.dumps(review, sort_keys=True).encode() if review_contents is None else review_contents
        review_ref = replace_member(review_path, body)
        inventory["selection"]["reviewManifest"] = asdict(review_ref)
        inventory["files"].sort(key=lambda entry: entry["path"])
        inventory["totalFileBytes"] = sum(entry["size"] for entry in inventory["files"])
        if change_inventory is not None:
            change_inventory(inventory)
        inventory_bytes = json.dumps(inventory, sort_keys=True).encode()
        digest = sha256(inventory_bytes).hexdigest()
        inventory_ref = StoredArtifact(f"attempts/{attempt}/export/inventories/{digest}.json", digest, len(inventory_bytes))
        members["inventory.json"] = inventory_bytes
        ordered = [("inventory.json", inventory_bytes), *[(path, members[path]) for path in sorted(members)
                                                        if path != "inventory.json"]]
        archive_ref = self.save_archive(archive_fixtures.ArchiveIntegrityTests.pack(ordered), attempt=attempt)
        self.assertEqual(_VERIFY_ARCHIVE(self.root, archive_ref, inventory_ref,
            run_id=RUN, attempt_id=attempt).archive, archive_ref)
        if check_preparation:
            self.assertEqual(_VERIFY_PREPARATION(self.root, archive_ref, inventory_ref,
                run_id=RUN, attempt_id=attempt).integrity.archive, archive_ref)
        metadata_entry = next(entry for entry in inventory["files"] if entry["path"] == metadata_path)
        return SimpleNamespace(archive=archive_ref, inventory=inventory_ref, members=members, review=review_ref,
            metadata=StoredArtifact(**{name: metadata_entry[name] for name in ("path", "sha256", "size")}),
            review_data=review, metadata_data=metadata, attempt=attempt)

    def reject(self, candidate):
        with self.assertRaises(review_records.ExportReviewError):
            self.verify(candidate, attempt=candidate.attempt)

    def test_real_modes_and_selections_return_exact_frozen_review_result(self):
        for key, contents in self.archive_bytes.items():
            attempt, selected = key
            with self.subTest(attempt=attempt, selected=selected):
                record = self.save_archive(contents, attempt=attempt)
                inventory = self.inventories[key]
                members = self.saved_members[key]
                data = json.loads(members["inventory.json"])
                review_ref = StoredArtifact(**data["selection"]["reviewManifest"])
                review = json.loads(members[review_ref.path])
                checked = integrity.ExportArchiveIntegrity(record, inventory, len(data["files"]), data["totalFileBytes"])
                prepared = preparation.ExportPreparationIntegrity(checked,
                    StoredArtifact(**data["selection"]["preparationManifest"]), "synthetic_fixture")
                result = self.verify(SimpleNamespace(archive=record, inventory=inventory), attempt=attempt)
                self.assertEqual(result, review_records.ExportReviewIntegrity(prepared, review_ref,
                    StoredArtifact(**review["rawMetadata"])))
                with self.assertRaises(FrozenInstanceError):
                    result.review = None
                self.assertFalse(hasattr(result, "visualEffectDemonstrated"))
                self.assertFalse(hasattr(result, "automatedWorkflowComplete"))

    def test_archive_only_verification_needs_no_external_evidence_or_media_decode(self):
        reads = []
        original_read, original_open = integrity.read_verified, os.open

        def track_read(root, record):
            reads.append(record)
            return original_read(root, record)

        def readonly_open(path, flags, *args, **kwargs):
            self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            return original_open(path, flags, *args, **kwargs)

        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        with patch.object(integrity, "read_verified", side_effect=track_read), \
                patch.object(os, "open", side_effect=readonly_open), \
                patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No ledger")), \
                patch.object(SubmissionCoordinator, "__init__", side_effect=AssertionError("No coordinator")), \
                patch.object(httpx.Client, "send", side_effect=AssertionError("No HTTP")), \
                patch.object(socket, "socket", side_effect=AssertionError("No network")), \
                patch.object(time, "time", side_effect=AssertionError("No clock")), \
                patch.object(subprocess, "run", side_effect=AssertionError("No FFmpeg")), \
                patch.object(Image, "open", side_effect=AssertionError("No image decode")), \
                patch.object(ZipFile, "extract", side_effect=AssertionError("No extraction")), \
                patch.object(ZipFile, "extractall", side_effect=AssertionError("No extraction")), \
                patch.object(Path, "write_bytes", side_effect=AssertionError("No writes")), \
                patch.object(Path, "glob", side_effect=AssertionError("No scan")), \
                patch.object(Path, "rglob", side_effect=AssertionError("No scan")):
            result = self.verify()
        self.assertEqual(result.review, self.review)
        self.assertEqual(reads, [self.archive])
        self.assertEqual(sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")), before)
        self.assertEqual(read_verified(self.run_root, self.archive), self.contents)

    def test_single_snapshot_retains_only_fixed_preparation_and_review_controls(self):
        reads, retained = [], []
        original_read, original_member = integrity.read_verified, integrity._read_member

        def change_after_read(root, record):
            reads.append(record)
            body = original_read(root, record)
            (Path(root) / record.path).write_bytes(b"changed after verification")
            return body

        def track_member(saved, member, record, *, retain=False):
            if retain:
                retained.append(member.filename)
            return original_member(saved, member, record, retain=retain)

        with patch.object(integrity, "read_verified", side_effect=change_after_read), \
                patch.object(integrity, "_read_member", side_effect=track_member):
            self.assertEqual(self.verify().review, self.review)
        expected = {"inventory.json", *[self.prefix + name for name in ("manifest.json", "frozen/config.json",
            "frozen/placement.json", "frozen/inputs.json", "frozen/prompt.txt", "review/manifest.json", "output/metadata.json")]}
        self.assertEqual(set(retained), expected)
        self.assertEqual(len(retained), 8)
        self.assertEqual(reads, [self.archive])

    def test_review_control_limits_reject_before_retained_reads(self):
        for field, path in (("review_contents", self.review.path), ("metadata_contents", self.metadata.path)):
            original = self.members[path]
            exact = original + b" " * (131072 - len(original))
            with self.subTest(field=field, boundary="exact"):
                candidate = self.candidate(**{field: exact})
                self.assertEqual(self.verify(candidate).review, candidate.review)
            candidate = self.candidate(**{field: exact + b" "})
            retained = []
            original_member = integrity._read_member

            def track_member(saved, member, record, *, retain=False):
                if retain:
                    retained.append(member.filename)
                return original_member(saved, member, record, retain=retain)

            with self.subTest(field=field, boundary="over"), patch.object(integrity, "_read_member", side_effect=track_member):
                self.reject(candidate)
            self.assertNotIn(path, retained)

    def test_every_payload_and_preparation_pass_before_review_json_is_interpreted(self):
        members = dict(self.members)
        path = sorted(name for name in members if name.startswith(self.prefix + "review/raw/"))[-1]
        body = members[path]
        members[path] = bytes([body[0] ^ 1]) + body[1:]
        record = self.save_archive(archive_fixtures.ArchiveIntegrityTests.pack(list(members.items())))
        with patch.object(review_records, "_parse_review_json") as parser:
            self.reject(SimpleNamespace(archive=record, inventory=self.inventory, attempt=ATTEMPT_A))
            parser.assert_not_called()
        candidate = self.candidate(change_preparation=lambda data: data.update(state="SUCCEEDED"), check_preparation=False)
        with patch.object(review_records, "_parse_review_json") as parser:
            self.reject(candidate)
            parser.assert_not_called()

    def test_review_json_identity_and_non_acceptance_fields_are_checked(self):
        body = self.members[self.review.path]
        for contents in (b"not JSON", b"[]", b"\xff", b'{"schemaVersion":"1.4",' + body[1:],
                         json.dumps(self.review_data | {"extra": float("nan")}).encode()):
            with self.subTest(json=contents[:30]):
                self.reject(self.candidate(review_contents=contents))
        changes = {"runId": "other-run", "attemptId": ATTEMPT_B, "schemaVersion": "other", "manifestType": "other",
            "mode": "B", "sourceKind": "clean_camera", "providerTaskId": "other-task", "payloadSha256": "f" * 64,
            "samplePixelFormat": "RGBA", "sampleAutomaticRotationApplied": True,
            "visualEffectDemonstrated": "PASS", "automatedWorkflowComplete": "PASS"}
        for field, value in changes.items():
            with self.subTest(field=field):
                self.reject(self.candidate(change_review=lambda data: data.update({field: value})))

    def test_review_media_and_preparation_references_match_full_inventory_records(self):
        for field in ("preparation", "sourceOriginal", "sourceNormalized", "providerVideoInput", "rawOutput", "rawMetadata"):
            with self.subTest(field=field):
                self.reject(self.candidate(change_review=lambda data: data[field].update(sha256="f" * 64)))
        changes = [lambda data: data["guide"]["artifact"].update(sha256="f" * 64),
            lambda data: data["guide"].update(kind="video"),
            lambda data: data.update(providerVideoInput=data["guide"]["artifact"]),
            lambda data: data["sourceOriginal"].update(size=float(data["sourceOriginal"]["size"])),
            lambda data: data["rawOutput"].update(path=f"attempts/{ATTEMPT_B}/output/raw.mp4")]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.reject(self.candidate(change_review=change))

    def test_review_sample_shape_and_exact_selected_member_set_are_checked(self):
        def reuse_source_image(data):
            prior = data["samples"][0]["source"]
            data["samples"][1]["source"].update(frameIndex=prior["frameIndex"], image=deepcopy(prior["image"]))

        changes = [lambda data: data["samples"].pop(), lambda data: data["samples"].reverse(), reuse_source_image,
            lambda data: data["samples"][0]["source"].update(frameIndex=True),
            lambda data: data["samples"][0]["source"]["image"].update(sha256="f" * 64),
            lambda data: data["samples"][0]["guide"].update(referenceSeconds=1)]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.reject(self.candidate(change_review=change))
        for change in (lambda data: data["samples"][0]["guide"].update(kind="static_keyframe"),
                       lambda data: data["samples"][0]["guide"].update(frameIndex=1)):
            self.reject(self.candidate(attempt=ATTEMPT_B, change_review=change))

    def test_raw_metadata_json_identity_geometry_and_raw_reference_are_checked(self):
        body = self.members[self.metadata.path]
        for contents in (b"[]", b"not JSON", b'{"schemaVersion":"1.4",' + body[1:]):
            with self.subTest(json=contents[:30]):
                self.reject(self.candidate(metadata_contents=contents))
        changes = [lambda data: data.update(runId="other-run"), lambda data: data.update(attemptId=ATTEMPT_B),
            lambda data: data.update(schemaVersion="other"), lambda data: data.update(manifestType="other"),
            lambda data: data.update(providerTaskId="other-task"), lambda data: data["raw"].update(sha256="f" * 64),
            lambda data: data.update(rawContentValidation="FAIL"), lambda data: data.update(visualEffectDemonstrated="PASS"),
            lambda data: data.update(automatedWorkflowComplete="PASS"), lambda data: data.update(width=True),
            lambda data: data.update(height=0), lambda data: data.update(width=4097),
            lambda data: data.update(width=4096, height=4096), lambda data: data.update(frameSeconds={})]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.reject(self.candidate(change_metadata=change))

    def test_raw_duration_and_native_timeline_are_bounded_and_strict(self):
        changes = [lambda data: data.update(durationSeconds=0), lambda data: data.update(durationSeconds=True),
            lambda data: data.update(durationSeconds=21), lambda data: data.update(durationSeconds=float("nan")),
            lambda data: data.update(frameSeconds=[]), lambda data: data["frameSeconds"].__setitem__(0, True),
            lambda data: data["frameSeconds"].__setitem__(0, -0.01),
            lambda data: data["frameSeconds"].__setitem__(1, data["frameSeconds"][0]),
            lambda data: data["frameSeconds"].reverse(),
            lambda data: data["frameSeconds"].__setitem__(-1, data["durationSeconds"] + 1),
            lambda data: data.update(frameSeconds=[index / 1000 for index in range(2401)])]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.reject(self.candidate(change_metadata=change))

    def test_raw_alignment_uses_declared_pts_native_bounds_and_earlier_ties(self):
        times = self.metadata_data["frameSeconds"]
        old_last = self.review_data["samples"][-1]["raw"]["image"]["path"]
        new_last = self.prefix + f"review/raw/frame-{len(times) - 1:06d}.png"

        def boundaries(data):
            first, last = data["samples"][0], data["samples"][-1]
            first["source"]["ptsSeconds"] = -0.000001
            first["raw"].update(offsetSeconds=times[0] + 0.000001, spanPosition="before_first", withinDeclaredDuration=False)
            last["source"]["ptsSeconds"] = 5.0
            last["raw"].update(frameIndex=len(times) - 1, ptsSeconds=times[-1], offsetSeconds=times[-1] - 5.0,
                               spanPosition="after_last", withinDeclaredDuration=False)

        candidate = self.candidate(change_review=boundaries, rename_members={old_last: new_last})
        self.assertEqual(self.verify(candidate).review, candidate.review)
        midpoint = times[1] / 2

        def earlier_tie(data):
            sample = data["samples"][0]
            sample["source"]["ptsSeconds"] = midpoint
            sample["raw"]["offsetSeconds"] = -midpoint

        candidate = self.candidate(change_review=earlier_tie)
        self.assertEqual(self.verify(candidate).review, candidate.review)

        def later_tie(data):
            earlier_tie(data)
            data["samples"][0]["raw"].update(frameIndex=1, ptsSeconds=times[1], offsetSeconds=times[1] - midpoint)

        first_image = self.review_data["samples"][0]["raw"]["image"]["path"]
        self.reject(self.candidate(change_review=later_tie,
            rename_members={first_image: self.prefix + "review/raw/frame-000001.png"}))

        def swap_raw_frames(data):
            left, right = data["samples"][1:3]
            left["raw"], right["raw"] = right["raw"], left["raw"]
            for sample in (left, right):
                sample["raw"]["offsetSeconds"] = sample["raw"]["ptsSeconds"] - sample["source"]["ptsSeconds"]

        self.reject(self.candidate(change_review=swap_raw_frames))
        self.reject(self.candidate(change_review=lambda data: data["samples"][0]["raw"].update(
            ptsSeconds=0.001, offsetSeconds=0.001)))
        self.reject(self.candidate(change_review=lambda data: data["samples"][0]["raw"].update(frameIndex=2399),
            rename_members={first_image: self.prefix + "review/raw/frame-002399.png"}))

    def test_raw_offsets_span_and_duration_labels_are_recomputed(self):
        changes = [lambda data: data["samples"][2]["raw"].update(offsetSeconds=0.1),
            lambda data: data["samples"][2]["raw"].update(spanPosition="before_first"),
            lambda data: data["samples"][2]["raw"].update(withinDeclaredDuration=False),
            lambda data: data["samples"][2]["raw"].update(withinDeclaredDuration=1)]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.reject(self.candidate(change_review=change))

    def test_private_references_match_canonical_paths_hash_size_and_provider_summaries(self):
        for field in ("requestCheck", "request", "frozenRequest", "providerSuccess"):
            with self.subTest(field=field, mismatch="path"):
                self.reject(self.candidate(change_review=lambda data: data[field].update(path=self.prefix + "private-other.json")))

            def change_hash(data):
                data[field]["sha256"] = "f" * 64
                if field == "requestCheck":
                    data[field]["path"] = self.prefix + "provider_checks/" + "f" * 64 + ".json"
                elif field == "providerSuccess":
                    data[field]["path"] = "provider_responses/" + "f" * 64 + ".json"

            with self.subTest(field=field, mismatch="hash"):
                self.reject(self.candidate(change_review=change_hash))
            with self.subTest(field=field, mismatch="size"):
                self.reject(self.candidate(change_review=lambda data: data[field].update(size=data[field]["size"] + 1)))
        self.reject(self.candidate(change_inventory=lambda data: data["provider"].update(requestBodySha256="f" * 64)))
        self.reject(self.candidate(change_review=lambda data: data.update(payloadSha256="f" * 64),
            change_inventory=lambda data: data["provider"].update(payloadSha256="f" * 64)))
        self.reject(self.candidate(
            change_review=lambda data: data["request"].update(size=float(data["request"]["size"])),
            change_inventory=lambda data: data["privateEvidenceHashes"]["request"].update(
                size=float(data["privateEvidenceHashes"]["request"]["size"]))))

    def test_shared_filesystem_view_and_metadata_readers_preserve_behavior(self):
        valid = self.candidate()
        invalid_view = self.candidate(change_review=lambda data: data["samples"][0]["raw"].update(offsetSeconds=0.5))
        invalid_metadata = self.candidate(change_metadata=lambda data: data.update(width=True))
        for label, candidate in (("valid", valid), ("view", invalid_view), ("metadata", invalid_metadata)):
            with self.subTest(case=label), TemporaryDirectory() as folder:
                root = Path(folder)
                for path, contents in candidate.members.items():
                    if path != "inventory.json":
                        publish_bytes(root, path, contents)
                review = candidate.review_data
                completed = CompletedOutput(RUN, ATTEMPT_A, review["providerTaskId"], review["providerSuccess"]["sha256"],
                    StoredArtifact(**review["rawOutput"]), candidate.metadata, 1001.0)
                if label == "valid":
                    self.assertIsNone(view._verify_views(root, review, self.prefix))
                    self.assertEqual(output_review._read_output_metadata(root, completed), candidate.metadata_data)
                    for field in ("raw", "metadata"):
                        foreign = replace(getattr(completed, field), path=f"attempts/{ATTEMPT_B}/output/other.json")
                        with self.subTest(wrong_path=field), patch.object(output_review, "_read_record") as reader:
                            with self.assertRaises(output_review.ReviewEvidenceError):
                                output_review._read_output_metadata(root, replace(completed, **{field: foreign}))
                            reader.assert_not_called()
                elif label == "view":
                    self.reject(candidate)
                    with self.assertRaises(view.ReviewPageError):
                        view._verify_views(root, review, self.prefix)
                else:
                    self.reject(candidate)
                    with self.assertRaises(output_review.ReviewEvidenceError):
                        output_review._read_output_metadata(root, completed)


if __name__ == "__main__":
    unittest.main()
