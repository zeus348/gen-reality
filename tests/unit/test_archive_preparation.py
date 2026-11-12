"""Check preparation declarations against one verified export ZIP snapshot."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import httpx

from core.budget import BudgetLedger
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference import prepared
from features.inference.submission import SubmissionCoordinator
from features.placement.math import Placement
from features.review import integrity, preparation
import test_archive_integrity as fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B
PREFIX = f"attempts/{ATTEMPT_A}/"


class ArchivePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integrity_fixture = type("PreparationIntegrityFixture", (fixtures.ArchiveIntegrityTests,), {})
        integrity_fixture.setUpClass()
        cls.addClassCleanup(integrity_fixture.doClassCleanups)
        cls.fixture = integrity_fixture
        cls.archive_bytes = integrity_fixture.archive_bytes
        cls.inventories = integrity_fixture.inventories

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.run_root = self.root / RUN
        self.contents = self.archive_bytes[ATTEMPT_A, False]
        self.archive = self.save_archive(self.contents)
        self.inventory = self.inventories[ATTEMPT_A, False]
        with ZipFile(BytesIO(self.contents)) as saved:
            self.members = {info.filename: saved.read(info) for info in saved.infolist()}
        self.original = json.loads(self.members["inventory.json"])
        self.manifest = StoredArtifact(**self.original["selection"]["preparationManifest"])
        self.manifest_data = json.loads(self.members[self.manifest.path])

    def save_archive(self, contents, *, attempt=ATTEMPT_A):
        digest = sha256(contents).hexdigest()
        path = f"attempts/{attempt}/export/archives/{digest}.zip"
        if (self.run_root / path).exists():
            return StoredArtifact(path, digest, len(contents))
        return publish_bytes(self.run_root, path, contents)

    def verify(self, candidate=None, *, attempt=ATTEMPT_A):
        record = self.archive if candidate is None else candidate.archive
        inventory = self.inventory if candidate is None else candidate.inventory
        return preparation.verify_export_preparation(self.root, record, inventory,
            run_id=RUN, attempt_id=attempt)

    def control(self, name):
        return json.loads(self.members[PREFIX + name])

    def candidate(self, *, controls=None, change_manifest=None, manifest_contents=None, change_inventory=None):
        """Refresh byte references while leaving the targeted semantic error intact."""
        members, inventory = dict(self.members), deepcopy(self.original)
        manifest = deepcopy(self.manifest_data)

        def replace_member(path, contents):
            record = StoredArtifact(path, sha256(contents).hexdigest(), len(contents))
            members[path] = contents
            entry = next(entry for entry in inventory["files"] if entry["path"] == path)
            entry.update(asdict(record))
            return record

        for name, contents in (controls or {}).items():
            manifest["artifacts"][name] = asdict(replace_member(PREFIX + name, contents))
        if change_manifest is not None:
            change_manifest(manifest)
        body = json.dumps(manifest, sort_keys=True).encode() if manifest_contents is None else manifest_contents
        manifest_ref = replace_member(self.manifest.path, body)
        inventory["selection"]["preparationManifest"] = asdict(manifest_ref)
        inventory["totalFileBytes"] = sum(entry["size"] for entry in inventory["files"])
        if change_inventory is not None:
            change_inventory(inventory)
        inventory_bytes = json.dumps(inventory, sort_keys=True).encode()
        digest = sha256(inventory_bytes).hexdigest()
        inventory_ref = StoredArtifact(PREFIX + f"export/inventories/{digest}.json", digest, len(inventory_bytes))
        members["inventory.json"] = inventory_bytes
        ordered = [("inventory.json", inventory_bytes), *[(path, members[path]) for path in sorted(members)
                                                        if path != "inventory.json"]]
        archive_ref = self.save_archive(fixtures.ArchiveIntegrityTests.pack(ordered))
        # All inner semantic fixtures remain valid archives with correct hashes.
        checked = integrity.verify_export_archive(self.root, archive_ref, inventory_ref,
            run_id=RUN, attempt_id=ATTEMPT_A)
        self.assertEqual(checked.archive, archive_ref)
        return SimpleNamespace(archive=archive_ref, inventory=inventory_ref, members=members,
                               manifest=manifest_ref, integrity=checked)

    def reject(self, candidate):
        with self.assertRaises(preparation.ExportPreparationError):
            self.verify(candidate)

    def test_real_modes_and_selections_return_exact_frozen_preparation_result(self):
        for key, contents in self.archive_bytes.items():
            attempt, selected = key
            with self.subTest(attempt=attempt, selected=selected):
                record = self.save_archive(contents, attempt=attempt)
                inventory = self.inventories[key]
                with ZipFile(BytesIO(contents)) as saved:
                    data = json.loads(saved.read("inventory.json"))
                candidate = SimpleNamespace(archive=record, inventory=inventory)
                result = self.verify(candidate, attempt=attempt)
                expected_integrity = integrity.ExportArchiveIntegrity(record, inventory,
                    len(data["files"]), data["totalFileBytes"])
                self.assertEqual(result, preparation.ExportPreparationIntegrity(expected_integrity,
                    StoredArtifact(**data["selection"]["preparationManifest"]), "synthetic_fixture"))
                with self.assertRaises(FrozenInstanceError):
                    result.source_kind = "clean_camera"
                self.assertFalse(hasattr(result, "visualEffectDemonstrated"))
                self.assertFalse(hasattr(result, "automatedWorkflowComplete"))

    def test_archive_only_operation_has_no_external_evidence_or_side_effects(self):
        reads = []
        original_read, original_open = integrity.read_verified, os.open

        def track_read(root, record):
            reads.append(record)
            return original_read(root, record)

        def readonly_open(path, flags, *args, **kwargs):
            self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            return original_open(path, flags, *args, **kwargs)

        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        self.assertFalse((self.run_root / self.manifest.path).exists())
        self.assertFalse((self.run_root / self.inventory.path).exists())
        with patch.object(integrity, "read_verified", side_effect=track_read), \
                patch.object(os, "open", side_effect=readonly_open), \
                patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No ledger")), \
                patch.object(SubmissionCoordinator, "__init__", side_effect=AssertionError("No coordinator")), \
                patch.object(httpx.Client, "send", side_effect=AssertionError("No HTTP")), \
                patch.object(socket, "socket", side_effect=AssertionError("No network")), \
                patch.object(time, "time", side_effect=AssertionError("No clock")), \
                patch.object(ZipFile, "extract", side_effect=AssertionError("No extraction")), \
                patch.object(ZipFile, "extractall", side_effect=AssertionError("No extraction")), \
                patch.object(Path, "write_bytes", side_effect=AssertionError("No writes")), \
                patch.object(Path, "glob", side_effect=AssertionError("No scan")), \
                patch.object(Path, "rglob", side_effect=AssertionError("No scan")):
            result = self.verify()
        self.assertEqual(result.preparation, self.manifest)
        self.assertEqual(reads, [self.archive])
        self.assertEqual(sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")), before)
        self.assertEqual(read_verified(self.run_root, self.archive), self.contents)

    def test_one_verified_snapshot_retains_only_bounded_preparation_controls(self):
        retained, reads = [], []
        original_read, original_member = integrity.read_verified, integrity._read_member

        def change_after_read(root, record):
            reads.append(record)
            contents = original_read(root, record)
            (Path(root) / record.path).write_bytes(b"changed after the verified read")
            return contents

        def track_member(saved, member, record, *, retain=False):
            if retain:
                retained.append(member.filename)
            return original_member(saved, member, record, retain=retain)

        with patch.object(integrity, "read_verified", side_effect=change_after_read), \
                patch.object(integrity, "_read_member", side_effect=track_member):
            result = self.verify()
        self.assertEqual(result.preparation, self.manifest)
        self.assertEqual(reads, [self.archive])
        self.assertEqual(set(retained), {"inventory.json", self.manifest.path,
            *[PREFIX + name for name in ("frozen/config.json", "frozen/placement.json", "frozen/inputs.json", "frozen/prompt.txt")]})
        self.assertEqual(len(retained), 6)
        self.assertEqual((self.run_root / self.archive.path).read_bytes(), b"changed after the verified read")

    def test_archive_integrity_failures_are_not_bypassed_and_errors_are_safe(self):
        for archive_ref, inventory_ref in [(replace(self.archive, size=True), self.inventory),
                (replace(self.archive, path="../private-token.zip"), self.inventory),
                (self.archive, replace(self.inventory, sha256="f" * 64))]:
            with self.subTest(archive=archive_ref, inventory=inventory_ref):
                self.reject(SimpleNamespace(archive=archive_ref, inventory=inventory_ref))
        (self.run_root / self.archive.path).write_bytes(self.contents[:-1] + b"X")
        with self.assertRaises(preparation.ExportPreparationError) as caught:
            self.verify()
        self.assertNotIn("private-token", str(caught.exception))
        self.assertNotIn(str(self.root), str(caught.exception))
        members = dict(self.members)
        sample = sorted(path for path in members if path.startswith(PREFIX + "review/raw/"))[-1]
        original = members[sample]
        members[sample] = bytes([original[0] ^ 1]) + original[1:]
        archive_ref = self.save_archive(fixtures.ArchiveIntegrityTests.pack(list(members.items())))
        with patch.object(preparation, "_parse_preparation_json") as parser:
            self.reject(SimpleNamespace(archive=archive_ref, inventory=self.inventory))
            parser.assert_not_called()

    def test_preparation_json_rejects_duplicate_nonfinite_and_malformed_values(self):
        original = self.members[self.manifest.path]
        duplicate = b'{"schemaVersion":"1.4",' + original[1:]
        for body in (duplicate, b"[]", b"not JSON", b"\xff", json.dumps(self.manifest_data | {"extra": float("nan")}).encode()):
            with self.subTest(body=body[:40]):
                self.reject(self.candidate(manifest_contents=body))
        for name in ("frozen/config.json", "frozen/placement.json", "frozen/inputs.json"):
            body = self.members[PREFIX + name]
            data = json.loads(body)
            first = next(iter(data))
            duplicate = ("{" + json.dumps(first) + ":" + json.dumps(data[first]) + ",").encode() + body[1:]
            with self.subTest(control=name):
                self.reject(self.candidate(controls={name: duplicate}))

    def test_manifest_identity_and_offline_claims_are_enforced(self):
        changes = [lambda data: data.update(runId="other-run"), lambda data: data.update(attemptId=ATTEMPT_B),
            lambda data: data.update(schemaVersion="other"), lambda data: data.update(manifestType="other"),
            lambda data: data.update(state="SUCCEEDED"), lambda data: data.update(releaseClassification="P0_FULL_GO"),
            lambda data: data.update(visualEffectDemonstrated="PASS"), lambda data: data.update(automatedWorkflowComplete="PASS"),
            lambda data: data["provider"].update(vendor="other"), lambda data: data["provider"].update(model="other"),
            lambda data: data["provider"].update(submitted=True), lambda data: data["provider"].update(taskId="fake-task")]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.reject(self.candidate(change_manifest=change))

    def test_all_nine_frozen_references_match_full_inventory_records(self):
        name = "frozen/guide_B.mp4"  # Mode A still retains and verifies its alternate local guide.
        original_ref = self.manifest_data["artifacts"][name]
        changes = [lambda data: data["artifacts"].pop(name),
            lambda data: data["artifacts"].update({"frozen/extra.bin": original_ref}),
            lambda data: data["artifacts"].update({"frozen/source_original.mov": original_ref}),
            lambda data: data["artifacts"][name].update(path=PREFIX + "frozen/keyframe_A.png"),
            lambda data: data["artifacts"][name].update(path=f"attempts/{ATTEMPT_B}/" + name),
            lambda data: data["artifacts"][name].update(sha256="f" * 64),
            lambda data: data["artifacts"][name].update(size=original_ref["size"] + 1),
            lambda data: data["artifacts"][name].update(size=float(original_ref["size"])),
            lambda data: data["artifacts"][name].update(size=True),
            lambda data: data["artifacts"][name].update(unexpected="private-token")]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.reject(self.candidate(change_manifest=change))

    def test_declared_source_kind_matches_manifest_controls_and_inventory(self):
        inputs = self.control("frozen/inputs.json") | {"sourceKind": "clean_camera"}
        controls = {"frozen/inputs.json": json.dumps(inputs).encode()}
        change_manifest = lambda data: data.update(sourceKind="clean_camera")
        candidate = self.candidate(controls=controls, change_manifest=change_manifest,
            change_inventory=lambda data: data.update(sourceKindDeclared="clean_camera"))
        self.assertEqual(self.verify(candidate).source_kind, "clean_camera")
        self.reject(self.candidate(controls=controls, change_manifest=change_manifest))
        self.reject(self.candidate(change_manifest=lambda data: data.update(sourceKind="unknown")))

    def test_input_declarations_and_exact_references_are_validated(self):
        original = self.control("frozen/inputs.json")
        changes = [lambda data: data.update(sourceKind="clean_camera"),
            lambda data: data.update(cleanSourceDeclared=False), lambda data: data.update(nonHumanDeclared=1),
            lambda data: data.update(source=data["asset"]), lambda data: data.update(normalizedSource=data["source"]),
            lambda data: data["asset"].update(sha256="f" * 64),
            lambda data: data["source"].update(size=float(data["source"]["size"]))]
        for index, change in enumerate(changes):
            data = deepcopy(original)
            change(data)
            with self.subTest(case=index):
                self.reject(self.candidate(controls={"frozen/inputs.json": json.dumps(data).encode()}))

    def test_frozen_configuration_uses_the_locked_schema(self):
        original = self.control("frozen/config.json")
        changes = [lambda data: data["sdk"].update(model="other"), lambda data: data["sdk"].update(version="other"),
            lambda data: data["asset"].update(alphaRequired=1), lambda data: data.update(unexpected="private-token")]
        for index, change in enumerate(changes):
            data = deepcopy(original)
            change(data)
            with self.subTest(case=index):
                self.reject(self.candidate(controls={"frozen/config.json": json.dumps(data).encode()}))

    def test_frozen_placement_keeps_numeric_and_exact_field_rules(self):
        original = self.control("frozen/placement.json")
        cases = [original | {"u": 2}, original | {"v": True}, original | {"heightFrac": 0},
            original | {"rotationDeg": 1}, original | {"extra": 0}, {key: value for key, value in original.items() if key != "u"}]
        for index, data in enumerate(cases):
            with self.subTest(case=index):
                self.reject(self.candidate(controls={"frozen/placement.json": json.dumps(data).encode()}))

    def test_prompt_obeys_utf8_character_and_byte_limits(self):
        boundary = ("\U0001f98a" * 1000).encode("utf-8")
        self.assertEqual(len(boundary), 4000)
        candidate = self.candidate(controls={"frozen/prompt.txt": boundary})
        self.assertEqual(self.verify(candidate).preparation, candidate.manifest)
        for contents in (b" ", b"x" * 1001, b"\xff", boundary + b"x"):
            with self.subTest(size=len(contents)):
                self.reject(self.candidate(controls={"frozen/prompt.txt": contents}))

    def test_control_size_limits_apply_before_retained_payload_reads(self):
        json_paths = [self.manifest.path, *[PREFIX + name for name in
                      ("frozen/config.json", "frozen/placement.json", "frozen/inputs.json")]]
        for path in json_paths:
            original = self.members[path]
            exact = original + b" " * (65536 - len(original))
            arguments = {"manifest_contents": exact} if path == self.manifest.path else {"controls": {path.removeprefix(PREFIX): exact}}
            with self.subTest(path=path, boundary="exact"):
                candidate = self.candidate(**arguments)
                self.assertEqual(self.verify(candidate).preparation, candidate.manifest)
            arguments = {"manifest_contents": exact + b" "} if path == self.manifest.path else {"controls": {path.removeprefix(PREFIX): exact + b" "}}
            candidate = self.candidate(**arguments)
            retained = []
            original_member = integrity._read_member

            def track_member(saved, member, record, *, retain=False):
                if retain:
                    retained.append(member.filename)
                return original_member(saved, member, record, retain=retain)

            with self.subTest(path=path, boundary="over"), patch.object(integrity, "_read_member", side_effect=track_member):
                self.reject(candidate)
            self.assertNotIn(path, retained)
        candidate = self.candidate(controls={"frozen/prompt.txt": ("\U0001f98a" * 1000).encode() + b"x"})
        with patch.object(integrity, "_read_member") as reader:
            # Preserve inventory parsing, but reject before this oversized control is retained.
            reader.side_effect = original_member
            self.reject(candidate)
            self.assertFalse(any(call.args[1].filename == PREFIX + "frozen/prompt.txt" and call.kwargs.get("retain")
                                 for call in reader.call_args_list))

    def test_filesystem_reader_preserves_behavior_and_shares_control_rejections(self):
        valid = self.candidate()
        config = self.control("frozen/config.json")
        config["sdk"]["model"] = "other"
        inputs = self.control("frozen/inputs.json")
        inputs["source"]["size"] = float(inputs["source"]["size"])
        invalid = [self.candidate(controls={"frozen/config.json": json.dumps(config).encode()}),
            self.candidate(controls={"frozen/inputs.json": json.dumps(inputs).encode()}),
            self.candidate(controls={"frozen/prompt.txt": b"x" * 1001})]
        for index, candidate in enumerate([valid, *invalid]):
            with self.subTest(case=index), TemporaryDirectory() as folder:
                root = Path(folder)
                for path, contents in candidate.members.items():
                    if path != "inventory.json":
                        publish_bytes(root, path, contents)
                if index:
                    self.reject(candidate)
                    with self.assertRaises(prepared.PreparationError):
                        prepared.read_prepared_attempt(root, candidate.manifest, run_id=RUN, attempt_id=ATTEMPT_A)
                else:
                    saved = prepared.read_prepared_attempt(root, candidate.manifest, run_id=RUN, attempt_id=ATTEMPT_A)
                    self.assertIsInstance(saved, prepared.PreparedAttempt)
                    self.assertEqual(saved.manifest, self.verify(candidate).preparation)
                    self.assertEqual((saved.run_id, saved.attempt_id, saved.source_kind), (RUN, ATTEMPT_A, "synthetic_fixture"))
                    self.assertEqual(saved.prompt, self.members[PREFIX + "frozen/prompt.txt"].decode())
                    placement = self.control("frozen/placement.json")
                    self.assertEqual(saved.placement, Placement(placement["u"], placement["v"], placement["heightFrac"], placement["rotationDeg"]))
                    self.assertEqual(saved.media_for_mode("A"), (saved.normalized_source, saved.keyframe_a))
                    self.assertEqual(saved.media_for_mode("B"), (saved.guide_b, None))


if __name__ == "__main__":
    unittest.main()
