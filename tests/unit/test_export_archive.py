"""Package real fixture inventories with deterministic, bounded ZIP output."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import stat
from tempfile import TemporaryDirectory, TemporaryFile
import time
import unittest
from unittest.mock import Mock, patch
from zipfile import ZIP_STORED, ZipFile
import zlib

import httpx

from core.budget import BudgetLedger
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference.submission import SubmissionCoordinator
from features.review import archive
from features.review.export import build_export_inventory
import test_export_inventory as fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B


class ExportArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inventory_fixture = type("ArchiveInventoryFixture", (fixtures.ExportInventoryTests,), {})
        inventory_fixture.setUpClass()
        cls.addClassCleanup(inventory_fixture.doClassCleanups)
        cls.fixture = inventory_fixture
        cls.base = inventory_fixture.base
        reader_fixture = inventory_fixture.fixture
        cls.inventories = {}
        ledger = BudgetLedger(cls.base / "budget.sqlite", clock=lambda: 5000.0)
        with SubmissionCoordinator(ledger) as owner:
            for attempt in (ATTEMPT_A, ATTEMPT_B):
                for selected in (False, True):
                    cls.inventories[attempt, selected] = build_export_inventory(owner, cls.base,
                        inventory_fixture.manifests[attempt], run_id=RUN, attempt_id=attempt,
                        scorecard_binding=reader_fixture.bindings[attempt] if selected else None)

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)
        self.run_root = self.root / RUN
        self.inventory = self.inventories[ATTEMPT_A, False]
        self.original = self.data(self.inventory)

    def data(self, record):
        return json.loads(read_verified(self.run_root, record))

    def build(self, *, attempt=ATTEMPT_A, selected=False, inventory=None, **changes):
        arguments = {"run_id": RUN, "attempt_id": attempt} | changes
        return archive.build_export_archive(self.root, inventory or self.inventories[attempt, selected], **arguments)

    def save_inventory(self, data, *, contents=None):
        body = json.dumps(data, sort_keys=True).encode() if contents is None else contents
        return publish_bytes(self.run_root,
            f"attempts/{ATTEMPT_A}/export/inventories/{sha256(body).hexdigest()}.json", body)

    def assert_no_archives(self):
        for attempt in (ATTEMPT_A, ATTEMPT_B):
            self.assertFalse((self.run_root / f"attempts/{attempt}/export/archives").exists())

    @staticmethod
    def payload_record(entry):
        return StoredArtifact(**{field: entry[field] for field in ("path", "sha256", "size")})

    def test_real_modes_and_selections_preserve_exact_members_crc_and_fixed_metadata(self):
        for attempt in (ATTEMPT_A, ATTEMPT_B):
            for selected in (False, True):
                with self.subTest(attempt=attempt, selected=selected):
                    inventory_ref = self.inventories[attempt, selected]
                    inventory_bytes = read_verified(self.run_root, inventory_ref)
                    inventory = json.loads(inventory_bytes)
                    record = self.build(attempt=attempt, selected=selected)
                    contents = read_verified(self.run_root, record)
                    self.assertEqual(record.path, f"attempts/{attempt}/export/archives/{sha256(contents).hexdigest()}.zip")
                    expected = {"inventory.json": inventory_bytes}
                    expected.update({entry["path"]: read_verified(self.run_root, self.payload_record(entry))
                                     for entry in inventory["files"]})
                    with ZipFile(BytesIO(contents)) as saved:
                        self.assertEqual(saved.namelist(), ["inventory.json", *sorted(name for name in expected if name != "inventory.json")])
                        self.assertEqual(saved.comment, b"")
                        self.assertIsNone(saved.testzip())
                        for member in saved.infolist():
                            actual = saved.read(member)
                            self.assertEqual(actual, expected[member.filename])
                            self.assertEqual(member.CRC, zlib.crc32(actual))
                            self.assertEqual((member.file_size, member.compress_size), (len(actual), len(actual)))
                            self.assertEqual(member.compress_type, ZIP_STORED)
                            self.assertEqual(member.date_time, (1980, 1, 1, 0, 0, 0))
                            self.assertEqual((member.create_system, member.create_version, member.extract_version), (3, 20, 20))
                            self.assertEqual((member.flag_bits, member.internal_attr), (0, 0))
                            self.assertEqual(member.external_attr, (stat.S_IFREG | 0o600) << 16)
                            self.assertEqual((member.extra, member.comment), (b"", b""))
                            self.assertFalse(member.is_dir())
                            self.assertNotIn(b"runway://", actual)
                            self.assertNotIn(b"private-output-token", actual)
                    self.assertEqual(inventory["visualEffectDemonstrated"], "NOT_RUN")
                    self.assertEqual(inventory["automatedWorkflowComplete"], "NOT_RUN")

    def test_inventory_bytes_are_not_reserialized(self):
        contents = b" \n" + json.dumps(self.original, separators=(",", ":")).encode() + b"\n\t"
        inventory = self.save_inventory(self.original, contents=contents)
        result = self.build(inventory=inventory)
        with ZipFile(BytesIO(read_verified(self.run_root, result))) as saved:
            self.assertEqual(saved.read("inventory.json"), contents)

    def test_repeat_ignores_file_times_and_current_clock(self):
        first = self.build()
        original = read_verified(self.run_root, first)
        for entry in self.original["files"]:
            os.utime(self.run_root / entry["path"], (1, 1))
        os.utime(self.run_root / self.inventory.path, (1, 1))
        with patch.object(time, "time", side_effect=AssertionError("No clock needed")):
            second = self.build()
        self.assertEqual(first, second)
        self.assertEqual(read_verified(self.run_root, second), original)

    def test_packaging_needs_no_private_provider_files_ledger_network_or_scan(self):
        review = self.data(StoredArtifact(**self.original["selection"]["reviewManifest"]))
        for name in ("requestCheck", "request", "frozenRequest", "providerSuccess"):
            (self.run_root / review[name]["path"]).unlink()
        (self.root / "budget.sqlite").unlink()
        shutil.rmtree(self.root / "provider-evidence")
        shutil.rmtree(self.run_root / f"attempts/{ATTEMPT_B}")
        publish_bytes(self.run_root, f"attempts/{ATTEMPT_A}/review/index.html", b"private unrelated HTML")
        with patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No ledger")), \
                patch.object(SubmissionCoordinator, "__init__", side_effect=AssertionError("No coordinator")), \
                patch.object(httpx.Client, "send", side_effect=AssertionError("No provider call")), \
                patch.object(Path, "glob", side_effect=AssertionError("No scan")), \
                patch.object(Path, "rglob", side_effect=AssertionError("No recursive scan")):
            result = self.build()
        with ZipFile(BytesIO(read_verified(self.run_root, result))) as saved:
            self.assertEqual(saved.namelist(), ["inventory.json", *[entry["path"] for entry in self.original["files"]]])
            self.assertNotIn("private unrelated HTML", str(saved.namelist()))

    def test_exact_size_cap_succeeds_and_overhead_rejects_before_payload_reads(self):
        result = self.build()
        with patch.object(archive, "MAX_ARCHIVE_BYTES", result.size):
            self.assertEqual(self.build(), result)
        payload_size = self.original["totalFileBytes"] + self.inventory.size
        self.assertGreater(result.size, payload_size)
        real_read = archive.read_verified
        for limit in (result.size - 1, payload_size):
            reads = []

            def record_read(root, record):
                reads.append(record.path)
                return real_read(root, record)

            with self.subTest(limit=limit), patch.object(archive, "MAX_ARCHIVE_BYTES", limit), \
                    patch.object(archive, "read_verified", side_effect=record_read), \
                    patch.object(archive, "TemporaryFile") as temporary, \
                    patch.object(archive, "_publish_or_verify") as publisher:
                with self.assertRaises(archive.ExportArchiveError):
                    self.build()
                temporary.assert_not_called()
                publisher.assert_not_called()
            self.assertEqual(reads, [self.inventory.path])

    def test_large_declared_payload_is_rejected_before_read_or_allocation(self):
        data = deepcopy(self.original)
        data["files"][0]["size"] = 536870912
        data["totalFileBytes"] = sum(entry["size"] for entry in data["files"])
        record = self.save_inventory(data)
        real_read = archive.read_verified
        reads = []

        def record_read(root, selected):
            reads.append(selected.path)
            return real_read(root, selected)

        with patch.object(archive, "read_verified", side_effect=record_read), patch.object(archive, "TemporaryFile") as temporary:
            with self.assertRaises(archive.ExportArchiveError):
                self.build(inventory=record)
            temporary.assert_not_called()
        self.assertEqual(reads, [record.path])
        self.assert_no_archives()

    def test_bounded_file_allows_header_rewrites_but_rejects_outside_writes_and_seeks(self):
        with TemporaryFile(mode="w+b") as temporary:
            limited = archive._LimitedFile(temporary, 5)
            self.assertEqual(limited.write(b"abc"), 3)
            self.assertEqual(limited.seek(0), 0)
            limited.write(b"X")
            limited.seek(5)
            with self.assertRaises(archive.ExportArchiveError):
                limited.write(b"Y")
            with self.assertRaises(archive.ExportArchiveError):
                limited.seek(6)
            temporary.seek(0)
            self.assertEqual(temporary.read(), b"Xbc")

    def test_unexpected_short_zip_fails_before_final_read_or_publication(self):
        real_write, real_temporary = archive._write_zip, archive.TemporaryFile
        tracked = []

        def omit_final_member(file, names, contents_for):
            chosen = names[:-1] if isinstance(file, archive._LimitedFile) else names
            return real_write(file, chosen, contents_for)

        @contextmanager
        def tracked_temporary(**arguments):
            with real_temporary(**arguments) as temporary:
                proxy = Mock(wraps=temporary)
                tracked.append(proxy)
                yield proxy

        with patch.object(archive, "_write_zip", side_effect=omit_final_member), \
                patch.object(archive, "TemporaryFile", tracked_temporary), \
                patch.object(archive, "_publish_or_verify") as publisher:
            with self.assertRaises(archive.ExportArchiveError):
                self.build()
            publisher.assert_not_called()
        self.assertEqual(len(tracked), 1)
        tracked[0].read.assert_not_called()
        self.assert_no_archives()

    def test_zip_uses_the_verified_snapshot_if_a_source_changes_after_read(self):
        raw = next(entry for entry in self.original["files"] if entry["roles"] == ["raw_output"])
        expected = read_verified(self.run_root, self.payload_record(raw))
        real_read = archive.read_verified

        def change_after_verified_read(root, record):
            contents = real_read(root, record)
            if record.path == raw["path"]:
                (Path(root) / record.path).write_bytes(b"changed after verified read")
            return contents

        with patch.object(archive, "read_verified", side_effect=change_after_verified_read):
            result = self.build()
        with ZipFile(BytesIO(read_verified(self.run_root, result))) as saved:
            self.assertEqual(saved.read(raw["path"]), expected)
        self.assertEqual((self.run_root / raw["path"]).read_bytes(), b"changed after verified read")

    def test_inventory_identity_hash_size_and_path_are_checked_before_packaging(self):
        copied = publish_bytes(self.run_root, f"attempts/{ATTEMPT_A}/export/copied.json",
                               read_verified(self.run_root, self.inventory))
        records = [copied, self.inventories[ATTEMPT_B, False], replace(self.inventory, sha256="f" * 64),
                   replace(self.inventory, size=True), replace(self.inventory, size=65537),
                   replace(self.inventory, path="../private-inventory.json")]
        for record in records:
            with self.subTest(record=record), self.assertRaises(archive.ExportArchiveError):
                self.build(inventory=record)
            self.assert_no_archives()
        for changes in ({"run_id": "../invalid"}, {"run_id": "other-run"}, {"attempt_id": ATTEMPT_B}):
            with self.subTest(changes=changes), self.assertRaises(archive.ExportArchiveError):
                self.build(**changes)
            self.assert_no_archives()

    def test_payload_names_roles_and_totals_reject_ambiguous_or_foreign_entries(self):
        cases = []
        duplicate = deepcopy(self.original)
        duplicate["files"].append(duplicate["files"][0] | {"roles": ["raw_output"]})
        duplicate["files"].sort(key=lambda entry: entry["path"])
        duplicate["totalFileBytes"] = sum(entry["size"] for entry in duplicate["files"])
        cases.append(duplicate)
        foreign = deepcopy(self.original)
        foreign["files"][0]["path"] = foreign["files"][0]["path"].replace(ATTEMPT_A, ATTEMPT_B)
        foreign["files"].sort(key=lambda entry: entry["path"])
        cases.append(foreign)
        reversed_files = deepcopy(self.original)
        reversed_files["files"].reverse()
        cases.append(reversed_files)
        reversed_roles = deepcopy(self.original)
        next(entry for entry in reversed_roles["files"] if len(entry["roles"]) > 1)["roles"].reverse()
        cases.append(reversed_roles)
        for field, value in (("path", "../private"), ("path", f"attempts/{ATTEMPT_A}/provider/request.json"),
                             ("size", float(self.original["files"][0]["size"])), ("roles", ["source_original"])):
            changed = deepcopy(self.original)
            changed["files"][0][field] = value
            cases.append(changed)
        cases.extend([self.original | {"totalFileBytes": self.original["totalFileBytes"] + 1},
                      self.original | {"totalFileBytes": float(self.original["totalFileBytes"])}])
        for index, data in enumerate(cases):
            record = self.save_inventory(data)
            with self.subTest(case=index), patch.object(archive, "TemporaryFile") as temporary:
                with self.assertRaises(archive.ExportArchiveError):
                    self.build(inventory=record)
                temporary.assert_not_called()
            self.assert_no_archives()

    def test_missing_required_payload_and_unselected_or_mismatched_cards_are_rejected(self):
        missing = deepcopy(self.original)
        missing["files"] = [entry for entry in missing["files"] if "guide_b" not in entry["roles"]]
        missing["totalFileBytes"] = sum(entry["size"] for entry in missing["files"])
        selected = self.data(self.inventories[ATTEMPT_A, True])
        unselected = deepcopy(self.original)
        unselected["files"] += [entry for entry in selected["files"]
                                if entry["roles"] in (["scorecard"], ["scorecard_binding"])]
        unselected["files"].sort(key=lambda entry: entry["path"])
        unselected["totalFileBytes"] = sum(entry["size"] for entry in unselected["files"])
        mismatched = deepcopy(selected)
        mismatched["selection"]["scorecard"]["sha256"] = "f" * 64
        for data in (missing, unselected, mismatched):
            record = self.save_inventory(data)
            with self.subTest(selection=data["selection"]), self.assertRaises(archive.ExportArchiveError):
                self.build(inventory=record)
            self.assert_no_archives()

    def test_invalid_json_schema_and_outcome_claims_are_rejected(self):
        bodies = [b"{malformed", b'{"runId":"one","runId":"two"}', b'{"value":NaN}']
        bodies += [json.dumps(self.original | change).encode() for change in (
            {"visualEffectDemonstrated": "PASS"}, {"automatedWorkflowComplete": "PASS"},
            {"provider": self.original["provider"] | {"outputUrl": "https://private.example/output"}})]
        for contents in bodies:
            record = self.save_inventory(self.original, contents=contents)
            with self.subTest(contents=contents[:40]), self.assertRaises(archive.ExportArchiveError):
                self.build(inventory=record)
            self.assert_no_archives()

    def test_missing_or_corrupt_listed_payload_is_never_published(self):
        selected = self.data(self.inventories[ATTEMPT_A, True])
        for role in ("source_original", "guide_b", "raw_output", "raw_sample", "scorecard"):
            entry = next(entry for entry in selected["files"] if role in entry["roles"])
            path = self.run_root / entry["path"]
            original = path.read_bytes()
            path.unlink()
            with self.subTest(role=role), self.assertRaises(archive.ExportArchiveError):
                self.build(selected=True)
            self.assert_no_archives()
            path.write_bytes(original)
        entry = next(entry for entry in selected["files"] if "raw_metadata" in entry["roles"])
        path = self.run_root / entry["path"]
        original = path.read_bytes()
        changed = original[:-1] + b" "
        self.assertNotEqual(changed, original)
        self.assertEqual(len(changed), len(original))
        path.write_bytes(changed)
        with self.assertRaises(archive.ExportArchiveError):
            self.build(selected=True)
        self.assert_no_archives()

    def test_payload_symlinks_are_not_followed(self):
        entry = self.original["files"][0]
        path = self.run_root / entry["path"]
        target = self.root / "same-content-outside-attempt"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(archive.ExportArchiveError):
            self.build()
        self.assert_no_archives()

    def test_publication_failure_is_safe_retryable_and_changed_existing_zip_is_preserved(self):
        with patch.object(archive, "_publish_or_verify", side_effect=OSError("private storage path")) as publisher:
            with self.assertRaises(archive.ExportArchiveError) as raised:
                self.build()
        self.assertEqual(publisher.call_count, 1)
        self.assertEqual(str(raised.exception), "Export archive could not be verified or published for this attempt.")
        self.assert_no_archives()
        result = self.build()
        self.assertEqual(self.build(), result)
        path = self.run_root / result.path
        path.write_bytes(b"changed immutable ZIP")
        with self.assertRaises(archive.ExportArchiveError):
            self.build()
        self.assertEqual(path.read_bytes(), b"changed immutable ZIP")


if __name__ == "__main__":
    unittest.main()
