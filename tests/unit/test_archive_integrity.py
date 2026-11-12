"""Verify actual export ZIP bytes independently of their packaging process."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import stat
import struct
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
import warnings
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipExtFile, ZipFile, ZipInfo

import httpx

from core.budget import BudgetLedger
from core.storage import StoredArtifact, publish_bytes, read_verified
from features.inference.submission import SubmissionCoordinator
from features.review import archive, integrity
import test_export_archive as fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B


class ArchiveIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive_fixture = type("IntegrityArchiveFixture", (fixtures.ExportArchiveTests,), {})
        archive_fixture.setUpClass()
        cls.addClassCleanup(archive_fixture.doClassCleanups)
        cls.fixture = archive_fixture
        cls.inventories = archive_fixture.inventories
        cls.archives, cls.archive_bytes = {}, {}
        for attempt in (ATTEMPT_A, ATTEMPT_B):
            for selected in (False, True):
                key = attempt, selected
                record = archive.build_export_archive(archive_fixture.base, cls.inventories[key],
                    run_id=RUN, attempt_id=attempt)
                cls.archives[key] = record
                cls.archive_bytes[key] = read_verified(archive_fixture.base / RUN, record)

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.run_root = self.root / RUN
        self.contents = self.archive_bytes[ATTEMPT_A, False]
        self.record = self.save_archive(self.contents)
        self.inventory = self.inventories[ATTEMPT_A, False]
        with ZipFile(BytesIO(self.contents)) as saved:
            self.members = [(info.filename, saved.read(info)) for info in saved.infolist()]
        self.inventory_bytes = self.members[0][1]
        self.original = json.loads(self.inventory_bytes)

    def save_archive(self, contents, *, attempt=ATTEMPT_A):
        digest = sha256(contents).hexdigest()
        path = f"attempts/{attempt}/export/archives/{digest}.zip"
        if (self.run_root / path).exists():
            return StoredArtifact(path, digest, len(contents))
        return publish_bytes(self.run_root, path, contents)

    @staticmethod
    def inventory_reference(contents):
        digest = sha256(contents).hexdigest()
        return StoredArtifact(f"attempts/{ATTEMPT_A}/export/inventories/{digest}.json", digest, len(contents))

    def verify(self, record=None, inventory=None, *, attempt=ATTEMPT_A, **changes):
        arguments = {"run_id": RUN, "attempt_id": attempt} | changes
        return integrity.verify_export_archive(self.root, record or self.record,
            inventory or self.inventory, **arguments)

    @staticmethod
    def pack(members, *, change_info=None, comment=b""):
        """Create independent ZIP fixtures without calling a production ZIP helper."""
        output = BytesIO()
        with warnings.catch_warnings(), ZipFile(output, "w", allowZip64=False) as saved:
            warnings.simplefilter("ignore", UserWarning)
            saved.comment = comment
            for index, (name, contents) in enumerate(members):
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_STORED
                info.create_system, info.create_version, info.extract_version = 3, 20, 20
                info.flag_bits, info.internal_attr = 0, 0
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                info.extra, info.comment = b"", b""
                if change_info is not None and index == 0:
                    change_info(info)
                saved.writestr(info, contents)
        return output.getvalue()

    def changed_inventory(self, change=None, *, contents=None):
        data = deepcopy(self.original)
        if change is not None:
            change(data)
        body = json.dumps(data, sort_keys=True).encode() if contents is None else contents
        return self.pack([("inventory.json", body), *self.members[1:]]), self.inventory_reference(body)

    def reject(self, contents, inventory=None, *, before_zip=False):
        record = self.save_archive(bytes(contents))
        if before_zip:
            with patch.object(integrity, "ZipFile") as parser:
                with self.assertRaises(integrity.ArchiveIntegrityError):
                    self.verify(record, inventory)
                parser.assert_not_called()
        else:
            with self.assertRaises(integrity.ArchiveIntegrityError):
                self.verify(record, inventory)

    @staticmethod
    def central_offsets(contents):
        end = len(contents) - 22
        offset = struct.unpack_from("<I", contents, end + 16)[0]
        while offset < end:
            if contents[offset:offset + 4] != b"PK\x01\x02":
                raise AssertionError("Fixture central directory is malformed")
            yield offset
            lengths = struct.unpack_from("<3H", contents, offset + 28)
            offset += 46 + sum(lengths)

    def test_real_modes_and_selections_return_exact_frozen_integrity_result(self):
        for key, contents in self.archive_bytes.items():
            attempt, selected = key
            with self.subTest(attempt=attempt, selected=selected):
                record = self.save_archive(contents, attempt=attempt)
                inventory = self.inventories[key]
                with ZipFile(BytesIO(contents)) as saved:
                    data = json.loads(saved.read("inventory.json"))
                actual = self.verify(record, inventory, attempt=attempt)
                self.assertEqual(actual, integrity.ExportArchiveIntegrity(record, inventory,
                    len(data["files"]), data["totalFileBytes"]))
                with self.assertRaises(FrozenInstanceError):
                    actual.total_payload_bytes = 0
                self.assertFalse(hasattr(actual, "visualEffectDemonstrated"))
                self.assertFalse(hasattr(actual, "automatedWorkflowComplete"))

    def test_only_archive_is_read_without_original_evidence_or_side_effects(self):
        reads = []
        original_read, original_open = integrity.read_verified, os.open

        def track_read(root, record):
            reads.append(record)
            return original_read(root, record)

        def readonly_open(path, flags, *args, **kwargs):
            self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            return original_open(path, flags, *args, **kwargs)

        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
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
                patch.object(Path, "rglob", side_effect=AssertionError("No scan")), \
                patch.object(Path, "glob", side_effect=AssertionError("No scan")):
            result = self.verify()
        self.assertEqual(result.archive, self.record)
        self.assertEqual(reads, [self.record])
        self.assertEqual(sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")), before)
        self.assertEqual(read_verified(self.run_root, self.record), self.contents)

    def test_malformed_references_and_ids_are_rejected_before_storage_reads(self):
        archive_cases = [replace(self.record, path="../secret.zip"), replace(self.record, sha256="f" * 64),
            replace(self.record, sha256=self.record.sha256.upper()), replace(self.record, size=float(self.record.size)),
            replace(self.record, size=True), replace(self.record, size=0), replace(self.record, size=-1),
            replace(self.record, size=536870913)]
        inventory_cases = [replace(self.inventory, path="../secret.json"), replace(self.inventory, sha256="G" * 64),
            replace(self.inventory, size=True), replace(self.inventory, size=0), replace(self.inventory, size=65537),
            self.inventories[ATTEMPT_B, False]]
        cases = [(record, self.inventory, {}) for record in archive_cases]
        cases += [(self.record, inventory, {}) for inventory in inventory_cases]
        cases += [(self.record, self.inventory, changes) for changes in
                  ({"run_id": "../bad"}, {"attempt_id": "att-invalid"})]
        for record, inventory, changes in cases:
            with self.subTest(record=record, inventory=inventory, changes=changes), \
                    patch.object(integrity, "read_verified") as reader:
                with self.assertRaises(integrity.ArchiveIntegrityError):
                    self.verify(record, inventory, **changes)
                reader.assert_not_called()

    def test_archive_size_cap_includes_overhead_and_precedes_read_and_parse(self):
        with patch.object(integrity, "MAX_ARCHIVE_BYTES", self.record.size):
            self.assertEqual(self.verify().archive, self.record)
        for limit in (self.record.size - 1, self.original["totalFileBytes"] + self.inventory.size):
            with self.subTest(limit=limit), patch.object(integrity, "MAX_ARCHIVE_BYTES", limit), \
                    patch.object(integrity, "read_verified") as reader, patch.object(integrity, "ZipFile") as parser:
                with self.assertRaises(integrity.ArchiveIntegrityError):
                    self.verify()
                reader.assert_not_called()
                parser.assert_not_called()

    def test_outer_archive_hash_size_and_regular_file_are_verified(self):
        path = self.run_root / self.record.path
        equivalent = self.inventory_bytes.replace(b"\n", b" ", 1)
        valid_alternative, alternative_inventory = self.changed_inventory(contents=equivalent)
        self.assertEqual(len(valid_alternative), self.record.size)
        path.write_bytes(valid_alternative)
        with self.assertRaises(integrity.ArchiveIntegrityError):
            self.verify(inventory=alternative_inventory)
        for contents in (self.contents[:-1], self.contents[:-1] + b"X"):
            with self.subTest(length=len(contents)):
                path.write_bytes(contents)
                with self.assertRaises(integrity.ArchiveIntegrityError):
                    self.verify()
        path.unlink()
        with self.assertRaises(integrity.ArchiveIntegrityError):
            self.verify()
        target = self.root / "private-token.zip"
        target.write_bytes(self.contents)
        path.symlink_to(target)
        with self.assertRaises(integrity.ArchiveIntegrityError) as caught:
            self.verify()
        self.assertNotIn("private-token", str(caught.exception))
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(integrity.ArchiveIntegrityError):
            self.verify()

    def test_exact_inventory_bytes_are_verified_even_when_json_is_equivalent(self):
        changed = self.inventory_bytes.replace(b"\n", b" ", 1)
        self.assertNotEqual(changed, self.inventory_bytes)
        self.assertEqual(len(changed), self.inventory.size)
        self.assertEqual(json.loads(changed), self.original)
        contents, reference = self.changed_inventory(contents=changed)
        record = self.save_archive(contents)
        self.assertEqual(self.verify(record, reference).inventory, reference)
        self.reject(contents)

    def test_inventory_json_schema_and_expected_identity_are_rechecked(self):
        cases = [self.changed_inventory(contents=b"not JSON"),
            self.changed_inventory(contents=b'{"mode":"A",' + self.inventory_bytes[1:]),
            self.changed_inventory(lambda data: data.update(unexpected="private-token")),
            self.changed_inventory(lambda data: data.update(runId="another-run")),
            self.changed_inventory(lambda data: data.update(attemptId=ATTEMPT_B)),
            self.changed_inventory(lambda data: data["provider"].update(completedUnix=float("nan"))),
            self.changed_inventory(lambda data: data.update(visualEffectDemonstrated="PASS"))]
        for index, (contents, inventory) in enumerate(cases):
            with self.subTest(case=index):
                self.reject(contents, inventory)

    def test_inventory_paths_roles_selection_and_totals_are_rechecked(self):
        changes = [lambda data: data["files"].reverse(),
            lambda data: data["files"][0].update(roles=["raw_output"]),
            lambda data: data["files"][0].update(path="attempts/" + ATTEMPT_B + "/frozen/config.json"),
            lambda data: data["selection"]["reviewManifest"].update(sha256="f" * 64),
            lambda data: data["selection"].update(scorecard=data["selection"]["reviewManifest"]),
            lambda data: data.update(totalFileBytes=data["totalFileBytes"] + 1),
            lambda data: data.update(mode="B")]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                contents, inventory = self.changed_inventory(change)
                self.reject(contents, inventory)

    def test_members_are_exact_ordered_unique_allowlisted_files(self):
        candidates = [self.members[:-1], self.members[1:], [*self.members, self.members[-1]],
            [self.members[0], *reversed(self.members[1:])],
            [*self.members, ("private.txt", b"private-token")],
            [*self.members, (f"attempts/{ATTEMPT_A}/provider/request.json", b"private-token")],
            [*self.members, ("../escape.txt", b"private-token")],
            [*self.members, ("/absolute.txt", b"private-token")],
            [*self.members, ("folder/", b"")]]
        for index, members in enumerate(candidates):
            with self.subTest(case=index):
                self.reject(self.pack(members))

    def test_fixed_member_metadata_is_required(self):
        changes = [("compress_type", ZIP_DEFLATED), ("date_time", (1981, 1, 1, 0, 0, 0)),
            ("create_system", 0), ("create_version", 21), ("extract_version", 21),
            ("internal_attr", 1), ("external_attr", (stat.S_IFLNK | 0o600) << 16),
            ("external_attr", (stat.S_IFREG | 0o644) << 16), ("extra", b"\x02\x00\x00\x00"),
            ("comment", b"private-token")]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                self.reject(self.pack(self.members, change_info=lambda info: setattr(info, field, value)))
        self.reject(self.pack(self.members, comment=b"private-token"))
        central = next(self.central_offsets(self.contents))
        for flag in (1, 8, 0x800):
            altered = bytearray(self.contents)
            struct.pack_into("<H", altered, 6, flag)
            struct.pack_into("<H", altered, central + 8, flag)
            with self.subTest(flag=flag):
                self.reject(altered)

    def test_local_headers_must_match_central_metadata_and_names(self):
        changes = [(4, "H", 21), (8, "H", ZIP_DEFLATED), (10, "H", 1),
            (14, "I", 0), (18, "I", self.inventory.size + 1),
            (22, "I", self.inventory.size + 1), (28, "H", 1)]
        for offset, format_code, value in changes:
            altered = bytearray(self.contents)
            struct.pack_into("<" + format_code, altered, offset, value)
            with self.subTest(offset=offset):
                self.reject(altered)
        for value in (ord("I"), 0):
            altered = bytearray(self.contents)
            altered[30] = value
            with self.subTest(filename_byte=value):
                self.reject(altered)
        central = next(self.central_offsets(self.contents))
        name_length = struct.unpack_from("<H", self.contents, central + 28)[0]
        name_end = central + 46 + name_length
        nul_name = bytearray(self.contents[:name_end] + b"\0" + self.contents[name_end:])
        struct.pack_into("<H", nul_name, central + 28, name_length + 1)
        directory_size = struct.unpack_from("<I", self.contents, len(self.contents) - 22 + 12)[0]
        struct.pack_into("<I", nul_name, len(nul_name) - 22 + 12, directory_size + 1)
        self.reject(nul_name)

    def test_crc_is_checked_even_when_payload_hashes_are_unchanged(self):
        altered = bytearray(self.contents)
        central = list(self.central_offsets(self.contents))[-1]
        local = struct.unpack_from("<I", altered, central + 42)[0]
        wrong_crc = struct.unpack_from("<I", altered, central + 16)[0] ^ 1
        struct.pack_into("<I", altered, central + 16, wrong_crc)
        struct.pack_into("<I", altered, local + 14, wrong_crc)
        self.reject(altered)

    def test_payload_sha256_is_checked_with_valid_crc_and_size(self):
        members = list(self.members)
        name, contents = members[-1]
        changed = bytes([contents[0] ^ 1]) + contents[1:]
        members[-1] = name, changed
        candidate = self.pack(members)
        with ZipFile(BytesIO(candidate)) as saved:
            self.assertIsNone(saved.testzip())
            self.assertEqual(len(saved.read(name)), len(contents))
        self.reject(candidate)

    def test_payload_member_sizes_match_inventory(self):
        reads = {}
        original_read = ZipExtFile.read

        def bounded_read(stream, size=-1):
            self.assertGreater(size, 0)
            self.assertLessEqual(size, 1024)
            contents = original_read(stream, size)
            reads.setdefault(stream.name, []).append(len(contents))
            return contents

        self.assertGreater(len(self.inventory_bytes), 1024)
        with patch.object(integrity, "READ_CHUNK_BYTES", 1024), patch.object(ZipExtFile, "read", bounded_read):
            self.verify()
        self.assertEqual(set(reads), {name for name, _ in self.members})
        for name, contents in self.members:
            self.assertEqual(sum(reads[name]), len(contents))
            self.assertEqual(reads[name][-1], 0)
        self.assertGreater(len(reads["inventory.json"]), 2)
        members = list(self.members)
        name, contents = members[-1]
        for changed in (contents[:-1], contents + b"X"):
            members[-1] = name, changed
            with self.subTest(size=len(changed)):
                self.reject(self.pack(members))

    def test_invalid_zip_envelopes_are_rejected_before_zipfile_parsing(self):
        end = len(self.contents) - 22
        cases = [self.contents + b"trailing", self.contents[:-1], b"not a ZIP"]
        for offset, format_code, value in [(4, "H", 1), (6, "H", 1), (8, "H", 1),
                (10, "H", 0xffff), (12, "I", 0xffffffff), (16, "I", 0xffffffff),
                (12, "I", 1), (20, "H", 1)]:
            altered = bytearray(self.contents)
            struct.pack_into("<" + format_code, altered, end + offset, value)
            cases.append(altered)
        wrong_count = bytearray(self.contents)
        actual_count = struct.unpack_from("<H", wrong_count, end + 10)[0]
        struct.pack_into("<2H", wrong_count, end + 8, actual_count - 1, actual_count - 1)
        cases.append(wrong_count)
        zip64_locator = bytearray(self.contents)
        zip64_locator[-42:-38] = b"PK\x06\x07"
        cases.append(zip64_locator)
        central_size = struct.unpack_from("<I", self.contents, end + 12)[0]
        oversized = bytearray(self.contents[:end] + bytes(128 * 1024) + self.contents[end:])
        struct.pack_into("<I", oversized, len(oversized) - 22 + 12, central_size + 128 * 1024)
        cases.append(oversized)
        for index, contents in enumerate(cases):
            with self.subTest(case=index):
                self.reject(contents, before_zip=True)

    def test_zip_layout_has_no_prefix_gaps_or_conflicting_offsets(self):
        end = len(self.contents) - 22
        central = struct.unpack_from("<I", self.contents, end + 16)[0]
        gap = b"private-token"
        prefixed = bytearray(gap + self.contents)
        struct.pack_into("<I", prefixed, len(prefixed) - 22 + 16, central + len(gap))
        for offset in self.central_offsets(self.contents):
            local = struct.unpack_from("<I", self.contents, offset + 42)[0]
            struct.pack_into("<I", prefixed, offset + len(gap) + 42, local + len(gap))
        gapped = bytearray(self.contents[:central] + gap + self.contents[central:])
        struct.pack_into("<I", gapped, len(gapped) - 22 + 16, central + len(gap))
        overlapping = bytearray(self.contents)
        second = list(self.central_offsets(self.contents))[1]
        struct.pack_into("<I", overlapping, second + 42, 0)
        for name, contents in (("prefix", prefixed), ("gap", gapped), ("overlap", overlapping)):
            with self.subTest(layout=name):
                self.reject(contents)


if __name__ == "__main__":
    unittest.main()
