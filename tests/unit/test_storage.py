"""Exercise real filesystem publication, concurrency, and hash verification."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.storage import ArtifactExistsError, StorageError, StoredArtifact, publish_bytes, read_verified


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "runs"

    def test_publish_and_verify_exact_bytes(self):
        contents = b"original fixture\x00\xff"
        record = publish_bytes(self.root, "run-1/attempts/a-1/raw.mp4", contents)
        self.assertEqual(record, StoredArtifact("run-1/attempts/a-1/raw.mp4",
                                               sha256(contents).hexdigest(), len(contents)))
        self.assertEqual(read_verified(self.root, record), contents)
        self.assertEqual(list(self.root.rglob(".pending-*")), [])

    def test_existing_attempt_cannot_be_replaced(self):
        first = publish_bytes(self.root, "attempt-1/guide.png", b"first")
        with self.assertRaises(ArtifactExistsError):
            publish_bytes(self.root, first.path, b"second")
        second = publish_bytes(self.root, "attempt-2/guide.png", b"second")
        self.assertEqual(read_verified(self.root, first), b"first")
        self.assertEqual(read_verified(self.root, second), b"second")

    def test_concurrent_publication_has_exactly_one_winner(self):
        def publish(index):
            try:
                return publish_bytes(self.root, "attempt/raw.mp4", str(index).encode())
            except ArtifactExistsError:
                return None
        with ThreadPoolExecutor(max_workers=20) as pool:
            records = list(pool.map(publish, range(20)))
        winners = [record for record in records if record is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(read_verified(self.root, winners[0]),
                         (self.root / "attempt/raw.mp4").read_bytes())
        self.assertEqual(list(self.root.rglob(".pending-*")), [])

    def test_unsafe_paths_are_rejected_for_write_and_read(self):
        for path in ["", "/tmp/file", "../escape", "a/../escape", "a//file", "./file",
                     "a/", "a\\file", "a/\x00file", "a/" * 17 + "file", "x" * 129]:
            with self.subTest(path=path):
                with self.assertRaises(StorageError):
                    publish_bytes(self.root, path, b"data")
                with self.assertRaises(StorageError):
                    read_verified(self.root, StoredArtifact(path, "0" * 64, 4))
        self.assertFalse(self.root.exists())

    def test_symlinks_are_not_followed(self):
        outside = Path(self.folder.name) / "outside"
        outside.mkdir()
        (outside / "file").write_bytes(b"safe")
        self.root.mkdir()
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        (self.root / "file").symlink_to(outside / "file")
        for path in ["link/file", "file"]:
            with self.subTest(path=path):
                with self.assertRaises(StorageError):
                    publish_bytes(self.root, path, b"bad")
                with self.assertRaises(StorageError):
                    read_verified(self.root, StoredArtifact(path, sha256(b"safe").hexdigest(), 4))
        self.assertEqual((outside / "file").read_bytes(), b"safe")

    def test_changed_size_and_hash_are_rejected(self):
        record = publish_bytes(self.root, "raw.mp4", b"first")
        for contents in [b"other", b"longer"]:
            with self.subTest(contents=contents):
                (self.root / record.path).write_bytes(contents)
                with self.assertRaises(StorageError):
                    read_verified(self.root, record)

    def test_invalid_records_are_rejected(self):
        record = publish_bytes(self.root, "raw.mp4", b"data")
        for changed in [replace(record, size=True), replace(record, size=-1),
                        replace(record, size=512 * 1024 * 1024 + 1),
                        replace(record, sha256="invalid")]:
            with self.subTest(record=changed):
                with self.assertRaisesRegex(StorageError, "record has an invalid"):
                    read_verified(self.root, changed)

    def test_fifo_is_rejected_without_blocking(self):
        self.root.mkdir()
        os.mkfifo(self.root / "pipe")
        with self.assertRaisesRegex(StorageError, "regular file"):
            read_verified(self.root, StoredArtifact("pipe", sha256(b"").hexdigest(), 0))

    def test_failed_publish_leaves_no_target_or_temporary_file(self):
        with patch("core.storage.os.link", side_effect=OSError("dummy-secret")):
            with self.assertRaises(StorageError) as caught:
                publish_bytes(self.root, "attempt/raw.mp4", b"complete")
        self.assertNotIn("dummy-secret", str(caught.exception))
        self.assertFalse((self.root / "attempt/raw.mp4").exists())
        self.assertEqual(list(self.root.rglob(".pending-*")), [])

    def test_publication_follows_file_sync_and_precedes_directory_sync(self):
        events = []
        sync = os.fsync
        link = os.link

        def record_sync(descriptor):
            events.append("file" if stat.S_ISREG(os.fstat(descriptor).st_mode) else "directory")
            sync(descriptor)

        def record_link(*args, **kwargs):
            self.assertIn("file", events)
            events.append("link")
            return link(*args, **kwargs)

        with patch("core.storage.os.fsync", side_effect=record_sync), \
                patch("core.storage.os.link", side_effect=record_link):
            publish_bytes(self.root, "raw.mp4", b"complete")
        self.assertLess(events.index("file"), events.index("link"))
        self.assertIn("directory", events[events.index("link") + 1:])

    def test_mutable_and_non_byte_contents_are_rejected(self):
        for contents in [bytearray(b"data"), memoryview(b"data"), "data", None]:
            with self.subTest(contents=contents):
                with self.assertRaisesRegex(StorageError, "contents must be bytes"):
                    publish_bytes(self.root, "raw.mp4", contents)

    def test_new_root_and_ancestor_entries_are_synced(self):
        root = self.root / "nested-root"
        synced = set()
        original = os.fsync

        def record_sync(descriptor):
            info = os.fstat(descriptor)
            synced.add((info.st_dev, info.st_ino))
            original(descriptor)

        with patch("core.storage.os.fsync", side_effect=record_sync):
            publish_bytes(root, "attempt/raw.mp4", b"complete")
        for parent in [Path(self.folder.name), self.root, root, root / "attempt"]:
            with self.subTest(parent=parent.name):
                info = parent.stat()
                self.assertIn((info.st_dev, info.st_ino), synced)

    def test_existing_directory_entries_are_synced_by_each_writer(self):
        (self.root / "attempt").mkdir(parents=True)
        synced = set()
        sync = os.fsync
        link = os.link
        root_info = self.root.stat()

        def record_sync(descriptor):
            info = os.fstat(descriptor)
            synced.add((info.st_dev, info.st_ino))
            sync(descriptor)

        def check_before_link(*args, **kwargs):
            self.assertIn((root_info.st_dev, root_info.st_ino), synced)
            return link(*args, **kwargs)

        with patch("core.storage.os.fsync", side_effect=record_sync), \
                patch("core.storage.os.link", side_effect=check_before_link):
            publish_bytes(self.root, "attempt/raw.mp4", b"complete")
