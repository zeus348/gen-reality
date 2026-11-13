"""Read registered starts without recovering jobs or changing local selections."""

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app import preparation_state as state
from core.budget import BudgetLedger
from core.run_start import start_run
from features.inference.submission import SubmissionCoordinator


# Fixture construction stays independent of the reader under test.
_State = state.State
_INSPECT = state._inspect_state
_CONNECT = sqlite3.connect


class RegisteredRunStartTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        self.output = self.root / "output"
        self.ledger = BudgetLedger(self.private / "budget.sqlite", clock=lambda: 1000.0)
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.state = _State(self.private, self.output, self.ledger.path)
        self.run_id = "run-" + "a" * 32
        self.start = start_run(self.output, run_id=self.run_id, clock=lambda: 1000.25)
        self.state.create(self.run_id, self.start.record)

    def read(self, **changes):
        arguments = {"state_dir": self.private, "output_root": self.output,
                     "ledger_path": self.ledger.path, "run_id": self.run_id} | changes
        return state.read_registered_run_start(**arguments)

    def assert_error(self, code, operation):
        with self.assertRaises(state.ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_registered_start_is_read_in_one_snapshot_without_recovery_or_writes(self):
        with self.state.transaction() as connection:
            for input_id, kind in (("source", "source"), ("asset", "asset")):
                connection.execute("INSERT INTO preparation_inputs VALUES (?,?,?,?)",
                                   (input_id, self.run_id, kind, "{}"))
            connection.execute("INSERT INTO preparation_attempts VALUES "
                               "(?,?,0,'source','asset','{}','prompt','BUILDING',NULL,NULL)",
                               ("att-" + "b" * 32, self.run_id))
        changes, transactions, connections = [], [], []

        def connect(*args, **kwargs):
            connection = _CONNECT(*args, **kwargs)
            connections.append((args, kwargs))

            def authorizer(action, *details):
                if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                              sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_TRIGGER,
                              sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_TRIGGER):
                    changes.append((action, details))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            connection.set_authorizer(authorizer)
            return connection

        def inspect(connection, owner):
            transactions.append(connection.in_transaction)
            return _INSPECT(connection, owner)

        with patch.object(state.sqlite3, "connect", side_effect=connect), \
                patch.object(state, "_inspect_state", side_effect=inspect), \
                patch.object(state.State, "__init__", side_effect=state.ServiceError("STATE_UNAVAILABLE")) as constructor:
            result = self.read()
        self.assertEqual(result, self.start.record)
        self.assertEqual(asdict(result), {"path": "run-start.json", "sha256": self.start.record.sha256,
                                          "size": self.start.record.size})
        self.assertEqual(changes, [])
        self.assertEqual(transactions, [True])
        self.assertEqual(len(connections), 1)
        self.assertTrue(connections[0][0][0].endswith("?mode=ro"))
        constructor.assert_not_called()
        with self.state.connection() as connection:
            row = connection.execute("SELECT status,error_code FROM preparation_attempts").fetchone()
            self.assertEqual(tuple(row), ("BUILDING", None))

    def test_unregistered_and_invalid_ids_do_not_select_another_run(self):
        self.assert_error("RUN_NOT_FOUND", lambda: self.read(run_id="run-" + "b" * 32))
        for value in (None, True, "a", self.run_id.upper(), self.run_id + "/child"):
            with self.subTest(value=value), \
                    patch.object(state.sqlite3, "connect", side_effect=state.ServiceError("STATE_UNAVAILABLE")) as opening:
                self.assert_error("INVALID_INPUT", lambda: self.read(run_id=value))
                opening.assert_not_called()

    def test_exact_state_schema_and_owner_are_required(self):
        self.assert_error("STATE_UNAVAILABLE", lambda: self.read(output_root=self.root / "other-output"))
        other_ledger = BudgetLedger(self.private / "other.sqlite")
        self.assert_error("STATE_UNAVAILABLE", lambda: self.read(ledger_path=other_ledger.path))
        with self.state.connection(write=True) as connection:
            connection.execute("DROP TRIGGER preparation_runs_revision")
        self.assert_error("STATE_UNAVAILABLE", self.read)
        with _CONNECT(self.state.path) as connection:
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_schema "
                                                "WHERE name='preparation_runs_revision'").fetchone())

    def test_page_size_and_journal_mode_are_verified_without_repair(self):
        for page_size, journal_mode in ((4096, "DELETE"), (8192, "WAL")):
            with self.subTest(page_size=page_size, journal_mode=journal_mode):
                with _CONNECT(self.state.path, isolation_level=None) as connection:
                    connection.execute("PRAGMA journal_mode=DELETE")
                    connection.execute(f"PRAGMA page_size={page_size}")
                    connection.execute("VACUUM")
                    connection.execute(f"PRAGMA journal_mode={journal_mode}")
                before = self.state.path.read_bytes()
                self.assert_error("STATE_UNAVAILABLE", self.read)
                self.assertEqual(self.state.path.read_bytes(), before)
                with _CONNECT(self.state.path) as connection:
                    self.assertEqual(connection.execute("PRAGMA page_size").fetchone()[0], page_size)
                    self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], journal_mode.lower())

    def test_state_reader_and_existing_writer_check_actual_shm_and_journal_paths(self):
        for suffix in ("-shm", "-journal"):
            for mode in ("directory", "public"):
                with self.subTest(suffix=suffix, mode=mode):
                    path = Path(str(self.state.path) + suffix)
                    path.unlink(missing_ok=True)
                    if mode == "directory":
                        path.mkdir(mode=0o700)
                    else:
                        path.write_bytes(b"keep these bytes")
                        path.chmod(0o644)
                    try:
                        with patch.object(state.sqlite3, "connect",
                                          side_effect=state.ServiceError("STATE_UNAVAILABLE")) as opening:
                            self.assert_error("STATE_UNAVAILABLE", self.read)
                            self.assert_error("STATE_UNAVAILABLE", lambda: state.State(
                                self.private, self.output, self.ledger.path))
                            opening.assert_not_called()
                        if mode == "public":
                            self.assertEqual(path.read_bytes(), b"keep these bytes")
                    finally:
                        path.rmdir() if mode == "directory" else path.unlink()

    def test_missing_symlink_or_oversized_state_cannot_open_sqlite(self):
        backup = self.state.path.with_suffix(".saved")
        self.state.path.rename(backup)
        try:
            for mode in ("missing", "symlink", "oversized"):
                with self.subTest(mode=mode):
                    if mode == "symlink":
                        self.state.path.symlink_to(backup)
                    elif mode == "oversized":
                        with self.state.path.open("wb") as file:
                            file.truncate(state.MAX_DATABASE_BYTES + 1)
                        self.state.path.chmod(0o600)
                    try:
                        with patch.object(state.sqlite3, "connect",
                                          side_effect=state.ServiceError("STATE_UNAVAILABLE")) as opening:
                            self.assert_error("STATE_UNAVAILABLE", self.read)
                            opening.assert_not_called()
                    finally:
                        self.state.path.unlink(missing_ok=True)
        finally:
            backup.rename(self.state.path)

    def test_registered_reference_is_bounded_and_strict(self):
        original = asdict(self.start.record)
        values = [dict(original, extra=True), dict(original, size=True),
                  dict(original, size=0), dict(original, sha256="A" * 64),
                  {"path": "a" * 1025, "sha256": "a" * 64, "size": 1}]
        for value in values:
            with self.subTest(value=value):
                with self.state.connection(write=True) as connection:
                    connection.execute("DROP TRIGGER preparation_runs_revision")
                    connection.execute("UPDATE preparation_runs SET start_record=?", (json.dumps(value),))
                    connection.execute(state._TRIGGERS["preparation_runs_revision"])
                self.assert_error("STATE_UNAVAILABLE", self.read)
        with self.state.connection(write=True) as connection:
            connection.execute("DROP TRIGGER preparation_runs_revision")
            connection.execute("UPDATE preparation_runs SET start_record=?", (json.dumps(original),))
            connection.execute(state._TRIGGERS["preparation_runs_revision"])
        self.assertEqual(self.read(), self.start.record)

    def test_state_owner_directory_and_sqlite_family_must_remain_private_and_separate(self):
        self.private.chmod(0o755)
        try:
            with patch.object(state.sqlite3, "connect",
                              side_effect=state.ServiceError("STATE_UNAVAILABLE")) as opening:
                self.assert_error("STATE_UNAVAILABLE", self.read)
                opening.assert_not_called()
        finally:
            self.private.chmod(0o700)
        with patch.object(state.sqlite3, "connect",
                          side_effect=state.ServiceError("STATE_UNAVAILABLE")) as opening:
            self.assert_error("STATE_UNAVAILABLE", lambda: self.read(ledger_path=self.state.path))
            opening.assert_not_called()


if __name__ == "__main__":
    unittest.main()
