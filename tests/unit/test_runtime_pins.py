"""Keep trusted receipt identities outside the run's replaceable files."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

import httpx

from app import submit as app_submit, resume as app_resume
from core import runtime_pins as pins
from core.budget import BudgetLedger
from core.storage import StoredArtifact, read_verified
from features.inference import runtime
from features.output import fetch
import test_runtime_evidence as evidence_tests


ATTEMPT = "att-" + "a" * 32
PAYLOAD = "b" * 64


class RuntimePinStoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.output = self.root / "runs"
        self.output.mkdir(mode=0o700)
        self.ledger_path = self.root / "budget.sqlite"
        self.ledger = BudgetLedger(self.ledger_path, clock=lambda: 1000.0)
        self.ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                               max_attempts=6, cost_cents=140)
        self.path = self.root / "trusted.sqlite"

    def store(self, **changes):
        return pins.RuntimePinStore(self.path, **({"output_root": self.output,
            "ledger_path": self.ledger_path, "run_id": "run-1"} | changes))

    def snapshot(self, **changes):
        return pins.read_runtime_pins(self.path, **({"output_root": self.output,
            "ledger_path": self.ledger_path, "run_id": "run-1"} | changes))

    def pin(self):
        return pins.RuntimePin(ATTEMPT, PAYLOAD, None, "runtime_create_context",
            StoredArtifact(f"attempts/{ATTEMPT}/runtime/create-context.json", "a" * 64, 120))

    def test_exact_pin_is_immutable_and_repeated_write_is_idempotent(self):
        store = self.store()
        pin = self.pin()
        self.assertFalse(store.matches(pin))
        store.pin(pin)
        first = self.snapshot()
        self.assertEqual((first.run_id, first.pins), ("run-1", (pin,)))
        self.assertRegex(first.sha256, r"^[0-9a-f]{64}$")
        self.assertTrue(store.matches(pin))
        store.pin(pin)
        self.assertEqual(self.snapshot(), first)
        reopened = self.store()
        self.assertTrue(reopened.matches(pin))
        self.assertEqual(self.snapshot(), first)

    def test_conflicting_path_cannot_replace_any_pinned_field(self):
        store = self.store()
        original = self.pin()
        store.pin(original)
        before = self.snapshot()
        for changed in (replace(original, payload_sha256="c" * 64),
                        replace(original, receipt=replace(original.receipt, sha256="d" * 64)),
                        replace(original, receipt=replace(original.receipt, size=121))):
            with self.subTest(pin=changed):
                with self.assertRaises(pins.RuntimePinError):
                    store.matches(changed)
                with self.assertRaises(pins.RuntimePinError):
                    store.pin(changed)
                self.assertEqual(self.snapshot(), before)

    def test_owner_is_checked_by_store_reopen_match_and_reader(self):
        store = self.store()
        store.pin(self.pin())
        before = self.path.read_bytes()
        other_output = self.root / "other-output"
        other_output.mkdir(mode=0o700)
        other_ledger = self.root / "other-budget.sqlite"
        BudgetLedger(other_ledger, clock=lambda: 1000.0)
        changes = [{"output_root": other_output}, {"ledger_path": other_ledger}, {"run_id": "other-run"}]
        for changed in changes:
            arguments = {"output_root": self.output, "ledger_path": self.ledger_path, "run_id": "run-1"} | changed
            with self.subTest(changed=changed):
                with self.assertRaises(pins.RuntimePinError):
                    store.require_owner(**arguments)
                with self.assertRaises(pins.RuntimePinError):
                    self.store(**changed)
                with self.assertRaises(pins.RuntimePinError):
                    self.snapshot(**changed)
                self.assertEqual(self.path.read_bytes(), before)

    def test_index_cannot_be_created_inside_output_root(self):
        inside = self.output / "trusted.sqlite"
        self.ledger_path.chmod(0o600)
        alias = self.root / "ledger-link.sqlite"
        os.link(self.ledger_path, alias)
        before = self.ledger_path.read_bytes()
        for path in (inside, self.ledger_path, alias):
            with self.subTest(path=path), self.assertRaises(pins.RuntimePinError):
                pins.RuntimePinStore(path, output_root=self.output, ledger_path=self.ledger_path, run_id="run-1")
            self.assertEqual(self.ledger_path.read_bytes(), before)
        self.assertFalse(inside.exists())

    def test_absent_ledger_sidecar_names_cannot_become_pin_indexes(self):
        before = self.ledger_path.read_bytes()
        for suffix in ("-journal", "-wal", "-shm"):
            candidate = Path(str(self.ledger_path) + suffix)
            self.assertFalse(candidate.exists())
            with self.subTest(suffix=suffix), patch.object(pins, "_connection",
                    side_effect=pins.RuntimePinError("Forbidden SQLite connection")) as connection:
                with self.assertRaises(pins.RuntimePinError):
                    pins.RuntimePinStore(candidate, output_root=self.output,
                                         ledger_path=self.ledger_path, run_id="run-1")
                connection.assert_not_called()
            self.assertFalse(candidate.exists())
            self.assertEqual(self.ledger_path.read_bytes(), before)

    def test_pin_sidecar_names_cannot_already_belong_to_a_ledger(self):
        for suffix in ("-journal", "-wal", "-shm"):
            ledger_path = Path(str(self.path) + suffix)
            BudgetLedger(ledger_path, clock=lambda: 1000.0)
            ledger_path.chmod(0o600)
            before = ledger_path.read_bytes()
            self.assertFalse(self.path.exists())
            with self.subTest(suffix=suffix), patch.object(pins, "_connection",
                    side_effect=pins.RuntimePinError("Forbidden SQLite connection")) as connection:
                with self.assertRaises(pins.RuntimePinError):
                    pins.RuntimePinStore(self.path, output_root=self.output,
                                         ledger_path=ledger_path, run_id="run-1")
                connection.assert_not_called()
            self.assertFalse(self.path.exists())
            self.assertEqual(ledger_path.read_bytes(), before)

    def test_new_cross_family_hardlinks_block_every_existing_index_operation(self):
        store = self.store()
        store.pin(self.pin())
        self.ledger_path.chmod(0o600)
        links = [(self.ledger_path, Path(str(self.path) + "-journal")),
                 (self.path, Path(str(self.ledger_path) + "-journal"))]
        for source, alias in links:
            os.link(source, alias)
            before = {path: path.read_bytes() for path in (self.ledger_path, self.path, alias)}
            try:
                operations = {"reopen": self.store, "read": self.snapshot,
                              "matches": lambda: store.matches(self.pin()), "pin": lambda: store.pin(self.pin())}
                for name, operation in operations.items():
                    with self.subTest(alias=alias.name, operation=name), patch.object(pins, "_connection",
                            side_effect=pins.RuntimePinError("Forbidden SQLite connection")) as connection:
                        with self.assertRaises(pins.RuntimePinError):
                            operation()
                        connection.assert_not_called()
                    self.assertEqual({path: path.read_bytes() for path in before}, before)
            finally:
                alias.unlink()
        self.assertEqual(self.snapshot().pins, (self.pin(),))

    def test_snapshot_digest_binds_exact_canonical_owner_and_pin_contents(self):
        store = self.store()
        original = self.pin()
        store.pin(original)
        expected_owner = ["runtime-pins-1", "run-1", str(self.output.resolve()), str(self.ledger_path.resolve())]
        expected_pin = {
            "attempt_id": ATTEMPT, "payload_sha256": PAYLOAD, "provider_task_id": None,
            "receipt_kind": "runtime_create_context",
            "receipt": {"path": f"attempts/{ATTEMPT}/runtime/create-context.json", "sha256": "a" * 64, "size": 120},
        }
        expected_bytes = json.dumps({"owner": expected_owner, "pins": [expected_pin]},
                                    sort_keys=True, separators=(",", ":")).encode("utf-8")
        original_digest = self.snapshot().sha256
        self.assertEqual(original_digest, sha256(expected_bytes).hexdigest())

        other_owner_path = self.root / "other-owner.sqlite"
        other_owner = pins.RuntimePinStore(other_owner_path, output_root=self.output,
                                           ledger_path=self.ledger_path, run_id="run-2")
        other_owner.pin(original)
        changed_owner = pins.read_runtime_pins(other_owner_path, output_root=self.output,
                                               ledger_path=self.ledger_path, run_id="run-2")
        self.assertNotEqual(changed_owner.sha256, original_digest)

        other_pin_path = self.root / "other-pin.sqlite"
        other_pin = pins.RuntimePinStore(other_pin_path, output_root=self.output,
                                         ledger_path=self.ledger_path, run_id="run-1")
        other_pin.pin(replace(original, payload_sha256="c" * 64))
        changed_pin = pins.read_runtime_pins(other_pin_path, output_root=self.output,
                                             ledger_path=self.ledger_path, run_id="run-1")
        self.assertNotEqual(changed_pin.sha256, original_digest)
        self.assertEqual(self.snapshot().sha256, original_digest)

    def test_index_requires_private_parent_and_existing_regular_ledger(self):
        public = self.root / "public"
        public.mkdir(mode=0o755)
        public.chmod(0o755)
        for path, ledger in ((public / "trusted.sqlite", self.ledger_path),
                             (self.path, self.root / "missing.sqlite"), (self.path, self.output)):
            with self.subTest(path=path, ledger=ledger), self.assertRaises(pins.RuntimePinError):
                pins.RuntimePinStore(path, output_root=self.output, ledger_path=ledger, run_id="run-1")
            self.assertFalse(path.exists())

    def test_symlink_index_is_rejected_without_changing_target(self):
        self.store().pin(self.pin())
        before = self.path.read_bytes()
        alias = self.root / "alias.sqlite"
        alias.symlink_to(self.path)
        for action in (lambda: pins.RuntimePinStore(alias, output_root=self.output,
                        ledger_path=self.ledger_path, run_id="run-1"),
                       lambda: pins.read_runtime_pins(alias, output_root=self.output,
                        ledger_path=self.ledger_path, run_id="run-1")):
            with self.assertRaises(pins.RuntimePinError):
                action()
        self.assertEqual(self.path.read_bytes(), before)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            sidecar.touch(mode=0o600)
            try:
                for action in (self.store, self.snapshot):
                    with self.subTest(sidecar=suffix), self.assertRaises(pins.RuntimePinError):
                        action()
                self.assertEqual(sidecar.read_bytes(), b"")
                self.assertEqual(self.path.read_bytes(), before)
            finally:
                sidecar.unlink()

    def test_missing_reader_does_not_create_index(self):
        with self.assertRaises(pins.RuntimePinError):
            self.snapshot()
        self.assertFalse(self.path.exists())

    def test_constructor_retry_syncs_parent_after_committed_initialization(self):
        sync = pins.os.fsync
        with patch.object(pins.os, "fsync", side_effect=OSError("fixture directory sync failed")):
            with self.assertRaises(pins.RuntimePinError):
                self.store()
        self.assertEqual(self.snapshot().pins, ())
        with patch.object(pins.os, "fsync", wraps=sync) as retried_sync:
            store = self.store()
        retried_sync.assert_called_once()
        store.pin(self.pin())
        before = self.snapshot()
        with patch.object(pins.os, "fsync", side_effect=OSError("fixture directory sync failed")):
            with self.assertRaises(pins.RuntimePinError):
                self.store()
        self.assertEqual(self.snapshot(), before)
        with patch.object(pins.os, "fsync", wraps=sync) as reopened_sync:
            self.store()
        reopened_sync.assert_called_once()
        self.assertEqual(self.snapshot(), before)

    def test_foreign_database_is_rejected_without_schema_or_row_changes(self):
        self.store().pin(self.pin())
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.execute("INSERT INTO unrelated VALUES ('preserve this row')")
        self.path.chmod(0o600)
        before = self.path.read_bytes()
        for action in (self.store, self.snapshot):
            with self.assertRaises(pins.RuntimePinError):
                action()
            self.assertEqual(self.path.read_bytes(), before)

    def test_changed_page_size_is_rejected_without_modifying_the_index(self):
        store = self.store()
        store.pin(self.pin())
        with sqlite3.connect(self.path, isolation_level=None) as connection:
            connection.execute("PRAGMA page_size=8192")
            connection.execute("VACUUM")
            self.assertEqual(connection.execute("PRAGMA page_size").fetchone()[0], 8192)
        before = self.path.read_bytes()
        for action in (self.store, self.snapshot, lambda: store.pin(self.pin())):
            with self.assertRaises(pins.RuntimePinError):
                action()
            self.assertEqual(self.path.read_bytes(), before)

    def test_owner_and_pin_rows_reject_update_delete_and_replace(self):
        store = self.store()
        store.pin(self.pin())
        before = self.snapshot()
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA recursive_triggers").fetchone()[0], 0)
            owner = connection.execute("SELECT * FROM runtime_pin_owner").fetchone()
            pin = connection.execute("SELECT * FROM runtime_pins").fetchone()
            statements = [
                ("UPDATE runtime_pin_owner SET run_id='other'", ()),
                ("DELETE FROM runtime_pin_owner", ()),
                ("INSERT OR REPLACE INTO runtime_pin_owner VALUES (?, ?, ?, ?, ?)", owner),
                ("UPDATE runtime_pins SET receipt_sha256=?", ("e" * 64,)),
                ("DELETE FROM runtime_pins", ()),
                ("INSERT OR REPLACE INTO runtime_pins VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pin),
            ]
            for statement, values in statements:
                with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(statement, values)
                connection.rollback()
        self.assertEqual(self.snapshot(), before)

    def test_reader_uses_readonly_connection_without_schema_or_row_writes(self):
        self.store().pin(self.pin())
        before = self.path.read_bytes()
        opened, denied = [], []
        connect = sqlite3.connect
        write_actions = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TRIGGER,
            sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_DROP_TRIGGER,
            sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}

        def authorize(action, *details):
            if action in write_actions:
                denied.append(action)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def read_connection(*args, **kwargs):
            opened.append((args, kwargs))
            connection = connect(*args, **kwargs)
            connection.set_authorizer(authorize)
            return connection

        with patch.object(pins.sqlite3, "connect", side_effect=read_connection):
            captured = self.snapshot()
        self.assertEqual(captured.pins, (self.pin(),))
        self.assertEqual(denied, [])
        self.assertTrue(opened)
        self.assertTrue(all(args[0].endswith("?mode=ro") and kwargs["uri"] for args, kwargs in opened))
        self.assertEqual(self.path.read_bytes(), before)

    def test_database_row_and_capture_bounds_reject_without_replacing_pins(self):
        store = self.store()
        first = self.pin()
        second_attempt = "att-" + "c" * 32
        second = replace(first, attempt_id=second_attempt,
            receipt=replace(first.receipt, path=f"attempts/{second_attempt}/runtime/create-context.json"))
        with patch.object(pins, "MAX_PINS", 1):
            store.pin(first)
            with self.assertRaises(pins.RuntimePinError):
                store.pin(second)
            self.assertEqual(self.snapshot().pins, (first,))
        store.pin(second)
        before = self.snapshot()
        for name, limit in (("MAX_PINS", 1), ("MAX_CAPTURE_BYTES", 1),
                            ("MAX_DATABASE_BYTES", self.path.stat().st_size - 1)):
            with self.subTest(limit=name), patch.object(pins, name, limit):
                with self.assertRaises(pins.RuntimePinError):
                    self.snapshot()
        self.assertEqual(self.snapshot(), before)

    def test_sqlite_cell_and_query_work_limits_are_enforced(self):
        self.store().pin(self.pin())
        with pins._connection(self.path) as connection:
            with self.assertRaises(sqlite3.DataError):
                connection.execute("SELECT ?", ("x" * (pins.MAX_CELL_BYTES + 1),)).fetchone()
        with patch.object(pins, "QUERY_STEP_INTERVAL", 1), patch.object(pins, "MAX_QUERY_STEPS", 0):
            with self.assertRaises(pins.RuntimePinError):
                self.snapshot()
        self.assertEqual(self.snapshot().pins, (self.pin(),))

    def test_locked_index_fails_within_bound_and_keeps_rows(self):
        store = self.store()
        store.pin(self.pin())
        before = self.snapshot()
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN EXCLUSIVE")
            started = time.monotonic()
            with self.assertRaises(pins.RuntimePinError):
                store.matches(self.pin())
            elapsed = time.monotonic() - started
            connection.rollback()
        self.assertLess(elapsed, 2.0)
        self.assertEqual(self.snapshot(), before)

    def test_invalid_pin_identity_and_receipt_shape_cannot_be_saved(self):
        store = self.store()
        original = self.pin()
        invalid = [replace(original, attempt_id="../other"), replace(original, payload_sha256="x"),
            replace(original, provider_task_id="task"), replace(original, receipt_kind="live_proof"),
            replace(original, receipt=replace(original.receipt, path="../escape.json")),
            replace(original, receipt=replace(original.receipt, sha256="A" * 64)),
            replace(original, receipt=replace(original.receipt, size=True)),
            replace(original, receipt=replace(original.receipt, size=16385))]
        for changed in invalid:
            with self.subTest(pin=changed), self.assertRaises(pins.RuntimePinError):
                store.pin(changed)
        self.assertEqual(self.snapshot().pins, ())


class RuntimePinIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = evidence_tests.RuntimeEvidenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.private = TemporaryDirectory()
        self.addCleanup(self.private.cleanup)
        self.path = Path(self.private.name) / "trusted.sqlite"

    def store(self):
        return pins.RuntimePinStore(self.path, output_root=self.fixture.root,
            ledger_path=self.fixture.ledger.path, run_id="run-1")

    def snapshot(self):
        return pins.read_runtime_pins(self.path, output_root=self.fixture.root,
            ledger_path=self.fixture.ledger.path, run_id="run-1")

    def recorder(self, store):
        return runtime.RuntimeEvidence(self.fixture.root, "run-1", ATTEMPT,
            clock=lambda: self.fixture.observed_unix, pin_store=store)

    def arguments(self):
        return ["--output-root", str(self.fixture.root), "--ledger", str(self.fixture.ledger.path),
                "--run-id", "run-1", "--attempt-id", ATTEMPT, "--trusted-runtime-index", str(self.path)]

    def submit_arguments(self):
        record = self.fixture.checked.evidence
        return self.arguments() + ["--check-path", record.path, "--check-sha256", record.sha256,
            "--check-size", str(record.size), "--api-key-env", "TEST_PIN_KEY"]

    def test_new_receipts_are_pinned_and_restoration_does_not_write_pins(self):
        store = self.store()
        owner = self.fixture.owner()
        recorder = self.recorder(store)
        result = self.fixture.send(owner, recorder)
        self.assertEqual(result.state, "QUEUED")
        before = self.snapshot()
        self.assertEqual(len(before.pins), 2)
        self.assertEqual({pin.receipt for pin in before.pins}, set(recorder.references))
        for pin in before.pins:
            document = json.loads(read_verified(self.fixture.run, pin.receipt))
            self.assertEqual((pin.attempt_id, pin.payload_sha256, pin.receipt_kind, pin.provider_task_id),
                (ATTEMPT, self.fixture.checked.frozen.payload_sha256,
                 document["manifestType"], document.get("providerTaskId")))
        restored = self.recorder(store)
        with patch.object(store, "pin", side_effect=AssertionError("Restore must not add trust")):
            restored.bind_existing(owner)
            self.assertEqual(restored.references, recorder.references)
            self.assertEqual(restored.summary()["issues"], [])
            context = json.loads(read_verified(self.fixture.run, recorder.references[0]))
            self.assertEqual(restored._publish("create-context.json", context), recorder.references[0])
        self.assertEqual(self.snapshot(), before)

    def test_successful_get_and_download_pin_exact_receipt_identities(self):
        store = self.store()
        owner = self.fixture.owner()
        recorder = self.recorder(store)
        self.fixture.send(owner, recorder)
        observed = self.fixture.task_get(recorder)
        payload = b"fixture downloaded bytes"
        result = fetch.download_task_output(evidence_tests.TASK, observed.evidence_sha256, self.fixture.run,
            allowed_hosts=frozenset({evidence_tests.HOST}), runtime=recorder,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload)))
        self.assertEqual(result, payload)
        captured = self.snapshot().pins
        self.assertEqual(len(captured), 5)
        self.assertEqual({pin.receipt for pin in captured}, set(recorder.references))
        with self.fixture.ledger._connection() as connection:
            reserved_payload = connection.execute(
                "SELECT payload_sha256 FROM reservations WHERE attempt_id=?", (ATTEMPT,)).fetchone()[0]
        for pin in captured:
            document = json.loads(read_verified(self.fixture.run, pin.receipt))
            self.assertEqual((pin.attempt_id, pin.payload_sha256, pin.receipt_kind, pin.provider_task_id),
                (ATTEMPT, reserved_payload, document["manifestType"], document.get("providerTaskId")))
            self.assertEqual((document["runId"], document["attemptId"], document["payloadSha256"]),
                             ("run-1", pin.attempt_id, pin.payload_sha256))
        self.assertEqual({pin.receipt_kind for pin in captured}, {
            "runtime_create_context", "runtime_create_result", "runtime_task_get",
            "runtime_download_intent", "runtime_download_result"})

    def test_reconciliation_preserves_original_unknown_receipt_task_fields(self):
        store = self.store()
        owner = self.fixture.owner()
        original = self.recorder(store)
        result = self.fixture.send(owner, original, status=400)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        before = self.snapshot()
        self.assertEqual(len(before.pins), 2)
        self.assertTrue(all(pin.provider_task_id is None for pin in before.pins))
        owner.record_verified_task(ATTEMPT, evidence_tests.TASK, "e" * 64)
        restored = self.recorder(store)
        restored.bind_existing(owner)
        self.assertEqual(restored.summary()["issues"], [])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.fixture.task_get(restored).task_id, evidence_tests.TASK)
        captured = self.snapshot().pins
        self.assertEqual(len(captured), 3)
        self.assertTrue(all(pin in captured for pin in before.pins))
        task_pin = next(pin for pin in captured if pin.receipt_kind == "runtime_task_get")
        self.assertEqual(task_pin.provider_task_id, evidence_tests.TASK)
        self.assertEqual(restored.summary()["issues"], [])

    def test_legacy_creation_receipts_cannot_backfill_trusted_pins(self):
        owner = self.fixture.owner()
        original = self.fixture.recorder()
        self.fixture.send(owner, original)
        store = self.store()
        restored = self.recorder(store)
        restored.bind_existing(owner)
        self.assertIn("PIN_MISSING", restored.summary()["issues"])
        self.assertEqual(self.snapshot().pins, ())
        # Even identical existing bytes are an observation, not a new publication.
        context = json.loads(read_verified(self.fixture.run, original.references[0]))
        with self.assertRaises(pins.RuntimePinError):
            restored._publish("create-context.json", context)
        self.assertEqual(self.snapshot().pins, ())
        observed = self.fixture.task_get(restored)
        self.assertEqual(observed.status, "SUCCEEDED")
        captured = self.snapshot().pins
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].receipt_kind, "runtime_task_get")

    def test_context_pin_failure_blocks_post_and_keeps_reservation(self):
        store = self.store()
        owner = self.fixture.owner()
        recorder = self.recorder(store)
        with patch.object(store, "pin", side_effect=pins.RuntimePinError("private pin path")):
            result = self.fixture.send(owner, recorder)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.fixture.requests, [])
        self.assertEqual(self.snapshot().pins, ())
        with self.fixture.ledger._connection() as connection:
            self.assertEqual(self.fixture.ledger._usage(connection, "run-1"), (1, 140, False))
        self.assertNotIn("private pin path", json.dumps(recorder.summary()))

    def test_changed_run_receipts_cannot_replace_original_trusted_pins(self):
        store = self.store()
        owner = self.fixture.owner()
        original = self.recorder(store)
        self.fixture.send(owner, original)
        before = self.snapshot()
        context_record, result_record = original.references
        context = json.loads(read_verified(self.fixture.run, context_record))
        context["transportClass"] = "DEFAULT_NETWORK"
        context_bytes = (json.dumps(context, sort_keys=True, indent=2) + "\n").encode()
        changed_context = StoredArtifact(context_record.path, sha256(context_bytes).hexdigest(), len(context_bytes))
        result = json.loads(read_verified(self.fixture.run, result_record))
        result["transportClass"] = "DEFAULT_NETWORK"
        result["createContext"] = asdict(changed_context)
        (self.fixture.run / context_record.path).write_bytes(context_bytes)
        (self.fixture.run / result_record.path).write_bytes((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
        restored = self.recorder(store)
        restored.bind_existing(owner)
        self.assertIn("PIN_MISMATCH", restored.summary()["issues"])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.fixture.task_get(restored).status, "SUCCEEDED")
        captured = self.snapshot().pins
        self.assertEqual(len(captured), 3)
        self.assertTrue(all(pin in captured for pin in before.pins))
        self.assertEqual(sum(pin.receipt_kind == "runtime_task_get" for pin in captured), 1)

    def test_actual_ledger_must_match_pin_owner_before_claim(self):
        other = Path(self.private.name) / "other-budget.sqlite"
        BudgetLedger(other, clock=lambda: 1001.0)
        store = pins.RuntimePinStore(self.path, output_root=self.fixture.root,
            ledger_path=other, run_id="run-1")
        recorder = self.recorder(store)
        owner = self.fixture.owner()
        with patch.object(owner, "_claim", wraps=owner._claim) as claim:
            with self.assertRaises(pins.RuntimePinError):
                self.fixture.send(owner, recorder)
        claim.assert_not_called()
        self.assertEqual(owner.get(ATTEMPT).state, "RESERVED")
        self.assertEqual(self.fixture.requests, [])

    def test_app_helpers_reject_wrong_ledger_before_opening_coordinator(self):
        other = Path(self.private.name) / "other-budget.sqlite"
        BudgetLedger(other, clock=lambda: 1001.0)
        store = pins.RuntimePinStore(self.path, output_root=self.fixture.root,
            ledger_path=other, run_id="run-1")
        recorder = self.recorder(store)
        actions = [
            (app_submit, lambda: app_submit.submit_saved_request(self.fixture.root, self.fixture.ledger.path,
                self.fixture.checked.evidence, run_id="run-1", attempt_id=ATTEMPT, runtime=recorder)),
            (app_resume, lambda: app_resume.resume_attempt(self.fixture.root, self.fixture.ledger.path,
                run_id="run-1", attempt_id=ATTEMPT, output_hosts=frozenset(), status_only=True, runtime=recorder)),
        ]
        for module, action in actions:
            with self.subTest(module=module.__name__), patch.object(module, "SubmissionCoordinator",
                    side_effect=pins.RuntimePinError("Forbidden coordinator creation")) as open_owner:
                with self.assertRaises(pins.RuntimePinError):
                    action()
                open_owner.assert_not_called()

    def test_result_pin_failure_preserves_known_task_and_context_pin(self):
        store = self.store()
        owner = self.fixture.owner()
        recorder = self.recorder(store)
        save = store.pin

        def fail_result(pin):
            if pin.receipt_kind == "runtime_create_result":
                raise pins.RuntimePinError("private result index")
            save(pin)

        with patch.object(store, "pin", side_effect=fail_result):
            result = self.fixture.send(owner, recorder)
        self.assertEqual((result.state, result.provider_task_id), ("QUEUED", evidence_tests.TASK))
        self.assertEqual([pin.receipt_kind for pin in self.snapshot().pins], ["runtime_create_context"])
        self.assertIn("PIN_RECORDING_FAILED", recorder.summary()["issues"])
        self.assertNotIn("private result index", json.dumps(recorder.summary()))

    def test_get_pin_failure_preserves_task_observation(self):
        store = self.store()
        owner = self.fixture.owner()
        recorder = self.recorder(store)
        self.fixture.send(owner, recorder)
        before = self.snapshot()
        with patch.object(store, "pin", side_effect=pins.RuntimePinError("private GET index")):
            observed = self.fixture.task_get(recorder)
        self.assertEqual((observed.task_id, observed.status), (evidence_tests.TASK, "SUCCEEDED"))
        self.assertEqual(self.snapshot(), before)
        self.assertIn("PIN_RECORDING_FAILED", recorder.summary()["issues"])
        self.assertNotIn("private GET index", json.dumps(recorder.summary()))

    def test_download_intent_pin_failure_prevents_output_get(self):
        store = self.store()
        owner = self.fixture.owner()
        recorder = self.recorder(store)
        self.fixture.send(owner, recorder)
        observed = self.fixture.task_get(recorder)
        before = self.snapshot()
        provider = Mock(return_value=httpx.Response(200, content=b"fixture output"))
        with patch.object(store, "pin", side_effect=pins.RuntimePinError("private intent index")):
            with self.assertRaises(fetch.DownloadError):
                fetch.download_task_output(evidence_tests.TASK, observed.evidence_sha256, self.fixture.run,
                    allowed_hosts=frozenset({evidence_tests.HOST}), transport=httpx.MockTransport(provider), runtime=recorder)
        provider.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_download_result_pin_failure_preserves_returned_body(self):
        store = self.store()
        owner = self.fixture.owner()
        recorder = self.recorder(store)
        self.fixture.send(owner, recorder)
        observed = self.fixture.task_get(recorder)
        save = store.pin
        payload = b"fixture body survives an index failure"

        def fail_result(pin):
            if pin.receipt_kind == "runtime_download_result":
                raise pins.RuntimePinError("private download index")
            save(pin)

        with patch.object(store, "pin", side_effect=fail_result):
            result = fetch.download_task_output(evidence_tests.TASK, observed.evidence_sha256, self.fixture.run,
                allowed_hosts=frozenset({evidence_tests.HOST}), runtime=recorder,
                transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload)))
        self.assertEqual(result, payload)
        kinds = [pin.receipt_kind for pin in self.snapshot().pins]
        self.assertIn("runtime_download_intent", kinds)
        self.assertNotIn("runtime_download_result", kinds)
        self.assertIn("PIN_RECORDING_FAILED", recorder.summary()["issues"])
        self.assertNotIn("private download index", json.dumps(recorder.summary()))

    def test_submit_cli_pins_new_receipts_without_printing_index_path(self):
        submit_saved = app_submit.submit_saved_request
        posts = []

        def provider(request):
            posts.append(request)
            return httpx.Response(200, json={"id": evidence_tests.TASK})

        def submit_fixture(*args, **kwargs):
            return submit_saved(*args, **kwargs, clock=lambda: 1001.0, transport=httpx.MockTransport(provider))

        output, errors = StringIO(), StringIO()
        with patch.dict("os.environ", {"TEST_PIN_KEY": evidence_tests.fixtures.KEY}), \
                patch.object(app_submit, "submit_saved_request", side_effect=submit_fixture), \
                redirect_stdout(output), redirect_stderr(errors):
            status = app_submit.main(self.submit_arguments())
        self.assertEqual(status, 0)
        result = json.loads(output.getvalue())
        self.assertEqual((result["state"], result["provider_task_id"]), ("QUEUED", evidence_tests.TASK))
        captured = self.snapshot()
        self.assertEqual(len(captured.pins), 2)
        self.assertEqual({pin.receipt for pin in captured.pins},
                         {StoredArtifact(**record) for record in result["runtime"]["receipts"]})
        self.assertEqual(len(posts), 1)
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn(str(self.path), output.getvalue())
        self.assertNotIn(evidence_tests.fixtures.KEY, output.getvalue())

    def test_resume_cli_restores_pins_without_printing_index_path_or_adding_rows(self):
        store = self.store()
        owner = self.fixture.owner()
        recorder = self.recorder(store)
        self.fixture.send(owner, recorder)
        owner.close()
        before = self.snapshot()
        output, errors = StringIO(), StringIO()
        matches = pins.RuntimePinStore.matches
        with patch.object(pins.RuntimePinStore, "matches", autospec=True, side_effect=matches) as checked, \
                patch.dict("os.environ", {}, clear=True), redirect_stdout(output), redirect_stderr(errors):
            status = app_resume.main(self.arguments() + ["--status-only"])
        self.assertEqual(status, 0)
        result = json.loads(output.getvalue())
        self.assertEqual((result["state"], result["provider_task_id"]), ("QUEUED", evidence_tests.TASK))
        self.assertEqual(result["runtime"]["issues"], [])
        self.assertEqual(len(result["runtime"]["receipts"]), 2)
        self.assertEqual(checked.call_count, 2)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.fixture.requests), 1)
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn(str(self.path), output.getvalue())

    def test_cli_rejects_index_inside_run_root_before_service_with_safe_error(self):
        private_path = self.fixture.root / "private-operator-index.sqlite"
        for module, arguments in ((app_submit, self.submit_arguments()),
                                  (app_resume, self.arguments() + ["--status-only"])):
            arguments[arguments.index("--trusted-runtime-index") + 1] = str(private_path)
            output, errors = StringIO(), StringIO()
            service = "submit_saved_request" if module is app_submit else "resume_attempt"
            with self.subTest(module=module.__name__), patch.object(module, service,
                    side_effect=ValueError("Forbidden service invocation")) as call, \
                    redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(module.main(arguments), 1)
            call.assert_not_called()
            self.assertEqual(output.getvalue(), "")
            self.assertIn("could not proceed", errors.getvalue())
            self.assertNotIn(str(private_path), errors.getvalue())
            self.assertFalse(private_path.exists())

