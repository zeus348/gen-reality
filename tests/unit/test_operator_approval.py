"""Allocate existing operator authority without restarting local server owners."""

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app import approve as command, operator_config, preparation, preparation_state
from core import budget
from core.storage import StoredArtifact, publish_bytes
from features.inference import pricing, submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASE_LEDGER = budget.BudgetLedger
_BASE_COORDINATOR = submission.SubmissionCoordinator
_BASE_SERVICE = preparation.PreparationService
_BASE_PRICE_WRITER = pricing.record_price_quote
_SOURCE_MODULES = ("app.preparation_state", "features.inference.pricing", "app.operator_config", "app.approve")


def _approval_sources():
    """Let the external fault harness supply exact child code in memory."""
    return {name: (PROJECT_ROOT / name.replace(".", "/")).with_suffix(".py").read_text(encoding="utf-8")
            for name in _SOURCE_MODULES}


_CHILD_START = """
import importlib
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
arguments = sys.argv[2:]
sys.path.insert(0, str(root))
sources = json.load(sys.stdin)
for name in ("app.preparation_state", "features.inference.pricing", "app.operator_config"):
    package, leaf = name.rsplit(".", 1)
    parent = importlib.import_module(package)
    filename = (root / name.replace(".", "/")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    setattr(parent, leaf, module)
    exec(compile(sources[name], str(filename), "exec"), module.__dict__)
sys.argv = ["app.approve", *arguments]
__file__ = str(root / "app" / "approve.py")
__package__ = "app"
__spec__ = None
exec(compile(sources["app.approve"], __file__, "exec"), globals())
"""


class OperatorApprovalTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    @contextmanager
    def fixture(self, *, started=1000.0, checked=1100.0):
        with TemporaryDirectory(dir=self.root) as folder:
            root = Path(folder)
            state, outputs = root / "private", root / "runs"
            state.mkdir(mode=0o700)
            ledger = _BASE_LEDGER(state / "budget.sqlite", clock=lambda: started)
            owner = _BASE_COORDINATOR(ledger)
            service = None
            try:
                service = _BASE_SERVICE(state, outputs, owner, clock=lambda: started)
                selected = service.create_run()
                fixture = SimpleNamespace(root=root, state=state, outputs=outputs,
                    ledger=ledger, owner=owner, service=service, selected=selected,
                    config_path=state / "operator-approval.json")
                fixture.quote = self.quote(fixture, checked=checked)
                fixture.config = {"schemaVersion": "1.4", "manifestType": "local_operator_approval_config",
                    "outputRoot": str(outputs), "approvedBy": "fixture operator", "priceQuote": asdict(fixture.quote.record)}
                self.write_config(fixture)
                yield fixture
            finally:
                if service is not None:
                    service.close()
                owner.close()

    def quote(self, fixture, *, checked, additional=7, checked_by="fixture price reviewer"):
        root = fixture.outputs / fixture.selected.run_id
        contents = b"Synthetic retained price evidence; no remote verification.\n"
        source = StoredArtifact("evidence/price.txt", sha256(contents).hexdigest(), len(contents))
        if (root / source.path).exists():
            self.assertEqual((root / source.path).read_bytes(), contents)
        else:
            self.assertEqual(publish_bytes(root, source.path, contents), source)
        return _BASE_PRICE_WRITER(root, source, checked_by=checked_by, checked_unix=checked,
            credits_per_second=17, minimum_credits=100, usd_per_credit="0.01",
            maximum_additional_cents=additional, clock=lambda: checked)

    def write_config(self, fixture, *, value=None, contents=None):
        body = contents if contents is not None else (json.dumps(
            fixture.config if value is None else value, sort_keys=True) + "\n").encode("utf-8")
        fixture.config_path.write_bytes(body)
        fixture.config_path.chmod(0o600)
        return body

    def call(self, fixture, *, clock=None, run_id=None):
        return command.approve_registered_run(fixture.state, run_id or fixture.selected.run_id,
                                              clock=(lambda: 1200.0) if clock is None else clock)

    def snapshot(self, fixture):
        result = {}
        for name in ("budget.sqlite", "preparation.sqlite"):
            with sqlite3.connect(fixture.state / name) as connection:
                result[name] = tuple(connection.iterdump())
        return result

    def assert_code(self, code, operation):
        with self.assertRaises(command.ApprovalError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)

    def reject_before_writer(self, fixture, code, *, run_id=None):
        clock = Mock(side_effect=RuntimeError("fixture clock must not run"))
        with patch.object(command, "BudgetLedger", side_effect=budget.BudgetError(
                "fixture writer must not open")) as writer:
            self.assert_code(code, lambda: self.call(fixture, clock=clock, run_id=run_id))
            writer.assert_not_called()
            clock.assert_not_called()

    def child(self, arguments):
        with TemporaryDirectory(prefix="approval-child-", dir=self.root) as folder:
            environment = dict(os.environ, TMPDIR=folder, PYTHONDONTWRITEBYTECODE="1")
            environment.pop("RUNWAY_API_KEY", None)
            try:
                return subprocess.run([sys.executable, "-c", _CHILD_START, str(PROJECT_ROOT), *arguments],
                    input=json.dumps(_approval_sources()), text=True, capture_output=True,
                    cwd=PROJECT_ROOT, env=environment, timeout=10, start_new_session=True, check=False)
            except subprocess.TimeoutExpired:
                self.fail("The bounded operator approval child did not finish.")

    def cli_json(self, contents):
        self.assertTrue(contents.strip(), "The operator command did not emit its expected JSON result.")
        try:
            return json.loads(contents)
        except json.JSONDecodeError:
            self.fail("The operator command did not emit valid JSON.")

    def test_registered_run_gets_exact_existing_caps_and_original_deadline_with_one_clock(self):
        with self.fixture() as fixture:
            clock = Mock(return_value=1200.0)
            before_state = self.snapshot(fixture)["preparation.sqlite"]
            owner_lock = fixture.owner._lock
            with patch.object(preparation_state, "State", side_effect=preparation_state.ServiceError(
                    "STATE_UNAVAILABLE")) as state_constructor, \
                    patch.object(submission, "SubmissionCoordinator", side_effect=submission.SubmissionError(
                        "fixture owner must not reopen")) as coordinator:
                result = self.call(fixture, clock=clock)
                state_constructor.assert_not_called()
                coordinator.assert_not_called()
            clock.assert_called_once_with()
            expected = {"runId": fixture.selected.run_id, "status": "APPROVED", "limitCents": 1000,
                "maxAttempts": 6, "costCents": 107, "startedUnix": 1200.0, "deadlineUnix": 11800.0,
                "runStart": asdict(fixture.selected.run_start), "priceQuote": asdict(fixture.quote.record)}
            self.assertEqual(result.record(), expected)
            self.assertEqual((result.run_id, result.status, result.run_start, result.price_quote),
                (fixture.selected.run_id, "APPROVED", fixture.selected.run_start, fixture.quote.record))
            with self.assertRaises(FrozenInstanceError):
                result.status = "changed"
            self.assertEqual(self.snapshot(fixture)["preparation.sqlite"], before_state)
            self.assertEqual(fixture.owner._lock, owner_lock)
            fixture.owner._require_open()
            with sqlite3.connect(fixture.ledger.path) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM approved_runs").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT event_kind FROM run_events").fetchall(), [("APPROVAL_RECORDED",)])
                for table in ("reservations", "actual_charges", "submission_jobs", "submission_events"):
                    self.assertEqual(connection.execute("SELECT count(*) FROM " + table).fetchone()[0], 0)
            self.assertNotIn(str(fixture.root), json.dumps(result.record()))

    def test_config_requires_exact_private_bounded_json_before_any_writer(self):
        with self.fixture() as fixture:
            original = fixture.config_path.read_bytes()
            cases = [b"", b" " * 16385, b"{", b"\xff", b"[]", original.rstrip()[:-1] + b',"approvedBy":"other"}',
                json.dumps(fixture.config | {"paidSubmission": True}).encode(),
                json.dumps(fixture.config | {"approvedBy": " "}).encode(),
                json.dumps(fixture.config | {"approvedBy": "x" * 201}).encode(),
                json.dumps(fixture.config | {"schemaVersion": "2"}).encode(),
                json.dumps(fixture.config | {"manifestType": "browser_approval"}).encode(),
                json.dumps(fixture.config | {"outputRoot": "relative/runs"}).encode(),
                json.dumps(fixture.config | {"outputRoot": str(fixture.outputs) + "/../runs"}).encode(),
                json.dumps(fixture.config | {"approvedBy": float("nan")}).encode(),
                json.dumps(fixture.config | {"priceQuote": asdict(fixture.quote.record) | {"size": True}}).encode(),
                json.dumps(fixture.config | {"priceQuote": asdict(fixture.quote.record) | {"path": "../price.json"}}).encode()]
            before = self.snapshot(fixture)
            for body in cases:
                with self.subTest(body=body[:70]):
                    self.write_config(fixture, contents=body)
                    self.reject_before_writer(fixture, "CONFIG_INVALID")
            self.write_config(fixture, contents=original)
            fixture.config_path.chmod(0o644)
            self.reject_before_writer(fixture, "CONFIG_INVALID")
            fixture.config_path.chmod(0o600)
            actual_uid = os.getuid()
            with patch.object(operator_config.os, "getuid", return_value=actual_uid + 1):
                self.reject_before_writer(fixture, "CONFIG_INVALID")
            fixture.config_path.unlink()
            self.reject_before_writer(fixture, "CONFIG_INVALID")
            target = fixture.root / "other-config.json"
            target.write_bytes(original)
            target.chmod(0o600)
            fixture.config_path.symlink_to(target)
            self.reject_before_writer(fixture, "CONFIG_INVALID")
            self.assertEqual(self.snapshot(fixture), before)

    def test_exact_config_byte_limit_is_accepted(self):
        with self.fixture() as fixture:
            original = fixture.config_path.read_bytes()
            self.write_config(fixture, contents=original + b" " * (16384 - len(original)))
            self.assertEqual(fixture.config_path.stat().st_size, 16384)
            self.assertEqual(self.call(fixture).status, "APPROVED")

    def test_missing_empty_foreign_or_incomplete_databases_cannot_be_repaired_into_approval(self):
        for kind in ("missing_ledger", "empty_ledger", "foreign_ledger", "missing_submission_events",
                     "missing_run_events", "missing_state", "empty_state", "ledger_symlink",
                     "ledger_uid", "journal_uid"):
            with self.subTest(kind=kind), self.fixture() as fixture:
                ledger = fixture.ledger.path
                if kind == "missing_ledger":
                    ledger.unlink()
                elif kind == "empty_ledger":
                    ledger.write_bytes(b"")
                elif kind == "foreign_ledger":
                    ledger.unlink()
                    with sqlite3.connect(ledger) as connection:
                        connection.execute("CREATE TABLE unrelated(value TEXT)")
                elif kind.startswith("missing_") and kind.removeprefix("missing_") in ("submission_events", "run_events"):
                    with sqlite3.connect(ledger) as connection:
                        connection.execute("DROP TABLE " + kind.removeprefix("missing_"))
                elif kind == "missing_state":
                    (fixture.state / "preparation.sqlite").unlink()
                elif kind == "empty_state":
                    (fixture.state / "preparation.sqlite").write_bytes(b"")
                elif kind == "ledger_symlink":
                    moved = fixture.root / "original-ledger.sqlite"
                    ledger.rename(moved)
                    ledger.symlink_to(moved)
                if kind in ("ledger_uid", "journal_uid"):
                    candidate = ledger if kind == "ledger_uid" else Path(str(ledger) + "-journal")
                    if kind == "journal_uid":
                        candidate.write_bytes(b"")
                    original_lstat = Path.lstat

                    def wrong_owner(path):
                        info = original_lstat(path)
                        if path == candidate:
                            values = list(info)
                            values[4] += 1
                            return os.stat_result(values)
                        return info

                    with patch.object(Path, "lstat", wrong_owner):
                        self.reject_before_writer(fixture, "STATE_UNAVAILABLE")
                else:
                    self.reject_before_writer(fixture, "STATE_UNAVAILABLE")
                if kind == "missing_ledger":
                    self.assertFalse(ledger.exists(), "Approval created a missing ledger.")
                if kind == "empty_ledger":
                    self.assertEqual(ledger.read_bytes(), b"", "Approval initialized an empty ledger.")

    def test_registration_owner_and_exact_start_reference_are_required(self):
        for kind in ("unknown_run", "wrong_output_root", "missing_start", "changed_valid_start", "bad_run_id"):
            with self.subTest(kind=kind), self.fixture() as fixture:
                run_id = fixture.selected.run_id
                if kind == "unknown_run":
                    run_id = "run-" + "f" * 32
                elif kind == "bad_run_id":
                    run_id = "../private-run"
                elif kind == "wrong_output_root":
                    other = fixture.root / "other-runs"
                    other.mkdir(mode=0o700)
                    self.write_config(fixture, value=fixture.config | {"outputRoot": str(other)})
                else:
                    path = fixture.outputs / run_id / "run-start.json"
                    if kind == "missing_start":
                        path.unlink()
                    else:
                        value = json.loads(path.read_bytes())
                        value["startedAtUnix"] += 1
                        value["deadlineAtUnix"] += 1
                        path.write_text(json.dumps(value))
                self.reject_before_writer(fixture, "CONFIG_INVALID" if kind == "bad_run_id" else "STATE_UNAVAILABLE",
                                          run_id=run_id)

    def test_price_record_source_and_recomputed_facts_are_verified_before_writer(self):
        for kind in ("missing_quote", "damaged_quote", "damaged_source", "reserved_cents", "base_cents",
                     "model", "price_url", "missing_fees", "credit_bool"):
            with self.subTest(kind=kind), self.fixture() as fixture:
                root = fixture.outputs / fixture.selected.run_id
                path = root / fixture.quote.record.path
                if kind == "missing_quote":
                    path.unlink()
                elif kind == "damaged_quote":
                    contents = path.read_bytes()
                    path.write_bytes(b"X" + contents[1:])
                elif kind == "damaged_source":
                    source = root / fixture.quote.source.path
                    contents = source.read_bytes()
                    source.write_bytes(b"X" + contents[1:])
                else:
                    value = json.loads(path.read_bytes())
                    if kind == "reserved_cents": value["reservedCents"] += 1
                    elif kind == "base_cents": value["baseCents"] += 1
                    elif kind == "model": value["model"] = "other-model"
                    elif kind == "price_url": value["priceSourceUrl"] = "https://invalid.example/price"
                    elif kind == "missing_fees": del value["maximumAdditionalCents"]
                    else: value["creditsPerSecond"] = True
                    contents = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
                    record = publish_bytes(root, "provider_pricing/" + sha256(contents).hexdigest() + ".json", contents)
                    self.write_config(fixture, value=fixture.config | {"priceQuote": asdict(record)})
                self.reject_before_writer(fixture, "PRICE_INVALID")

    def test_new_approval_checks_one_current_price_and_original_experiment_window(self):
        for now, code in ((999.0, "APPROVAL_TIME_INVALID"), (11800.0, "APPROVAL_TIME_INVALID"),
                          (1099.0, "PRICE_STALE"), (1400.001, "PRICE_STALE"),
                          (True, "APPROVAL_TIME_INVALID"), (float("nan"), "APPROVAL_TIME_INVALID"),
                          (float("inf"), "APPROVAL_TIME_INVALID"), (-1, "APPROVAL_TIME_INVALID"),
                          (10 ** 400, "APPROVAL_TIME_INVALID")):
            with self.subTest(now=str(now)[:40]), self.fixture() as fixture:
                before = self.snapshot(fixture)
                clock = Mock(return_value=now)
                self.assert_code(code, lambda: self.call(fixture, clock=clock))
                clock.assert_called_once_with()
                self.assertEqual(self.snapshot(fixture), before, "A failed approval-time check changed ledger rows.")
        with self.fixture() as fixture:
            clock = Mock(return_value=1400.0)
            self.assertEqual(self.call(fixture, clock=clock).record()["startedUnix"], 1400.0)
            clock.assert_called_once_with()

    def test_exact_stopped_repeat_preserves_live_submission_without_clock_writer_or_recovery(self):
        with self.fixture() as fixture:
            first = self.call(fixture)
            fixture.ledger.clock = lambda: 1201.0
            reserved = fixture.ledger.reserve(fixture.selected.run_id, "held attempt", "a" * 64)
            self.assertTrue(fixture.owner._claim(reserved.attempt_id))
            fixture.ledger.clock = lambda: 11800.0
            with self.assertRaises(budget.BudgetExceeded):
                fixture.ledger.reserve(fixture.selected.run_id, "expired attempt", "a" * 64)
            before = self.snapshot(fixture)
            clock = Mock(side_effect=RuntimeError("fixture clock must not run"))
            with patch.object(command, "BudgetLedger", side_effect=budget.BudgetError(
                    "fixture writer must not open")) as writer, \
                    patch.object(pricing, "record_price_quote", side_effect=pricing.PricingError(
                        "fixture quote must not be published")) as publisher, \
                    patch.object(preparation_state, "State", side_effect=preparation_state.ServiceError(
                        "STATE_UNAVAILABLE")) as state_constructor, \
                    patch.object(submission, "SubmissionCoordinator", side_effect=submission.SubmissionError(
                        "fixture owner must not reopen")) as coordinator:
                repeated = self.call(fixture, clock=clock)
                for forbidden in (clock, writer, publisher, state_constructor, coordinator):
                    forbidden.assert_not_called()
            self.assertEqual(repeated.record(), first.record() | {"status": "ALREADY_APPROVED"})
            self.assertEqual(repeated.approval.time_stopped, 1)
            self.assertEqual(fixture.owner.get(reserved.attempt_id).state, "SUBMITTING")
            self.assertEqual(self.snapshot(fixture), before)
            fixture.owner._require_open()

    def test_matching_cost_repeat_returns_selected_quote_without_claiming_original_quote_identity(self):
        with self.fixture() as fixture:
            first = self.call(fixture)
            other = self.quote(fixture, checked=1100.0, checked_by="different retained price reviewer")
            self.assertNotEqual(other.record, first.price_quote)
            self.write_config(fixture, value=fixture.config | {"priceQuote": asdict(other.record)})
            before = self.snapshot(fixture)
            clock = Mock(side_effect=RuntimeError("fixture clock must not run"))
            with patch.object(command, "BudgetLedger", side_effect=budget.BudgetError(
                    "fixture writer must not open")) as writer:
                repeated = self.call(fixture, clock=clock)
                clock.assert_not_called()
                writer.assert_not_called()
            self.assertEqual(repeated.status, "ALREADY_APPROVED")
            self.assertEqual(repeated.price_quote, other.record)
            self.assertEqual(repeated.approval, first.approval)
            self.assertEqual(set(repeated.record()), set(first.record()))
            self.assertEqual(self.snapshot(fixture), before)

    def test_retained_terms_cannot_be_changed_into_an_exact_repeat(self):
        for field, value in (("approved_by", "different operator"), ("limit_cents", 999),
                             ("max_attempts", 5), ("cost_cents", 108),
                             ("started_unix", 999.0), ("deadline_unix", 11799.0)):
            with self.subTest(field=field), self.fixture() as fixture:
                self.call(fixture)
                with sqlite3.connect(fixture.ledger.path) as connection:
                    connection.execute("UPDATE approved_runs SET " + field + "=? WHERE run_id=?",
                                       (value, fixture.selected.run_id))
                before = self.snapshot(fixture)
                self.reject_before_writer(fixture, "APPROVAL_MISMATCH")
                self.assertEqual(self.snapshot(fixture), before)

    def test_other_registered_run_cannot_allocate_the_same_dedicated_authority(self):
        with self.fixture() as fixture:
            self.call(fixture)
            fixture.selected = fixture.service.create_run()
            fixture.quote = self.quote(fixture, checked=1100.0)
            self.write_config(fixture, value=fixture.config | {"priceQuote": asdict(fixture.quote.record)})
            before = self.snapshot(fixture)
            self.assert_code("APPROVAL_DENIED", lambda: self.call(fixture))
            self.assertEqual(self.snapshot(fixture), before)

    def test_cli_reports_approved_retained_and_safe_errors_without_reopening_owner(self):
        now = time.time()
        with self.fixture(started=now - 30, checked=now - 10) as fixture:
            arguments = ["--state-dir", str(fixture.state), "--run-id", fixture.selected.run_id]
            first = self.child(arguments)
            self.assertEqual((first.returncode, first.stderr), (0, ""), first.stderr)
            record = self.cli_json(first.stdout)
            self.assertEqual((record["runId"], record["status"], record["costCents"]),
                             (fixture.selected.run_id, "APPROVED", 107))
            before = self.snapshot(fixture)
            second = self.child(arguments)
            self.assertEqual((second.returncode, second.stderr), (2, ""), second.stderr)
            self.assertEqual(self.cli_json(second.stdout), record | {"status": "ALREADY_APPROVED"})
            self.assertEqual(self.snapshot(fixture), before)
            fixture.owner._require_open()
            for bad, code in (([], "INVALID_ARGUMENTS"), (arguments + ["--limit-cents", "1000"], "INVALID_ARGUMENTS"),
                              (["--state-dir", str(fixture.root / "missing-private-state"),
                                "--run-id", fixture.selected.run_id], "CONFIG_INVALID")):
                with self.subTest(code=code):
                    result = self.child(bad)
                    self.assertEqual((result.returncode, result.stdout), (1, ""))
                    self.assertEqual(self.cli_json(result.stderr), {"error": code})
                    self.assertNotIn(str(fixture.root), result.stdout + result.stderr)
            self.assertFalse((fixture.root / "missing-private-state").exists())

    def test_fifo_configuration_fails_in_a_bounded_child_without_opening_writer(self):
        with self.fixture() as fixture:
            fixture.config_path.unlink()
            os.mkfifo(fixture.config_path, mode=0o600)
            before = self.snapshot(fixture)
            target = fixture.config_path.lstat()
            original_open = os.open
            requested_flags = []

            def observe_open(path, flags, *arguments, **keywords):
                if os.fspath(path) in (fixture.config_path.name, str(fixture.config_path)):
                    info = os.stat(path, dir_fd=keywords.get("dir_fd"), follow_symlinks=False)
                    if (info.st_dev, info.st_ino) == (target.st_dev, target.st_ino):
                        requested_flags.append(flags)
                        # A missing production flag must produce an assertion,
                        # not block this process while opening the real FIFO.
                        flags |= os.O_NONBLOCK
                return original_open(path, flags, *arguments, **keywords)

            with patch.object(os, "open", observe_open):
                self.reject_before_writer(fixture, "CONFIG_INVALID")
            self.assertEqual(len(requested_flags), 1, "The fixed configuration FIFO was not opened exactly once.")
            self.assertTrue(requested_flags[0] & os.O_NONBLOCK,
                            "The configuration reader did not request a nonblocking FIFO open.")
            result = self.child(["--state-dir", str(fixture.state), "--run-id", fixture.selected.run_id])
            self.assertEqual((result.returncode, result.stdout), (1, ""))
            self.assertEqual(self.cli_json(result.stderr), {"error": "CONFIG_INVALID"})
            self.assertEqual(self.snapshot(fixture), before)


if __name__ == "__main__":
    unittest.main()
