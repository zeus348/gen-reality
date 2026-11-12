"""Keep the experiment deadline fixed before preparation and spending."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import Mock, patch

import httpx

from app import prepare, submit
from core import run_start, budget
from core.storage import read_verified
from features.inference import submission
from features.placement.math import Placement
import test_preflight as fixtures


def _run_start_source():
    """Use the tested module's source for the isolated reader process."""
    return Path(run_start.__file__).read_text(encoding="utf-8")


class RunStartTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / "run-1/run-start.json"

    def document(self, **changes):
        return {"schemaVersion": "1.4", "manifestType": "experiment_start", "runId": "run-1",
                "startedAtUnix": 1000.25, "deadlineAtUnix": 11800.25,
                "maxElapsedSeconds": 10800} | changes

    def write(self, contents):
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_bytes(contents)

    def test_start_is_immutable_and_repeat_needs_no_clock(self):
        clock = Mock(return_value=1000.25)
        started = run_start.start_run(self.root, run_id="run-1", clock=clock)
        clock.assert_called_once_with()
        self.assertEqual((started.run_id, started.started_unix, started.deadline_unix),
                         ("run-1", 1000.25, 11800.25))
        contents = self.path.read_bytes()
        self.assertEqual(json.loads(contents), self.document())
        self.assertEqual(started.record.path, "run-start.json")
        self.assertEqual((started.record.sha256, started.record.size), (sha256(contents).hexdigest(), len(contents)))
        self.assertEqual(read_verified(self.path.parent, started.record), contents)
        failed_clock = Mock(side_effect=RuntimeError("clock unavailable"))
        self.assertEqual(run_start.start_run(self.root, run_id="run-1", clock=failed_clock), started)
        self.assertEqual(run_start.read_run_start(self.root, run_id="run-1"), started)
        failed_clock.assert_not_called()
        self.assertEqual(self.path.read_bytes(), contents)

    def test_concurrent_starters_return_one_durable_winner(self):
        barrier = Barrier(12)
        publish = run_start.publish_bytes

        def publish_together(*args):
            # All callers pass the empty-root check before any record is linked.
            barrier.wait(timeout=10)
            return publish(*args)

        def start(index):
            try:
                return run_start.start_run(self.root, run_id="run-1", clock=lambda: 1000.0 + index)
            except run_start.RunStartError:
                # Another writer's incomplete temporary file may block this call.
                return None

        with patch.object(run_start, "publish_bytes", side_effect=publish_together), \
                ThreadPoolExecutor(max_workers=12) as pool:
            records = list(pool.map(start, range(12)))
        successful = [record for record in records if record is not None]
        self.assertEqual(len(successful), 12)
        winner = successful[0]
        self.assertTrue(all(record == winner for record in successful))
        failed_clock = Mock(side_effect=RuntimeError("clock unavailable"))
        self.assertEqual(run_start.start_run(self.root, run_id="run-1", clock=failed_clock), winner)
        failed_clock.assert_not_called()
        self.assertEqual(winner, run_start.read_run_start(self.root, run_id="run-1"))
        self.assertEqual([path.name for path in self.path.parent.iterdir()], ["run-start.json"])
        self.assertIn(winner.started_unix, [1000.0 + index for index in range(12)])

    def test_nonempty_legacy_run_is_not_adopted_or_changed(self):
        self.path.parent.mkdir()
        legacy = self.path.parent / "attempts"
        legacy.mkdir()
        original = legacy / "original.bin"
        original.write_bytes(b"retained historical evidence")
        clock = Mock(return_value=1000.0)
        with self.assertRaises(run_start.RunStartError):
            run_start.start_run(self.root, run_id="run-1", clock=clock)
        self.assertFalse(self.path.exists())
        self.assertEqual(original.read_bytes(), b"retained historical evidence")

    def test_missing_reader_does_not_create_directories(self):
        with self.assertRaises(run_start.RunStartError):
            run_start.read_run_start(self.root, run_id="missing")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_ids_and_clock_values_cannot_publish_start(self):
        for run_id in ("../escape", "", "x" * 65, True, "a/b"):
            with self.subTest(run_id=run_id), self.assertRaises(run_start.RunStartError):
                run_start.start_run(self.root, run_id=run_id, clock=lambda: 1000.0)
        for now in (True, None, "1000", -1, float("nan"), float("inf"), 253402290000):
            with self.subTest(clock=now), self.assertRaises(run_start.RunStartError):
                run_start.start_run(self.root, run_id="run-1", clock=lambda: now)
            self.assertFalse(self.path.exists())

    def test_latest_supported_start_preserves_full_duration(self):
        started = run_start.start_run(self.root, run_id="run-1", clock=lambda: 253402289999)
        self.assertEqual(started.started_unix, 253402289999)
        self.assertEqual(started.deadline_unix, 253402300799)
        self.assertEqual(run_start.read_run_start(self.root, run_id="run-1"), started)

    def test_reader_rejects_wrong_identity_extra_fields_and_invalid_times(self):
        changes = [{"schemaVersion": "other"}, {"manifestType": "approval"}, {"runId": "other"},
                   {"accepted": True}, {"maxElapsedSeconds": 10799}, {"maxElapsedSeconds": True},
                   {"maxElapsedSeconds": 10800.0},
                   {"startedAtUnix": True}, {"startedAtUnix": -1}, {"startedAtUnix": "1000.25"},
                   {"startedAtUnix": float("nan")}, {"deadlineAtUnix": float("inf")},
                   {"deadlineAtUnix": 11801.25}, {"deadlineAtUnix": 1000.25},
                   {"deadlineAtUnix": True}]
        for fields in changes:
            contents = json.dumps(self.document(**fields)).encode()
            self.write(contents)
            with self.subTest(fields=fields), self.assertRaises(run_start.RunStartError):
                run_start.read_run_start(self.root, run_id="run-1")
            self.assertEqual(self.path.read_bytes(), contents)

    def test_reader_rejects_missing_fields_duplicate_keys_and_invalid_json(self):
        missing = self.document()
        del missing["maxElapsedSeconds"]
        duplicated = json.dumps(self.document()).replace('"runId": "run-1"',
                                                         '"runId": "run-1", "runId": "run-1"')
        for contents in (json.dumps(missing).encode(), duplicated.encode(), b"", b"[]", b"null", b"{", b"\xff",
                         json.dumps(self.document()).encode("utf-16")):
            self.write(contents)
            with self.subTest(contents=contents), self.assertRaises(run_start.RunStartError):
                run_start.read_run_start(self.root, run_id="run-1")

    def test_reader_accepts_bound_and_rejects_one_extra_byte(self):
        contents = json.dumps(self.document()).encode()
        self.write(contents + b" " * (4096 - len(contents)))
        record = run_start.read_run_start(self.root, run_id="run-1")
        self.assertEqual(record.record.size, 4096)
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaises(run_start.RunStartError):
            run_start.read_run_start(self.root, run_id="run-1")

    def test_reader_rejects_symlinks_directories_and_fifo(self):
        target = self.root / "target.json"
        target.write_bytes(json.dumps(self.document()).encode())
        self.path.parent.mkdir()
        self.path.symlink_to(target)
        with self.assertRaises(run_start.RunStartError):
            run_start.read_run_start(self.root, run_id="run-1")
        self.path.unlink()
        self.path.mkdir()
        with self.assertRaises(run_start.RunStartError):
            run_start.read_run_start(self.root, run_id="run-1")
        self.path.rmdir()
        os.mkfifo(self.path)
        child = """
from pathlib import Path
import sys
from core import run_start

exec(compile(sys.stdin.read(), run_start.__file__, "exec"), run_start.__dict__)
try:
    run_start.read_run_start(Path(sys.argv[1]), run_id="run-1")
except run_start.RunStartError:
    raise SystemExit(0)
raise AssertionError("The reader accepted a FIFO.")
"""
        try:
            result = subprocess.run([sys.executable, "-c", child, str(self.root)],
                input=_run_start_source(), cwd=Path(__file__).resolve().parents[2],
                text=True, capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            self.fail("Run-start reader blocked on a FIFO for more than five seconds.")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_bytes(), json.dumps(self.document()).encode())

    def test_run_directory_symlink_is_never_followed(self):
        target = self.root / "private-target"
        target.mkdir()
        self.path.parent.symlink_to(target, target_is_directory=True)
        for action in (lambda: run_start.start_run(self.root, run_id="run-1", clock=lambda: 1000.0),
                       lambda: run_start.read_run_start(self.root, run_id="run-1")):
            with self.assertRaises(run_start.RunStartError):
                action()
        self.assertEqual(list(target.iterdir()), [])

    def test_preparation_starts_before_media_and_returns_the_same_record(self):
        observed = []
        result = {"runId": "run-1", "attemptId": "att-1", "state": "PREPARED"}

        def prepare_media(*args, **kwargs):
            observed.append(run_start.read_run_start(self.root, run_id="run-1"))
            return result.copy()

        arguments = (self.root / "source.mp4", self.root / "asset.png", self.root,
                     "run-1", "att-1", Placement(0.5, 0.5, 0.25))
        with patch.object(prepare, "prepare_attempt", side_effect=prepare_media) as media:
            prepared = prepare.prepare_run_attempt(*arguments, non_human=True, clean_source=True,
                                                    source_kind="synthetic_fixture", clock=lambda: 1000.25)
        media.assert_called_once_with(*arguments, non_human=True, clean_source=True,
                                      source_kind="synthetic_fixture", prompt=prepare.DEFAULT_PROMPT)
        self.assertEqual(len(observed), 1)
        self.assertEqual(prepared, result | {"runStart": asdict(observed[0].record)})

    def test_media_failure_keeps_start_and_retry_cannot_reset_it(self):
        arguments = (self.root / "source.mp4", self.root / "asset.png", self.root,
                     "run-1", "att-1", Placement(0.5, 0.5, 0.25))
        with patch.object(prepare, "prepare_attempt", side_effect=OSError("fixture media failure")):
            with self.assertRaises(OSError):
                prepare.prepare_run_attempt(*arguments, non_human=True, clean_source=True, clock=lambda: 1000.25)
        started = run_start.read_run_start(self.root, run_id="run-1")
        with patch.object(prepare, "prepare_attempt", return_value={}):
            result = prepare.prepare_run_attempt(*arguments, non_human=True, clean_source=True, clock=lambda: 2000.0)
        self.assertEqual(result["runStart"], asdict(started.record))
        self.assertEqual(run_start.read_run_start(self.root, run_id="run-1"), started)

    def test_existing_start_must_be_durable_before_media_can_begin(self):
        started = run_start.start_run(self.root, run_id="run-1", clock=lambda: 1000.25)
        contents = self.path.read_bytes()
        clock = Mock(side_effect=RuntimeError("clock must not be read again"))
        with patch.object(run_start.os, "fsync", side_effect=OSError("fixture sync failure")), \
                patch.object(prepare, "prepare_attempt", return_value={}) as media:
            with self.assertRaises(run_start.RunStartError):
                prepare.prepare_run_attempt(self.root / "source.mp4", self.root / "asset.png", self.root,
                    "run-1", "att-1", Placement(0.5, 0.5, 0.25), non_human=True, clean_source=True, clock=clock)
            # Ordinary historical reads do not need a write barrier.
            self.assertEqual(run_start.read_run_start(self.root, run_id="run-1"), started)
        clock.assert_not_called()
        media.assert_not_called()
        self.assertEqual(self.path.read_bytes(), contents)

    def test_invalid_start_blocks_media(self):
        self.write(b"invalid saved start")
        with patch.object(prepare, "prepare_attempt", return_value={}) as media:
            with self.assertRaises(run_start.RunStartError):
                prepare.prepare_run_attempt(self.root / "source.mp4", self.root / "asset.png", self.root,
                    "run-1", "att-1", Placement(0.5, 0.5, 0.25), non_human=True, clean_source=True)
        media.assert_not_called()

    def test_interruption_before_start_publication_is_retryable_before_media(self):
        arguments = (self.root / "source.mp4", self.root / "asset.png", self.root,
                     "run-1", "att-1", Placement(0.5, 0.5, 0.25))
        with patch.object(run_start, "publish_bytes", side_effect=KeyboardInterrupt), \
                patch.object(prepare, "prepare_attempt", return_value={}) as media:
            with self.assertRaises(KeyboardInterrupt):
                prepare.prepare_run_attempt(*arguments, non_human=True, clean_source=True, clock=lambda: 1000.0)
        media.assert_not_called()
        self.assertFalse(self.path.exists())
        with patch.object(prepare, "prepare_attempt", return_value={}):
            result = prepare.prepare_run_attempt(*arguments, non_human=True, clean_source=True, clock=lambda: 2000.0)
        saved = run_start.read_run_start(self.root, run_id="run-1")
        self.assertEqual(saved.started_unix, 2000.0)
        self.assertEqual(result["runStart"], asdict(saved.record))

    def test_failure_after_publication_preserves_start_and_returns_safe_error(self):
        publish = run_start.publish_bytes

        def publish_then_fail(*args):
            publish(*args)
            raise OSError("private-fixture-path")

        with patch.object(run_start, "publish_bytes", side_effect=publish_then_fail):
            with self.assertRaises(run_start.RunStartError) as error:
                run_start.start_run(self.root, run_id="run-1", clock=lambda: 1000.25)
        self.assertNotIn("private-fixture-path", str(error.exception))
        contents = self.path.read_bytes()
        failed_clock = Mock(side_effect=RuntimeError("clock unavailable"))
        saved = run_start.start_run(self.root, run_id="run-1", clock=failed_clock)
        self.assertEqual(saved.started_unix, 1000.25)
        self.assertEqual(self.path.read_bytes(), contents)
        failed_clock.assert_not_called()

    def test_stranded_publication_file_is_kept_and_blocks_adoption(self):
        self.path.parent.mkdir()
        pending = self.path.parent / ".pending-fixture"
        pending.write_bytes(b"interrupted publication")
        with patch.object(prepare, "prepare_attempt", return_value={}) as media:
            with self.assertRaises(run_start.RunStartError):
                prepare.prepare_run_attempt(self.root / "source.mp4", self.root / "asset.png", self.root,
                    "run-1", "att-1", Placement(0.5, 0.5, 0.25), non_human=True, clean_source=True,
                    clock=lambda: 2000.0)
        media.assert_not_called()
        self.assertEqual(pending.read_bytes(), b"interrupted publication")
        self.assertFalse(self.path.exists())


class RunDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def approve(self, ledger, **changes):
        ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                           max_attempts=6, cost_cents=140, **changes)

    def approval(self, ledger):
        with ledger._connection() as connection:
            return dict(connection.execute("SELECT * FROM approved_runs").fetchone())

    def test_delayed_approval_uses_original_fractional_deadline_and_one_clock_read(self):
        started = run_start.start_run(self.root, run_id="run-1", clock=lambda: 1000.25)
        clock = Mock(return_value=1100.125)
        ledger = budget.BudgetLedger(self.root / "budget.sqlite", clock=clock)
        self.approve(ledger, deadline_unix_cap=started.deadline_unix)
        row = self.approval(ledger)
        self.assertEqual((row["started_unix"], row["deadline_unix"], row["last_checked_unix"]),
                         (1100.125, 11800.25, 1100.125))
        clock.assert_called_once_with()

    def test_shorter_approval_duration_is_not_extended_by_later_cap(self):
        ledger = budget.BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1100.125)
        self.approve(ledger, max_elapsed_minutes=1, deadline_unix_cap=11800.25)
        self.assertEqual(self.approval(ledger)["deadline_unix"], 1160.125)

    def test_sub_minute_fractional_cap_is_saved_in_approval_and_event(self):
        clock = Mock(return_value=1000.25)
        ledger = budget.BudgetLedger(self.root / "budget.sqlite", clock=clock)
        self.approve(ledger, deadline_unix_cap=1000.75)
        self.assertEqual(self.approval(ledger)["deadline_unix"], 1000.75)
        with ledger._connection() as connection:
            events = connection.execute("SELECT event_kind, recorded_unix, record_json FROM run_events").fetchall()
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["event_kind"], events[0]["recorded_unix"]),
                         ("APPROVAL_RECORDED", 1000.25))
        self.assertEqual(json.loads(events[0]["record_json"])["deadline_unix"], 1000.75)
        clock.assert_called_once_with()

    def test_uncapped_low_level_approval_keeps_existing_duration(self):
        ledger = budget.BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1100.125)
        self.approve(ledger)
        self.assertEqual(self.approval(ledger)["deadline_unix"], 11900.125)

    def test_invalid_or_elapsed_cap_creates_no_approval_or_event(self):
        ledger = budget.BudgetLedger(self.root / "budget.sqlite", clock=lambda: 1000.25)
        for cap in (True, "11800", -1, float("nan"), float("inf"), 253402300800, 1000.25, 1000.0):
            with self.subTest(cap=cap), self.assertRaises(budget.BudgetError):
                self.approve(ledger, deadline_unix_cap=cap)
            with ledger._connection() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM approved_runs").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM run_events").fetchone()[0], 0)

    def test_exact_experiment_deadline_blocks_final_send_and_stays_stopped(self):
        started = run_start.start_run(self.root, run_id="run-1", clock=lambda: 1000.25)
        now = 11700.0
        ledger = budget.BudgetLedger(self.root / "budget.sqlite", clock=lambda: now)
        self.approve(ledger, deadline_unix_cap=started.deadline_unix)
        attempt = ledger.reserve("run-1", "first", "a" * 64).attempt_id
        post = Mock(return_value="fixture-task")
        with submission.SubmissionCoordinator(ledger) as owner:
            now = 11800.0

            def create():
                nonlocal now
                now = 11800.25
                owner.require_send_allowed(attempt)
                return post()

            self.assertEqual(owner.submit_once(attempt, create).state, "SUBMISSION_UNKNOWN")
            post.assert_not_called()
            self.assertEqual(self.approval(ledger)["time_stopped"], 1)
            now = 11700.0
            with self.assertRaises(budget.BudgetExceeded):
                ledger.reserve("run-1", "after rollback", "a" * 64)
            with ledger._connection() as connection:
                self.assertEqual(ledger._usage(connection, "run-1"), (1, 140, False))

    def test_send_just_before_original_deadline_can_record_later_response(self):
        started = run_start.start_run(self.root, run_id="run-1", clock=lambda: 1000.25)
        now = 11700.0
        ledger = budget.BudgetLedger(self.root / "budget.sqlite", clock=lambda: now)
        self.approve(ledger, deadline_unix_cap=started.deadline_unix)
        attempt = ledger.reserve("run-1", "first", "a" * 64).attempt_id
        post = Mock(return_value="fixture-task")
        with submission.SubmissionCoordinator(ledger) as owner:
            now = 11800.249

            def create():
                nonlocal now
                owner.require_send_allowed(attempt)
                task = post()
                now = 11810.0
                return task

            result = owner.submit_once(attempt, create)
            self.assertEqual((result.state, result.provider_task_id), ("QUEUED", "fixture-task"))
            post.assert_called_once_with()


class StartedSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PreflightTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.checked = self.fixture.check()
        self.now = 1001.0
        self.path = self.root / "budget.sqlite"
        self.posts = []

    def approve(self, *, capped=True, now=1001.0):
        ledger = budget.BudgetLedger(self.path, clock=lambda: now)
        changes = {"deadline_unix_cap": 11800.0} if capped else {}
        ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=1000,
                           max_attempts=6, cost_cents=140, **changes)
        return ledger

    def send(self, *, clock=None, uncertain=False):
        def provider(request):
            self.posts.append(request)
            if uncertain:
                raise httpx.ReadTimeout("fixture uncertain result")
            return httpx.Response(200, json={"id": "fixture-task"})

        return submit.submit_saved_request(self.root, self.path, self.checked.evidence,
            run_id="run-1", attempt_id=fixtures.ATTEMPT, api_key_env="TEST_RUN_START_KEY",
            clock=clock or (lambda: self.now), transport=httpx.MockTransport(provider))

    def counts(self, ledger):
        with ledger._connection() as connection:
            return tuple(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                         for table in ("reservations", "submission_jobs"))

    def test_capped_saved_request_uses_exact_checked_payload(self):
        ledger = self.approve()
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}):
            result = self.send()
        self.assertEqual((result.state, result.provider_task_id), ("QUEUED", "fixture-task"))
        self.assertEqual([request.content for request in self.posts], [self.checked.frozen.request_body])
        self.assertEqual(self.counts(ledger), (1, 1))

    def test_approval_at_saved_start_is_allowed(self):
        ledger = self.approve(now=1000.0)
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}):
            self.assertEqual(self.send().state, "QUEUED")
        self.assertEqual(self.counts(ledger), (1, 1))
        self.assertEqual(len(self.posts), 1)

    def test_missing_start_blocks_new_reservation_and_post(self):
        ledger = self.approve()
        (self.fixture.run / "run-start.json").unlink()
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}), self.assertRaises(ValueError):
            self.send()
        self.assertEqual(self.counts(ledger), (0, 0))
        self.assertEqual(self.posts, [])

    def test_bad_start_blocks_existing_reservation_without_claim(self):
        ledger = self.approve()
        ledger.reserve("run-1", "existing", self.checked.frozen.payload_sha256, attempt_id=fixtures.ATTEMPT)
        (self.fixture.run / "run-start.json").write_bytes(b"invalid saved start")
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}), self.assertRaises(ValueError):
            self.send()
        self.assertEqual(self.counts(ledger), (1, 0))
        self.assertEqual(self.posts, [])

    def test_uncapped_new_approval_cannot_reserve(self):
        ledger = self.approve(capped=False)
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}), self.assertRaises(budget.BudgetError):
            self.send()
        self.assertEqual(self.counts(ledger), (0, 0))
        self.assertEqual(self.posts, [])

    def test_uncapped_existing_reservation_cannot_claim_or_post(self):
        ledger = self.approve(capped=False)
        ledger.reserve("run-1", "existing", self.checked.frozen.payload_sha256, attempt_id=fixtures.ATTEMPT)
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}), self.assertRaises(budget.BudgetError):
            self.send()
        self.assertEqual(self.counts(ledger), (1, 0))
        self.assertEqual(self.posts, [])

    def test_approval_before_experiment_start_cannot_reserve(self):
        ledger = self.approve(now=999.0)
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}), self.assertRaises(budget.BudgetError):
            self.send()
        self.assertEqual(self.counts(ledger), (0, 0))
        self.assertEqual(self.posts, [])

    def test_known_task_remains_keyless_when_start_file_is_gone(self):
        ledger = self.approve()
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}):
            result = self.send()
        (self.fixture.run / "run-start.json").unlink()
        clock = Mock(side_effect=RuntimeError("clock unavailable"))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.send(clock=clock), result)
        clock.assert_not_called()
        self.assertEqual(self.counts(ledger), (1, 1))
        self.assertEqual(len(self.posts), 1)

    def test_unknown_remains_keyless_when_start_file_is_gone(self):
        ledger = self.approve()
        with patch.dict("os.environ", {"TEST_RUN_START_KEY": fixtures.KEY}):
            result = self.send(uncertain=True)
        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        (self.fixture.run / "run-start.json").unlink()
        clock = Mock(side_effect=RuntimeError("clock unavailable"))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.send(clock=clock), result)
        clock.assert_not_called()
        self.assertEqual(self.counts(ledger), (1, 1))
        self.assertEqual(len(self.posts), 1)
