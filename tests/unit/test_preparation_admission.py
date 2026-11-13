"""Keep one media operation admitted until its actual worker has finished."""

import asyncio
from concurrent.futures import Future
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import patch

from PIL import Image

from app import preparation
from core.budget import BudgetLedger
from features.inference.submission import SubmissionCoordinator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _asset_png():
    image = Image.new("RGBA", (4, 4), (210, 120, 30, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    contents = BytesIO()
    image.save(contents, format="PNG")
    return contents.getvalue()


async def _chunks(contents, reads):
    reads.append(len(contents))
    yield contents


def _service_sources():
    """Supply child source so fault checks never need to edit product files."""
    return {name: (PROJECT_ROOT / name.replace(".", "/")).with_suffix(".py").read_text(encoding="utf-8")
            for name in ("app.preparation_state", "app.preparation_media", "app.preparation")}


_CHILD_START = """
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
scenario = sys.argv[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests" / "unit"))
import app
sources = json.load(sys.stdin)
for name in ("app.preparation_state", "app.preparation_media", "app.preparation"):
    filename = (root / name.replace(".", "/")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    setattr(app, name.rsplit(".", 1)[1], module)
    exec(compile(sources[name], str(filename), "exec"), module.__dict__)
import test_preparation_admission as checks
print(json.dumps(checks._bounded_scenario(scenario), sort_keys=True))
"""


def _bounded_scenario(scenario):
    """Run blocking admission and shutdown faults only in a bounded child."""
    async def exercise(service, run):
        if scenario == "busy":
            reads = []
            async with service.media_operation(run.run_id, expected_revision=0):
                try:
                    async with service.media_operation(run.run_id, expected_revision=0) as media:
                        await media.import_asset(_chunks(_asset_png(), reads), non_human=True)
                except preparation.ServiceError as error:
                    code = error.code
                else:
                    raise AssertionError("Two media operations were admitted.")
                staged = len(list(service.state_dir.glob(".preparation-*")))
            async with service.media_operation(run.run_id, expected_revision=0) as media:
                result = await media.import_asset(_chunks(_asset_png(), []), non_human=True)
            return {"code": code, "reads": len(reads), "staged": staged, "revision": result.revision}

        if scenario == "unused":
            reads = []
            async with service.media_operation(run.run_id, expected_revision=0) as media:
                service.close()
                try:
                    await media.import_asset(_chunks(_asset_png(), reads), non_human=True)
                except preparation.ServiceError as error:
                    code = error.code
                else:
                    raise AssertionError("A closed service accepted an unused handle.")
            return {"code": code, "reads": len(reads)}

        if scenario == "pending":
            submitted = asyncio.Event()
            future = Future()
            calls = []

            def submit(function, *arguments):
                calls.append((function, arguments))
                submitted.set()
                return future

            def finish():
                if not future.done() and future.set_running_or_notify_cancel():
                    function, arguments = calls[0]
                    try:
                        future.set_result(function(*arguments))
                    except BaseException as error:
                        future.set_exception(error)

            async def request():
                async with service.media_operation(run.run_id, expected_revision=0) as media:
                    return await media.import_asset(_chunks(_asset_png(), []), non_human=True)

            with patch.object(service._executor, "submit", side_effect=submit):
                pending = asyncio.create_task(request())
                try:
                    await asyncio.wait_for(submitted.wait(), timeout=5)
                    path = Path(calls[0][1][1])
                    pending.cancel()
                    try:
                        await pending
                    except asyncio.CancelledError:
                        pass
                    if future.done() or future.cancelled():
                        raise AssertionError("Request cancellation canceled accepted worker work.")
                    if not path.is_file():
                        raise AssertionError("Request cancellation removed pending worker input.")
                    finish()
                    future.result(timeout=1)
                    return {"revision": service.get_run(run.run_id).revision,
                            "staging_removed": not path.parent.exists()}
                finally:
                    if calls:
                        finish()
                    if not pending.done():
                        pending.cancel()
                        try:
                            await pending
                        except asyncio.CancelledError:
                            pass

        if scenario == "body":
            reading, resume = asyncio.Event(), asyncio.Event()
            reads = []
            contents = _asset_png()

            async def body():
                reads.append("first")
                yield contents[:16]
                reading.set()
                await resume.wait()
                reads.append("second")
                yield contents[16:]
                reads.append("extra")
                yield b"x"

            async def request():
                async with service.media_operation(run.run_id, expected_revision=0) as media:
                    return await media.import_asset(body(), non_human=True)

            with patch.object(service, "_import_asset", wraps=service._import_asset) as worker:
                pending = asyncio.create_task(request())
                try:
                    await asyncio.wait_for(reading.wait(), timeout=5)
                    await asyncio.wait_for(asyncio.to_thread(service.close), timeout=5)
                    staged_while_closed = len(list(service.state_dir.glob(".preparation-*")))
                    resume.set()
                    try:
                        await pending
                    except preparation.ServiceError as error:
                        code = error.code
                    else:
                        raise AssertionError("A closed service submitted the remaining body.")
                    return {"code": code, "reads": reads, "worker_calls": worker.call_count,
                            "staged_while_closed": staged_while_closed,
                            "staged_after_unwind": len(list(service.state_dir.glob(".preparation-*")))}
                finally:
                    resume.set()
                    if not pending.done():
                        pending.cancel()
                        try:
                            await pending
                        except asyncio.CancelledError:
                            pass

        entered, release, finished = Event(), Event(), Event()
        shutdown_entered, closed = Event(), Event()
        paths, at_close = [], []
        original_worker, original_shutdown = service._import_asset, service._executor.shutdown

        def worker(snapshot, path, non_human):
            paths.append(Path(path))
            entered.set()
            if not release.wait(5):
                raise AssertionError("The shutdown test did not release the worker.")
            try:
                return original_worker(snapshot, path, non_human)
            finally:
                finished.set()

        def shutdown(*args, **kwargs):
            shutdown_entered.set()
            return original_shutdown(*args, **kwargs)

        async def request():
            async with service.media_operation(run.run_id, expected_revision=0) as media:
                await media.import_asset(_chunks(_asset_png(), []), non_human=True)

        def close():
            try:
                service.close()
                at_close.append((finished.is_set(), not paths[0].exists()))
            finally:
                closed.set()

        closer = None
        with patch.object(service, "_import_asset", side_effect=worker), \
                patch.object(service._executor, "shutdown", side_effect=shutdown):
            pending = asyncio.create_task(request())
            try:
                if not await asyncio.to_thread(entered.wait, 5):
                    raise AssertionError("The media worker did not start.")
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass
                if not paths[0].is_file():
                    raise AssertionError("Cancellation removed a running worker's input.")
                closer = Thread(target=close)
                closer.start()
                if not await asyncio.to_thread(shutdown_entered.wait, 5):
                    raise AssertionError("Shutdown did not start draining the worker.")
                try:
                    async with service.media_operation(run.run_id, expected_revision=0):
                        raise AssertionError("Shutdown accepted another operation.")
                except preparation.ServiceError as error:
                    if error.code != "SERVICE_CLOSED":
                        raise AssertionError("Shutdown did not close admission before waiting.") from None
                release.set()
                if not await asyncio.to_thread(closed.wait, 5):
                    raise AssertionError("Shutdown did not finish after the worker.")
                closer.join(timeout=1)
                if closer.is_alive():
                    raise AssertionError("The shutdown thread did not exit.")
                return {"at_close": at_close, "staging_removed": not paths[0].parent.exists()}
            finally:
                release.set()
                if not pending.done():
                    pending.cancel()
                    try:
                        await pending
                    except asyncio.CancelledError:
                        pass
                if closer is not None:
                    closer.join(timeout=6)

    with TemporaryDirectory() as folder:
        state, output = Path(folder) / "state", Path(folder) / "runs"
        state.mkdir(mode=0o700)
        output.mkdir(mode=0o700)
        ledger = BudgetLedger(state / "budget.sqlite", clock=lambda: 1000.0)
        with SubmissionCoordinator(ledger) as owner:
            service = preparation.PreparationService(state, output, owner, clock=lambda: 1000.0)
            try:
                return asyncio.run(exercise(service, service.create_run()))
            finally:
                service.close()


class PreparationAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.state, self.output = Path(self.folder.name) / "state", Path(self.folder.name) / "runs"
        self.state.mkdir(mode=0o700)
        self.output.mkdir(mode=0o700)
        self.ledger = BudgetLedger(self.state / "budget.sqlite", clock=lambda: 1000.0)
        self.owner = SubmissionCoordinator(self.ledger)
        self.addCleanup(self.owner.close)
        self.asset = _asset_png()

    def service(self):
        service = preparation.PreparationService(self.state, self.output, self.owner, clock=lambda: 1000.0)
        self.addCleanup(service.close)
        return service

    async def import_asset(self, service, run_id, revision):
        async with service.media_operation(run_id, expected_revision=revision) as media:
            return await media.import_asset(_chunks(self.asset, []), non_human=True)

    async def wait_event(self, event):
        self.assertTrue(await asyncio.to_thread(event.wait, 5), "A bounded test event was not signaled.")

    def assert_no_staging(self):
        self.assertEqual(list(self.state.glob(".preparation-*")), [])

    def test_busy_admission_reads_no_chunks_and_stages_no_file(self):
        self.assertEqual(self.run_child("busy"),
                         {"code": "MEDIA_BUSY", "reads": 0, "staged": 0, "revision": 1})

    async def test_missing_run_and_stale_revision_fail_before_body_reads(self):
        service = self.service()
        run = service.create_run()
        for run_id, revision, expected in [("run-" + "f" * 32, 0, "RUN_NOT_FOUND"),
                                           (run.run_id, 1, "REVISION_CONFLICT")]:
            with self.subTest(code=expected):
                reads = []
                with self.assertRaises(preparation.ServiceError) as caught:
                    async with service.media_operation(run_id, expected_revision=revision) as media:
                        await media.import_asset(_chunks(self.asset, reads), non_human=True)
                self.assertEqual(caught.exception.code, expected)
                self.assertEqual(reads, [])
                self.assert_no_staging()
        self.assertEqual((await self.import_asset(service, run.run_id, 0)).revision, 1)

    async def test_context_parse_failure_releases_unused_admission(self):
        service = self.service()
        run = service.create_run()
        with self.assertRaises(preparation.ServiceError) as caught:
            async with service.media_operation(run.run_id, expected_revision=0):
                raise ValueError("fixture parse error")
        self.assertEqual(caught.exception.code, "STATE_UNAVAILABLE")
        self.assertNotIn("fixture parse error", str(caught.exception))
        self.assert_no_staging()
        self.assertEqual((await self.import_asset(service, run.run_id, 0)).revision, 1)

    async def test_body_disconnect_cleans_staging_and_releases_admission(self):
        service = self.service()
        run = service.create_run()

        async def disconnected():
            yield self.asset[:16]
            raise ConnectionResetError("fixture disconnected body")

        with patch.object(service, "_import_asset", wraps=service._import_asset) as worker:
            with self.assertRaises(preparation.ServiceError) as caught:
                async with service.media_operation(run.run_id, expected_revision=0) as media:
                    await media.import_asset(disconnected(), non_human=True)
            self.assertEqual(caught.exception.code, "STATE_UNAVAILABLE")
            worker.assert_not_called()
        self.assert_no_staging()
        self.assertEqual((await self.import_asset(service, run.run_id, 0)).revision, 1)

    async def test_body_cancellation_before_submit_removes_staging_and_releases_slot(self):
        service = self.service()
        run = service.create_run()
        reading = asyncio.Event()
        resume = asyncio.Event()

        async def body():
            yield self.asset[:16]
            reading.set()
            await resume.wait()
            yield self.asset[16:]

        async def request():
            async with service.media_operation(run.run_id, expected_revision=0) as media:
                return await media.import_asset(body(), non_human=True)

        with patch.object(service, "_import_asset", wraps=service._import_asset) as worker:
            pending = asyncio.create_task(request())
            try:
                await asyncio.wait_for(reading.wait(), timeout=5)
                self.assertEqual(len(list(self.state.glob(".preparation-*"))), 1)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                worker.assert_not_called()
                self.assert_no_staging()
            finally:
                resume.set()
                if not pending.done():
                    pending.cancel()
        self.assertEqual((await self.import_asset(service, run.run_id, 0)).revision, 1)

    async def test_failed_executor_submit_releases_exactly_one_slot(self):
        service = self.service()
        run = service.create_run()
        with patch.object(service._executor, "submit", side_effect=RuntimeError("fixture submit failure")):
            with self.assertRaises(preparation.ServiceError) as caught:
                await self.import_asset(service, run.run_id, 0)
            self.assertEqual(caught.exception.code, "STATE_UNAVAILABLE")
        self.assert_no_staging()
        async with service.media_operation(run.run_id, expected_revision=0):
            with self.assertRaises(preparation.ServiceError) as busy:
                async with service.media_operation(run.run_id, expected_revision=0):
                    self.fail("A second operation was admitted.")
            self.assertEqual(busy.exception.code, "MEDIA_BUSY")
        self.assertEqual((await self.import_asset(service, run.run_id, 0)).revision, 1)

    async def test_repeated_cancellation_keeps_worker_input_and_slot_until_actual_completion(self):
        service = self.service()
        run = service.create_run()
        entered, release, completed = Event(), Event(), Event()
        self.addCleanup(release.set)
        paths, futures = [], []
        original_worker, original_submit = service._import_asset, service._executor.submit

        def worker(snapshot, path, non_human):
            paths.append(Path(path))
            entered.set()
            if not release.wait(5):
                raise AssertionError("The test did not release the media worker.")
            return original_worker(snapshot, path, non_human)

        def submit(*args, **kwargs):
            future = original_submit(*args, **kwargs)
            futures.append(future)
            return future

        with patch.object(service, "_import_asset", side_effect=worker), \
                patch.object(service._executor, "submit", side_effect=submit):
            pending = asyncio.create_task(self.import_asset(service, run.run_id, 0))
            try:
                await self.wait_event(entered)
                futures[0].add_done_callback(lambda future: completed.set())
                pending.cancel()
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertTrue(paths[0].is_file())
                self.assertEqual(paths[0].read_bytes(), self.asset)
                self.assertFalse(futures[0].done())
                self.assertFalse(futures[0].cancelled())
                reads = []
                with self.assertRaises(preparation.ServiceError) as caught:
                    async with service.media_operation(run.run_id, expected_revision=0) as media:
                        await media.import_asset(_chunks(self.asset, reads), non_human=True)
                self.assertEqual(caught.exception.code, "MEDIA_BUSY")
                self.assertEqual(reads, [])
                release.set()
                await self.wait_event(completed)
                self.assertFalse(paths[0].parent.exists())
                self.assertEqual(service.get_run(run.run_id).revision, 1)
            finally:
                release.set()
                if not pending.done():
                    pending.cancel()
                    try:
                        await pending
                    except asyncio.CancelledError:
                        pass
        self.assertEqual((await self.import_asset(service, run.run_id, 1)).revision, 2)

    async def test_immediate_completion_and_used_handle_do_not_over_release(self):
        service = self.service()
        run = service.create_run()

        def immediate_submit(function, *args, **kwargs):
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)
            return future

        held = None
        try:
            with patch.object(service._executor, "submit", side_effect=immediate_submit):
                async with service.media_operation(run.run_id, expected_revision=0) as media:
                    result = await media.import_asset(_chunks(self.asset, []), non_human=True)
                    self.assertEqual(result.revision, 1)
                    self.assert_no_staging()
                    reads = []
                    with self.assertRaises(preparation.ServiceError) as caught:
                        await media.import_asset(_chunks(self.asset, reads), non_human=True)
                    self.assertEqual(caught.exception.code, "OPERATION_USED")
                    self.assertEqual(reads, [])
                    held = service.media_operation(run.run_id, expected_revision=1)
                    await held.__aenter__()
            with self.assertRaises(preparation.ServiceError) as caught:
                async with service.media_operation(run.run_id, expected_revision=1):
                    self.fail("Exiting the used handle released another handle's slot.")
            self.assertEqual(caught.exception.code, "MEDIA_BUSY")
        finally:
            if held is not None:
                await held.__aexit__(None, None, None)
        self.assertEqual((await self.import_asset(service, run.run_id, 1)).revision, 2)

    async def test_worker_failure_cleans_stage_and_returns_only_safe_error(self):
        service = self.service()
        run = service.create_run()
        with patch.object(service, "_import_asset", side_effect=RuntimeError("private worker detail")):
            with self.assertRaises(preparation.ServiceError) as caught:
                await self.import_asset(service, run.run_id, 0)
            self.assertEqual(caught.exception.code, "STATE_UNAVAILABLE")
            self.assertNotIn("private worker detail", str(caught.exception))
        self.assert_no_staging()
        self.assertEqual(service.get_run(run.run_id).revision, 0)
        self.assertEqual((await self.import_asset(service, run.run_id, 0)).revision, 1)

    def run_child(self, scenario):
        try:
            result = subprocess.run([sys.executable, "-c", _CHILD_START, str(PROJECT_ROOT), scenario],
                                    input=json.dumps(_service_sources()).encode("utf-8"),
                                    cwd=PROJECT_ROOT, capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            self.fail("Preparation admission or shutdown exceeded the bounded child time limit.")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        return json.loads(result.stdout)

    def test_shutdown_waits_for_actual_worker_and_cleanup_before_returning(self):
        result = self.run_child("worker")
        self.assertEqual(result, {"at_close": [[True, True]], "staging_removed": True})

    def test_shutdown_rejects_unused_handle_without_waiting_for_its_context(self):
        result = self.run_child("unused")
        self.assertEqual(result, {"code": "SERVICE_CLOSED", "reads": 0})

    def test_shutdown_returns_during_body_then_rejects_late_chunks_and_submit(self):
        result = self.run_child("body")
        self.assertEqual(result, {"code": "SERVICE_CLOSED", "reads": ["first", "second"],
                                 "worker_calls": 0, "staged_while_closed": 1, "staged_after_unwind": 0})

    def test_cancellation_before_worker_start_keeps_the_accepted_future(self):
        result = self.run_child("pending")
        self.assertEqual(result, {"revision": 1, "staging_removed": True})


if __name__ == "__main__":
    unittest.main()
