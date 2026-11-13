"""Keep preview and historical reads inside the one local media admission."""

import asyncio
from concurrent.futures import Future
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import unittest
from unittest.mock import patch

from PIL import Image

from app import preparation
from core.budget import BudgetLedger
from features.inference.submission import SubmissionCoordinator
import test_local_preparation as fixtures


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _service_sources():
    """Let the external fault harness supply child code without file edits."""
    return {name: (PROJECT_ROOT / name.replace(".", "/")).with_suffix(".py").read_text(encoding="utf-8")
            for name in ("app.preparation_state", "app.preparation_media", "app.preparation")}


_CHILD_START = """
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
scenario, kind = sys.argv[2:4]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests" / "unit"))
import app
payload = json.load(sys.stdin)
for name in ("app.preparation_state", "app.preparation_media", "app.preparation"):
    filename = (root / name.replace(".", "/")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    setattr(app, name.rsplit(".", 1)[1], module)
    exec(compile(payload["sources"][name], str(filename), "exec"), module.__dict__)
import test_preview_admission as checks
print(json.dumps(checks._run_case(payload["fixture"], scenario, kind), sort_keys=True))
"""


def _media():
    image = Image.new("RGBA", (4, 4), (220, 100, 30, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    contents = output.getvalue()
    return preparation.LocalMedia(contents, "image/png", sha256(contents).hexdigest())


async def _read(service, selected, kind):
    if kind == "preview":
        return await service.preview(selected["run_id"], expected_revision=selected["revision"], role="source")
    return await service.prepared_media(selected["run_id"], selected["attempt_id"], role="keyframe-a")


async def _busy_calls(service, selected, media):
    """A held read must reject imports and both kinds of read before work."""
    reads = []

    async def body():
        reads.append(1)
        yield media.contents

    async def import_asset():
        async with service.media_operation(selected["run_id"], expected_revision=selected["revision"]) as operation:
            await operation.import_asset(body(), non_human=True)

    for kind in ("import", "preview", "history"):
        try:
            if kind == "import":
                await import_asset()
            else:
                await _read(service, selected, kind)
        except preparation.ServiceError as error:
            if error.code != "MEDIA_BUSY":
                raise AssertionError("A held read did not report MEDIA_BUSY for " + kind + ".") from None
        else:
            raise AssertionError("A held read admitted another " + kind + " operation.")
    if reads:
        raise AssertionError("A busy import consumed body bytes.")


async def _wait(event, message):
    if not await asyncio.to_thread(event.wait, 5):
        raise AssertionError(message)


async def _cancel(task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return
    raise AssertionError("Canceling a read did not cancel its request task.")


async def _running_case(service, selected, kind, expected):
    entered, release, completed = Event(), Event(), Event()
    futures = []
    original_submit = service._executor.submit

    def worker(*arguments):
        entered.set()
        if not release.wait(5):
            raise AssertionError("The read worker was not released.")
        return expected

    def submit(*arguments, **keywords):
        future = original_submit(*arguments, **keywords)
        futures.append(future)
        return future

    with patch.object(service, "_preview_media", side_effect=worker), \
            patch.object(service, "_read_prepared_media", side_effect=worker), \
            patch.object(service._executor, "submit", side_effect=submit):
        pending = asyncio.create_task(_read(service, selected, kind))
        try:
            await _wait(entered, "The read worker did not start.")
            futures[0].add_done_callback(lambda future: completed.set())
            pending.cancel()
            await _cancel(pending)
            if futures[0].done() or futures[0].cancelled():
                raise AssertionError("Canceling a read completed its still-running worker.")
            await _busy_calls(service, selected, expected)
            release.set()
            await _wait(completed, "The actual read worker did not finish cleanup.")
            if futures[0].result(timeout=1) != expected:
                raise AssertionError("The completed read changed the fixed media bytes or hash.")
            other = "history" if kind == "preview" else "preview"
            if await _read(service, selected, other) != expected:
                raise AssertionError("The released read slot could not serve the other read.")
            return {"blocked": ["import", "preview", "history"], "completed": True}
        finally:
            release.set()
            if not pending.done():
                await _cancel(pending)


async def _pending_case(service, selected, kind, expected):
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

    with patch.object(service, "_preview_media", return_value=expected), \
            patch.object(service, "_read_prepared_media", return_value=expected), \
            patch.object(service._executor, "submit", side_effect=submit):
        pending = asyncio.create_task(_read(service, selected, kind))
        try:
            await asyncio.wait_for(submitted.wait(), timeout=5)
            await _cancel(pending)
            if future.done() or future.cancelled():
                raise AssertionError("Read cancellation canceled an accepted pending worker.")
            await _busy_calls(service, selected, expected)
            finish()
            if future.result(timeout=1) != expected:
                raise AssertionError("The pending read lost its fixed media result.")
            async with service.media_operation(selected["run_id"], expected_revision=selected["revision"]):
                pass
            return {"kept_pending": True, "completed": True}
        finally:
            if calls:
                finish()
            if not pending.done():
                await _cancel(pending)


async def _shutdown_case(service, selected, kind, expected):
    entered, release, worker_finished = Event(), Event(), Event()
    shutdown_entered, close_returned, closed = Event(), Event(), Event()
    futures, at_close, close_errors = [], [], []
    original_submit, original_shutdown = service._executor.submit, service._executor.shutdown

    def worker(*arguments):
        entered.set()
        if not release.wait(5):
            raise AssertionError("The shutdown read worker was not released.")
        worker_finished.set()
        return expected

    def submit(*arguments, **keywords):
        future = original_submit(*arguments, **keywords)
        futures.append(future)
        return future

    def shutdown(*arguments, **keywords):
        shutdown_entered.set()
        return original_shutdown(*arguments, **keywords)

    close_code = service.close.__func__.__code__

    def observe_close_return(frame, event, argument):
        # Observe the production return before a caller-line scheduling pause
        # can let the worker finish and hide an early return.
        if event == "return" and frame.f_code is close_code:
            slot_free = service._slot.acquire(blocking=False)
            if slot_free:
                service._slot.release()
            at_close.append([release.is_set(), worker_finished.is_set(), futures[0].done(), slot_free])
            close_returned.set()

    def close():
        previous_profile = sys.getprofile()
        sys.setprofile(observe_close_return)
        try:
            service.close()
        except BaseException as error:
            close_errors.append(type(error).__name__)
        finally:
            sys.setprofile(previous_profile)
            closed.set()

    closer = None
    with patch.object(service, "_preview_media", side_effect=worker), \
            patch.object(service, "_read_prepared_media", side_effect=worker), \
            patch.object(service._executor, "submit", side_effect=submit), \
            patch.object(service._executor, "shutdown", side_effect=shutdown):
        pending = asyncio.create_task(_read(service, selected, kind))
        try:
            await _wait(entered, "The shutdown read worker did not start.")
            await _cancel(pending)
            closer = Thread(target=close)
            closer.start()
            await _wait(shutdown_entered, "Shutdown did not start waiting for the read.")
            for read_kind in ("preview", "history"):
                try:
                    await _read(service, selected, read_kind)
                except preparation.ServiceError as error:
                    if error.code != "SERVICE_CLOSED":
                        raise AssertionError("Shutdown did not close read admission before waiting.") from None
                else:
                    raise AssertionError("Shutdown admitted another media read.")
            if await asyncio.to_thread(close_returned.wait, 0.2):
                raise AssertionError("Production close returned before the gated read worker was released.")
            if release.is_set() or worker_finished.is_set() or futures[0].done():
                raise AssertionError("The read worker did not remain gated during the close wait.")
            release.set()
            await _wait(closed, "Shutdown did not finish after the read.")
            closer.join(timeout=1)
            if closer.is_alive():
                raise AssertionError("The read shutdown thread did not exit.")
            if close_errors:
                raise AssertionError("Production close did not return normally.")
            for read_kind in ("preview", "history"):
                try:
                    await _read(service, selected, read_kind)
                except preparation.ServiceError as error:
                    if error.code != "SERVICE_CLOSED":
                        raise AssertionError("Completed shutdown did not keep read admission closed.") from None
                else:
                    raise AssertionError("Completed shutdown accepted another media read.")
            if at_close != [[True, True, True, True]]:
                raise AssertionError("Production close did not return after the read and cleanup.")
            return {"at_close": at_close, "closed_reads": ["preview", "history"]}
        finally:
            release.set()
            try:
                if not pending.done():
                    await _cancel(pending)
            finally:
                if closer is not None:
                    closer.join(timeout=6)


def _run_case(selected, scenario, kind):
    ledger = BudgetLedger(Path(selected["state"]) / "budget.sqlite", clock=lambda: 1000.0)
    with SubmissionCoordinator(ledger) as owner:
        service = preparation.PreparationService(Path(selected["state"]), Path(selected["outputs"]), owner,
                                                 clock=lambda: 1000.0)
        try:
            cases = {"running": _running_case, "pending": _pending_case, "shutdown": _shutdown_case}
            return asyncio.run(cases[scenario](service, selected, kind, _media()))
        finally:
            service.close()


class PreviewAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = type("PreviewAdmissionFixture", (fixtures.LocalPreparationTests,), {})
        cls.baseline.setUpClass()
        cls.addClassCleanup(cls.baseline.doClassCleanups)

    def setUp(self):
        # Build registered real inputs and history before the faulted body runs.
        baseline = self.baseline("runTest")
        baseline.setUp()
        self.addCleanup(baseline.doCleanups)
        service = baseline.service()
        run = baseline.ready(service)
        with patch.object(fixtures.prepare, "compile_guidance", side_effect=baseline.cached_guidance):
            run = baseline.build(service, run)
        self.selected = {"state": str(baseline.state), "outputs": str(baseline.outputs),
                         "run_id": run.run_id, "revision": run.revision,
                         "attempt_id": run.attempts[-1].attempt_id}
        service.close()
        service.coordinator.close()

    def run_child(self, scenario, kind):
        payload = {"sources": _service_sources(), "fixture": self.selected}
        try:
            result = subprocess.run([sys.executable, "-c", _CHILD_START, str(PROJECT_ROOT), scenario, kind],
                                    input=json.dumps(payload).encode("utf-8"), cwd=PROJECT_ROOT,
                                    capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            self.fail("Media read admission or shutdown exceeded the bounded child time limit.")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        return json.loads(result.stdout)

    def test_cancelled_preview_holds_shared_admission_until_its_worker_finishes(self):
        self.assertEqual(self.run_child("running", "preview"),
                         {"blocked": ["import", "preview", "history"], "completed": True})

    def test_cancelled_history_read_holds_shared_admission_until_its_worker_finishes(self):
        self.assertEqual(self.run_child("running", "history"),
                         {"blocked": ["import", "preview", "history"], "completed": True})

    def test_cancelled_pending_reads_keep_their_accepted_future(self):
        for kind in ("preview", "history"):
            with self.subTest(read=kind):
                self.assertEqual(self.run_child("pending", kind), {"kept_pending": True, "completed": True})

    def test_shutdown_drains_each_read_before_releasing_the_service(self):
        for kind in ("preview", "history"):
            with self.subTest(read=kind):
                self.assertEqual(self.run_child("shutdown", kind),
                                 {"at_close": [[True, True, True, True]], "closed_reads": ["preview", "history"]})


if __name__ == "__main__":
    unittest.main()
