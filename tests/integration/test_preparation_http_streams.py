"""Check the local upload boundary with raw, lazy ASGI request bodies."""

import asyncio
from contextlib import asynccontextmanager, redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from app import local_session, preparation, preparation_routes as routes, server


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _server_sources():
    """Allow child fault checks to replace source in memory, without file edits."""
    return {name: (PROJECT_ROOT / name.replace(".", "/")).with_suffix(".py").read_text(encoding="utf-8")
            for name in ("app.local_session", "app.preparation_routes", "app.server")}


_CHILD_START = """
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests" / "integration"))
import app
sources = json.load(sys.stdin)
for name in ("app.local_session", "app.preparation_routes", "app.server"):
    filename = (root / name.replace(".", "/")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    setattr(app, name.rsplit(".", 1)[1], module)
    exec(compile(sources[name], str(filename), "exec"), module.__dict__)
import test_preparation_http_streams as checks
checks._run_case(sys.argv[2])
print(json.dumps({"passed": sys.argv[2]}))
"""


def _png():
    image = Image.new("RGBA", (4, 4), (180, 70, 20, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@asynccontextmanager
async def _fixture():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        state = root / "state"
        state.mkdir(mode=0o700)
        config = server.LocalServerConfig(state, root / "runs")
        app = server.create_app(config)
        failure = None
        with redirect_stdout(StringIO()):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url=config.origin) as client:
                    response = await client.post("/session", json={"token": config.token_path.read_text().strip()},
                                                 headers={"Origin": config.origin})
                    if response.status_code != 200:
                        raise AssertionError("The stream fixture could not log in.")
                    session = await client.get("/session")
                    service = app.state.preparation
                    fixture = SimpleNamespace(app=app, config=config, service=service,
                        selected=service.create_run(), csrf=session.json()["csrfToken"],
                        cookie=local_session.COOKIE_NAME + "=" + client.cookies.get(local_session.COOKIE_NAME))
                    try:
                        yield fixture
                    except AssertionError as error:
                        failure = (error, error.__traceback__)
        if failure is not None:
            raise failure[0].with_traceback(failure[1])


def _headers(fixture, kind="asset", revision=0):
    headers = [(b"host", b"127.0.0.1:8765"), (b"origin", fixture.config.origin.encode()),
               (b"cookie", fixture.cookie.encode()), (b"x-csrf-token", fixture.csrf.encode()),
               (b"x-expected-revision", str(revision).encode())]
    if kind == "asset":
        return headers + [(b"content-type", b"image/png"), (b"x-non-human", b"true")]
    return headers + [(b"content-type", b"video/mp4"), (b"x-clean-source", b"true"),
                      (b"x-source-kind", b"synthetic_fixture")]


class _Body:
    def __init__(self, *chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.reads = 0

    async def __call__(self):
        index = self.reads
        self.reads += 1
        if index >= len(self.chunks):
            raise AssertionError("The request read beyond its final body message.")
        if self.delay:
            await asyncio.sleep(self.delay)
        return {"type": "http.request", "body": self.chunks[index],
                "more_body": index + 1 < len(self.chunks)}


async def _raw(fixture, body, *, kind="asset", path=None, headers=None, method="POST"):
    path = path or f"/runs/{fixture.selected.run_id}/{kind}"
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "scheme": "http", "method": method, "path": path, "raw_path": path.encode(),
             "query_string": b"", "root_path": "", "server": ("127.0.0.1", 8765),
             "client": ("127.0.0.1", 12345),
             "headers": _headers(fixture, kind) if headers is None else headers}
    messages = []

    async def send(message):
        messages.append(message)

    await fixture.app(scope, body, send)
    starts = [item for item in messages if item["type"] == "http.response.start"]
    if len(starts) != 1:
        raise AssertionError("The request did not produce exactly one response.")
    response_headers = {name.lower(): value for name, value in starts[0].get("headers", [])}
    if response_headers.get(b"content-type") != b"application/json":
        raise AssertionError("The upload boundary did not return an application/json response.")
    contents = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
    return starts[0]["status"], json.loads(contents)


def _expect(response, status, code=None):
    if response[0] != status:
        raise AssertionError(f"Expected HTTP {status}; received {response!r}.")
    if code is not None and response[1] != {"error": {"code": code}}:
        raise AssertionError("The response did not contain the exact safe error code " + code + ".")


def _clean_before_worker(fixture):
    if fixture.service.get_run(fixture.selected.run_id).revision != 0:
        raise AssertionError("A rejected receive changed the selected input revision.")
    if list(fixture.config.state_dir.glob(".preparation-*")):
        raise AssertionError("A rejected receive left a staged upload behind.")


async def _slot_is_free(fixture):
    async with fixture.service.media_operation(fixture.selected.run_id, expected_revision=0):
        pass


async def _zero_reads():
    async with _fixture() as fixture:
        valid = _headers(fixture)
        cases = [
            ([(name, value) for name, value in valid if name != b"cookie"], None, 401, "SESSION_REQUIRED"),
            ([(name, b"0" * 64 if name == b"x-csrf-token" else value) for name, value in valid],
             None, 403, "CSRF_INVALID"),
            ([(name, value) for name, value in valid if name != b"x-expected-revision"],
             None, 400, "INVALID_REQUEST"),
            (valid + [(b"x-expected-revision", b"0")], None, 400, "INVALID_REQUEST"),
            ([(name, b"text/plain" if name == b"content-type" else value) for name, value in valid],
             None, 415, "UNSUPPORTED_MEDIA_TYPE"),
            (valid, "/runs/run-" + "0" * 32 + "/asset", 404, "RUN_NOT_FOUND"),
            (_headers(fixture, revision=1), None, 409, "REVISION_CONFLICT"),
        ]
        for headers, path, status, code in cases:
            body = _Body(_png())
            _expect(await _raw(fixture, body, path=path, headers=headers), status, code)
            if body.reads != 0:
                raise AssertionError("A rejected upload consumed body bytes: " + code + ".")
        async with fixture.service.media_operation(fixture.selected.run_id, expected_revision=0):
            body = _Body(_png())
            _expect(await _raw(fixture, body), 409, "MEDIA_BUSY")
            if body.reads != 0:
                raise AssertionError("A busy upload consumed body bytes.")
        _expect(await _raw(fixture, _Body(_png())), 200)


async def _limits():
    if (local_session.MAX_INPUT_BYTES, local_session.MAX_FILE_BYTES, local_session.MAX_BODY_BYTES) != (
            256 * 1024 * 1024, 32 * 1024 * 1024, 4096):
        raise AssertionError("The published source, asset or control body limit changed.")
    with patch.object(local_session, "MAX_INPUT_BYTES", 11), patch.object(local_session, "MAX_FILE_BYTES", 7):
        for path, method, expected in (
                ("/runs/run-" + "a" * 32 + "/source", "POST", 11),
                ("/runs/run-" + "a" * 32 + "/asset", "POST", 7),
                ("/runs/run-" + "a" * 32 + "/source", "PUT", 4096),
                ("/runs/run-bad/asset", "POST", 4096),
                ("/runs/run-" + "a" * 32 + "/asset/", "POST", 4096),
                ("/runs", "POST", 4096)):
            if local_session.request_body_limit({"path": path, "method": method}) != expected:
                raise AssertionError("The shared route body rule granted an incorrect limit.")
        async with _fixture() as fixture:
            # Decoder behavior is separate: retain real admission, staging and byte counting.
            with patch.object(fixture.service, "_import_asset", return_value=fixture.selected), \
                    patch.object(fixture.service, "_import_source", return_value=fixture.selected):
                for kind, limit in (("source", 11), ("asset", 7)):
                    for declared in (False, True):
                        headers = _headers(fixture, kind)
                        if declared:
                            headers.append((b"content-length", str(limit).encode()))
                        body = _Body(b"x" * (limit - 1), b"x")
                        _expect(await _raw(fixture, body, kind=kind, headers=headers), 200)
                        if body.reads != 2:
                            raise AssertionError("An exact-limit upload was not consumed in full.")
                    declared = _Body(b"x" * (limit + 1))
                    headers = _headers(fixture, kind) + [(b"content-length", str(limit + 1).encode())]
                    _expect(await _raw(fixture, declared, kind=kind, headers=headers), 413, "BODY_TOO_LARGE")
                    if declared.reads:
                        raise AssertionError("An oversized declared body was read.")
                    for declared_length in (None, b"1"):
                        headers = _headers(fixture, kind)
                        if declared_length is not None:
                            headers.append((b"content-length", declared_length))
                        body = _Body(b"x" * limit, b"x")
                        _expect(await _raw(fixture, body, kind=kind, headers=headers), 413, "BODY_TOO_LARGE")
                        _clean_before_worker(fixture)
                        await _slot_is_free(fixture)
                headers = [(name, value) for name, value in _headers(fixture)
                           if name not in (b"content-type", b"x-non-human", b"x-expected-revision")]
                headers.append((b"content-type", b"application/json"))
                _expect(await _raw(fixture, _Body(b"{}" + b" " * 4094), path="/runs", headers=headers), 201)
                _expect(await _raw(fixture, _Body(b"{}" + b" " * 4095), path="/runs", headers=headers),
                        413, "BODY_TOO_LARGE")


async def _deadline(total):
    with patch.object(routes, "UPLOAD_RECEIVE_SECONDS", 0.08 if total else 0.03):
        async with _fixture() as fixture:
            worker = Mock(return_value=fixture.selected)
            contents = _png()
            body = (_Body(*(contents[index:index + 1] for index in range(len(contents))), delay=0.025)
                    if total else _Body(contents[:1], contents[1:]))

            async def receive():
                if total:
                    body.delay = 0 if body.reads == 0 else 0.025
                elif body.reads:
                    await asyncio.sleep(0.06)
                return await body()

            with patch.object(fixture.service, "_import_asset", worker):
                _expect(await _raw(fixture, receive), 408, "UPLOAD_RECEIVE_TIMEOUT")
            worker.assert_not_called()
            if total and not 2 <= body.reads < len(contents):
                raise AssertionError("The total receive deadline did not stop a continuing body stream.")
            if not total and body.reads != 1:
                raise AssertionError("The stalled receive did not stop after the first body chunk.")
            _clean_before_worker(fixture)
            await _slot_is_free(fixture)


async def _wait(event, message):
    if not await asyncio.to_thread(event.wait, 3):
        raise AssertionError(message)


async def _cancel(task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return
    raise AssertionError("External request cancellation did not propagate.")


async def _worker(cancel):
    entered, release, completed = Event(), Event(), Event()
    with patch.object(routes, "UPLOAD_RECEIVE_SECONDS", 0.03):
        async with _fixture() as fixture:
            original = fixture.service._import_asset
            original_submit = fixture.service._executor.submit
            futures = []

            def worker(*arguments):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("The upload worker was not released.")
                return original(*arguments)

            def submit(*arguments, **keywords):
                future = original_submit(*arguments, **keywords)
                futures.append(future)
                return future

            with patch.object(fixture.service, "_import_asset", side_effect=worker), \
                    patch.object(fixture.service._executor, "submit", side_effect=submit):
                pending = asyncio.create_task(_raw(fixture, _Body(_png())))
                try:
                    await _wait(entered, "The upload worker did not start after body completion.")
                    futures[0].add_done_callback(lambda future: completed.set())
                    if cancel:
                        await _cancel(pending)
                    else:
                        await asyncio.sleep(0.09)
                        if pending.done():
                            raise AssertionError("The receive deadline ended a request whose worker was running.")
                    if futures[0].done() or futures[0].cancelled():
                        raise AssertionError("The held upload worker completed before release.")
                    if not list(fixture.config.state_dir.glob(".preparation-*")):
                        raise AssertionError("The upload lost staging before its actual worker finished.")
                    body = _Body(_png())
                    _expect(await _raw(fixture, body), 409, "MEDIA_BUSY")
                    if body.reads:
                        raise AssertionError("A second upload read bytes while the actual worker held its slot.")
                    release.set()
                    await _wait(completed, "The actual upload worker did not finish cleanup.")
                    if not cancel:
                        _expect(await pending, 200)
                    if fixture.service.get_run(fixture.selected.run_id).revision != 1:
                        raise AssertionError("The completed actual worker did not preserve its uploaded asset.")
                    if list(fixture.config.state_dir.glob(".preparation-*")):
                        raise AssertionError("The completed worker left its staged upload behind.")
                    async with fixture.service.media_operation(fixture.selected.run_id, expected_revision=1):
                        pass
                finally:
                    release.set()
                    if not pending.done():
                        await _cancel(pending)


async def _consumer_failure():
    retained = []

    async def reject(operation, chunks, *, non_human):
        retained.append(chunks)
        await anext(chunks)
        raise preparation.ServiceError("INVALID_INPUT")

    with patch.object(routes, "UPLOAD_RECEIVE_SECONDS", 0.03):
        async with _fixture() as fixture:
            try:
                with patch.object(preparation._MediaOperation, "import_asset", reject):
                    _expect(await _raw(fixture, _Body(_png(), b"unused")), 422, "INVALID_INPUT")
                try:
                    await asyncio.sleep(0.09)
                except asyncio.CancelledError:
                    raise AssertionError("A consumer failure left the receive timer active.") from None
                if len(retained) != 1 or retained[0].ag_frame is not None:
                    raise AssertionError("The failed consumer's retained stream was not closed.")
                _clean_before_worker(fixture)
                await _slot_is_free(fixture)
            finally:
                for chunks in retained:
                    await chunks.aclose()


async def _unrelated_timeout():
    async with _fixture() as fixture:
        async def receive():
            raise TimeoutError("private receive detail")

        worker = Mock(return_value=fixture.selected)
        with patch.object(fixture.service, "_import_asset", worker):
            _expect(await _raw(fixture, receive), 500, "STATE_UNAVAILABLE")
        worker.assert_not_called()
        _clean_before_worker(fixture)
        await _slot_is_free(fixture)


async def _receive_cancel():
    async with _fixture() as fixture:
        waiting = asyncio.Event()
        gate = asyncio.Event()
        reads = 0

        async def receive():
            nonlocal reads
            reads += 1
            if reads == 1:
                return {"type": "http.request", "body": _png()[:1], "more_body": True}
            waiting.set()
            await gate.wait()
            return {"type": "http.request", "body": _png()[1:], "more_body": False}

        worker = Mock(return_value=fixture.selected)
        with patch.object(fixture.service, "_import_asset", worker):
            pending = asyncio.create_task(_raw(fixture, receive))
            try:
                await asyncio.wait_for(waiting.wait(), 3)
                await _cancel(pending)
                worker.assert_not_called()
                _clean_before_worker(fixture)
                await _slot_is_free(fixture)
            finally:
                gate.set()
                if not pending.done():
                    await _cancel(pending)


def _run_case(case):
    if routes.UPLOAD_RECEIVE_SECONDS != 300:
        raise AssertionError("The published cooperative receive deadline is not 300 seconds.")
    cases = {"zero_reads": _zero_reads, "limits": _limits,
             "stalled_deadline": lambda: _deadline(False), "total_deadline": lambda: _deadline(True),
             "slow_worker": lambda: _worker(False), "worker_cancel": lambda: _worker(True),
             "consumer_failure": _consumer_failure, "unrelated_timeout": _unrelated_timeout,
             "receive_cancel": _receive_cancel}
    asyncio.run(cases[case]())


class PreparationHttpStreamTests(unittest.TestCase):
    def _child(self, case):
        try:
            with TemporaryDirectory(prefix="http-stream-child-") as folder:
                environment = dict(os.environ, TMPDIR=folder, PYTHONDONTWRITEBYTECODE="1")
                result = subprocess.run([sys.executable, "-c", _CHILD_START, str(PROJECT_ROOT), case],
                    input=json.dumps(_server_sources()), text=True, capture_output=True,
                    timeout=20, check=False, start_new_session=True, env=environment)
        except subprocess.TimeoutExpired:
            self.fail("The bounded HTTP stream child did not finish.")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), {"passed": case})

    def test_rejected_uploads_consume_no_body_bytes(self):
        self._child("zero_reads")

    def test_shared_route_limits_accept_equality_and_reject_one_extra_byte(self):
        self._child("limits")

    def test_stalled_receive_deadline_cleans_staging_before_worker(self):
        self._child("stalled_deadline")

    def test_receive_deadline_bounds_total_body_time_not_each_chunk(self):
        self._child("total_deadline")

    def test_finished_body_does_not_time_out_slow_worker(self):
        self._child("slow_worker")

    def test_consumer_failure_closes_retained_stream_and_disarms_deadline(self):
        self._child("consumer_failure")

    def test_unrelated_timeout_is_not_a_receive_deadline(self):
        self._child("unrelated_timeout")

    def test_external_cancellation_during_receive_propagates_and_cleans_staging(self):
        self._child("receive_cancel")

    def test_cancelled_http_waiter_keeps_actual_worker_staging_and_slot(self):
        self._child("worker_cancel")


if __name__ == "__main__":
    unittest.main()
