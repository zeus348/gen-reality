"""Check the local operator session without running an experiment or provider."""

import base64
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
from starlette.testclient import TestClient

from app import local_session, server
from core import budget
from features.inference import submission


COOKIE = "gen_reality_session"
PRIVATE = "private-sentinel-token-and-path"


def _server_sources():
    """Let fault checks replace only production source in a bounded child."""
    return {
        name: {"path": module.__file__, "source": Path(module.__file__).read_text()}
        for name, module in (("app.local_session", local_session), ("app.server", server))
    }


CHILD_BOOTSTRAP = """
import json
from pathlib import Path
import sys
import types
import app
data = json.loads(sys.stdin.buffer.read())
exec(data.get('prelude', ''))
for name, item in data['sources'].items():
    module = types.ModuleType(name)
    module.__file__ = item['path']
    module.__package__ = 'app'
    sys.modules[name] = module
    setattr(app, name.rsplit('.', 1)[1], module)
    exec(compile(item['source'], item['path'], 'exec'), module.__dict__)
exec(compile(data['scenario'], '<bootstrap-test>', 'exec'))
"""


class LocalBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def config(self, name="state", **changes):
        state = self.root / name
        state.mkdir(mode=0o700)
        values = {"state_dir": state, "output_root": self.root / "outputs", "port": 8765}
        values.update(changes)
        return server.LocalServerConfig(**values)

    def client(self, app, config):
        return TestClient(app, base_url=config.origin, raise_server_exceptions=False)

    def login(self, client, config, token=None):
        if token is None:
            token = config.token_path.read_text().strip()
        return client.post("/session", json={"token": token}, headers={"Origin": config.origin})

    def cookie(self, client):
        return COOKIE + "=" + client.cookies.get(COOKIE)

    def session(self, client, cookie=None):
        return client.get("/session", headers={} if cookie is None else {"Cookie": cookie})

    def error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.headers.get("content-type"), "application/json")
        self.assertEqual(response.json(), {"error": {"code": code}})
        self.assertNotIn("set-cookie", response.headers)

    def raw(self, app, client, config, *, method="POST", path="/session", headers=(), chunks=()):
        """Send unnormalized headers and real ASGI chunks through the whole app."""
        async def request():
            messages = [
                {"type": "http.request", "body": body, "more_body": index < len(chunks) - 1}
                for index, body in enumerate(chunks)
            ] or [{"type": "http.request", "body": b"", "more_body": False}]
            sent = []

            async def receive():
                if messages:
                    return messages.pop(0)
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message)

            scope = {
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": method, "scheme": "http", "path": path,
                "raw_path": path.encode(), "query_string": b"", "root_path": "",
                "headers": list(headers), "client": ("127.0.0.1", 32123),
                "server": ("127.0.0.1", config.port), "state": {},
            }
            await app(scope, receive, send)
            start = next(item for item in sent if item["type"] == "http.response.start")
            body = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
            return httpx.Response(start["status"], headers=start.get("headers", []), content=body)

        return client.portal.call(request)

    def headers(self, config, *extra):
        return [(b"host", f"127.0.0.1:{config.port}".encode()),
                (b"origin", config.origin.encode()), (b"content-type", b"application/json"), *extra]

    def child(self, scenario, *arguments, prelude=""):
        try:
            return subprocess.run(
                [sys.executable, "-c", CHILD_BOOTSTRAP, *map(str, arguments)],
                input=json.dumps({"sources": _server_sources(), "scenario": scenario,
                                  "prelude": prelude}).encode(),
                capture_output=True, timeout=12,
            )
        except subprocess.TimeoutExpired:
            self.fail("The bounded bootstrap child did not finish.")

    def test_login_lookup_logout_and_cookie_policy(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            token_bytes = config.token_path.read_bytes()
            self.assertRegex(token_bytes, rb"\A[0-9a-f]{64}\n\Z")
            self.assertEqual(stat.S_IMODE(config.token_path.stat().st_mode), 0o600)
            self.error(self.session(client), 401, "SESSION_REQUIRED")
            response = self.login(client, config)
            self.assertEqual(response.json(), {"authenticated": True})
            self.assertEqual(response.status_code, 200)
            cookie = response.headers["set-cookie"]
            self.assertIn("httponly", cookie.lower())
            self.assertIn("samesite=strict", cookie.lower())
            self.assertIn("path=/", cookie.lower())
            self.assertIn("max-age=3600", cookie.lower())
            self.assertNotIn("domain=", cookie.lower())
            self.assertNotIn("secure", cookie.lower())
            self.assertNotIn(token_bytes.decode().strip(), cookie)
            value = client.cookies.get(COOKIE).split(".")[0]
            document = json.loads(base64.b64decode(value + "=" * (-len(value) % 4)))
            self.assertEqual(set(document), {"sessionId", "csrfToken"})
            self.assertNotIn(str(config.state_dir), json.dumps(document))
            current = self.session(client)
            self.assertEqual(current.status_code, 200)
            self.assertEqual(set(current.json()), {"authenticated", "csrfToken"})
            self.assertIs(current.json()["authenticated"], True)
            self.assertRegex(current.json()["csrfToken"], r"\A[0-9a-f]{64}\Z")
            self.assertEqual(document["csrfToken"], current.json()["csrfToken"])
            saved_cookie = self.cookie(client)
            logout = client.delete("/session", headers={"Origin": config.origin,
                                                        "X-CSRF-Token": current.json()["csrfToken"]})
            self.assertEqual(logout.json(), {"authenticated": False})
            self.assertEqual(logout.status_code, 200)
            self.assertIn("expires=", logout.headers["set-cookie"].lower())
            self.error(self.session(client, saved_cookie), 401, "SESSION_REQUIRED")

    def test_static_routes_and_security_headers_do_not_expose_authority(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            token = config.token_path.read_text().strip()
            for path in ("/", "/app.js", "/session"):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 401 if path == "/session" else 200)
                    self.assertIn("no-store", response.headers["cache-control"])
                    self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                    self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
                    for private in (token, str(config.state_dir), str(config.output_root), "RUNWAY_API_KEY"):
                        self.assertNotIn(private, response.text)
            for path in ("/openapi.json", "/docs", "/redoc", "/runs", "/attempts"):
                response = client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(set(response.json()), {"error"})
            self.assertEqual(client.put("/session", headers={"Origin": config.origin}).status_code, 405)

    def test_host_headers_are_exact_and_checked_before_cookie_decode(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            cases = [[], [(b"host", b"localhost:8765")], [(b"host", b"127.0.0.1")],
                     [(b"host", b"127.0.0.1:8766")], [(b"host", b"[::1]:8765")],
                     [(b"host", b"127.0.0.1:8765"), (b"host", b"127.0.0.1:8765")],
                     [(b"host", b"127.0.0.1:8765.attacker.invalid")]]
            for headers in cases:
                with self.subTest(headers=headers):
                    with patch("itsdangerous.TimestampSigner.unsign", side_effect=ValueError(PRIVATE)) as decode:
                        response = self.raw(app, client, config, method="GET",
                                            headers=headers + [(b"cookie", (COOKIE + "=malformed").encode())])
                        self.error(response, 400, "HOST_NOT_ALLOWED")
                        decode.assert_not_called()

    def test_write_and_supplied_read_origins_are_exact(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            self.assertEqual(self.login(client, config).status_code, 200)
            cookie = self.cookie(client)
            for origins in ([], [b"null"], [b"http://localhost:8765"],
                            [b"http://127.0.0.1:8766"], [config.origin.encode()] * 2):
                with self.subTest(origins=origins):
                    headers = [(b"host", b"127.0.0.1:8765"), (b"cookie", cookie.encode())]
                    headers.extend((b"origin", value) for value in origins)
                    self.error(self.raw(app, client, config, method="DELETE", headers=headers),
                               403, "ORIGIN_NOT_ALLOWED")
                    self.assertEqual(self.session(client).status_code, 200)
            self.error(client.get("/session", headers={"Origin": "https://attacker.invalid"}),
                       403, "ORIGIN_NOT_ALLOWED")
            self.assertEqual(self.session(client).status_code, 200)

    def test_header_and_cookie_bounds_reject_before_session_decoding(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            cases = [
                ([(b"host", b"127.0.0.1:8765"), (b"x-padding", b"a" * 16384)],
                 431, "HEADERS_TOO_LARGE"),
                ([(b"host", b"127.0.0.1:8765")] + [(b"x-extra", b"a")] * 64,
                 431, "HEADERS_TOO_LARGE"),
                ([(b"host", b"127.0.0.1:8765"), (b"cookie", b"a" * 4097)],
                 431, "HEADERS_TOO_LARGE"),
                ([(b"host", b"127.0.0.1:8765"), (b"cookie", b"a=b"), (b"cookie", b"c=d")],
                 400, "INVALID_REQUEST"),
                ([(b"host", b"127.0.0.1:8765"),
                  (b"cookie", (COOKIE + "=first; " + COOKIE + "=second").encode())],
                 400, "INVALID_REQUEST"),
                ([(b"host", b"127.0.0.1:8765"), (b"x-invalid", b"line\nbreak")],
                 400, "INVALID_REQUEST"),
            ]
            for headers, status, code in cases:
                with self.subTest(code=code, headers=len(headers)):
                    self.error(self.raw(app, client, config, method="GET", headers=headers), status, code)

    def test_forwarding_headers_cannot_override_host_or_origin(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            for header in ((b"forwarded", b"host=127.0.0.1:8765;proto=http"),
                           (b"x-forwarded-host", b"127.0.0.1:8765"),
                           (b"x-forwarded-proto", b"http")):
                with self.subTest(header=header):
                    self.error(self.raw(app, client, config, method="GET",
                                        headers=[(b"host", b"attacker.invalid"), header]),
                               400, "HOST_NOT_ALLOWED")
            headers = self.headers(config, (b"x-forwarded-host", b"attacker.invalid"))
            response = self.raw(app, client, config, method="GET", headers=headers)
            self.error(response, 400, "HOST_NOT_ALLOWED")

    def test_factory_defers_files_and_one_owner_lives_until_shutdown(self):
        config = self.config()
        with patch.object(server, "SubmissionCoordinator", wraps=submission.SubmissionCoordinator) as construct:
            app = server.create_app(config)
            self.assertEqual(list(config.state_dir.iterdir()), [])
            self.assertFalse(config.output_root.exists())
            construct.assert_not_called()
            with self.client(app, config) as client:
                owner = app.state.coordinator
                owner._require_open()
                construct.assert_called_once()
                self.login(client, config)
                self.session(client)
                construct.assert_called_once()
                with sqlite3.connect(config.ledger_path) as connection:
                    self.assertEqual(connection.execute("SELECT count(*) FROM approved_runs").fetchone()[0], 0)
                    self.assertEqual(connection.execute("SELECT count(*) FROM reservations").fetchone()[0], 0)
                original_token = config.token_path.read_bytes()
                child = self.child("""
from app import server
from starlette.testclient import TestClient
config = server.LocalServerConfig(Path(sys.argv[1]), Path(sys.argv[2]))
try:
    with TestClient(server.create_app(config), base_url=config.origin):
        raise AssertionError('Competing owner was accepted.')
except ValueError:
    print('owner-rejected')
""", config.state_dir, config.output_root)
                self.assertEqual(child.returncode, 0, child.stderr.decode())
                self.assertEqual(child.stdout.strip(), b"owner-rejected")
                self.assertEqual(config.token_path.read_bytes(), original_token)
            with self.assertRaises(submission.SubmissionError):
                owner._require_open()
        with submission.SubmissionCoordinator(budget.BudgetLedger(config.ledger_path)):
            pass
        self.assertFalse(config.output_root.exists())

    def test_startup_recovers_abandoned_submission_and_keeps_held_reserve(self):
        config = self.config()
        ledger = budget.BudgetLedger(config.ledger_path, clock=lambda: 1000.0)
        ledger.approve_run("run-1", approved_by="fixture operator", limit_cents=140,
                           max_attempts=1, cost_cents=140, max_elapsed_minutes=1)
        attempt = ledger.reserve("run-1", "fixture-key", "a" * 64).attempt_id
        with submission.SubmissionCoordinator(ledger) as old_owner:
            self.assertTrue(old_owner._claim(attempt))
            self.assertEqual(old_owner.get(attempt).state, "SUBMITTING")
        app = server.create_app(config)
        with self.client(app, config):
            self.assertEqual(app.state.coordinator.get(attempt).state, "SUBMISSION_UNKNOWN")
            with sqlite3.connect(config.ledger_path) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM approved_runs").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT count(*) FROM actual_charges").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT cost_cents FROM reservations").fetchone()[0], 140)
                self.assertEqual(connection.execute(
                    "SELECT event_kind FROM submission_events ORDER BY event_id").fetchall(),
                    [("CLAIMED",), ("OWNER_RECOVERED_UNKNOWN",)])

    def test_configuration_rejects_unsafe_state_and_invalid_ports_without_writes(self):
        for index, port in enumerate((True, 0, 1023, 65536, 8765.0, "8765")):
            with self.subTest(port=port):
                state = self.root / f"port-{index}"
                state.mkdir(mode=0o700)
                with self.assertRaises(ValueError):
                    server.create_app(server.LocalServerConfig(state, self.root / "outputs", port))
                self.assertEqual(list(state.iterdir()), [])
        private = self.root / "private"
        private.mkdir(mode=0o700)
        public = self.root / "public"
        public.mkdir(mode=0o755)
        public.chmod(0o755)
        linked = self.root / "linked"
        linked.symlink_to(private, target_is_directory=True)
        for state, output in ((self.root / "missing", self.root / "outputs"),
                              (public, self.root / "outputs"),
                              (linked, self.root / "outputs"),
                              (private, self.root), (private, private)):
            with self.subTest(state=state, output=output):
                with self.assertRaises(ValueError):
                    server.create_app(server.LocalServerConfig(state, output))
                self.assertEqual(list(private.iterdir()), [])

    def test_token_symlink_is_rejected_and_failed_startup_releases_owner(self):
        config = self.config()
        target = self.root / "private-token-target"
        target.write_text(PRIVATE)
        target.chmod(0o600)
        config.token_path.symlink_to(target)
        with self.assertRaises(ValueError):
            with self.client(server.create_app(config), config):
                pass
        self.assertEqual(target.read_text(), PRIVATE)
        self.assertTrue(config.token_path.is_symlink())
        with submission.SubmissionCoordinator(budget.BudgetLedger(config.ledger_path)):
            pass

    def test_failed_token_publication_releases_owner_and_clears_session(self):
        config = self.config()
        app = server.create_app(config)
        with patch.object(server, "_publish_login_token", side_effect=OSError(PRIVATE)) as publish:
            with self.assertRaises(server.LocalServerError) as caught:
                with self.client(app, config):
                    pass
            publish.assert_called_once()
        self.assertNotIn(PRIVATE, str(caught.exception))
        self.assertIsNone(app.state.coordinator)
        self.assertIsNone(app.state.sessions.bootstrap_token)
        self.assertFalse(config.token_path.exists())
        with submission.SubmissionCoordinator(budget.BudgetLedger(config.ledger_path)):
            pass

    def test_token_fifo_is_rejected_in_a_bounded_child(self):
        config = self.config()
        os.mkfifo(config.token_path, 0o600)
        child = self.child("""
from app import server
from starlette.testclient import TestClient
config = server.LocalServerConfig(Path(sys.argv[1]), Path(sys.argv[2]))
try:
    with TestClient(server.create_app(config), base_url=config.origin):
        raise AssertionError('FIFO token path was accepted.')
except ValueError:
    print('fifo-rejected')
""", config.state_dir, config.output_root)
        self.assertEqual(child.returncode, 0, child.stderr.decode())
        self.assertEqual(child.stdout.strip(), b"fifo-rejected")
        self.assertTrue(stat.S_ISFIFO(config.token_path.stat().st_mode))
        with submission.SubmissionCoordinator(budget.BudgetLedger(config.ledger_path)):
            pass

    def test_import_has_no_ledger_owner_or_provider_configuration_side_effect(self):
        child = self.child("""
ledger.assert_not_called()
owner.assert_not_called()
assert list(Path(sys.argv[1]).iterdir()) == []
print('import-only')
""", self.root, prelude="""
from unittest.mock import Mock
from core import budget
from features.inference import submission
ledger = Mock(side_effect=AssertionError('Import constructed a ledger.'))
owner = Mock(side_effect=AssertionError('Import constructed an owner.'))
budget.BudgetLedger = ledger
submission.SubmissionCoordinator = owner
""")
        self.assertEqual(child.returncode, 0, child.stderr.decode())
        self.assertEqual(child.stdout.strip(), b"import-only")

    def test_unexpected_endpoint_error_has_one_safe_json_response(self):
        config = self.config()
        app = server.create_app(config)

        async def fail():
            raise RuntimeError(PRIVATE + str(config.state_dir))

        app.add_api_route("/fixture-failure", fail, methods=["GET"])
        with self.client(app, config) as client:
            response = client.get("/fixture-failure")
            self.error(response, 500, "INTERNAL_ERROR")
            for private in (PRIVATE, str(config.state_dir), "RuntimeError", "Traceback"):
                self.assertNotIn(private, response.text)

    def test_cli_sets_loopback_single_owner_options_and_never_prints_token(self):
        config = self.config(port=9876)
        output = io.StringIO()
        errors = io.StringIO()

        def serve(app, **options):
            with self.client(app, config):
                pass

        with patch.dict(os.environ, {"WEB_CONCURRENCY": "99", "FORWARDED_ALLOW_IPS": "*",
                                     "UVICORN_HOST": "0.0.0.0"}), \
                patch.object(server.uvicorn, "run", side_effect=serve) as run, \
                patch("sys.stdout", output), patch("sys.stderr", errors):
            result = server.main(["--state-dir", str(config.state_dir),
                                  "--output-root", str(config.output_root), "--port", "9876"])
        self.assertEqual(result, 0)
        run.assert_called_once()
        options = run.call_args.kwargs
        self.assertEqual({key: options[key] for key in (
            "host", "port", "workers", "lifespan", "loop", "http", "ws", "reload",
            "proxy_headers", "access_log")}, {
                "host": "127.0.0.1", "port": 9876, "workers": 1, "lifespan": "on",
                "loop": "asyncio", "http": "h11", "ws": "none", "reload": False,
                "proxy_headers": False, "access_log": False,
            })
        self.assertIn(config.origin, output.getvalue() + errors.getvalue())
        self.assertIn(str(config.token_path), output.getvalue() + errors.getvalue())
        self.assertNotIn(config.token_path.read_text().strip(), output.getvalue() + errors.getvalue())

    def test_cli_errors_are_fixed_and_do_not_echo_arguments_or_exception_text(self):
        config = self.config()
        for arguments in (["--state-dir", PRIVATE, "--output-root", str(config.output_root), "--port", PRIVATE],
                          ["--state-dir", str(config.state_dir), "--output-root", str(config.output_root)]):
            with self.subTest(arguments=arguments):
                output = io.StringIO()
                errors = io.StringIO()
                with patch.object(server.uvicorn, "run", side_effect=RuntimeError(PRIVATE)), \
                        patch("sys.stdout", output), patch("sys.stderr", errors):
                    result = server.main(arguments)
                self.assertEqual(result, 1)
                self.assertEqual(output.getvalue(), "")
                self.assertEqual(errors.getvalue(), "Local server could not start safely.\n")

    def test_token_publication_syncs_file_before_replace_and_directory_after(self):
        config = self.config()
        previous = "a" * 64 + "\n"
        config.token_path.write_text(previous)
        config.token_path.chmod(0o600)
        original_sync = os.fsync
        observations = []

        def sync(descriptor):
            observations.append((stat.S_ISREG(os.fstat(descriptor).st_mode), config.token_path.read_text()))
            original_sync(descriptor)

        with patch.object(server.os, "fsync", side_effect=sync):
            server._publish_login_token(config, "b" * 64)
        self.assertEqual(observations[0], (True, previous))
        self.assertTrue(any(not regular and contents == "b" * 64 + "\n"
                            for regular, contents in observations[1:]))
        self.assertEqual(config.token_path.read_text(), "b" * 64 + "\n")
        self.assertEqual(stat.S_IMODE(config.token_path.stat().st_mode), 0o600)
        self.assertEqual(sorted(path.name for path in config.state_dir.iterdir()), [config.token_path.name])

    def test_failed_token_file_sync_keeps_previous_bytes_and_removes_temporary_file(self):
        config = self.config()
        previous = "a" * 64 + "\n"
        config.token_path.write_text(previous)
        config.token_path.chmod(0o600)
        original_sync = os.fsync

        def sync(descriptor):
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError(PRIVATE)
            original_sync(descriptor)

        with patch.object(server.os, "fsync", side_effect=sync):
            with self.assertRaises(OSError):
                server._publish_login_token(config, "b" * 64)
        self.assertEqual(config.token_path.read_text(), previous)
        self.assertEqual(sorted(path.name for path in config.state_dir.iterdir()), [config.token_path.name])

    def test_malformed_session_fields_fail_closed_without_type_errors(self):
        sessions = local_session.SessionState(bootstrap_token="a" * 64)
        cookie = {}
        self.assertTrue(sessions.login("a" * 64, cookie))
        for field in ("sessionId", "csrfToken"):
            for value in ("é" * 64, 1, True, None, [], {}, "a" * 63, "A" * 64):
                with self.subTest(field=field, value=value):
                    malformed = {**cookie, field: value}
                    self.assertFalse(sessions.authenticated(malformed))
                    self.assertTrue(sessions.authenticated(cookie))

    def test_csrf_is_required_unique_and_bound_to_current_session(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            self.login(client, config)
            csrf = self.session(client).json()["csrfToken"]
            cookie = self.cookie(client).encode()
            for tokens in ([], [b"wrong"], [b"\xff"], [csrf.encode()] * 2):
                with self.subTest(tokens=tokens):
                    headers = self.headers(config, (b"cookie", cookie))
                    headers.extend((b"x-csrf-token", token) for token in tokens)
                    self.error(self.raw(app, client, config, method="DELETE", headers=headers),
                               403, "CSRF_INVALID")
                    self.assertEqual(self.session(client).status_code, 200)

    def test_replacement_login_and_tampered_cookie_cannot_reuse_session(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            self.login(client, config)
            first_cookie = self.cookie(client)
            first_csrf = self.session(client).json()["csrfToken"]
            self.login(client, config)
            second = self.session(client)
            self.assertNotEqual(first_csrf, second.json()["csrfToken"])
            self.error(self.session(client, first_cookie), 401, "SESSION_REQUIRED")
            self.error(self.session(client, self.cookie(client) + "tampered"), 401, "SESSION_REQUIRED")
            self.assertEqual(self.session(client).status_code, 200)

    def test_cookie_expires_after_the_one_hour_boundary(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            with patch("itsdangerous.timed.TimestampSigner.get_timestamp", return_value=1000):
                self.login(client, config)
            with patch("itsdangerous.timed.TimestampSigner.get_timestamp", return_value=4600):
                self.assertEqual(self.session(client).status_code, 200)
            with patch("itsdangerous.timed.TimestampSigner.get_timestamp", return_value=4601):
                self.error(self.session(client), 401, "SESSION_REQUIRED")

    def test_restart_rotates_bootstrap_token_and_signing_key(self):
        config = self.config()
        first = server.create_app(config)
        with self.client(first, config) as client:
            token = config.token_path.read_text().strip()
            self.login(client, config)
            old_cookie = self.cookie(client)
        second = server.create_app(config)
        with self.client(second, config) as client:
            self.assertNotEqual(config.token_path.read_text().strip(), token)
            self.error(self.login(client, config, token), 401, "LOGIN_FAILED")
            self.error(self.session(client, old_cookie), 401, "SESSION_REQUIRED")
            self.assertEqual(self.login(client, config).status_code, 200)

    def test_actual_chunk_limit_allows_exact_cap_and_rejects_one_extra_byte(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            body = json.dumps({"token": config.token_path.read_text().strip()}).encode()
            exact = body + b" " * (4096 - len(body))
            for length in (None, b"1"):
                with self.subTest(content_length=length):
                    headers = self.headers(config)
                    if length is not None:
                        headers.append((b"content-length", length))
                    response = self.raw(app, client, config, headers=headers,
                                        chunks=[exact[:2000], exact[2000:]])
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json(), {"authenticated": True})
                    cookie = response.headers["set-cookie"].split(";", 1)[0]
                    self.error(self.raw(app, client, config, headers=headers,
                                        chunks=[exact[:2000], exact[2000:], b" "]),
                               413, "BODY_TOO_LARGE")
                    self.assertEqual(self.session(client, cookie).status_code, 200)

    def test_invalid_or_oversized_declared_lengths_are_rejected(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            body = json.dumps({"token": config.token_path.read_text().strip()}).encode()
            length = str(len(body)).encode()
            other_length = str(len(body) + 1).encode()
            for values, status, code in (([b"4097"], 413, "BODY_TOO_LARGE"),
                                         ([b"-1"], 400, "INVALID_REQUEST"),
                                         ([b"no"], 400, "INVALID_REQUEST"),
                                         ([length, length], 400, "INVALID_REQUEST"),
                                         ([length, other_length], 400, "INVALID_REQUEST")):
                with self.subTest(lengths=values):
                    headers = self.headers(config)
                    headers.extend((b"content-length", value) for value in values)
                    self.error(self.raw(app, client, config, headers=headers, chunks=[body]), status, code)

    def test_bad_login_bodies_preserve_the_current_session(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            self.login(client, config)
            original = self.session(client).json()
            token = config.token_path.read_text().strip()
            bodies = [b"", b"{", b"\xff", b"{}", b"[]", b"null", b'{"token":true}',
                      b'{"token":1}', b'{"token":null}', b'{"token":"a","token":"b"}',
                      json.dumps({"token": token, "unexpected": True}).encode(),
                      json.dumps({"token": token}).encode("utf-16")]
            for body in bodies:
                with self.subTest(body=body[:60]):
                    self.error(client.post("/session", content=body,
                                           headers={"Origin": config.origin, "Content-Type": "application/json"}),
                               400, "INVALID_REQUEST")
                    self.assertEqual(self.session(client).json(), original)
            self.error(self.login(client, config, "a" * 64), 401, "LOGIN_FAILED")
            self.assertEqual(self.session(client).json(), original)

    def test_login_media_type_and_logout_body_are_checked_before_mutation(self):
        config = self.config()
        app = server.create_app(config)
        with self.client(app, config) as client:
            for media_type in (None, "text/plain", "application/x-www-form-urlencoded"):
                with self.subTest(media_type=media_type):
                    headers = {"Origin": config.origin}
                    if media_type is not None:
                        headers["Content-Type"] = media_type
                    self.error(client.post("/session", content=b"{}", headers=headers),
                               415, "UNSUPPORTED_MEDIA_TYPE")
            self.login(client, config)
            csrf = self.session(client).json()["csrfToken"]
            self.error(client.request("DELETE", "/session", content=b" ",
                                      headers={"Origin": config.origin, "X-CSRF-Token": csrf}),
                       400, "INVALID_REQUEST")
            self.assertEqual(self.session(client).status_code, 200)
