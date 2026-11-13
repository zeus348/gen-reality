"""Exercise operator sign-in with a real loopback server and Chromium."""

from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import time
import unittest
from urllib.parse import urlsplit

from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STARTUP_TIMEOUT_SECONDS = 20


def _server_sources():
    """Let the external fault check replace child code without editing files."""
    return {name: (PROJECT_ROOT / name.replace(".", "/")).with_suffix(".py").read_text(encoding="utf-8")
            for name in ("app.local_session", "app.server")}


_CHILD_START = """
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
arguments = sys.argv[2:]
sys.path.insert(0, str(root))
sources = json.load(sys.stdin)
import app

name = "app.local_session"
filename = root / "app" / "local_session.py"
spec = importlib.util.spec_from_file_location(name, filename)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
setattr(app, "local_session", module)
exec(compile(sources[name], str(filename), "exec"), module.__dict__)

# Run the ordinary argument parser and __main__ guard with replaceable code.
sys.argv = ["app.server", *arguments]
__file__ = str(root / "app" / "server.py")
__package__ = "app"
__spec__ = None
exec(compile(sources["app.server"], __file__, "exec"), globals())
"""


def _wait_for_server(process, port, token_file):
    """Wait for this child to serve after its private token is durable."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("The local server exited during startup.")
        if token_file.is_file():
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.25)
            try:
                connection.request("GET", "/")
                response = connection.getresponse()
                if response.status == 200 and process.poll() is None:
                    return
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
        time.sleep(0.05)
    raise AssertionError("The local server did not become ready within the time limit.")


@contextmanager
def _running_server(state_dir, output_root):
    """Start a bounded child with the real CLI options and release it on failure."""
    sources = _server_sources()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    # The CLI owns its listening socket. Release the temporary port reservation
    # immediately before launch; a competing bind makes this bounded test fail.
    arguments = [sys.executable, "-c", _CHILD_START, str(PROJECT_ROOT),
                 "--state-dir", str(state_dir), "--output-root", str(output_root), "--port", str(port)]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    with TemporaryFile() as logs:
        process = subprocess.Popen(arguments, cwd=PROJECT_ROOT, env=environment,
                                   stdin=subprocess.PIPE, stdout=logs, stderr=subprocess.STDOUT)
        server = {"origin": f"http://127.0.0.1:{port}",
                  "token_file": state_dir / "operator-login.txt", "logs": b""}
        try:
            process.stdin.write(json.dumps(sources).encode("utf-8"))
            process.stdin.close()
            _wait_for_server(process, port, server["token_file"])
            yield server
        finally:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            logs.seek(0)
            server["logs"] = logs.read(65_537)


class LocalSessionBrowserTests(unittest.TestCase):
    def test_operator_can_sign_in_reload_and_sign_out_over_loopback(self):
        with TemporaryDirectory() as folder:
            state_dir, output_root = Path(folder) / "operator", Path(folder) / "runs"
            state_dir.mkdir(mode=0o700)
            output_root.mkdir(mode=0o700)
            with _running_server(state_dir, output_root) as server:
                token_bytes = server["token_file"].read_bytes()
                self.assertTrue(re.fullmatch(rb"[0-9a-f]{64}\n", token_bytes) is not None,
                                "The operator credential has an invalid format.")
                self.assertEqual(server["token_file"].stat().st_mode & 0o777, 0o600)
                token = token_bytes.decode("ascii").strip()
                origin = server["origin"]
                requests, blocked, errors = [], [], []
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        executable_path=os.environ.get("GEN_REALITY_BROWSER_BIN", "/usr/bin/chromium"),
                        headless=True)
                    try:
                        context = browser.new_context(service_workers="block")
                        context.set_default_timeout(5_000)

                        def restrict_request(route):
                            url = route.request.url
                            requests.append(url)
                            parsed = urlsplit(url)
                            if parsed.scheme == "http" and parsed.netloc == urlsplit(origin).netloc:
                                route.continue_()
                            else:
                                blocked.append(url)
                                route.abort()

                        context.route("**/*", restrict_request)
                        page = context.new_page()
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        response = page.goto(origin, wait_until="domcontentloaded", timeout=10_000)
                        self.assertEqual(response.status, 200)
                        status = page.get_by_test_id("session-status")
                        expect(status).to_have_text("Not signed in.")
                        try:
                            page.get_by_test_id("login-token").fill(token)
                        except PlaywrightError:
                            # Playwright can include fill arguments in its error log.
                            raise AssertionError("The operator login field was unavailable.") from None
                        page.get_by_test_id("login-button").click()
                        expect(status).to_have_text("Signed in.")
                        session = page.evaluate("""async () => {
                            const response = await fetch('/session', {signal: AbortSignal.timeout(5000)});
                            return {status: response.status, body: await response.json()};
                        }""")
                        self.assertEqual(session["status"], 200)
                        self.assertEqual(sorted(session["body"]), ["authenticated", "csrfToken"])
                        self.assertIs(session["body"]["authenticated"], True)
                        self.assertIsInstance(session["body"]["csrfToken"], str)
                        self.assertGreaterEqual(len(session["body"]["csrfToken"]), 32)
                        cookies = context.cookies(origin)
                        self.assertEqual(len(cookies), 1)
                        self.assertTrue(cookies[0]["httpOnly"])
                        self.assertEqual(cookies[0]["sameSite"], "Strict")
                        self.assertEqual(cookies[0]["path"], "/")
                        page.reload(wait_until="domcontentloaded")
                        expect(status).to_have_text("Signed in.")
                        page.get_by_test_id("logout-button").click()
                        expect(status).to_have_text("Not signed in.")
                        denied = page.evaluate("""async () => {
                            const response = await fetch('/session', {signal: AbortSignal.timeout(5000)});
                            return {status: response.status, body: await response.json()};
                        }""")
                        self.assertEqual(denied["status"], 401)
                        self.assertEqual(set(denied["body"]), {"error"})
                        self.assertEqual(len(context.cookies(origin)), 0, "Logout did not clear the session cookie.")
                        self.assertEqual(len(blocked), 0, "The browser attempted a request outside the local origin.")
                        self.assertEqual(len(errors), 0, "The page raised a JavaScript error.")
                        self.assertGreater(len(requests), 0)
                        self.assertFalse(any(token in url for url in requests), "A URL contained the operator token.")
                    finally:
                        browser.close()
            self.assertLessEqual(len(server["logs"]), 65_536, "The server produced excessive logs.")
            self.assertFalse(token.encode("ascii") in server["logs"], "Server logs contained the operator token.")


if __name__ == "__main__":
    unittest.main()
