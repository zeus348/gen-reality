"""Serve the private operator login on one fixed loopback origin."""

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import HTMLResponse, JSONResponse, Response
import uvicorn

from app import local_session
from app.preparation import PreparationService
from core.budget import BudgetLedger
from core.json_data import unique_fields
from core.storage import _parent_directory
from features.inference.submission import SubmissionCoordinator


class LocalServerError(ValueError):
    """The private local server could not start safely."""


@dataclass(frozen=True)
class LocalServerConfig:
    state_dir: Path
    output_root: Path
    port: int = 8765

    @property
    def ledger_path(self):
        return self.state_dir / "budget.sqlite"

    @property
    def token_path(self):
        return self.state_dir / "operator-login.txt"

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.port}"


def _check_config(config: LocalServerConfig):
    if (type(config) is not LocalServerConfig or type(config.port) is not int
            or not 1024 <= config.port <= 65535
            or not isinstance(config.state_dir, Path) or not isinstance(config.output_root, Path)):
        raise LocalServerError("Local server configuration is invalid.")
    state = Path(os.path.abspath(config.state_dir))
    output = Path(os.path.abspath(config.output_root))
    if state == output or state.is_relative_to(output):
        raise LocalServerError("Local state must be outside the output directory.")
    with _parent_directory(state, ["operator-login.txt"], create=False) as directory:
        info = os.fstat(directory)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise LocalServerError("Local state requires a private directory owned by this operator.")
        for name in ("budget.sqlite", "budget.sqlite-wal", "budget.sqlite-shm",
                     "budget.sqlite-journal", "budget.sqlite.submission.lock", "operator-login.txt"):
            try:
                item = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(item.st_mode) or item.st_uid != os.getuid():
                raise LocalServerError("Local state contains an unsafe file.")
            if name == "operator-login.txt" and item.st_mode & 0o077:
                raise LocalServerError("Operator login file must remain private.")
    current = Path(output.anchor)
    for part in output.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if not stat.S_ISDIR(info.st_mode):
            raise LocalServerError("Output path must contain only directories.")
    return LocalServerConfig(state, output, config.port)


def _publish_login_token(config: LocalServerConfig, token: str):
    """Replace only the fixed private login file, after owning the ledger."""
    with _parent_directory(config.state_dir, [config.token_path.name], create=False) as directory:
        temporary = ".login-" + uuid4().hex
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write((token + "\n").encode("ascii"))
                file.flush()
                os.fsync(file.fileno())
            try:
                existing = os.stat(config.token_path.name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None and (not stat.S_ISREG(existing.st_mode)
                                         or existing.st_uid != os.getuid()
                                         or existing.st_mode & 0o077):
                raise LocalServerError("Operator login file cannot be replaced safely.")
            os.replace(temporary, config.token_path.name, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
            os.fsync(directory)


def _reject_constant(value):
    raise ValueError("JSON constants are not allowed.")


async def _login_body(request: Request):
    contents = await request.body()
    try:
        value = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields,
                           parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        return None
    if (type(value) is not dict or set(value) != {"token"} or type(value["token"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value["token"]) is None):
        return None
    return value["token"]


def create_app(config: LocalServerConfig) -> FastAPI:
    """Build middleware now; acquire files and exclusive ownership in lifespan."""
    try:
        config = _check_config(config)
        asset_root = Path(__file__).parent / "static"
        page = (asset_root / "login.html").read_text(encoding="utf-8")
        script = (asset_root / "login.js").read_bytes()
    except Exception:
        raise LocalServerError("Local server configuration could not be loaded.") from None
    signing_key = secrets.token_hex(32)
    sessions = local_session.SessionState()

    @asynccontextmanager
    async def lifespan(app):
        owner = None
        preparation = None
        try:
            _check_config(config)
            ledger = BudgetLedger(config.ledger_path)
            owner = SubmissionCoordinator(ledger)
            preparation = PreparationService(config.state_dir, config.output_root, owner)
            token = secrets.token_hex(32)
            _publish_login_token(config, token)
            sessions.bootstrap_token = token
            app.state.coordinator = owner
            app.state.preparation = preparation
            print(f"Local server: {config.origin}", flush=True)
            print(f"Operator login file: {config.token_path}", flush=True)
            yield
        except Exception:
            raise LocalServerError("Local server could not start or stop safely.") from None
        finally:
            sessions.clear()
            app.state.coordinator = None
            app.state.preparation = None
            try:
                if preparation is not None:
                    preparation.close()
            finally:
                if owner is not None:
                    owner.close()

    app = FastAPI(lifespan=lifespan, debug=False, openapi_url=None, docs_url=None, redoc_url=None)
    app.state.coordinator = None
    app.state.preparation = None
    app.state.sessions = sessions
    app.add_middleware(SessionMiddleware, secret_key=signing_key,
                       session_cookie=local_session.COOKIE_NAME,
                       max_age=local_session.SESSION_SECONDS, same_site="strict", https_only=False)
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=local_session.MAX_BODY_BYTES)
    app.add_middleware(local_session.LocalBoundary, origin=config.origin)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        codes = {400: "INVALID_REQUEST", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
                 413: "BODY_TOO_LARGE"}
        return local_session.error_response(error.status_code,
            codes.get(error.status_code, "INVALID_REQUEST"))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return local_session.error_response(400, "INVALID_REQUEST")

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        return local_session.error_response(500, "INTERNAL_ERROR")

    @app.get("/", response_class=HTMLResponse)
    async def login_page():
        return HTMLResponse(page)

    @app.get("/app.js")
    async def login_script():
        return Response(script, media_type="text/javascript")

    @app.post("/session")
    async def login(request: Request):
        token = await _login_body(request)
        if token is None:
            return local_session.error_response(400, "INVALID_REQUEST")
        if not sessions.login(token, request.session):
            return local_session.error_response(401, "LOGIN_FAILED")
        return JSONResponse({"authenticated": True})

    @app.get("/session")
    async def session(request: Request):
        if not sessions.authenticated(request.session):
            return local_session.error_response(401, "SESSION_REQUIRED")
        return JSONResponse({"authenticated": True, "csrfToken": sessions.csrf_token})

    @app.delete("/session")
    async def logout(request: Request):
        if not sessions.authenticated(request.session):
            return local_session.error_response(401, "SESSION_REQUIRED")
        tokens = request.headers.getlist("x-csrf-token")
        if (len(tokens) != 1 or re.fullmatch(r"[0-9a-f]{64}", tokens[0]) is None
                or not hmac.compare_digest(tokens[0], sessions.csrf_token)):
            return local_session.error_response(403, "CSRF_INVALID")
        if await request.body():
            return local_session.error_response(400, "INVALID_REQUEST")
        sessions.session_id = None
        sessions.csrf_token = None
        request.session.clear()
        return JSONResponse({"authenticated": False})

    return app


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise LocalServerError("Local server arguments are invalid.")


def main(argv=None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    try:
        arguments = parser.parse_args(argv)
        config = LocalServerConfig(arguments.state_dir, arguments.output_root, arguments.port)
        app = create_app(config)
        try:
            uvicorn.run(app, host="127.0.0.1", port=config.port, workers=1, lifespan="on",
                        loop="asyncio", http="h11", ws="none", reload=False, proxy_headers=False,
                        access_log=False, log_level="critical", server_header=False)
        except SystemExit as error:
            if error.code:
                raise LocalServerError("Local server did not start.") from None
    except Exception:
        print("Local server could not start safely.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
