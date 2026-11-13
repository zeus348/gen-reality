"""Protect the fixed loopback routes before reading a request body."""

from dataclasses import dataclass
import hmac
import re
import secrets

from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.responses import JSONResponse

from features.asset.ingest import MAX_FILE_BYTES
from features.media.ingest import MAX_INPUT_BYTES


MAX_BODY_BYTES = 4096
MAX_HEADER_BYTES = 16 * 1024
MAX_COOKIE_BYTES = 4096
MAX_HEADERS = 64
SESSION_SECONDS = 3600
COOKIE_NAME = "gen_reality_session"
SECURITY_HEADERS = {
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "content-security-policy": (
        "default-src 'none'; script-src 'self'; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}


def error_response(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code}}, status_code=status,
                        headers=SECURITY_HEADERS)


def require_session(request, *, write=False):
    """Check the current login before a protected route reads its body."""
    sessions = request.app.state.sessions
    if not sessions.authenticated(request.session):
        return error_response(401, "SESSION_REQUIRED")
    if write:
        tokens = request.headers.getlist("x-csrf-token")
        if (len(tokens) != 1 or re.fullmatch(r"[0-9a-f]{64}", tokens[0]) is None
                or not hmac.compare_digest(tokens[0], sessions.csrf_token)):
            return error_response(403, "CSRF_INVALID")
    return None


def request_body_limit(scope):
    """Grant larger bodies only to the two fixed upload routes."""
    if scope.get("method") == "POST":
        match = re.fullmatch(r"/runs/run-[0-9a-f]{32}/(source|asset)", scope.get("path", ""))
        if match is not None:
            return MAX_INPUT_BYTES if match[1] == "source" else MAX_FILE_BYTES
    return MAX_BODY_BYTES


class RouteBodyLimit:
    """Select the shipped byte counter without reading or buffering a body."""

    def __init__(self, app):
        self.app = app
        self.limits = {limit: RequestBodyLimitMiddleware(app, max_body_size=limit)
                       for limit in (MAX_BODY_BYTES, MAX_INPUT_BYTES, MAX_FILE_BYTES)}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        await self.limits[request_body_limit(scope)](scope, receive, send)


@dataclass
class SessionState:
    """Keep one current login. Cookies contain no budget or provider authority."""

    bootstrap_token: str | None = None
    session_id: str | None = None
    csrf_token: str | None = None

    def login(self, token: str, cookie: dict) -> bool:
        if self.bootstrap_token is None or not hmac.compare_digest(token, self.bootstrap_token):
            return False
        self.session_id = secrets.token_hex(32)
        self.csrf_token = secrets.token_hex(32)
        cookie.clear()
        cookie.update(sessionId=self.session_id, csrfToken=self.csrf_token)
        return True

    def authenticated(self, cookie: dict) -> bool:
        if self.session_id is None or self.csrf_token is None:
            return False
        if set(cookie) != {"sessionId", "csrfToken"}:
            return False
        session_id, csrf_token = cookie["sessionId"], cookie["csrfToken"]
        return (type(session_id) is str and type(csrf_token) is str
                and re.fullmatch(r"[0-9a-f]{64}", session_id) is not None
                and re.fullmatch(r"[0-9a-f]{64}", csrf_token) is not None
                and hmac.compare_digest(session_id, self.session_id)
                and hmac.compare_digest(csrf_token, self.csrf_token))

    def clear(self):
        self.bootstrap_token = None
        self.session_id = None
        self.csrf_token = None


def _request_headers(scope):
    """Bound raw headers before session decoding, including duplicate names."""
    rows = scope.get("headers", [])
    if len(rows) > MAX_HEADERS:
        return None, error_response(431, "HEADERS_TOO_LARGE")
    total = 0
    headers = {}
    for name, value in rows:
        if type(name) is not bytes or type(value) is not bytes:
            return None, error_response(400, "INVALID_REQUEST")
        total += len(name) + len(value) + 4
        if total > MAX_HEADER_BYTES:
            return None, error_response(431, "HEADERS_TOO_LARGE")
        name = name.lower()
        if (re.fullmatch(rb"[!#$%&'*+.^_`|~0-9a-z-]+", name) is None
                or any(byte < 32 or byte == 127 for byte in value)):
            return None, error_response(400, "INVALID_REQUEST")
        headers.setdefault(name, []).append(value)
    cookies = headers.get(b"cookie", [])
    if len(cookies) > 1:
        return None, error_response(400, "INVALID_REQUEST")
    if cookies:
        if len(cookies[0]) > MAX_COOKIE_BYTES:
            return None, error_response(431, "HEADERS_TOO_LARGE")
        names = [part.strip().split(b"=", 1)[0] for part in cookies[0].split(b";")]
        if names.count(COOKIE_NAME.encode("ascii")) > 1:
            return None, error_response(400, "INVALID_REQUEST")
    return headers, None


class LocalBoundary:
    """Require one exact local Host and Origin; keep early errors safe."""

    def __init__(self, app, *, origin: str):
        self.app = app
        self.origin = origin.encode("ascii")
        self.host = self.origin.removeprefix(b"http://")

    def _check(self, scope):
        headers, error = _request_headers(scope)
        if error is not None:
            return error
        if (headers.get(b"host") != [self.host]
                or any(name in headers for name in (
                    b"forwarded", b"x-forwarded-host", b"x-forwarded-proto", b"x-forwarded-for"))):
            return error_response(400, "HOST_NOT_ALLOWED")
        origins = headers.get(b"origin", [])
        writes = scope["method"] not in {"GET", "HEAD"}
        if ((origins and origins != [self.origin]) or (writes and not origins)):
            return error_response(403, "ORIGIN_NOT_ALLOWED")
        if scope.get("query_string"):
            return error_response(400, "INVALID_REQUEST")
        lengths = headers.get(b"content-length", [])
        transfer = headers.get(b"transfer-encoding", [])
        if (len(lengths) > 1 or len(transfer) > 1 or (lengths and transfer)
                or (transfer and transfer != [b"chunked"])):
            return error_response(400, "INVALID_REQUEST")
        if lengths:
            if re.fullmatch(rb"[0-9]{1,20}", lengths[0]) is None:
                return error_response(400, "INVALID_REQUEST")
            if int(lengths[0]) > request_body_limit(scope):
                return error_response(413, "BODY_TOO_LARGE")
        if scope["method"] == "POST" and scope["path"] == "/session":
            content_types = headers.get(b"content-type", [])
            if (len(content_types) != 1 or content_types[0].lower().replace(b" ", b"")
                    not in (b"application/json", b"application/json;charset=utf-8")):
                return error_response(415, "UNSUPPORTED_MEDIA_TYPE")
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def safe_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                fixed = {name.encode("ascii"): value.encode("ascii")
                         for name, value in SECURITY_HEADERS.items()}
                headers = [(name, value) for name, value in message.get("headers", [])
                           if name.lower() not in fixed]
                message = {**message, "headers": headers + list(fixed.items())}
            await send(message)

        try:
            error = self._check(scope)
            if error is not None:
                await error(scope, receive, safe_send)
                return
            await self.app(scope, receive, safe_send)
        except Exception:
            if started:
                raise RuntimeError("Local response could not finish.") from None
            await error_response(500, "INTERNAL_ERROR")(scope, receive, safe_send)
