"""Download the first output from saved success evidence for a known task."""

import ipaddress
from io import BytesIO
import json
import logging
from pathlib import Path
import re
import time
from typing import Callable

import httpx

from core.json_data import unique_fields
from core.storage import StoredArtifact, read_verified
from features.inference.retrieve import MAX_TASK_RESPONSE_BYTES
from features.inference.runtime import RuntimeEvidence
from features.inference.submission import SubmissionCoordinator
from features.output.validate import MAX_RAW_BYTES


DOWNLOAD_SECONDS = 60.0


class DownloadError(ValueError):
    """Output could not be downloaded from the recorded task."""


def _redact_httpx_url(record: logging.LogRecord) -> bool:
    """Keep method and status logs without exposing URL paths or signatures."""
    if isinstance(record.args, tuple):
        record.args = tuple(f"{value.scheme}://{value.host}/[redacted]" if isinstance(value, httpx.URL)
                            else value for value in record.args)
    return True


def _output_url(task_id: str, evidence_sha256: str, evidence_root: Path,
                allowed_hosts: frozenset[str]) -> httpx.URL:
    SubmissionCoordinator._task_id(task_id)
    if type(evidence_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None:
        raise ValueError("Success evidence needs a SHA-256 hash.")
    if (type(allowed_hosts) is not frozenset or not allowed_hosts
            or any(type(host) is not str for host in allowed_hosts)):
        raise ValueError("Output hosts must be an explicit server-approved set.")
    path = f"provider_responses/{evidence_sha256}.json"
    size = (Path(evidence_root) / path).lstat().st_size
    if not 0 < size <= MAX_TASK_RESPONSE_BYTES:
        raise ValueError("Success evidence exceeds the response size limit.")
    body = read_verified(evidence_root, StoredArtifact(path, evidence_sha256, size))
    data = json.loads(body, object_pairs_hook=unique_fields)
    if type(data) is not dict or data.get("id") != task_id or data.get("status") != "SUCCEEDED":
        raise ValueError("Success evidence does not match the task.")
    outputs = data.get("output")
    if type(outputs) is not list or not outputs or type(outputs[0]) is not str:
        raise ValueError("Success evidence has no output URL.")
    value = outputs[0]
    if len(value) > 8192 or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        raise ValueError("Output URL is invalid.")
    url = httpx.URL(value)
    if (url.scheme != "https" or url.port not in (None, 443) or url.userinfo or url.fragment
            or not url.host or url.host not in allowed_hosts):
        raise ValueError("Output URL must use an approved HTTPS host on port 443.")
    try:
        ipaddress.ip_address(url.host)
    except ValueError:
        return url
    raise ValueError("Output URL must use an approved host name, not an IP address.")


def download_task_output(task_id: str, evidence_sha256: str, evidence_root: Path, *,
                         allowed_hosts: frozenset[str],
                         transport: httpx.BaseTransport | None = None,
                         clock: Callable[[], float] = time.monotonic,
                         runtime: RuntimeEvidence | None = None) -> bytes:
    """GET one output without provider credentials, redirects, or retries.

    The server owns the evidence root and exact host allowlist. Never obtain the
    allowlist from a client or the task response. This function selects the first
    output URL. Pass its bytes to OutputCompleter for MP4 validation and storage.
    Bodies must use identity encoding and fit the raw-video size limit.
    Check a 60-second deadline between chunks; a stalled read can take another
    30 seconds. A failed download leaves the provider task and its cost unchanged.
    """
    operation = None
    try:
        if runtime is not None:
            runtime.require_evidence_root(evidence_root)
        url = _output_url(task_id, evidence_sha256, evidence_root, allowed_hosts)
        # Keep this filter installed. Removing it could expose concurrent GETs.
        logging.getLogger("httpx").addFilter(_redact_httpx_url)
        deadline = clock() + DOWNLOAD_SECONDS
        timeout = httpx.Timeout(30.0, connect=10.0)
        with httpx.Client(transport=transport, timeout=timeout, follow_redirects=False) as client:
            if runtime is not None:
                operation = runtime.download(task_id, evidence_sha256, url.host, transport, client, url)
            if operation is not None:
                operation.start()
            with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
                if operation is not None:
                    operation.observe_response(response)
                if response.status_code != 200:
                    raise ValueError("Output GET did not succeed.")
                if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                    raise ValueError("Output GET returned encoded content.")
                length = response.headers.get("content-length")
                if length is not None and not 0 < int(length) <= MAX_RAW_BYTES:
                    raise ValueError("Output size exceeds the raw-video limit.")
                buffer, size = BytesIO(), 0
                # Do not aggregate small chunks before checking the deadline.
                for chunk in response.iter_bytes():
                    if clock() >= deadline:
                        raise TimeoutError("Output GET reached its deadline.")
                    size += len(chunk)
                    if size > MAX_RAW_BYTES:
                        raise ValueError("Output size exceeds the raw-video limit.")
                    buffer.write(chunk)
                if operation is not None:
                    operation.body_finished()
                if clock() >= deadline or size == 0 or (length is not None and size != int(length)):
                    raise ValueError("Output GET did not finish with the expected bytes in time.")
                contents = buffer.getvalue()
                if operation is not None:
                    operation.complete(contents)
                return contents
    except Exception:
        raise DownloadError("Output GET failed; resume the same task.") from None
    finally:
        if operation is not None:
            operation.complete()
