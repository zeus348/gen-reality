"""Upload verified frozen bytes and retain their Runway URI mapping."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import Callable

import httpx

from core.json_data import unique_fields
from core.storage import ArtifactExistsError, StoredArtifact, publish_bytes, read_verified
from features.inference.runway import SDK_VERSION, ProviderError, _require_upload_uri, make_sdk_client


MIN_UPLOAD_BYTES = 512
MAX_UPLOAD_BYTES = 200_000_000
MAX_UPLOAD_RESPONSE_BYTES = 1024 * 1024
UPLOAD_LIFETIME_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class UploadedArtifact:
    source: StoredArtifact
    uri: str
    started_unix: float
    completed_unix: float
    expires_unix: float
    evidence: StoredArtifact


class _UploadMetadataStream(httpx.SyncByteStream):
    """Limit metadata bytes and combine tiny chunks before HTTPX buffers them."""

    def __init__(self, stream: httpx.SyncByteStream):
        self.stream = stream

    def __iter__(self):
        buffer, size = BytesIO(), 0
        for chunk in self.stream:
            size += len(chunk)
            if size > MAX_UPLOAD_RESPONSE_BYTES:
                raise ValueError("Upload metadata exceeds the response size limit.")
            buffer.write(chunk)
            if buffer.tell() >= 64 * 1024:
                yield buffer.getvalue()
                buffer = BytesIO()
        if buffer.tell():
            yield buffer.getvalue()

    def close(self):
        self.stream.close()


def upload_frozen_artifact(source_root: Path, source: StoredArtifact, api_key: str,
                           evidence_root: Path, *, allowed_upload_hosts: frozenset[str],
                           transport: httpx.BaseTransport | None = None,
                           clock: Callable[[], float] = time.time) -> UploadedArtifact:
    """Upload one frozen PNG or MP4 through the pinned SDK, with no task creation.

    The caller verifies preparation, account access, and the server-owned host
    allowlist first. The SDK starts an upload, then uses a separate HTTPX client
    for the multipart POST. Both clients have retries and redirects disabled.
    Read and verify the file once; pass those bytes to the SDK without reopening
    the path. A failed call never repeats automatically. A later explicit upload
    needs a new provider URI. Expiry is conservatively measured from call start.
    """
    try:
        if (type(allowed_upload_hosts) is not frozenset or not allowed_upload_hosts
                or any(type(host) is not str for host in allowed_upload_hosts)):
            raise ValueError("Upload hosts must be an explicit server-approved set.")
        if (type(source.size) is not int or not MIN_UPLOAD_BYTES <= source.size <= MAX_UPLOAD_BYTES
                or Path(source.path).suffix not in (".mp4", ".png")):
            raise ValueError("Upload needs a frozen PNG or MP4 of 512 bytes to 200 MB.")
        contents = read_verified(source_root, source)
        filename = Path(source.path).name
        started = clock()
        if type(started) not in (int, float) or not math.isfinite(started) or started < 0:
            raise ValueError("Upload start time is invalid.")

        def check_response(response):
            if not response.is_success:
                return
            if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                raise ValueError("Upload metadata must use identity encoding.")
            response.stream = _UploadMetadataStream(response.stream)
            body = response.read()
            if len(body) > MAX_UPLOAD_RESPONSE_BYTES:
                raise ValueError("Upload metadata exceeds the response size limit.")
            data = json.loads(body, object_pairs_hook=unique_fields)
            uri = data["runwayUri"]
            _require_upload_uri(uri)
            if api_key in uri:
                raise ValueError("Upload URI contains a credential.")
            url = httpx.URL(data["uploadUrl"])
            if (url.scheme != "https" or url.port not in (None, 443) or url.userinfo or url.fragment
                    or url.host not in allowed_upload_hosts):
                raise ValueError("Upload URL must use an approved HTTPS host on port 443.")

        with make_sdk_client(api_key, transport=transport, check_response=check_response) as client:
            response = client.uploads.create_ephemeral(file=(filename, contents),
                                                       extra_headers={"Accept-Encoding": "identity"},
                                                       timeout=httpx.Timeout(60.0, connect=10.0))
        _require_upload_uri(response.uri)
        completed = clock()
        expires = started + UPLOAD_LIFETIME_SECONDS
        if type(completed) not in (int, float) or not math.isfinite(completed) or not started <= completed < expires:
            raise ValueError("Upload completion time is invalid or the URI has expired.")
        record = {"schemaVersion": "1.4", "manifestType": "provider_upload", "sdkVersion": SDK_VERSION,
                  "source": asdict(source), "uri": response.uri,
                  "startedAtUnix": started, "completedAtUnix": completed, "expiresAtUnix": expires}
        body = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        digest = sha256(body).hexdigest()
        path = f"provider_uploads/{digest}.json"
        try:
            evidence = publish_bytes(evidence_root, path, body)
        except ArtifactExistsError:
            evidence = StoredArtifact(path, digest, len(body))
            read_verified(evidence_root, evidence)
        return UploadedArtifact(source, response.uri, started, completed, expires, evidence)
    except Exception:
        raise ProviderError("Upload or evidence recording failed; no generation was submitted.") from None
