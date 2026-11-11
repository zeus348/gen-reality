"""Retrieve a known task with the SDK and save its response evidence."""

from hashlib import sha256
import json
import math
from pathlib import Path

import httpx
from runwayml import APITimeoutError

from core.json_data import unique_fields
from core.storage import ArtifactExistsError, StoredArtifact, publish_bytes, read_verified
from features.inference.polling import PROVIDER_STATUSES, TaskObservation
from features.inference.runway import ProviderError, make_sdk_client
from features.inference.submission import SubmissionCoordinator


MAX_TASK_RESPONSE_BYTES = 1024 * 1024


def retrieve_task_once(task_id: str, api_key: str, evidence_root: Path, *,
                       timeout_seconds: float = 10.0,
                       transport: httpx.BaseTransport | None = None) -> TaskObservation:
    """Perform one GET, then store a valid successful JSON body by its hash.

    This reads task metadata, not the output video. Successful bodies are limited
    to 1 MiB. Errors disclose no response details, and no create method is used.
    Identical responses share a verified file. Never replace existing evidence.
    """
    SubmissionCoordinator._task_id(task_id)
    if (type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 30
            or not math.isfinite(timeout_seconds)):
        raise ValueError("Task GET timeout must be greater than zero and at most 30 seconds.")
    try:
        with make_sdk_client(api_key, transport=transport) as client:
            with client.tasks.with_streaming_response.retrieve(task_id, timeout=timeout_seconds) as response:
                chunks, size = [], 0
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    size += len(chunk)
                    if size > MAX_TASK_RESPONSE_BYTES:
                        raise ProviderError("Provider task response is too large.")
                    chunks.append(chunk)
                body = b"".join(chunks)
        data = json.loads(body, object_pairs_hook=unique_fields)
        if (type(data) is not dict or data.get("id") != task_id
                or data.get("status") not in PROVIDER_STATUSES):
            raise ProviderError("Provider task response is invalid.")
        if api_key in json.dumps(data, ensure_ascii=False):
            raise ProviderError("Provider task response contains a credential.")
        digest = sha256(body).hexdigest()
        path = f"provider_responses/{digest}.json"
        try:
            publish_bytes(evidence_root, path, body)
        except ArtifactExistsError:
            read_verified(evidence_root, StoredArtifact(path, digest, len(body)))
        return TaskObservation(task_id, data["status"], digest)
    except (APITimeoutError, httpx.TimeoutException):
        raise TimeoutError("Task GET timed out; resume with the same task ID.") from None
    except Exception:
        raise ProviderError("Task GET or evidence recording failed; the task remains unchanged.") from None
