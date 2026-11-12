"""Bind a paid reservation to frozen request bytes and evidence hashes."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Callable

import httpx

from core.budget import BudgetError, IdempotencyConflict
from core.storage import publish_bytes
from features.inference.runway import API_URL, SDK_VERSION, ProviderError, RunwayCreateAdapter, create_arguments
from features.inference.runtime import RuntimeEvidence
from features.inference.submission import Submission, SubmissionCoordinator, SubmissionError


@dataclass(frozen=True)
class FrozenCreate:
    """Trusted inputs verified by the caller; hashes identify external evidence.

    The preparation hash identifies its manifest, which names the frozen media.
    The pricing hash identifies price-check evidence. Account checks are separate
    so a fresh account GET does not change a reserved request's identity.
    These hashes bind identity; this record does not verify those checks itself.
    """

    mode: str
    video_uri: str
    prompt: str
    preparation_sha256: str
    pricing_sha256: str
    estimated_cents: int
    keyframe_uri: str | None = None

    def __post_init__(self):
        if self.mode not in ("A", "B"):
            raise ValueError("Mode must be A or B.")
        if (self.mode == "A") != (self.keyframe_uri is not None):
            raise ValueError("Mode A needs one keyframe; Mode B must omit it.")
        create_arguments(self.video_uri, self.prompt, self.keyframe_uri)
        for evidence in (self.preparation_sha256, self.pricing_sha256):
            if type(evidence) is not str or re.fullmatch(r"[0-9a-f]{64}", evidence) is None:
                raise ValueError("Preparation and pricing evidence need lowercase SHA-256 hashes.")
        if type(self.estimated_cents) is not int or not 1 <= self.estimated_cents <= 1000:
            raise ValueError("Estimated cost must be 1 to 1000 integer USD cents.")

    @property
    def request_body(self) -> bytes:
        payload = {"model": "aleph2", "videoUri": self.video_uri}
        if self.mode == "A":
            payload["keyframes"] = [{"uri": self.keyframe_uri, "seconds": 0}]
        payload["promptText"] = self.prompt
        # HTTPX also serializes the SDK request. The send step checks exact bytes.
        return httpx.Request("POST", API_URL + "/v1/video_to_video", json=payload).content

    @property
    def record_bytes(self) -> bytes:
        record = {"mode": self.mode, "sdkVersion": SDK_VERSION,
                  "preparationSha256": self.preparation_sha256, "pricingSha256": self.pricing_sha256,
                  "estimatedCents": self.estimated_cents,
                  "requestBodySha256": sha256(self.request_body).hexdigest()}
        return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    @property
    def payload_sha256(self) -> str:
        """Use this hash when reserving the attempt in BudgetLedger."""
        return sha256(self.record_bytes).hexdigest()


def submit_frozen(coordinator: SubmissionCoordinator, attempt_id: str, frozen: FrozenCreate,
                  output_root: Path, api_key: str, *, transport: httpx.BaseTransport | None = None,
                  before_send: Callable[[], None] | None = None,
                  runtime: RuntimeEvidence | None = None) -> Submission:
    """Submit matching reserved inputs and publish request evidence before the POST.

    Only trusted server code may provide the root, API key, and frozen inputs.
    Duplicate calls return the stored state, even without a current API key.
    A recording failure after SUBMITTING remains unknown and cannot be replayed.
    Application callers use submit_checked to supply the final evidence gate.
    """
    with coordinator.ledger._connection() as connection:
        row = connection.execute("SELECT * FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone()
        if row is None:
            raise SubmissionError("Submission has no matching reservation.")
        reservation = coordinator.ledger._reservation(row)
    if runtime is not None:
        runtime.require_identity(output_root, reservation.run_id, attempt_id)
    if reservation.payload_sha256 != frozen.payload_sha256:
        raise IdempotencyConflict("Frozen request does not match the reserved payload.")
    if reservation.cost_cents != frozen.estimated_cents:
        raise BudgetError("Frozen cost does not match the approved quote.")
    existing = coordinator.get(attempt_id)
    if existing.state != "RESERVED":
        return existing

    expected_body = frozen.request_body
    record = frozen.record_bytes
    root = Path(output_root) / reservation.run_id
    prefix = f"attempts/{attempt_id}/provider/"

    def record_request(body):
        if body != expected_body:
            raise ProviderError("SDK request differs from the frozen request.")
        frozen_record = publish_bytes(root, prefix + "frozen_request.json", record)
        request_record = publish_bytes(root, prefix + "request.json", body)
        if runtime is not None:
            runtime.prepare_create(request_record, frozen_record, transport, adapter.http_client)
        if before_send is not None:
            before_send()

    with RunwayCreateAdapter(api_key, record_request=record_request, transport=transport,
            check_response=runtime.observe_create_response if runtime is not None else None) as adapter:
        result = coordinator.submit_once(attempt_id, lambda: adapter.create_task(
            frozen.video_uri, frozen.prompt, keyframe_uri=frozen.keyframe_uri))
        if runtime is not None:
            runtime.finish_create(result)
        return result
