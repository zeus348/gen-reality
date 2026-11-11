"""Check account credits and model limits without creating a generation."""

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
from features.inference.runway import API_URL, SDK_VERSION, ProviderError, make_sdk_client


MAX_ACCOUNT_RESPONSE_BYTES = 1024 * 1024
ACCOUNT_GET_SECONDS = 10.0
ACCOUNT_MAX_AGE_SECONDS = 60.0


@dataclass(frozen=True)
class AccountCheck:
    checked_unix: float
    credit_balance: int
    required_credits: int
    blocked_reasons: tuple[str, ...]
    response: StoredArtifact
    record: StoredArtifact

    def require_current(self, now_unix: float) -> None:
        """Require a recent passing check; this does not authorize spending."""
        if (self.blocked_reasons or type(now_unix) not in (int, float) or not math.isfinite(now_unix)
                or not 0 <= now_unix - self.checked_unix <= ACCOUNT_MAX_AGE_SECONDS):
            raise ProviderError("Account check is blocked or stale; check the account again.")


def _count(value) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("Account counts must be nonnegative integers.")
    return value


def _account_limits(data: dict, required_credits: int) -> tuple[int, tuple[str, ...]]:
    balance = _count(data["creditBalance"])
    models, usage = data["tier"]["models"], data["usage"]["models"]
    if type(models) is not dict or type(usage) is not dict:
        raise ValueError("Account model limits and usage are missing.")
    reasons = []
    if balance < required_credits:
        reasons.append("INSUFFICIENT_CREDITS")
    if "aleph2" not in models:
        reasons.append("MODEL_UNAVAILABLE")
    else:
        limits = models["aleph2"]
        concurrent, daily = limits["maxConcurrentGenerations"], limits["maxDailyGenerations"]
        used = _count(usage["aleph2"]["dailyGenerations"])
        if concurrent is not None and _count(concurrent) == 0:
            reasons.append("MODEL_CONCURRENCY_DISABLED")
        if daily is not None and used >= _count(daily):
            reasons.append("MODEL_DAILY_LIMIT_REACHED")
    return balance, tuple(reasons)


def _record_bytes(root: Path, group: str, body: bytes) -> StoredArtifact:
    digest = sha256(body).hexdigest()
    path = f"provider_account/{group}/{digest}.json"
    try:
        return publish_bytes(root, path, body)
    except ArtifactExistsError:
        result = StoredArtifact(path, digest, len(body))
        read_verified(root, result)
        return result


def check_account_once(api_key: str, required_credits: int, evidence_root: Path, *,
                        transport: httpx.BaseTransport | None = None,
                        clock: Callable[[], float] = time.time,
                        elapsed_clock: Callable[[], float] = time.monotonic) -> AccountCheck:
    """GET the SDK account endpoint and retain the response and check result.

    Trusted server code supplies the current quote in credits. This checks the
    balance, Aleph 2 model limits, and daily usage; it cannot guarantee a future
    provider response. Missing limits or usage fail closed. Explicit null limits
    mean unlimited. Repeat this check before each paid submission, together with
    current pricing and budget authorization. No price or approval is inferred.
    Successful bodies must use identity encoding and fit within 1 MiB. Check
    a ten-second deadline between chunks; a stalled read can take another ten.
    """
    try:
        if _count(required_credits) == 0:
            raise ValueError("Required credits must be positive.")
        checked = clock()
        if type(checked) not in (int, float) or not math.isfinite(checked) or checked < 0:
            raise ValueError("Account check time is invalid.")
        deadline = elapsed_clock() + ACCOUNT_GET_SECONDS
        with make_sdk_client(api_key, transport=transport) as client:
            with client.organization.with_streaming_response.retrieve(
                    timeout=ACCOUNT_GET_SECONDS, extra_headers={"Accept-Encoding": "identity"}) as response:
                if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                    raise ValueError("Account response must use identity encoding.")
                buffer, size = BytesIO(), 0
                for chunk in response.iter_bytes():
                    if elapsed_clock() >= deadline:
                        raise TimeoutError("Account GET reached its deadline.")
                    size += len(chunk)
                    if size > MAX_ACCOUNT_RESPONSE_BYTES:
                        raise ValueError("Account response exceeds the size limit.")
                    buffer.write(chunk)
                if elapsed_clock() >= deadline:
                    raise TimeoutError("Account GET reached its deadline.")
                body = buffer.getvalue()
        data = json.loads(body, object_pairs_hook=unique_fields)
        if api_key in json.dumps(data, ensure_ascii=False, allow_nan=False):
            raise ValueError("Account response contains a credential.")
        balance, reasons = _account_limits(data, required_credits)
        completed = clock()
        if (type(completed) not in (int, float) or not math.isfinite(completed)
                or not checked <= completed <= checked + ACCOUNT_MAX_AGE_SECONDS):
            raise ValueError("Account check time moved backwards or the check is stale.")
        response_record = _record_bytes(evidence_root, "responses", body)
        details = {"schemaVersion": "1.4", "manifestType": "provider_account_check",
                   "apiOrigin": API_URL, "sdkVersion": SDK_VERSION, "model": "aleph2",
                   "checkedAtUnix": checked, "completedAtUnix": completed, "creditBalance": balance,
                   "requiredCredits": required_credits, "blockedReasons": reasons,
                   "response": asdict(response_record), "spendingAuthorized": False}
        record_body = (json.dumps(details, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        record = _record_bytes(evidence_root, "checks", record_body)
        return AccountCheck(checked, balance, required_credits, reasons, response_record, record)
    except Exception:
        raise ProviderError("Account check or evidence recording failed; no generation was submitted.") from None
