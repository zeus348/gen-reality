"""Bind verified preparation and provider evidence before reserving a request."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Callable

from core.budget import BudgetLedger
from core.json_data import unique_fields
from core.storage import ArtifactExistsError, StoredArtifact, publish_bytes, read_verified
from features.inference.account import ACCOUNT_MAX_AGE_SECONDS, AccountCheck, _account_limits
from features.inference.frozen import FrozenCreate
from features.inference.prepared import read_prepared_attempt
from features.inference.pricing import PRICE_MAX_AGE_SECONDS, PRICE_SOURCE_URL, PriceQuote
from features.inference.runway import API_URL, SDK_VERSION, _require_upload_uri
from features.inference.upload import UPLOAD_LIFETIME_SECONDS, UploadedArtifact


class PreflightError(ValueError):
    """Preparation and provider evidence do not support a current request."""


@dataclass(frozen=True)
class CheckedRequest:
    run_id: str
    attempt_id: str
    source_kind: str
    frozen: FrozenCreate
    checked_unix: float
    valid_until_unix: float
    evidence: StoredArtifact

    def require_current(self, now_unix: float) -> None:
        _timestamp(now_unix)
        if not self.checked_unix <= now_unix < self.valid_until_unix:
            raise PreflightError("Request check has expired or its clock moved backwards.")


def _timestamp(value):
    if (type(value) not in (int, float) or not 0 <= value <= 253_402_300_799
            or not math.isfinite(value)):
        raise PreflightError("Request check needs a valid UTC timestamp.")


def _read_record(root: Path, record: StoredArtifact, limit: int = 64 * 1024) -> dict:
    if type(record.size) is not int or not 0 < record.size <= limit:
        raise PreflightError("Provider evidence exceeds its size limit.")
    data = json.loads(read_verified(root, record).decode("utf-8"), object_pairs_hook=unique_fields)
    if type(data) is not dict:
        raise PreflightError("Provider evidence must be a JSON object.")
    json.dumps(data, allow_nan=False)
    return data


def _match_fields(data: dict, expected: dict) -> None:
    # JSON comparison distinguishes booleans from integer counts and sizes.
    actual = {name: data[name] for name in expected}
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise PreflightError("Provider evidence does not match the supplied record.")


def read_checked_request(run_root: Path, evidence: StoredArtifact, *,
                         run_id: str, attempt_id: str) -> CheckedRequest:
    """Restore a saved request from a trusted producer's file reference.

    Verify identity without requiring freshness so known or uncertain work can
    be inspected after expiry. This reader grants no spending authority.
    """
    try:
        BudgetLedger._run_id(run_id)
        BudgetLedger._attempt_id(attempt_id)
        expected_path = f"attempts/{attempt_id}/provider_checks/{evidence.sha256}.json"
        if evidence.path != expected_path:
            raise PreflightError("Request check path does not match its attempt and hash.")
        data = _read_record(run_root, evidence)
        frozen = FrozenCreate(**data["frozen"])
        _match_fields(data, {"schemaVersion": "1.4", "manifestType": "checked_provider_request",
                            "runId": run_id, "attemptId": attempt_id,
                            "payloadSha256": frozen.payload_sha256, "spendingAuthorized": False})
        if data["sourceKind"] not in ("clean_camera", "synthetic_fixture"):
            raise PreflightError("Request check has an invalid source kind.")
        checked, expires = data["checkedAtUnix"], data["validUntilUnix"]
        _timestamp(checked)
        _timestamp(expires)
        if checked >= expires:
            raise PreflightError("Request check has an invalid time interval.")
        return CheckedRequest(run_id, attempt_id, data["sourceKind"], frozen, checked, expires, evidence)
    except Exception:
        raise PreflightError("Saved request check could not be verified.") from None


def _check_upload(root: Path, upload: UploadedArtifact, source: StoredArtifact, now: float) -> None:
    if upload.source != source:
        raise PreflightError("Upload does not match the selected frozen media.")
    _require_upload_uri(upload.uri)
    for value in (upload.started_unix, upload.completed_unix, upload.expires_unix):
        _timestamp(value)
    if (upload.expires_unix != upload.started_unix + UPLOAD_LIFETIME_SECONDS
            or not upload.started_unix <= upload.completed_unix <= now < upload.expires_unix):
        raise PreflightError("Upload has expired or its recorded times are invalid.")
    _match_fields(_read_record(root, upload.evidence), {
        "schemaVersion": "1.4", "manifestType": "provider_upload", "sdkVersion": SDK_VERSION,
        "source": asdict(source), "uri": upload.uri, "startedAtUnix": upload.started_unix,
        "completedAtUnix": upload.completed_unix, "expiresAtUnix": upload.expires_unix})


def check_prepared_request(run_root: Path, manifest: StoredArtifact, *, run_id: str,
                           attempt_id: str, mode: str, video: UploadedArtifact,
                           keyframe: UploadedArtifact | None, price: PriceQuote, account: AccountCheck,
                           evidence_root: Path, clock: Callable[[], float] = time.time) -> CheckedRequest:
    """Use records from the trusted preparation, upload, price, and account writers.

    This function makes no provider calls and grants no spending authority.
    It retains enough frozen fields to reconstruct the request after a restart.
    Account refreshes get new check records without changing the frozen payload.
    Price interpretation and provider authenticity remain producer responsibilities.
    Recheck freshness at the final send boundary before a new paid POST.
    """
    try:
        prepared = read_prepared_attempt(run_root, manifest, run_id=run_id, attempt_id=attempt_id)
        selected_video, selected_keyframe = prepared.media_for_mode(mode)
        now = clock()
        _timestamp(now)
        price.require_current(now)
        account.require_current(now)
        price_data = _read_record(evidence_root, price.record)
        _match_fields(price_data, {"schemaVersion": "1.4", "manifestType": "operator_price_quote",
            "model": "aleph2", "clipSeconds": 5, "priceSourceUrl": PRICE_SOURCE_URL,
            "checkedAtUnix": price.checked_unix, "requiredCredits": price.required_credits,
            "reservedCents": price.reserved_cents, "source": asdict(price.source), "spendingAuthorized": False})
        read_verified(evidence_root, price.source)
        if account.required_credits != price.required_credits:
            raise PreflightError("Account check must cover the quoted credits.")
        _match_fields(_read_record(evidence_root, account.record), {
            "schemaVersion": "1.4", "manifestType": "provider_account_check", "apiOrigin": API_URL,
            "sdkVersion": SDK_VERSION, "model": "aleph2", "checkedAtUnix": account.checked_unix,
            "creditBalance": account.credit_balance, "requiredCredits": account.required_credits,
            "blockedReasons": list(account.blocked_reasons), "response": asdict(account.response),
            "spendingAuthorized": False})
        balance, reasons = _account_limits(_read_record(evidence_root, account.response, 1024 * 1024),
                                           price.required_credits)
        if balance != account.credit_balance or reasons != account.blocked_reasons or reasons:
            raise PreflightError("Account response does not support the quoted credits.")
        _check_upload(evidence_root, video, selected_video, now)
        if (selected_keyframe is None) != (keyframe is None):
            raise PreflightError("Mode A needs its keyframe upload; Mode B must omit it.")
        expires = [price.checked_unix + PRICE_MAX_AGE_SECONDS,
                   account.checked_unix + ACCOUNT_MAX_AGE_SECONDS, video.expires_unix]
        if keyframe is not None:
            _check_upload(evidence_root, keyframe, selected_keyframe, now)
            expires.append(keyframe.expires_unix)
        frozen = FrozenCreate(mode, video.uri, prepared.prompt, manifest.sha256, price.record.sha256,
                              price.reserved_cents, keyframe.uri if keyframe else None)
        valid_until = min(expires)
        finished = clock()
        _timestamp(finished)
        if not now <= finished < valid_until:
            raise PreflightError("Request evidence expired during verification.")
        details = {"schemaVersion": "1.4", "manifestType": "checked_provider_request",
                   "runId": run_id, "attemptId": attempt_id, "sourceKind": prepared.source_kind,
                   "checkedAtUnix": now, "validUntilUnix": valid_until, "frozen": asdict(frozen),
                   "payloadSha256": frozen.payload_sha256, "preparation": asdict(manifest),
                   "videoUpload": asdict(video.evidence), "keyframeUpload": asdict(keyframe.evidence) if keyframe else None,
                   "price": asdict(price.record), "account": asdict(account.record), "spendingAuthorized": False}
        body = (json.dumps(details, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        path = f"attempts/{attempt_id}/provider_checks/{sha256(body).hexdigest()}.json"
        try:
            evidence = publish_bytes(run_root, path, body)
        except ArtifactExistsError:
            evidence = StoredArtifact(path, sha256(body).hexdigest(), len(body))
            read_verified(run_root, evidence)
        return CheckedRequest(run_id, attempt_id, prepared.source_kind, frozen, now, valid_until, evidence)
    except Exception:
        raise PreflightError("Prepared request or its provider evidence could not be verified.") from None
