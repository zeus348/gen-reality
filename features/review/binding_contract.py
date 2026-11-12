"""Pure declared scorecard, native endpoint, chronology, and receipt checks."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Context, Decimal, ROUND_HALF_EVEN
from hashlib import sha256
import json
import math

from core.storage import StoredArtifact
from features.output.review_contract import MAX_REVIEW_JSON_BYTES
from features.review.score import MAX_SCORECARD_BYTES, ScorecardAssessment, assess_scorecard


MAX_BINDING_BYTES = 64 * 1024


class BindingContractError(ValueError):
    """Declared scorecard data does not support its binding receipt."""


@dataclass(frozen=True)
class _AssessedCard:
    record: StoredArtifact
    data: dict
    assessment: ScorecardAssessment


def _utc_seconds(value: str) -> Decimal:
    """Convert already-validated UTC text without float rounding or arithmetic context."""
    reviewed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elapsed = reviewed - datetime(1970, 1, 1, tzinfo=timezone.utc)
    whole_seconds = elapsed.days * 86400 + elapsed.seconds
    microseconds = whole_seconds * 1000000 + elapsed.microseconds
    context = Context(prec=32, rounding=ROUND_HALF_EVEN, Emin=-99, Emax=99,
                      capitals=1, clamp=0, flags=[], traps=[])
    return Decimal(microseconds).scaleb(-6, context=context)


def _assess_bound_card(contents: bytes, *, run_id: str, attempt_id: str) -> _AssessedCard:
    assessment = assess_scorecard(contents, run_id=run_id, attempt_id=attempt_id)
    if assessment.rubric_verdict == "NOT_RUN":
        raise BindingContractError("A complete reviewer name, UTC time, and signature are required.")
    data = json.loads(contents, parse_float=Decimal)
    digest = sha256(contents).hexdigest()
    record = StoredArtifact(f"attempts/{attempt_id}/review/scorecards/{digest}.json", digest, len(contents))
    return _AssessedCard(record, data, assessment)


def _exact_native_times(contents: bytes) -> set[Decimal]:
    """Read exact numeric tokens after metadata identity and timeline validation."""
    if type(contents) is not bytes or not 0 < len(contents) <= MAX_REVIEW_JSON_BYTES:
        raise BindingContractError("Raw metadata exceeds its size limit.")
    data = json.loads(contents, parse_float=Decimal)
    return {Decimal(value) for value in data["frameSeconds"]}


def _make_receipt(assessed: _AssessedCard, review_manifest: StoredArtifact, review: dict, *,
                   native_times: set[Decimal], completed_unix: float) -> dict:
    """Bind an assessed original card to already-verified review records."""
    card, assessment = assessed.data, assessed.assessment
    for failure in card["hardFailures"]:
        if failure["startPTS"] not in native_times or failure["endPTS"] not in native_times:
            raise BindingContractError("Failure endpoints must equal saved native frame timestamps.")
    if (type(completed_unix) not in (int, float) or not math.isfinite(completed_unix)
            or not 0 <= completed_unix <= 253402300799):
        raise BindingContractError("Completion time is invalid.")
    if _utc_seconds(card["reviewedAtUtc"]) < Decimal(str(completed_unix)):
        raise BindingContractError("Review time must be at or after output completion.")
    return {
        "schemaVersion": "1.4", "manifestType": "bound_review_scorecard",
        "runId": card["runId"], "attemptId": card["attemptId"], "mode": review["mode"],
        "providerTaskId": review["providerTaskId"], "reviewManifest": asdict(review_manifest),
        "scorecard": asdict(assessed.record), "rawOutput": review["rawOutput"],
        "rawMetadata": review["rawMetadata"], "providerSuccess": review["providerSuccess"],
        "completedUnix": completed_unix, "reviewedAtUtc": card["reviewedAtUtc"],
        "rubricVerdict": assessment.rubric_verdict,
        "qualifyingFailureIndexes": list(assessment.qualifying_failure_indexes),
        "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
    }


def _receipt_bytes(receipt: dict) -> bytes:
    return (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
