"""Bind declared human review data to one saved, completed attempt."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Context, Decimal, ROUND_HALF_EVEN
from hashlib import sha256
import json
import math
from pathlib import Path

from core.storage import StoredArtifact, read_verified
from features.inference.preflight import _match_fields, _read_record
from features.inference.submission import SubmissionCoordinator
from features.output.align import _validate_timeline
from features.output.complete import CompletedOutput, OutputCompleter, _publish_or_verify, read_completed_output
from features.output.review import _read_output_metadata, _verify_success_response
from features.output.validate import MAX_RAW_FRAMES
from features.output.view import _verify_views
from features.review.score import MAX_SCORECARD_BYTES, ScorecardAssessment, assess_scorecard


class ScorecardBindingError(ValueError):
    """A scorecard cannot be bound to the saved review evidence."""


@dataclass(frozen=True)
class BoundScorecard:
    """Verified references and a declared rubric result, without a human proof."""

    record: StoredArtifact
    scorecard: StoredArtifact
    review_manifest: StoredArtifact
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


def _verify_review(root: Path, manifest: StoredArtifact, *, run_id: str, attempt_id: str) -> dict:
    prefix = f"attempts/{attempt_id}/"
    if manifest.path != prefix + "review/manifest.json":
        raise ScorecardBindingError("Review manifest must belong to the requested attempt.")
    data = _read_record(root, manifest, 128 * 1024)
    _match_fields(data, {"schemaVersion": "1.4", "manifestType": "indexed_review_sources",
                        "runId": run_id, "attemptId": attempt_id, "samplePixelFormat": "8-bit RGB",
                        "sampleAutomaticRotationApplied": False,
                        "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN"})
    if data["mode"] not in ("A", "B") or data["sourceKind"] not in ("clean_camera", "synthetic_fixture"):
        raise ScorecardBindingError("Review mode or source kind is invalid.")
    _verify_views(root, data, prefix)
    return data


def _native_times(root: Path, completed: CompletedOutput) -> set[Decimal]:
    metadata = _read_output_metadata(root, completed)
    duration = metadata["durationSeconds"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 20:
        raise ScorecardBindingError("Raw metadata duration is invalid.")
    _validate_timeline(tuple(metadata["frameSeconds"]), minimum_count=1,
                       maximum_count=MAX_RAW_FRAMES, duration_seconds=duration)
    # The saved JSON has already passed its size, hash, identity, and numeric
    # checks. Read its decimal tokens exactly for endpoint membership.
    exact_metadata = json.loads(read_verified(root, completed.metadata), parse_float=Decimal)
    return {Decimal(value) for value in exact_metadata["frameSeconds"]}


def _binding_receipt(root: Path, review_manifest: StoredArtifact, contents: bytes,
                     completed: CompletedOutput | None, *, run_id: str,
                     attempt_id: str) -> tuple[dict, ScorecardAssessment]:
    """Verify evidence and construct the receipt without a clock, ledger, or writes."""
    assessment = assess_scorecard(contents, run_id=run_id, attempt_id=attempt_id)
    if assessment.rubric_verdict == "NOT_RUN":
        raise ScorecardBindingError("A complete reviewer name, UTC time, and signature are required.")
    card = json.loads(contents, parse_float=Decimal)
    data = _verify_review(root, review_manifest, run_id=run_id, attempt_id=attempt_id)
    if completed is None or completed.run_id != run_id or completed.attempt_id != attempt_id:
        raise ScorecardBindingError("Review requires completed output for this run and attempt.")
    success = _verify_success_response(root, completed)
    _match_fields(data, {"providerTaskId": completed.provider_task_id,
                        "rawOutput": asdict(completed.raw), "rawMetadata": asdict(completed.metadata),
                        "providerSuccess": asdict(success)})
    native_times = _native_times(root, completed)
    for failure in card["hardFailures"]:
        if failure["startPTS"] not in native_times or failure["endPTS"] not in native_times:
            raise ScorecardBindingError("Failure endpoints must equal saved native frame timestamps.")
    completed_at = completed.completed_unix
    if (type(completed_at) not in (int, float) or not math.isfinite(completed_at)
            or not 0 <= completed_at <= 253402300799):
        raise ScorecardBindingError("Completion time is invalid.")
    if _utc_seconds(card["reviewedAtUtc"]) < Decimal(str(completed_at)):
        raise ScorecardBindingError("Review time must be at or after output completion.")
    card_record = StoredArtifact(f"attempts/{attempt_id}/review/scorecards/{sha256(contents).hexdigest()}.json",
                                 sha256(contents).hexdigest(), len(contents))
    receipt = {
        "schemaVersion": "1.4", "manifestType": "bound_review_scorecard",
        "runId": run_id, "attemptId": attempt_id, "mode": data["mode"],
        "providerTaskId": completed.provider_task_id, "reviewManifest": asdict(review_manifest),
        "scorecard": asdict(card_record), "rawOutput": asdict(completed.raw),
        "rawMetadata": asdict(completed.metadata), "providerSuccess": asdict(success),
        "completedUnix": completed_at, "reviewedAtUtc": card["reviewedAtUtc"],
        "rubricVerdict": assessment.rubric_verdict,
        "qualifyingFailureIndexes": list(assessment.qualifying_failure_indexes),
        "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
    }
    return receipt, assessment


def _receipt_bytes(receipt: dict) -> bytes:
    return (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_scorecard_binding(coordinator: SubmissionCoordinator, output_root: Path,
                           binding_ref: StoredArtifact, *, run_id: str,
                           attempt_id: str) -> BoundScorecard:
    """Verify one explicitly selected historical binding without publishing.

    Use an already-open coordinator and trusted reference. This reader never
    opens a coordinator, recovers jobs, creates tables, or reads the current
    clock. Historical review time must still follow the recorded completion.
    Current task, media, native endpoints, and declared rubric are rechecked.
    Canonical receipt bytes must match exactly; extra fields are not accepted.
    These checks prove neither human participation nor live provider provenance.
    """
    try:
        with coordinator._mutex:
            coordinator._require_open()
            coordinator.ledger._run_id(run_id)
            coordinator.ledger._attempt_id(attempt_id)
            root = Path(output_root) / run_id
            prefix = f"attempts/{attempt_id}/review/"
            if binding_ref.path != prefix + f"scorecard-bindings/{binding_ref.sha256}.json":
                raise ScorecardBindingError("Binding path does not match its attempt and hash.")
            data = _read_record(root, binding_ref, 64 * 1024)
            scorecard = StoredArtifact(**data["scorecard"])
            if (scorecard.path != prefix + f"scorecards/{scorecard.sha256}.json"
                    or type(scorecard.size) is not int or not 0 < scorecard.size <= MAX_SCORECARD_BYTES):
                raise ScorecardBindingError("Scorecard path or size is invalid.")
            contents = read_verified(root, scorecard)
            review_manifest = StoredArtifact(**data["reviewManifest"])
            completed = read_completed_output(coordinator, output_root, attempt_id)
            expected, assessment = _binding_receipt(root, review_manifest, contents, completed,
                                                    run_id=run_id, attempt_id=attempt_id)
            if read_verified(root, binding_ref) != _receipt_bytes(expected):
                raise ScorecardBindingError("Binding does not match its complete verified receipt.")
            return BoundScorecard(binding_ref, scorecard, review_manifest, assessment)
    except Exception:
        raise ScorecardBindingError("Saved scorecard binding could not be verified for this attempt.") from None


def bind_scorecard(coordinator: SubmissionCoordinator, output_root: Path,
                   review_manifest: StoredArtifact, contents: bytes, *,
                   run_id: str, attempt_id: str) -> StoredArtifact:
    """Preserve a signed scorecard and publish its content-addressed receipt last.

    Supply the trusted manifest returned by build_review_evidence and retain the
    returned binding reference. Corrected cards create separate records; this
    function never selects a latest review. The existing ledger supplies the
    trusted clock. Matching retries reuse the same bytes and receipt.

    Every declared interval endpoint must equal a saved native frame PTS. A
    final display end is not inferred from FPS or container duration. The
    receipt binds records and a declared rubric result; it does not prove a real
    person reviewed them or that generation was live. Both global outcomes stay
    NOT_RUN. Existing review pages and manifests are left unchanged.
    """
    try:
        # Keep unsigned and malformed inputs outside the writer's database path.
        assessment = assess_scorecard(contents, run_id=run_id, attempt_id=attempt_id)
        if assessment.rubric_verdict == "NOT_RUN":
            raise ScorecardBindingError("A complete reviewer name, UTC time, and signature are required.")
        with coordinator._mutex:
            coordinator._require_open()
            root = Path(output_root) / run_id
            completed = OutputCompleter(coordinator, output_root).get(attempt_id)
            receipt, _ = _binding_receipt(root, review_manifest, contents, completed,
                                          run_id=run_id, attempt_id=attempt_id)
            if _utc_seconds(receipt["reviewedAtUtc"]) > Decimal(str(coordinator.ledger._now())):
                raise ScorecardBindingError("Review time must be between output completion and the trusted current time.")

            # All evidence and declarations pass before the first publication.
            prefix = f"attempts/{attempt_id}/review/"
            _publish_or_verify(root, receipt["scorecard"]["path"], contents)
            receipt_bytes = _receipt_bytes(receipt)
            return _publish_or_verify(root,
                prefix + f"scorecard-bindings/{sha256(receipt_bytes).hexdigest()}.json", receipt_bytes)
    except ScorecardBindingError:
        raise
    except Exception:
        raise ScorecardBindingError("Scorecard evidence could not be verified or published for this attempt.") from None
