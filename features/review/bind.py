"""Bind declared human review data to one saved, completed attempt."""

from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from core.storage import StoredArtifact, read_verified
from features.inference.preflight import _match_fields, _read_record
from features.inference.submission import SubmissionCoordinator
from features.output.complete import CompletedOutput, OutputCompleter, _publish_or_verify, read_completed_output
from features.output.review import _read_output_metadata, _verify_success_response
from features.output.view import _verify_views
from features.output.review_contract import ReviewContractError, _check_review_header, _raw_times
from features.review.score import MAX_SCORECARD_BYTES, ScorecardAssessment, assess_scorecard
from features.review.binding_contract import (
    BindingContractError, _assess_bound_card, _exact_native_times, _make_receipt, _receipt_bytes, _utc_seconds,
)


class ScorecardBindingError(ValueError):
    """A scorecard cannot be bound to the saved review evidence."""


@dataclass(frozen=True)
class BoundScorecard:
    """Verified references and a declared rubric result, without a human proof."""

    record: StoredArtifact
    scorecard: StoredArtifact
    review_manifest: StoredArtifact
    assessment: ScorecardAssessment


def _verify_review(root: Path, manifest: StoredArtifact, *, run_id: str, attempt_id: str) -> dict:
    prefix = f"attempts/{attempt_id}/"
    if manifest.path != prefix + "review/manifest.json":
        raise ScorecardBindingError("Review manifest must belong to the requested attempt.")
    data = _read_record(root, manifest, 128 * 1024)
    try:
        _check_review_header(data, run_id=run_id, attempt_id=attempt_id)
    except ReviewContractError as error:
        raise ScorecardBindingError(str(error)) from None
    _verify_views(root, data, prefix)
    return data


def _native_times(root: Path, completed: CompletedOutput) -> set[Decimal]:
    metadata = _read_output_metadata(root, completed)
    try:
        _raw_times(metadata)
    except ReviewContractError as error:
        raise ScorecardBindingError(str(error)) from None
    # The saved JSON has already passed its size, hash, identity, and numeric
    # checks. Read its decimal tokens exactly for endpoint membership.
    return _exact_native_times(read_verified(root, completed.metadata))


def _binding_receipt(root: Path, review_manifest: StoredArtifact, contents: bytes,
                     completed: CompletedOutput | None, *, run_id: str,
                     attempt_id: str) -> tuple[dict, ScorecardAssessment]:
    """Verify evidence and construct the receipt without a clock, ledger, or writes."""
    try:
        assessed = _assess_bound_card(contents, run_id=run_id, attempt_id=attempt_id)
    except BindingContractError as error:
        raise ScorecardBindingError(str(error)) from None
    data = _verify_review(root, review_manifest, run_id=run_id, attempt_id=attempt_id)
    if completed is None or completed.run_id != run_id or completed.attempt_id != attempt_id:
        raise ScorecardBindingError("Review requires completed output for this run and attempt.")
    success = _verify_success_response(root, completed)
    _match_fields(data, {"providerTaskId": completed.provider_task_id,
                        "rawOutput": asdict(completed.raw), "rawMetadata": asdict(completed.metadata),
                        "providerSuccess": asdict(success)})
    native_times = _native_times(root, completed)
    try:
        receipt = _make_receipt(assessed, review_manifest, data, native_times=native_times,
                                completed_unix=completed.completed_unix)
    except BindingContractError as error:
        raise ScorecardBindingError(str(error)) from None
    return receipt, assessed.assessment


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
