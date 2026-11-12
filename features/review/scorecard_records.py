"""Verify an explicitly selected historical scorecard and its archived receipt."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.storage import StoredArtifact
from features.output.review_contract import _parse_review_json
from features.review.binding_contract import (
    _assess_bound_card, _exact_native_times, _make_receipt, _receipt_bytes,
)
from features.review.integrity import _read_verified_archive
from features.review.review_records import ExportReviewIntegrity, _check_review_snapshot
from features.review.score import ScorecardAssessment


class ExportScorecardError(ValueError):
    """The selected archived scorecard does not match its complete binding."""


@dataclass(frozen=True)
class ExportScorecardIntegrity:
    """Selected declared rubric evidence, without human or release authentication."""

    review: ExportReviewIntegrity
    selection_status: Literal["SELECTED", "NOT_SELECTED"]
    binding: StoredArtifact | None
    scorecard: StoredArtifact | None
    assessment: ScorecardAssessment | None


def verify_export_scorecard(output_root: Path, archive_ref: StoredArtifact,
                            inventory_ref: StoredArtifact, *, run_id: str,
                            attempt_id: str) -> ExportScorecardIntegrity:
    """Verify the selected original card and canonical receipt in one snapshot.

    Finish archive, preparation, and review checks first. Selection comes only
    from the validated inventory; retain at most one card and binding of 64 KiB
    each. No selection is explicit and never replaced with a latest record.

    Recompute the declared rubric, require exact saved native endpoints, and
    require completion at or before review time. Historical checks use no
    current clock. A complete rubric FAIL can be consistent evidence. No human,
    provider, decoded-media, ledger, or global-outcome claim is established.
    """
    try:
        snapshot = _read_verified_archive(output_root, archive_ref, inventory_ref,
            run_id=run_id, attempt_id=attempt_id, include_scorecard=True)
        verified_review = _check_review_snapshot(snapshot, run_id=run_id, attempt_id=attempt_id)
        selection = snapshot.inventory["selection"]
        if selection["scorecardBinding"] is None:
            return ExportScorecardIntegrity(verified_review, "NOT_SELECTED", None, None, None)
        binding = StoredArtifact(**selection["scorecardBinding"])
        scorecard = StoredArtifact(**selection["scorecard"])
        assessed = _assess_bound_card(snapshot.controls[scorecard.path], run_id=run_id, attempt_id=attempt_id)
        if assessed.record != scorecard:
            raise ExportScorecardError("Original scorecard bytes differ from the selected record.")
        review = _parse_review_json(snapshot.controls[verified_review.review.path])
        native_times = _exact_native_times(snapshot.controls[verified_review.raw_metadata.path])
        expected = _make_receipt(assessed, verified_review.review, review, native_times=native_times,
                                 completed_unix=snapshot.inventory["provider"]["completedUnix"])
        if snapshot.controls[binding.path] != _receipt_bytes(expected):
            raise ExportScorecardError("Selected binding differs from its complete canonical receipt.")
        return ExportScorecardIntegrity(verified_review, "SELECTED", binding, scorecard, assessed.assessment)
    except Exception:
        raise ExportScorecardError("Archived scorecard binding could not be verified for this attempt.") from None
