"""Read declared attempt facts from one fully checked historical export ZIP."""

from dataclasses import dataclass
from pathlib import Path

from core.storage import StoredArtifact
from features.output.review_contract import _parse_review_json
from features.review.integrity import _read_verified_archive
from features.review.scorecard_records import ExportScorecardIntegrity, _check_scorecard_snapshot


class ExportAttemptFactsError(ValueError):
    """The selected archive cannot supply consistent declared attempt facts."""


@dataclass(frozen=True)
class ExportAttemptFacts:
    """Verified local declarations and references, without authenticity claims."""

    run_id: str
    attempt_id: str
    mode: str
    source_kind_declared: str
    sdk_version: str
    provider_task_id: str
    payload_sha256: str
    request_body_sha256: str
    reserved_cents: int
    completed_unix: float
    raw_output: StoredArtifact
    request_check: StoredArtifact
    request: StoredArtifact
    frozen_request: StoredArtifact
    provider_success: StoredArtifact
    graph: ExportScorecardIntegrity


def read_export_attempt_facts(output_root: Path, archive_ref: StoredArtifact,
                              inventory_ref: StoredArtifact, *, run_id: str,
                              attempt_id: str) -> ExportAttemptFacts:
    """Check one archive snapshot, then return its fixed fields for later joins.

    Retained caller references select the ZIP and inventory. Existing checks
    verify every archived byte and the preparation, review, and selected-card
    relationships before any facts are returned. An absent card stays unselected;
    a selected unsigned card remains invalid under the binding contract.

    Private evidence references are declarations, not reads of their contents.
    This reader does not inspect a ledger, decode media, authenticate a provider
    or human, or assign global outcomes. It needs only the selected ZIP file.
    """
    try:
        snapshot = _read_verified_archive(output_root, archive_ref, inventory_ref,
            run_id=run_id, attempt_id=attempt_id, include_scorecard=True)
        graph = _check_scorecard_snapshot(snapshot, run_id=run_id, attempt_id=attempt_id)
        inventory = snapshot.inventory
        provider = inventory["provider"]
        review = _parse_review_json(snapshot.controls[graph.review.review.path])
        return ExportAttemptFacts(
            run_id=run_id, attempt_id=attempt_id, mode=inventory["mode"],
            source_kind_declared=graph.review.preparation.source_kind,
            sdk_version=provider["sdkVersion"], provider_task_id=provider["taskId"],
            payload_sha256=provider["payloadSha256"], request_body_sha256=provider["requestBodySha256"],
            reserved_cents=provider["reservedCents"], completed_unix=provider["completedUnix"],
            raw_output=snapshot.records[f"attempts/{attempt_id}/output/raw.mp4"],
            request_check=StoredArtifact(**review["requestCheck"]),
            request=StoredArtifact(**review["request"]), frozen_request=StoredArtifact(**review["frozenRequest"]),
            provider_success=StoredArtifact(**review["providerSuccess"]), graph=graph,
        )
    except Exception:
        raise ExportAttemptFactsError("Archived attempt facts could not be verified for this selection.") from None
