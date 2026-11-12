"""Check declared review links and raw timestamps in one verified ZIP snapshot."""

from dataclasses import asdict, dataclass
from pathlib import Path

from core.storage import StoredArtifact
from features.inference.preflight import _match_fields
from features.output.align import _nearest_index
from features.output.review_contract import (
    _check_output_metadata, _check_review_header, _parse_review_json, _raw_times, _view_records,
)
from features.review.integrity import _read_verified_archive
from features.review.preparation import ExportPreparationIntegrity, _check_preparation_snapshot


VIEW_ROLES = frozenset({"source_original", "source_normalized", "raw_output", "provider_video_input",
                        "submitted_guide", "source_sample", "guide_sample", "raw_sample"})


class ExportReviewError(ValueError):
    """Archived review declarations do not agree with their linked records."""


@dataclass(frozen=True)
class ExportReviewIntegrity:
    """Verified record relationships, without a media or release verdict."""

    preparation: ExportPreparationIntegrity
    review: StoredArtifact
    raw_metadata: StoredArtifact


def _check_private_references(review: dict, inventory: dict, prefix: str) -> None:
    summaries = inventory["privateEvidenceHashes"]
    paths = {"requestCheck": prefix + f"provider_checks/{summaries['requestCheck']['sha256']}.json",
             "request": prefix + "provider/request.json", "frozenRequest": prefix + "provider/frozen_request.json",
             "providerSuccess": f"provider_responses/{summaries['providerSuccess']['sha256']}.json"}
    for name, path in paths.items():
        record = review[name]
        if type(record["size"]) is not int:
            raise ExportReviewError("Private reference sizes must be integers.")
        _match_fields(review, {name: summaries[name] | {"path": path}})
    _match_fields(inventory["provider"], {
        "payloadSha256": summaries["frozenRequest"]["sha256"],
        "requestBodySha256": summaries["request"]["sha256"],
    })


def _check_raw_samples(review: dict, metadata: dict) -> None:
    times = _raw_times(metadata)
    duration = metadata["durationSeconds"]
    for sample in review["samples"]:
        source_time, raw = sample["source"]["ptsSeconds"], sample["raw"]
        index = raw["frameIndex"]
        if index >= len(times) or index != _nearest_index(times, source_time) or raw["ptsSeconds"] != times[index]:
            raise ExportReviewError("Raw sample does not match the saved native timeline.")
        if source_time < times[0]:
            position = "before_first"
        elif source_time > times[-1]:
            position = "after_last"
        else:
            position = "within_span"
        _match_fields(raw, {"spanPosition": position, "withinDeclaredDuration": 0 <= source_time < duration})
        # The shared view checks already verify offset = raw PTS - source PTS.


def verify_export_review(output_root: Path, archive_ref: StoredArtifact,
                         inventory_ref: StoredArtifact, *, run_id: str,
                         attempt_id: str) -> ExportReviewIntegrity:
    """Verify preparation, review references and recorded raw timing in one ZIP.

    Only two additional fixed JSON records, at most 128 KiB each, are retained.
    The entire archive is checked before any preparation or review semantics.
    Private evidence remains hash/size references; its bytes are never read.

    Raw selection is checked against the saved raw timeline and declared source
    PTS. Full source and guide timelines are not retained and are not invented.
    No media decoding, scorecard assessment, provider/human authentication,
    ledger audit, or global outcome evaluation occurs.
    """
    try:
        snapshot = _read_verified_archive(output_root, archive_ref, inventory_ref,
            run_id=run_id, attempt_id=attempt_id, include_review=True)
        preparation = _check_preparation_snapshot(snapshot, run_id=run_id, attempt_id=attempt_id)
        inventory = snapshot.inventory
        prefix = f"attempts/{attempt_id}/"
        review_ref = StoredArtifact(**inventory["selection"]["reviewManifest"])
        metadata_ref = snapshot.records[prefix + "output/metadata.json"]
        raw_ref = snapshot.records[prefix + "output/raw.mp4"]
        review = _parse_review_json(snapshot.controls[review_ref.path])
        metadata = _parse_review_json(snapshot.controls[metadata_ref.path])
        _check_review_header(review, run_id=run_id, attempt_id=attempt_id)
        _match_fields(review, {"mode": inventory["mode"], "sourceKind": preparation.source_kind,
                              "preparation": asdict(preparation.preparation),
                              "providerTaskId": inventory["provider"]["taskId"],
                              "payloadSha256": inventory["provider"]["payloadSha256"],
                              "rawOutput": asdict(raw_ref), "rawMetadata": asdict(metadata_ref)})
        _check_private_references(review, inventory, prefix)
        views = _view_records(review, prefix)
        expected_paths = {entry["path"] for entry in inventory["files"] if VIEW_ROLES.intersection(entry["roles"])}
        if set(views) != expected_paths or any(record != snapshot.records[path] for path, record in views.items()):
            raise ExportReviewError("Review media and samples differ from the inventory.")
        _check_output_metadata(metadata, raw_ref, metadata_ref, run_id=run_id,
                               attempt_id=attempt_id, provider_task_id=inventory["provider"]["taskId"])
        _match_fields(metadata, {"visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN"})
        _check_raw_samples(review, metadata)
        return ExportReviewIntegrity(preparation, review_ref, metadata_ref)
    except Exception:
        raise ExportReviewError("Archived review records could not be verified for this attempt.") from None
