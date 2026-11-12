"""Publish an explicit completed-attempt inventory without copying private evidence."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from core.storage import StoredArtifact
from features.inference.preflight import _match_fields, _read_record, _timestamp, read_checked_request
from features.inference.prepared import _read_json, read_prepared_attempt
from features.inference.submission import SubmissionCoordinator
from features.output.complete import _publish_or_verify, read_completed_output
from features.output.review import _verify_expected_bytes, _verify_success_response
from features.review.bind import BoundScorecard, _native_times, _verify_review, read_scorecard_binding


MAX_INVENTORY_BYTES = 64 * 1024
SCHEMA_PATH = Path(__file__).with_name("inventory.schema.json")
_VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
FROZEN_ROLES = {
    "frozen/source_original.mp4": "source_original", "frozen/source_original.mov": "source_original",
    "frozen/source_normalized.mp4": "source_normalized", "frozen/asset_original.png": "asset_original",
    "frozen/placement.json": "placement", "frozen/config.json": "configuration",
    "frozen/prompt.txt": "prompt", "frozen/inputs.json": "input_declarations",
    "frozen/keyframe_A.png": "guide_a", "frozen/guide_B.mp4": "guide_b",
}


class ExportInventoryError(ValueError):
    """The saved attempt cannot support this export inventory."""


def _collect_files(preparation: dict, review: dict, preparation_manifest: StoredArtifact,
                   review_manifest: StoredArtifact, bound: BoundScorecard | None) -> list[dict]:
    """Select known records only; merge roles when the exact same file is reused."""
    files = {}

    def add(record: StoredArtifact, role: str):
        fields = asdict(record)
        previous = files.get(record.path)
        if previous is None:
            files[record.path] = fields | {"roles": {role}}
        else:
            _match_fields(previous, fields)
            previous["roles"].add(role)

    add(preparation_manifest, "preparation_manifest")
    for name, record in preparation["artifacts"].items():
        add(StoredArtifact(**record), FROZEN_ROLES[name])
    add(review_manifest, "review_manifest")
    for name, role in (("rawOutput", "raw_output"), ("rawMetadata", "raw_metadata"),
                       ("providerVideoInput", "provider_video_input")):
        add(StoredArtifact(**review[name]), role)
    add(StoredArtifact(**review["guide"]["artifact"]), "submitted_guide")
    for sample in review["samples"]:
        for name in ("source", "guide", "raw"):
            add(StoredArtifact(**sample[name]["image"]), name + "_sample")
    if bound is not None:
        add(bound.record, "scorecard_binding")
        add(bound.scorecard, "scorecard")
    return [files[path] | {"roles": sorted(files[path]["roles"])} for path in sorted(files)]


def build_export_inventory(coordinator: SubmissionCoordinator, output_root: Path,
                            review_manifest: StoredArtifact, *, run_id: str, attempt_id: str,
                            scorecard_binding: StoredArtifact | None = None) -> StoredArtifact:
    """Verify one completed attempt, then publish one content-addressed inventory.

    Use an existing open coordinator and trusted references. Keep all original
    bytes in place. The explicit files list governs future ZIP payload members;
    that ZIP must also carry these exact inventory bytes as inventory.json.
    Private request/check/response bytes are excluded. Their verified local
    hashes are recorded separately and do not establish provider authenticity.

    No clock, freshness check, coordinator recovery, or provider call is needed.
    A selected historical binding is rechecked; none is selected automatically.
    Aggregate payload size is recorded without claiming that a ZIP will fit the
    512 MiB artifact limit. Full-run, billing, timing, and release audits remain
    separate. Both global outcomes stay NOT_RUN.
    """
    try:
        with coordinator._mutex:
            coordinator._require_open()
            coordinator.ledger._run_id(run_id)
            coordinator.ledger._attempt_id(attempt_id)
            root = Path(output_root) / run_id
            prefix = f"attempts/{attempt_id}/"
            review = _verify_review(root, review_manifest, run_id=run_id, attempt_id=attempt_id)
            preparation_manifest = StoredArtifact(**review["preparation"])
            prepared = read_prepared_attempt(root, preparation_manifest, run_id=run_id, attempt_id=attempt_id)
            preparation = _read_json(root, preparation_manifest)
            request_check = StoredArtifact(**review["requestCheck"])
            checked = read_checked_request(root, request_check, run_id=run_id, attempt_id=attempt_id)
            if (checked.frozen.preparation_sha256 != preparation_manifest.sha256
                    or checked.source_kind != prepared.source_kind or checked.frozen.prompt != prepared.prompt):
                raise ExportInventoryError("Saved request does not match the prepared snapshot.")
            _match_fields(_read_record(root, request_check), {"preparation": asdict(preparation_manifest)})
            with coordinator.ledger._connection() as connection:
                reservation = connection.execute("SELECT * FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone()
            if (reservation is None or reservation["run_id"] != run_id
                    or reservation["payload_sha256"] != checked.frozen.payload_sha256
                    or reservation["cost_cents"] != checked.frozen.estimated_cents):
                raise ExportInventoryError("Saved request does not match its reservation.")
            request = _verify_expected_bytes(root, prefix + "provider/request.json", checked.frozen.request_body)
            frozen_request = _verify_expected_bytes(root, prefix + "provider/frozen_request.json", checked.frozen.record_bytes)
            completed = read_completed_output(coordinator, output_root, attempt_id)
            if completed is None or completed.run_id != run_id or completed.attempt_id != attempt_id:
                raise ExportInventoryError("Inventory requires completed output for this run and attempt.")
            coordinator._task_id(completed.provider_task_id)
            _timestamp(completed.completed_unix)
            _native_times(root, completed)
            success = _verify_success_response(root, completed)
            video, _ = prepared.media_for_mode(checked.frozen.mode)
            guide = prepared.keyframe_a if checked.frozen.mode == "A" else prepared.guide_b
            _match_fields(review, {
                "mode": checked.frozen.mode, "sourceKind": prepared.source_kind,
                "payloadSha256": checked.frozen.payload_sha256,
                "request": asdict(request), "frozenRequest": asdict(frozen_request),
                "providerTaskId": completed.provider_task_id, "providerSuccess": asdict(success),
                "rawOutput": asdict(completed.raw), "rawMetadata": asdict(completed.metadata),
                "sourceOriginal": asdict(prepared.source_original), "sourceNormalized": asdict(prepared.normalized_source),
                "providerVideoInput": asdict(video),
                "guide": {"kind": "static_keyframe" if checked.frozen.mode == "A" else "video", "artifact": asdict(guide)},
            })
            bound = None
            if scorecard_binding is not None:
                bound = read_scorecard_binding(coordinator, output_root, scorecard_binding,
                                               run_id=run_id, attempt_id=attempt_id)
                if bound.review_manifest != review_manifest:
                    raise ExportInventoryError("Selected scorecard belongs to another review manifest.")
            files = _collect_files(preparation, review, preparation_manifest, review_manifest, bound)
            private_records = {"requestCheck": request_check, "request": request,
                               "frozenRequest": frozen_request, "providerSuccess": success}
            inventory = {
                "schemaVersion": "1.4", "manifestType": "completed_attempt_export_inventory",
                "runId": run_id, "attemptId": attempt_id, "mode": checked.frozen.mode,
                "sourceKindDeclared": prepared.source_kind, "files": files,
                "selection": {"preparationManifest": asdict(preparation_manifest), "reviewManifest": asdict(review_manifest),
                              "scorecardBinding": asdict(bound.record) if bound else None,
                              "scorecard": asdict(bound.scorecard) if bound else None},
                "privateEvidenceHashes": {name: {"sha256": record.sha256, "size": record.size}
                                          for name, record in private_records.items()},
                "provider": {"vendor": "Runway", "model": "aleph2", "route": "POST /v1/video_to_video",
                             "sdkVersion": json.loads(checked.frozen.record_bytes)["sdkVersion"],
                             "taskId": completed.provider_task_id, "recordedStatus": "SUCCEEDED",
                             "payloadSha256": checked.frozen.payload_sha256, "requestBodySha256": request.sha256,
                             "reservedCents": reservation["cost_cents"], "completedUnix": completed.completed_unix},
                "totalFileBytes": sum(record["size"] for record in files),
                "availability": {"scorecard": "SELECTED" if bound else "NOT_SELECTED",
                                 "providerProvenance": "NOT_ESTABLISHED", "humanProvenance": "NOT_ESTABLISHED",
                                 "actualChargeAudit": "NOT_RUN", "runBudgetAudit": "NOT_RUN",
                                 "elapsedBudgetAudit": "NOT_RUN", "recordedTestAudit": "NOT_RUN"},
                "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
            }
            _VALIDATOR.validate(inventory)
            contents = (json.dumps(inventory, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            if len(contents) > MAX_INVENTORY_BYTES:
                raise ExportInventoryError("Inventory exceeds its size limit.")
            return _publish_or_verify(root, prefix + f"export/inventories/{sha256(contents).hexdigest()}.json", contents)
    except Exception:
        raise ExportInventoryError("Export inventory could not be verified or published for this attempt.") from None
