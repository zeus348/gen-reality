"""Build immutable review samples from one verified submitted attempt."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from core.storage import StoredArtifact, read_verified
from features.inference.preflight import _match_fields, _read_record, read_checked_request
from features.inference.prepared import read_prepared_attempt
from features.inference.retrieve import MAX_TASK_RESPONSE_BYTES
from features.inference.submission import SubmissionCoordinator
from features.media.ingest import MAX_INPUT_PIXELS, verify_normalized
from features.output.align import plan_review_frames
from features.output.complete import CompletedOutput, OutputCompleter, _publish_or_verify
from features.output.extract import _read_png, extract_frame_png


class ReviewEvidenceError(ValueError):
    """The saved attempt cannot support an immutable review set."""


def _verify_expected_bytes(root: Path, path: str, contents: bytes) -> StoredArtifact:
    record = StoredArtifact(path, sha256(contents).hexdigest(), len(contents))
    read_verified(root, record)
    return record


def _read_output_metadata(root: Path, completed: CompletedOutput) -> dict:
    prefix = f"attempts/{completed.attempt_id}/output/"
    if completed.raw.path != prefix + "raw.mp4" or completed.metadata.path != prefix + "metadata.json":
        raise ReviewEvidenceError("Completed output paths must belong to the same attempt.")
    data = _read_record(root, completed.metadata, 128 * 1024)
    _match_fields(data, {"schemaVersion": "1.4", "manifestType": "validated_raw_output",
                        "runId": completed.run_id, "attemptId": completed.attempt_id,
                        "providerTaskId": completed.provider_task_id, "raw": asdict(completed.raw),
                        "rawContentValidation": "PASS"})
    width, height = data["width"], data["height"]
    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= 4096 or not 1 <= height <= 4096
            or width * height > MAX_INPUT_PIXELS or type(data["frameSeconds"]) is not list):
        raise ReviewEvidenceError("Completed output metadata has invalid frame geometry or timestamps.")
    return data


def _verify_success_response(root: Path, completed: CompletedOutput) -> StoredArtifact:
    if (type(completed.provider_evidence_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", completed.provider_evidence_sha256) is None):
        raise ReviewEvidenceError("Provider success evidence needs a SHA-256 hash.")
    path = f"provider_responses/{completed.provider_evidence_sha256}.json"
    record = StoredArtifact(path, completed.provider_evidence_sha256, (root / path).lstat().st_size)
    data = _read_record(root, record, MAX_TASK_RESPONSE_BYTES)
    _match_fields(data, {"id": completed.provider_task_id, "status": "SUCCEEDED"})
    return record


def _normalized_times(contents: bytes) -> tuple[float, ...]:
    with TemporaryDirectory(prefix="gen-reality-review-timing-") as folder:
        path = Path(folder) / "video.mp4"
        path.write_bytes(contents)
        return verify_normalized(path)


def _verify_keyframe(contents: bytes) -> None:
    with TemporaryDirectory(prefix="gen-reality-review-keyframe-") as folder:
        path = Path(folder) / "keyframe.png"
        path.write_bytes(contents)
        _read_png(path, 1280, 720)


def _sample_video(root: Path, prefix: str, view: str, video: StoredArtifact,
                  indexes: tuple[int, ...], *, width: int, height: int,
                  frame_count: int) -> dict[int, StoredArtifact]:
    contents = read_verified(root, video)
    samples = {}
    for index in sorted(set(indexes)):
        png = extract_frame_png(contents, index, width=width, height=height, frame_count=frame_count)
        path = f"{prefix}{view}/frame-{index:06d}.png"
        samples[index] = _publish_or_verify(root, path, png)
    return samples


def build_review_evidence(coordinator: SubmissionCoordinator, output_root: Path,
                          preparation_manifest: StoredArtifact, request_check: StoredArtifact, *,
                          run_id: str, attempt_id: str) -> StoredArtifact:
    """Verify submitted identity, publish selected PNGs, then publish the manifest.

    Use one open coordinator and trusted artifact references. Expired request
    checks remain useful because this function performs no provider work. Keep
    every original artifact; matching existing review bytes are verified and
    reused. A failed build may leave samples that the same inputs can resume.
    The source sample basis is the normalized clip. Mode A reuses its original
    static keyframe; Mode B samples its submitted guide. Raw samples retain
    decoded geometry without rotation metadata. No acceptance verdict is made.
    """
    try:
        with coordinator._mutex:
            coordinator._require_open()
            coordinator.ledger._run_id(run_id)
            coordinator.ledger._attempt_id(attempt_id)
            root = Path(output_root) / run_id
            prefix = f"attempts/{attempt_id}/"
            prepared = read_prepared_attempt(root, preparation_manifest, run_id=run_id, attempt_id=attempt_id)
            checked = read_checked_request(root, request_check, run_id=run_id, attempt_id=attempt_id)
            if (checked.frozen.preparation_sha256 != preparation_manifest.sha256
                    or checked.source_kind != prepared.source_kind or checked.frozen.prompt != prepared.prompt):
                raise ReviewEvidenceError("Saved request does not match the prepared snapshot.")
            _match_fields(_read_record(root, request_check), {"preparation": asdict(preparation_manifest)})
            with coordinator.ledger._connection() as connection:
                reservation = connection.execute("SELECT * FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone()
            if (reservation is None or reservation["run_id"] != run_id
                    or reservation["payload_sha256"] != checked.frozen.payload_sha256
                    or reservation["cost_cents"] != checked.frozen.estimated_cents):
                raise ReviewEvidenceError("Saved request does not match the existing reservation.")
            request = _verify_expected_bytes(root, prefix + "provider/request.json", checked.frozen.request_body)
            frozen_request = _verify_expected_bytes(root, prefix + "provider/frozen_request.json", checked.frozen.record_bytes)
            completed = OutputCompleter(coordinator, output_root).get(attempt_id)
            if completed is None or completed.run_id != run_id or completed.attempt_id != attempt_id:
                raise ReviewEvidenceError("Review requires completed output for the same attempt.")
            metadata = _read_output_metadata(root, completed)
            success_response = _verify_success_response(root, completed)

            source_times = _normalized_times(read_verified(root, prepared.normalized_source))
            raw_times = tuple(metadata["frameSeconds"])
            matches = plan_review_frames(source_times, raw_times, raw_duration_seconds=metadata["durationSeconds"])
            mode = checked.frozen.mode
            guide = prepared.keyframe_a if mode == "A" else prepared.guide_b
            if mode == "A":
                _verify_keyframe(read_verified(root, guide))
                guide_times = None
            else:
                guide_times = _normalized_times(read_verified(root, guide))

            # All submitted identities and media references are checked before publication.
            review_prefix = prefix + "review/"
            source_indexes = tuple(match.source_index for match in matches)
            source_images = _sample_video(root, review_prefix, "source", prepared.normalized_source, source_indexes,
                                          width=1280, height=720, frame_count=len(source_times))
            guide_images = {} if mode == "A" else _sample_video(
                root, review_prefix, "guide", guide, source_indexes,
                width=1280, height=720, frame_count=len(guide_times))
            raw_images = _sample_video(root, review_prefix, "raw", completed.raw,
                tuple(match.raw_index for match in matches), width=metadata["width"], height=metadata["height"],
                frame_count=len(raw_times))
            samples = []
            for match in matches:
                guide_sample = {"kind": "static_keyframe", "referenceSeconds": 0.0, "image": asdict(guide)}
                if mode == "B":
                    guide_sample = {"kind": "video_frame", "frameIndex": match.source_index,
                                    "ptsSeconds": guide_times[match.source_index],
                                    "image": asdict(guide_images[match.source_index])}
                samples.append({
                    "requestedSeconds": match.requested_seconds,
                    "source": {"frameIndex": match.source_index, "ptsSeconds": match.source_seconds,
                               "image": asdict(source_images[match.source_index])},
                    "guide": guide_sample,
                    "raw": {"frameIndex": match.raw_index, "ptsSeconds": match.raw_seconds,
                            "image": asdict(raw_images[match.raw_index]), "offsetSeconds": match.offset_seconds,
                            "spanPosition": match.raw_span_position,
                            "withinDeclaredDuration": match.within_declared_duration},
                })
            provider_video, _ = prepared.media_for_mode(mode)
            manifest = {
                "schemaVersion": "1.4", "manifestType": "indexed_review_sources",
                "runId": run_id, "attemptId": attempt_id, "mode": mode, "sourceKind": prepared.source_kind,
                "preparation": asdict(preparation_manifest), "requestCheck": asdict(request_check),
                "payloadSha256": checked.frozen.payload_sha256, "request": asdict(request),
                "frozenRequest": asdict(frozen_request), "providerTaskId": completed.provider_task_id,
                "providerSuccess": asdict(success_response), "rawOutput": asdict(completed.raw),
                "rawMetadata": asdict(completed.metadata), "sourceOriginal": asdict(prepared.source_original),
                "sourceNormalized": asdict(prepared.normalized_source), "providerVideoInput": asdict(provider_video),
                "guide": {"kind": "static_keyframe" if mode == "A" else "video", "artifact": asdict(guide)},
                "samples": samples, "samplePixelFormat": "8-bit RGB", "sampleAutomaticRotationApplied": False,
                "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
            }
            body = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            return _publish_or_verify(root, review_prefix + "manifest.json", body)
    except Exception:
        raise ReviewEvidenceError("Review evidence could not be verified or published for this attempt.") from None
