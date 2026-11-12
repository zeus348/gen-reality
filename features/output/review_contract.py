"""Pure record checks shared by local review readers and archive verification."""

from dataclasses import asdict
import json
import math

from core.json_data import unique_fields
from core.storage import StoredArtifact
from features.inference.preflight import _match_fields
from features.media.ingest import MAX_INPUT_PIXELS
from features.output.align import _nearest_index, _raw_elapsed_times
from features.output.validate import _check_origin


MAX_REVIEW_JSON_BYTES = 128 * 1024
REVIEW_CONTROL_LIMITS = {"review/manifest.json": MAX_REVIEW_JSON_BYTES,
                         "output/metadata.json": MAX_REVIEW_JSON_BYTES}


class ReviewContractError(ValueError):
    """Declared review records do not satisfy their consistency contract."""


def _parse_review_json(contents: bytes) -> dict:
    if type(contents) is not bytes or not 0 < len(contents) <= MAX_REVIEW_JSON_BYTES:
        raise ReviewContractError("Review JSON exceeds its size limit.")
    data = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields)
    if type(data) is not dict:
        raise ReviewContractError("Review JSON must be an object.")
    json.dumps(data, allow_nan=False)
    return data


def _check_review_header(data: dict, *, run_id: str, attempt_id: str) -> None:
    _match_fields(data, {"schemaVersion": "1.4", "manifestType": "indexed_review_sources",
                        "runId": run_id, "attemptId": attempt_id, "samplePixelFormat": "8-bit RGB",
                        "sampleAutomaticRotationApplied": False,
                        "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN"})
    if data["mode"] not in ("A", "B") or data["sourceKind"] not in ("clean_camera", "synthetic_fixture"):
        raise ReviewContractError("Review mode or source kind is invalid.")


def _check_output_paths(raw: StoredArtifact, metadata: StoredArtifact, *, attempt_id: str) -> None:
    prefix = f"attempts/{attempt_id}/output/"
    if raw.path != prefix + "raw.mp4" or metadata.path != prefix + "metadata.json":
        raise ReviewContractError("Completed output paths must belong to the same attempt.")


def _check_output_metadata(data: dict, raw: StoredArtifact, metadata: StoredArtifact, *,
                            run_id: str, attempt_id: str, provider_task_id: str) -> None:
    _check_output_paths(raw, metadata, attempt_id=attempt_id)
    _match_fields(data, {"schemaVersion": "1.4", "manifestType": "validated_raw_output",
                        "runId": run_id, "attemptId": attempt_id,
                        "providerTaskId": provider_task_id, "raw": asdict(raw),
                        "rawContentValidation": "PASS"})
    width, height = data["width"], data["height"]
    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= 4096 or not 1 <= height <= 4096
            or width * height > MAX_INPUT_PIXELS or type(data["frameSeconds"]) is not list):
        raise ReviewContractError("Completed output metadata has invalid frame geometry or timestamps.")


def _raw_times(metadata: dict) -> tuple[float, ...]:
    times = tuple(metadata["frameSeconds"])
    try:
        _raw_elapsed_times(times, metadata["durationSeconds"], _raw_origin(metadata, "timelineOriginSeconds"))
    except ValueError as error:
        raise ReviewContractError(str(error)) from None
    return times


def _raw_origin(data: dict, field: str) -> float:
    """Saved records without a declared origin retain their original zero basis."""
    try:
        return _check_origin(data.get(field, 0.0))
    except ValueError as error:
        raise ReviewContractError(str(error)) from None


def _check_raw_samples(review: dict, metadata: dict) -> None:
    """Join declared origins, native sample indexes, and elapsed timing labels."""
    times = _raw_times(metadata)
    origin = _raw_origin(metadata, "timelineOriginSeconds")
    if _raw_origin(review, "rawTimelineOriginSeconds") != origin:
        raise ReviewContractError("Review and raw metadata use different timeline origins.")
    elapsed = tuple(value - origin for value in times)
    duration = metadata["durationSeconds"]
    for sample in review["samples"]:
        source_time, raw = sample["source"]["ptsSeconds"], sample["raw"]
        index = raw["frameIndex"]
        if index >= len(times) or index != _nearest_index(elapsed, source_time) or raw["ptsSeconds"] != times[index]:
            raise ReviewContractError("Raw sample does not match the saved native timeline.")
        if source_time < elapsed[0]:
            position = "before_first"
        elif source_time > elapsed[-1]:
            position = "after_last"
        else:
            position = "within_span"
        _match_fields(raw, {"spanPosition": position, "withinDeclaredDuration": 0 <= source_time < duration})
        # The shared view checks verify offset = (raw PTS - origin) - source PTS.


def _number(value, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not minimum <= value <= maximum or not math.isfinite(value):
        raise ReviewContractError("Review timing is invalid.")
    return value


def _index(value, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ReviewContractError("Review frame index is invalid.")
    return value


def _view_records(data: dict, prefix: str) -> dict[str, StoredArtifact]:
    """Check declared view fields and return their unique artifact records."""
    verified = {}

    def check(record_data, paths, limit=256 * 1024 * 1024):
        record = StoredArtifact(**record_data)
        if record.path not in paths or type(record.size) is not int or not 0 < record.size <= limit:
            raise ReviewContractError("Review media does not match its expected path or size.")
        if record.path in verified:
            if verified[record.path] != record:
                raise ReviewContractError("Review media has conflicting file records.")
        else:
            verified[record.path] = record

    mode = data["mode"]
    origin = _raw_origin(data, "rawTimelineOriginSeconds")
    check(data["sourceOriginal"], {prefix + "frozen/source_original.mp4", prefix + "frozen/source_original.mov"})
    check(data["sourceNormalized"], {prefix + "frozen/source_normalized.mp4"})
    check(data["rawOutput"], {prefix + "output/raw.mp4"})
    guide = data["guide"]
    guide_kind = "static_keyframe" if mode == "A" else "video"
    if guide["kind"] != guide_kind:
        raise ReviewContractError("Review guide does not match its mode.")
    guide_name = "keyframe_A.png" if mode == "A" else "guide_B.mp4"
    check(guide["artifact"], {prefix + "frozen/" + guide_name},
          32 * 1024 * 1024 if mode == "A" else 256 * 1024 * 1024)
    _match_fields(data, {"providerVideoInput": data["sourceNormalized"] if mode == "A" else guide["artifact"]})
    samples = data["samples"]
    if type(samples) is not list or len(samples) != 5:
        raise ReviewContractError("Review needs five ordered samples.")
    for second, sample in enumerate(samples):
        if _number(sample["requestedSeconds"], 0, 4) != second:
            raise ReviewContractError("Review samples must use source times zero through four.")
        source, raw, guide_sample = sample["source"], sample["raw"], sample["guide"]
        source_index = _index(source["frameIndex"], 119)
        raw_index = _index(raw["frameIndex"], 2399)
        source_time = _number(source["ptsSeconds"], -0.00001, 5)
        raw_time = _number(raw["ptsSeconds"], origin, origin + 20)
        raw_elapsed = raw_time - origin
        _number(raw_elapsed, 0, 20)
        offset = _number(raw["offsetSeconds"], -21, 21)
        if not math.isclose(offset, raw_elapsed - source_time, rel_tol=0, abs_tol=1e-9):
            raise ReviewContractError("Raw offset does not match the saved timestamps.")
        if (raw["spanPosition"] not in ("before_first", "within_span", "after_last")
                or type(raw["withinDeclaredDuration"]) is not bool):
            raise ReviewContractError("Raw timing labels are invalid.")
        check(source["image"], {prefix + f"review/source/frame-{source_index:06d}.png"}, 32 * 1024 * 1024)
        check(raw["image"], {prefix + f"review/raw/frame-{raw_index:06d}.png"}, 32 * 1024 * 1024)
        if mode == "A":
            if (guide_sample["kind"] != "static_keyframe"
                    or _number(guide_sample["referenceSeconds"], 0, 0) != 0):
                raise ReviewContractError("Mode A samples must use the submitted static keyframe.")
            _match_fields(guide_sample, {"image": guide["artifact"]})
        else:
            if (guide_sample["kind"] != "video_frame"
                    or _index(guide_sample["frameIndex"], 119) != source_index):
                raise ReviewContractError("Mode B samples must use the matched guide frames.")
            _number(guide_sample["ptsSeconds"], -0.00001, 5)
            check(guide_sample["image"], {prefix + f"review/guide/frame-{source_index:06d}.png"}, 32 * 1024 * 1024)
    return verified


