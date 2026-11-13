"""Save and read the fixed validated-input cache; never normalize on reuse."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import re

from app.preparation_state import MAX_RECORD_BYTES, ServiceError, decode, encode, reference
from core.storage import MAX_ARTIFACT_BYTES, StoredArtifact, publish_bytes, read_verified
from features.asset.ingest import MAX_DIMENSION, MAX_FILE_BYTES, MAX_PIXELS, ValidatedAsset
from features.media.ingest import MAX_INPUT_BYTES, NormalizedVideo


@dataclass(frozen=True)
class SourceSelection:
    input_id: str
    record: StoredArtifact
    original: StoredArtifact
    normalized: StoredArtifact
    source_suffix: str
    source_kind: str


@dataclass(frozen=True)
class AssetSelection:
    input_id: str
    record: StoredArtifact
    original: StoredArtifact
    width: int
    height: int


def _save_record(root, input_id, data):
    body = (encode(data) + "\n").encode("utf-8")
    return publish_bytes(root, f"inputs/{input_id}/{sha256(body).hexdigest()}.json", body)


def save_source(root, run_id, input_id, video, suffix, source_kind):
    prefix = f"inputs/{input_id}/"
    original = publish_bytes(root, prefix + "source_original" + suffix, video.original_bytes)
    normalized = publish_bytes(root, prefix + "source_normalized.mp4", video.video_bytes)
    return _save_record(root, input_id, {
        "schemaVersion": "1.4", "manifestType": "validated_source_input", "runId": run_id,
        "inputId": input_id, "sourceSuffix": suffix, "sourceKind": source_kind,
        "cleanSourceDeclared": True, "original": asdict(original), "normalized": asdict(normalized),
        "frameSeconds": list(video.frame_seconds),
    })


def save_asset(root, run_id, input_id, asset):
    prefix = f"inputs/{input_id}/"
    original = publish_bytes(root, prefix + "asset_original.png", asset.original_bytes)
    rgba = publish_bytes(root, prefix + "asset.rgba", asset.rgba)
    return _save_record(root, input_id, {
        "schemaVersion": "1.4", "manifestType": "validated_asset_input", "runId": run_id,
        "inputId": input_id, "nonHumanDeclared": True, "original": asdict(original),
        "rgba": asdict(rgba), "width": asset.width, "height": asset.height,
    })


def _read_record(root, run_id, input_id, record, kind):
    prefix = "src" if kind == "source" else "asset"
    if (type(input_id) is not str or re.fullmatch(prefix + r"-[0-9a-f]{32}", input_id) is None
            or record.path != f"inputs/{input_id}/{record.sha256}.json"
            or type(record.size) is not int or not 0 < record.size <= MAX_RECORD_BYTES):
        raise ServiceError("STATE_UNAVAILABLE")
    data = decode(read_verified(root, record).decode("utf-8"))
    if (type(data) is not dict or data.get("schemaVersion") != "1.4"
            or data.get("manifestType") != f"validated_{kind}_input"
            or data.get("runId") != run_id or data.get("inputId") != input_id):
        raise ServiceError("STATE_UNAVAILABLE")
    return data


def _file(data, name, path, maximum):
    record = reference(data[name])
    if record.path != path or record.size > maximum:
        raise ServiceError("STATE_UNAVAILABLE")
    return record


def load_source(root, run_id, input_id, record, *, with_bytes=False):
    data = _read_record(root, run_id, input_id, record, "source")
    fields = {"schemaVersion", "manifestType", "runId", "inputId", "sourceSuffix", "sourceKind",
              "cleanSourceDeclared", "original", "normalized", "frameSeconds"}
    if (set(data) != fields or data["sourceSuffix"] not in (".mp4", ".mov")
            or data["sourceKind"] not in ("clean_camera", "synthetic_fixture")
            or data["cleanSourceDeclared"] is not True):
        raise ServiceError("STATE_UNAVAILABLE")
    times = data["frameSeconds"]
    if (type(times) is not list or len(times) != 120
            or any(type(value) not in (int, float) or not -0.00001 <= value <= 5
                   or not math.isfinite(value) or abs(value - index / 24) > 0.00001
                   for index, value in enumerate(times))):
        raise ServiceError("STATE_UNAVAILABLE")
    prefix = f"inputs/{input_id}/"
    original = _file(data, "original", prefix + "source_original" + data["sourceSuffix"], MAX_INPUT_BYTES)
    normalized = _file(data, "normalized", prefix + "source_normalized.mp4", MAX_ARTIFACT_BYTES)
    selected = SourceSelection(input_id, record, original, normalized, data["sourceSuffix"], data["sourceKind"])
    video = None
    if with_bytes:
        video = NormalizedVideo(read_verified(root, original), original.sha256,
                                read_verified(root, normalized), normalized.sha256, tuple(times))
    return selected, video


def load_asset(root, run_id, input_id, record, *, with_bytes=False):
    data = _read_record(root, run_id, input_id, record, "asset")
    fields = {"schemaVersion", "manifestType", "runId", "inputId", "nonHumanDeclared",
              "original", "rgba", "width", "height"}
    if set(data) != fields or data["nonHumanDeclared"] is not True:
        raise ServiceError("STATE_UNAVAILABLE")
    width, height = data["width"], data["height"]
    if (type(width) is not int or type(height) is not int or not 1 <= width <= MAX_DIMENSION
            or not 1 <= height <= MAX_DIMENSION or width * height > MAX_PIXELS):
        raise ServiceError("STATE_UNAVAILABLE")
    prefix = f"inputs/{input_id}/"
    original = _file(data, "original", prefix + "asset_original.png", MAX_FILE_BYTES)
    rgba = _file(data, "rgba", prefix + "asset.rgba", MAX_PIXELS * 4)
    if rgba.size != width * height * 4:
        raise ServiceError("STATE_UNAVAILABLE")
    selected = AssetSelection(input_id, record, original, width, height)
    asset = None
    if with_bytes:
        asset = ValidatedAsset(read_verified(root, original), original.sha256,
                               width, height, read_verified(root, rgba))
    return selected, asset
