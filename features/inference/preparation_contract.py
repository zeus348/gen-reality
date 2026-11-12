"""Pure checks shared by filesystem and archived preparation readers."""

from dataclasses import asdict
import json
import re

from jsonschema import Draft202012Validator

from core.config import SCHEMA_PATH
from core.json_data import unique_fields
from core.storage import StoredArtifact
from features.placement.math import Placement


MAX_PREPARATION_JSON_BYTES = 64 * 1024
COMMON_FILES = frozenset({"frozen/placement.json", "frozen/source_normalized.mp4",
                          "frozen/asset_original.png", "frozen/config.json", "frozen/prompt.txt",
                          "frozen/keyframe_A.png", "frozen/guide_B.mp4", "frozen/inputs.json"})
ORIGINAL_FILES = frozenset({"frozen/source_original.mp4", "frozen/source_original.mov"})
PREPARATION_CONTROL_LIMITS = {
    "manifest.json": MAX_PREPARATION_JSON_BYTES,
    "frozen/config.json": MAX_PREPARATION_JSON_BYTES,
    "frozen/placement.json": MAX_PREPARATION_JSON_BYTES,
    "frozen/inputs.json": MAX_PREPARATION_JSON_BYTES,
    "frozen/prompt.txt": 4000,
}
_CONFIG_VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _parse_preparation_json(contents: bytes):
    if type(contents) is not bytes or not 0 < len(contents) <= MAX_PREPARATION_JSON_BYTES:
        raise ValueError("Preparation JSON exceeds its size limit.")
    data = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields)
    json.dumps(data, allow_nan=False)
    return data


def _check_preparation_identity(manifest: StoredArtifact, *, run_id: str, attempt_id: str) -> str:
    for identifier in (run_id, attempt_id):
        if type(identifier) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier) is None:
            raise ValueError("Preparation needs valid run and attempt IDs.")
    prefix = f"attempts/{attempt_id}/"
    if manifest.path != prefix + "manifest.json":
        raise ValueError("Preparation manifest belongs to a different attempt.")
    return prefix


def _check_preparation_manifest(data: dict, manifest: StoredArtifact, *, run_id: str,
                                attempt_id: str) -> tuple[str, str, dict[str, StoredArtifact]]:
    prefix = _check_preparation_identity(manifest, run_id=run_id, attempt_id=attempt_id)
    expected = {"schemaVersion": "1.4", "manifestType": "offline_preparation", "state": "PREPARED",
                "runId": run_id, "attemptId": attempt_id,
                "visualEffectDemonstrated": "NOT_RUN", "automatedWorkflowComplete": "NOT_RUN",
                "releaseClassification": "LOCAL_PIPELINE_ONLY"}
    if type(data) is not dict or any(data.get(name) != value for name, value in expected.items()):
        raise ValueError("Preparation manifest fields are inconsistent.")
    provider = data["provider"]
    if (provider["vendor"] != "Runway" or provider["model"] != "aleph2"
            or provider["taskId"] is not None or provider["submitted"] is not False):
        raise ValueError("Offline preparation must have no submitted provider task.")
    source_kind = data["sourceKind"]
    if source_kind not in ("clean_camera", "synthetic_fixture"):
        raise ValueError("Preparation source kind is invalid.")
    files = data["artifacts"]
    if type(files) is not dict:
        raise ValueError("Preparation has no artifact map.")
    originals = set(files) & ORIGINAL_FILES
    if len(originals) != 1 or set(files) != COMMON_FILES | originals:
        raise ValueError("Preparation must record all nine expected files.")
    original_name = next(iter(originals))
    records = {}
    for name, value in files.items():
        record = StoredArtifact(**value)
        if record.path != prefix + name or type(record.size) is not int or record.size <= 0:
            raise ValueError("Frozen file belongs to another attempt or has an invalid size.")
        records[name] = record
    return original_name, source_kind, records


def _check_preparation_controls(controls: dict, records: dict[str, StoredArtifact], *,
                                original_name: str, source_kind: str) -> tuple[Placement, str]:
    _CONFIG_VALIDATOR.validate(controls["frozen/config.json"])
    placement = controls["frozen/placement.json"]
    if set(placement) != {"u", "v", "heightFrac", "rotationDeg"}:
        raise ValueError("Frozen placement fields are invalid.")
    position = Placement(placement["u"], placement["v"], placement["heightFrac"], placement["rotationDeg"])
    prompt = controls["frozen/prompt.txt"]
    if not prompt.strip() or len(prompt) > 1000:
        raise ValueError("Frozen prompt must contain 1 to 1000 characters.")
    inputs = controls["frozen/inputs.json"]
    if (inputs["sourceKind"] != source_kind or inputs["cleanSourceDeclared"] is not True
            or inputs["nonHumanDeclared"] is not True):
        raise ValueError("Frozen input declarations are inconsistent.")
    for field, name in [("source", original_name), ("normalizedSource", "frozen/source_normalized.mp4"),
                        ("asset", "frozen/asset_original.png")]:
        reference = inputs[field]
        if (type(reference) is not dict or type(reference.get("size")) is not int
                or reference != asdict(records[name])):
            raise ValueError("Frozen input reference does not match its artifact record.")
    return position, prompt
