"""Read and verify a frozen offline preparation before provider work."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator

from core.config import SCHEMA_PATH
from core.json_data import unique_fields
from core.storage import StoredArtifact, read_verified
from features.placement.math import Placement


MAX_PREPARATION_JSON_BYTES = 64 * 1024
COMMON_FILES = frozenset({"frozen/placement.json", "frozen/source_normalized.mp4",
                          "frozen/asset_original.png", "frozen/config.json", "frozen/prompt.txt",
                          "frozen/keyframe_A.png", "frozen/guide_B.mp4", "frozen/inputs.json"})
ORIGINAL_FILES = frozenset({"frozen/source_original.mp4", "frozen/source_original.mov"})


class PreparationError(ValueError):
    """A complete frozen preparation could not be verified."""


@dataclass(frozen=True)
class PreparedAttempt:
    run_id: str
    attempt_id: str
    source_kind: str
    manifest: StoredArtifact
    source_original: StoredArtifact
    normalized_source: StoredArtifact
    asset_original: StoredArtifact
    keyframe_a: StoredArtifact
    guide_b: StoredArtifact
    placement: Placement
    prompt: str

    def media_for_mode(self, mode: str) -> tuple[StoredArtifact, StoredArtifact | None]:
        if mode == "A":
            return self.normalized_source, self.keyframe_a
        if mode == "B":
            return self.guide_b, None
        raise PreparationError("Mode must be A or B.")


def _read_json(root: Path, record: StoredArtifact):
    if type(record.size) is not int or not 0 < record.size <= MAX_PREPARATION_JSON_BYTES:
        raise ValueError("Preparation JSON exceeds its size limit.")
    data = json.loads(read_verified(root, record).decode("utf-8"), object_pairs_hook=unique_fields)
    json.dumps(data, allow_nan=False)
    return data


def read_prepared_attempt(run_root: Path, manifest: StoredArtifact, *,
                           run_id: str, attempt_id: str) -> PreparedAttempt:
    """Verify records from the trusted preparation writer and return frozen inputs.

    The caller supplies the expected run, attempt, and manifest hash. Every file
    must keep its recorded bytes and belong to this attempt. Read binary files
    one at a time; retain only the small control records. Media decoding and
    guidance compilation belong to preparation. This checks identity and record
    consistency; it does not establish live-provider provenance or visual quality.
    """
    try:
        for identifier in (run_id, attempt_id):
            if type(identifier) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier) is None:
                raise ValueError("Preparation needs valid run and attempt IDs.")
        prefix = f"attempts/{attempt_id}/"
        if manifest.path != prefix + "manifest.json":
            raise ValueError("Preparation manifest belongs to a different attempt.")
        data = _read_json(run_root, manifest)
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
        records, controls = {}, {}
        for name, value in files.items():
            record = StoredArtifact(**value)
            if record.path != prefix + name or type(record.size) is not int or record.size <= 0:
                raise ValueError("Frozen file belongs to another attempt or has an invalid size.")
            records[name] = record
            if name.endswith(".json"):
                controls[name] = _read_json(run_root, record)
            elif name == "frozen/prompt.txt":
                if record.size > 4000:
                    raise ValueError("Frozen prompt is too large.")
                controls[name] = read_verified(run_root, record).decode("utf-8")
            else:
                read_verified(run_root, record)
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(controls["frozen/config.json"])
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
        return PreparedAttempt(run_id, attempt_id, source_kind, manifest, records[original_name],
                               records["frozen/source_normalized.mp4"], records["frozen/asset_original.png"],
                               records["frozen/keyframe_A.png"], records["frozen/guide_B.mp4"], position, prompt)
    except Exception:
        raise PreparationError("Prepared snapshot could not be verified.") from None
