"""Read and verify a frozen offline preparation before provider work."""

from dataclasses import dataclass
from pathlib import Path

from core.storage import StoredArtifact, read_verified
from features.placement.math import Placement
from features.inference.preparation_contract import (
    COMMON_FILES, ORIGINAL_FILES, MAX_PREPARATION_JSON_BYTES, PREPARATION_CONTROL_LIMITS,
    SCHEMA_PATH, _check_preparation_identity, _check_preparation_manifest,
    _check_preparation_controls, _parse_preparation_json,
)


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
    return _parse_preparation_json(read_verified(root, record))


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
        _check_preparation_identity(manifest, run_id=run_id, attempt_id=attempt_id)
        data = _read_json(run_root, manifest)
        original_name, source_kind, records = _check_preparation_manifest(
            data, manifest, run_id=run_id, attempt_id=attempt_id)
        controls = {}
        for name, record in records.items():
            if name.endswith(".json"):
                controls[name] = _read_json(run_root, record)
            elif name == "frozen/prompt.txt":
                if record.size > PREPARATION_CONTROL_LIMITS[name]:
                    raise ValueError("Frozen prompt is too large.")
                controls[name] = read_verified(run_root, record).decode("utf-8")
            else:
                read_verified(run_root, record)
        position, prompt = _check_preparation_controls(
            controls, records, original_name=original_name, source_kind=source_kind)
        return PreparedAttempt(run_id, attempt_id, source_kind, manifest, records[original_name],
                               records["frozen/source_normalized.mp4"], records["frozen/asset_original.png"],
                               records["frozen/keyframe_A.png"], records["frozen/guide_B.mp4"], position, prompt)
    except Exception:
        raise PreparationError("Prepared snapshot could not be verified.") from None
