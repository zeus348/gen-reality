"""Shared, side-effect-free contract checks for completed-attempt inventories."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator

from core.storage import StoredArtifact, _path_parts
from features.inference.preflight import _match_fields
from features.output.validate import MAX_RAW_FRAMES


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


class InventoryContractError(ValueError):
    """An inventory does not satisfy its payload contract."""


def _roles_for(record: StoredArtifact, prefix: str, mode: str) -> list[str]:
    relative = record.path.removeprefix(prefix)
    if relative in FROZEN_ROLES:
        roles = {FROZEN_ROLES[relative]}
        if relative == "frozen/source_normalized.mp4" and mode == "A":
            roles.add("provider_video_input")
        if relative == "frozen/keyframe_A.png" and mode == "A":
            roles.update(("submitted_guide", "guide_sample"))
        if relative == "frozen/guide_B.mp4" and mode == "B":
            roles.update(("provider_video_input", "submitted_guide"))
        return sorted(roles)
    fixed = {"manifest.json": "preparation_manifest", "review/manifest.json": "review_manifest",
             "output/raw.mp4": "raw_output", "output/metadata.json": "raw_metadata"}
    if relative in fixed:
        return [fixed[relative]]
    sample = re.fullmatch(r"review/(source|guide|raw)/frame-([0-9]{6})\.png", relative)
    if sample is not None:
        view, index = sample[1], int(sample[2])
        if index >= (MAX_RAW_FRAMES if view == "raw" else 120) or (view == "guide" and mode != "B"):
            raise InventoryContractError("Sample path does not match its mode or frame bounds.")
        return [view + "_sample"]
    for directory, role in (("scorecards", "scorecard"), ("scorecard-bindings", "scorecard_binding")):
        if relative == f"review/{directory}/{record.sha256}.json":
            return [role]
    raise InventoryContractError("Payload path has no allowed role.")


def _check_inventory(data: dict, *, run_id: str, attempt_id: str) -> dict[str, StoredArtifact]:
    _VALIDATOR.validate(data)
    _match_fields(data, {"runId": run_id, "attemptId": attempt_id})
    prefix = f"attempts/{attempt_id}/"
    records, counts = {}, Counter()
    paths = [entry["path"] for entry in data["files"]]
    if paths != sorted(set(paths)):
        raise InventoryContractError("Payload paths must be unique and sorted.")
    for entry in data["files"]:
        record = StoredArtifact(**{name: entry[name] for name in ("path", "sha256", "size")})
        _path_parts(record.path)
        if not record.path.startswith(prefix) or type(record.size) is not int:
            raise InventoryContractError("Payload records must belong to this attempt and use integer sizes.")
        if entry["roles"] != _roles_for(record, prefix, data["mode"]):
            raise InventoryContractError("Payload roles do not match the path and mode.")
        records[record.path] = record
        counts.update(entry["roles"])
    required = set(FROZEN_ROLES.values()) | {"preparation_manifest", "review_manifest", "raw_output", "raw_metadata"}
    if (any(counts[role] != 1 for role in required) or counts["source_sample"] != 5
            or counts["guide_sample"] != (1 if data["mode"] == "A" else 5)
            or not 1 <= counts["raw_sample"] <= 5):
        raise InventoryContractError("Inventory does not contain the required completed-attempt payload.")
    selection = data["selection"]
    for name, selected in selection.items():
        if selected is None:
            continue
        record = records.get(selected["path"])
        if record is None:
            raise InventoryContractError("Selected record is missing from the payload.")
        _match_fields(selected, asdict(record))
    _match_fields(selection["preparationManifest"], {"path": prefix + "manifest.json"})
    _match_fields(selection["reviewManifest"], {"path": prefix + "review/manifest.json"})
    selected_count = 0 if selection["scorecardBinding"] is None else 1
    if counts["scorecard"] != selected_count or counts["scorecard_binding"] != selected_count:
        raise InventoryContractError("Payload contains an unselected scorecard or binding.")
    if type(data["totalFileBytes"]) is not int or data["totalFileBytes"] != sum(record.size for record in records.values()):
        raise InventoryContractError("Payload byte total does not match its records.")
    return records


