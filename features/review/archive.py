"""Package one trusted historical inventory and its verified payload bytes."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re
import stat
from tempfile import TemporaryFile
from typing import Callable
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from core.budget import BudgetLedger
from core.storage import MAX_ARTIFACT_BYTES, StoredArtifact, _path_parts, read_verified
from features.inference.preflight import _match_fields, _read_record
from features.output.complete import _publish_or_verify
from features.output.validate import MAX_RAW_FRAMES
from features.review.export import FROZEN_ROLES, MAX_INVENTORY_BYTES, _VALIDATOR


MAX_ARCHIVE_BYTES = MAX_ARTIFACT_BYTES


class ExportArchiveError(ValueError):
    """The selected inventory cannot be packaged safely."""


class _LimitedFile:
    """Keep ZIP writes and seeks inside the calculated archive size."""

    def __init__(self, file, limit: int):
        self.file = file
        self.limit = limit

    def write(self, contents):
        if self.file.tell() + len(contents) > self.limit:
            raise ExportArchiveError("ZIP data exceeds its calculated size.")
        return self.file.write(contents)

    def seek(self, offset, whence=os.SEEK_SET):
        position = self.file.seek(offset, whence)
        if not 0 <= position <= self.limit:
            raise ExportArchiveError("ZIP position exceeds its calculated size.")
        return position

    def tell(self):
        return self.file.tell()

    def flush(self):
        self.file.flush()


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
            raise ExportArchiveError("Sample path does not match its mode or frame bounds.")
        return [view + "_sample"]
    for directory, role in (("scorecards", "scorecard"), ("scorecard-bindings", "scorecard_binding")):
        if relative == f"review/{directory}/{record.sha256}.json":
            return [role]
    raise ExportArchiveError("Payload path has no allowed role.")


def _check_inventory(data: dict, *, run_id: str, attempt_id: str) -> dict[str, StoredArtifact]:
    _VALIDATOR.validate(data)
    _match_fields(data, {"runId": run_id, "attemptId": attempt_id})
    prefix = f"attempts/{attempt_id}/"
    records, counts = {}, Counter()
    paths = [entry["path"] for entry in data["files"]]
    if paths != sorted(set(paths)):
        raise ExportArchiveError("Payload paths must be unique and sorted.")
    for entry in data["files"]:
        record = StoredArtifact(**{name: entry[name] for name in ("path", "sha256", "size")})
        _path_parts(record.path)
        if not record.path.startswith(prefix) or type(record.size) is not int:
            raise ExportArchiveError("Payload records must belong to this attempt and use integer sizes.")
        if entry["roles"] != _roles_for(record, prefix, data["mode"]):
            raise ExportArchiveError("Payload roles do not match the path and mode.")
        records[record.path] = record
        counts.update(entry["roles"])
    required = set(FROZEN_ROLES.values()) | {"preparation_manifest", "review_manifest", "raw_output", "raw_metadata"}
    if (any(counts[role] != 1 for role in required) or counts["source_sample"] != 5
            or counts["guide_sample"] != (1 if data["mode"] == "A" else 5)
            or not 1 <= counts["raw_sample"] <= 5):
        raise ExportArchiveError("Inventory does not contain the required completed-attempt payload.")
    selection = data["selection"]
    for name, selected in selection.items():
        if selected is None:
            continue
        record = records.get(selected["path"])
        if record is None:
            raise ExportArchiveError("Selected record is missing from the payload.")
        _match_fields(selected, asdict(record))
    _match_fields(selection["preparationManifest"], {"path": prefix + "manifest.json"})
    _match_fields(selection["reviewManifest"], {"path": prefix + "review/manifest.json"})
    selected_count = 0 if selection["scorecardBinding"] is None else 1
    if counts["scorecard"] != selected_count or counts["scorecard_binding"] != selected_count:
        raise ExportArchiveError("Payload contains an unselected scorecard or binding.")
    if type(data["totalFileBytes"]) is not int or data["totalFileBytes"] != sum(record.size for record in records.values()):
        raise ExportArchiveError("Payload byte total does not match its records.")
    return records


def _member_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.internal_attr = 0
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.extra = b""
    info.comment = b""
    return info


def _write_zip(file, names: tuple[str, ...], contents_for: Callable[[str], bytes]) -> None:
    with ZipFile(file, "w", compression=ZIP_STORED, allowZip64=False) as archive:
        archive.comment = b""
        for name in names:
            archive.writestr(_member_info(name), contents_for(name))


def build_export_archive(output_root: Path, inventory_ref: StoredArtifact, *,
                          run_id: str, attempt_id: str) -> StoredArtifact:
    """Publish deterministic ZIP bytes from a trusted private root and inventory.

    The archive starts with the exact inventory.json bytes, then sorted payload
    members. It needs only those verified files, with no ledger, private provider
    records, current clock, or directory scan. This packages evidence; it does
    not re-audit the full evidence graph, human review, or provider authenticity.

    Measure ZIP_STORED overhead before reading payloads. A private temporary file
    holds construction; one verified payload is held at a time. Final publication
    uses the existing byte-based immutable storage API and the 512 MiB cap.
    """
    try:
        BudgetLedger._run_id(run_id)
        BudgetLedger._attempt_id(attempt_id)
        root = Path(output_root) / run_id
        prefix = f"attempts/{attempt_id}/export/"
        if inventory_ref.path != prefix + f"inventories/{inventory_ref.sha256}.json":
            raise ExportArchiveError("Inventory path does not match its attempt and hash.")
        data = _read_record(root, inventory_ref, MAX_INVENTORY_BYTES)
        records = _check_inventory(data, run_id=run_id, attempt_id=attempt_id)
        inventory_bytes = read_verified(root, inventory_ref)
        names = ("inventory.json", *records)
        with BytesIO() as empty:
            _write_zip(empty, names, lambda name: b"")
            expected_size = empty.tell() + len(inventory_bytes) + data["totalFileBytes"]
        if expected_size > MAX_ARCHIVE_BYTES:
            raise ExportArchiveError("ZIP including its metadata exceeds 512 MiB.")

        def contents_for(name):
            return inventory_bytes if name == "inventory.json" else read_verified(root, records[name])

        with TemporaryFile(mode="w+b") as temporary:
            _write_zip(_LimitedFile(temporary, expected_size), names, contents_for)
            actual_size = temporary.seek(0, os.SEEK_END)
            if actual_size != expected_size or actual_size > MAX_ARCHIVE_BYTES:
                raise ExportArchiveError("ZIP size does not match its bounded calculation.")
            temporary.seek(0)
            contents = temporary.read(expected_size + 1)
            if len(contents) != expected_size:
                raise ExportArchiveError("ZIP bytes changed before publication.")
        return _publish_or_verify(root, prefix + f"archives/{sha256(contents).hexdigest()}.zip", contents)
    except Exception:
        raise ExportArchiveError("Export archive could not be verified or published for this attempt.") from None
