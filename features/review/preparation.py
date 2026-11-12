"""Check preparation records within one fully verified archive snapshot."""

from dataclasses import dataclass
from pathlib import Path

from core.storage import StoredArtifact
from features.inference.preparation_contract import (
    _check_preparation_manifest, _check_preparation_controls, _parse_preparation_json,
)
from features.review.integrity import ExportArchiveIntegrity, _read_verified_archive


class ExportPreparationError(ValueError):
    """Archived preparation records are inconsistent with the verified inventory."""


@dataclass(frozen=True)
class ExportPreparationIntegrity:
    """Preparation consistency, without media, provider, human, or release claims."""

    integrity: ExportArchiveIntegrity
    preparation: StoredArtifact
    source_kind: str


def _check_preparation_snapshot(snapshot, *, run_id: str, attempt_id: str) -> ExportPreparationIntegrity:
    """Check retained preparation controls after the whole archive is verified."""
    manifest = StoredArtifact(**snapshot.inventory["selection"]["preparationManifest"])
    data = _parse_preparation_json(snapshot.controls[manifest.path])
    original_name, source_kind, records = _check_preparation_manifest(
        data, manifest, run_id=run_id, attempt_id=attempt_id)
    if source_kind != snapshot.inventory["sourceKindDeclared"]:
        raise ExportPreparationError("Preparation source kind differs from its inventory.")
    for record in records.values():
        if record != snapshot.records.get(record.path):
            raise ExportPreparationError("Frozen reference differs from its inventory record.")
    controls = {}
    for name, record in records.items():
        if name.endswith(".json"):
            controls[name] = _parse_preparation_json(snapshot.controls[record.path])
        elif name == "frozen/prompt.txt":
            controls[name] = snapshot.controls[record.path].decode("utf-8")
    _check_preparation_controls(controls, records, original_name=original_name, source_kind=source_kind)
    return ExportPreparationIntegrity(snapshot.integrity, manifest, source_kind)


def verify_export_preparation(output_root: Path, archive_ref: StoredArtifact,
                              inventory_ref: StoredArtifact, *, run_id: str,
                              attempt_id: str) -> ExportPreparationIntegrity:
    """Verify the archive and its preparation records without original files.

    Retain only four JSON records of at most 64 KiB each and the prompt of at
    most 4000 bytes. Finish every archive CRC and hash check before parsing the
    retained controls. Shared preparation rules check the offline identity,
    nine frozen references, locked configuration, placement, declarations and
    prompt. Each frozen reference must equal its inventory record.

    This does not decode media or check the review, raw metadata, selected
    scorecard, private provider evidence, ledger, provenance, or global outcomes.
    """
    try:
        snapshot = _read_verified_archive(output_root, archive_ref, inventory_ref,
            run_id=run_id, attempt_id=attempt_id, include_preparation=True)
        return _check_preparation_snapshot(snapshot, run_id=run_id, attempt_id=attempt_id)
    except Exception:
        raise ExportPreparationError("Archived preparation could not be verified for this attempt.") from None
