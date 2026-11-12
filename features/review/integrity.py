"""Verify one exported ZIP against caller-retained artifact references."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import stat
from struct import Struct
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from core.budget import BudgetLedger
from core.json_data import unique_fields
from core.storage import MAX_ARTIFACT_BYTES, StoredArtifact, read_verified
from features.inference.preparation_contract import PREPARATION_CONTROL_LIMITS
from features.output.review_contract import REVIEW_CONTROL_LIMITS
from features.review.binding_contract import MAX_BINDING_BYTES, MAX_SCORECARD_BYTES
from features.review.inventory_contract import MAX_INVENTORY_BYTES, _check_inventory


MAX_ARCHIVE_BYTES = MAX_ARTIFACT_BYTES
MAX_MEMBERS = 31  # The inventory and at most thirty payload members.
MAX_MEMBER_NAME_BYTES = 1024
READ_CHUNK_BYTES = 64 * 1024
_END = Struct("<4s4H2IH")
_LOCAL = Struct("<4s5H3I2H")
# Read only the signature and name/extra/comment lengths in each 46-byte header.
_CENTRAL_SHAPE = Struct("<4s24x3H12x")


class ArchiveIntegrityError(ValueError):
    """The archive does not match its references and bounded ZIP contract."""


@dataclass(frozen=True)
class ExportArchiveIntegrity:
    """Verified byte counts and references, without provenance or release claims."""

    archive: StoredArtifact
    inventory: StoredArtifact
    payload_file_count: int
    total_payload_bytes: int


@dataclass(frozen=True)
class _VerifiedArchive:
    integrity: ExportArchiveIntegrity
    inventory: dict
    records: dict[str, StoredArtifact]
    controls: dict[str, bytes]


def _check_reference(record: StoredArtifact, expected_directory: str, suffix: str, limit: int) -> None:
    if (not isinstance(record, StoredArtifact) or type(record.size) is not int
            or not 0 < record.size <= limit or type(record.sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", record.sha256) is None
            or record.path != f"{expected_directory}/{record.sha256}.{suffix}"):
        raise ArchiveIntegrityError("Artifact reference has an invalid path, size, or hash.")


def _check_envelope(contents: bytes) -> tuple[int, int, int]:
    """Bound the central directory before ZipFile can allocate or parse it."""
    if len(contents) < _END.size:
        raise ArchiveIntegrityError("ZIP end record is missing.")
    signature, disk, directory_disk, disk_count, count, size, offset, comment_size = _END.unpack_from(
        contents, len(contents) - _END.size)
    if (signature != b"PK\x05\x06" or disk != 0 or directory_disk != 0
            or not 1 <= count <= MAX_MEMBERS or disk_count != count or comment_size != 0
            or not count * _CENTRAL_SHAPE.size <= size <= count * (_CENTRAL_SHAPE.size + MAX_MEMBER_NAME_BYTES)
            or offset < _LOCAL.size or offset + size + _END.size != len(contents)):
        raise ArchiveIntegrityError("ZIP end record exceeds the bounded single-disk contract.")
    # ZipFile recognizes this locator even when opened with allowZip64=False.
    # Reject it before it can replace the bounded directory fields above.
    if contents[-(_END.size + 20):-(_END.size + 16)] == b"PK\x06\x07":
        raise ArchiveIntegrityError("ZIP64 is not permitted.")
    # Do not trust the advertised count: check the bounded record lengths too.
    # ZipFile still owns parsing each entry and reading its content.
    position = offset
    directory_end = offset + size
    for _ in range(count):
        if position + _CENTRAL_SHAPE.size > directory_end:
            raise ArchiveIntegrityError("ZIP directory record is truncated.")
        signature, name_size, extra_size, entry_comment_size = _CENTRAL_SHAPE.unpack_from(contents, position)
        if (signature != b"PK\x01\x02" or not 0 < name_size <= MAX_MEMBER_NAME_BYTES
                or extra_size != 0 or entry_comment_size != 0):
            raise ArchiveIntegrityError("ZIP directory record exceeds the canonical bounds.")
        position += _CENTRAL_SHAPE.size + name_size
    if position != directory_end:
        raise ArchiveIntegrityError("ZIP directory does not contain its declared entries.")
    return count, size, offset


def _check_members(contents: bytes, members: list[ZipInfo], directory_size: int,
                   directory_offset: int) -> list[str]:
    """Check the canonical metadata and every local extent without extracting."""
    names = []
    next_offset = 0
    expected_directory_size = 0
    for member in members:
        name = member.filename.encode("ascii")
        if (not 0 < len(name) <= MAX_MEMBER_NAME_BYTES or member.orig_filename != member.filename
                or member.filename in names or member.is_dir()
                or member.header_offset != next_offset
                or member.compress_type != ZIP_STORED or member.file_size != member.compress_size
                or member.file_size <= 0 or member.date_time != (1980, 1, 1, 0, 0, 0)
                or (member.create_system, member.create_version, member.extract_version, member.reserved) != (3, 20, 20, 0)
                or (member.flag_bits, member.volume, member.internal_attr) != (0, 0, 0)
                or member.external_attr != (stat.S_IFREG | 0o600) << 16
                or member.extra != b"" or member.comment != b""):
            raise ArchiveIntegrityError("ZIP member metadata is not canonical.")
        # DOS date 0x21 is 1980-01-01; its time word is zero.
        expected_header = (b"PK\x03\x04", 20, 0, ZIP_STORED, 0, 0x21, member.CRC,
                           member.compress_size, member.file_size, len(name), 0)
        if _LOCAL.unpack_from(contents, next_offset) != expected_header:
            raise ArchiveIntegrityError("ZIP local header differs from its directory record.")
        name_start = next_offset + _LOCAL.size
        if contents[name_start:name_start + len(name)] != name:
            raise ArchiveIntegrityError("ZIP local and directory names differ.")
        next_offset = name_start + len(name) + member.file_size
        if next_offset > directory_offset:
            raise ArchiveIntegrityError("ZIP member extends into its directory.")
        expected_directory_size += _CENTRAL_SHAPE.size + len(name)
        names.append(member.filename)
    if next_offset != directory_offset or expected_directory_size != directory_size:
        raise ArchiveIntegrityError("ZIP contains bytes outside its declared members.")
    return names


def _read_member(archive: ZipFile, member: ZipInfo, record: StoredArtifact, *, retain: bool = False) -> bytes:
    """Read to EOF for zipfile's CRC check, hashing bounded chunks independently."""
    if member.file_size != record.size:
        raise ArchiveIntegrityError("ZIP member size differs from its trusted record.")
    digest = sha256()
    total = 0
    retained = bytearray()
    with archive.open(member, "r") as stream:
        while chunk := stream.read(min(READ_CHUNK_BYTES, record.size + 1 - total)):
            total += len(chunk)
            if total > record.size:
                raise ArchiveIntegrityError("ZIP member exceeds its trusted size.")
            digest.update(chunk)
            if retain:
                retained.extend(chunk)
    if total != record.size or digest.hexdigest() != record.sha256:
        raise ArchiveIntegrityError("ZIP member bytes differ from their trusted record.")
    return bytes(retained)


def _read_verified_archive(output_root: Path, archive_ref: StoredArtifact,
                            inventory_ref: StoredArtifact, *, run_id: str,
                            attempt_id: str, include_preparation: bool = False,
                            include_review: bool = False, include_scorecard: bool = False) -> _VerifiedArchive:
    """Verify one snapshot; retain only the requested fixed groups of controls."""
    try:
        BudgetLedger._run_id(run_id)
        BudgetLedger._attempt_id(attempt_id)
        prefix = f"attempts/{attempt_id}/export"
        _check_reference(archive_ref, prefix + "/archives", "zip", MAX_ARCHIVE_BYTES)
        _check_reference(inventory_ref, prefix + "/inventories", "json", MAX_INVENTORY_BYTES)
        contents = read_verified(Path(output_root) / run_id, archive_ref)
        count, directory_size, directory_offset = _check_envelope(contents)
        with ZipFile(BytesIO(contents), "r", allowZip64=False) as archive:
            members = archive.infolist()
            if len(members) != count or archive.comment != b"":
                raise ArchiveIntegrityError("ZIP member count or comment differs from its end record.")
            names = _check_members(contents, members, directory_size, directory_offset)
            if names[0] != "inventory.json":
                raise ArchiveIntegrityError("ZIP must start with inventory.json.")
            inventory_bytes = _read_member(archive, members[0], inventory_ref, retain=True)
            data = json.loads(inventory_bytes.decode("utf-8"), object_pairs_hook=unique_fields)
            json.dumps(data, allow_nan=False)
            records = _check_inventory(data, run_id=run_id, attempt_id=attempt_id)
            if names != ["inventory.json", *records]:
                raise ArchiveIntegrityError("ZIP members do not match the selected inventory.")
            control_limits = {}
            if include_preparation or include_review or include_scorecard:
                control_limits.update(PREPARATION_CONTROL_LIMITS)
            if include_review or include_scorecard:
                control_limits.update(REVIEW_CONTROL_LIMITS)
            limits = {f"attempts/{attempt_id}/{name}": limit for name, limit in control_limits.items()}
            if include_scorecard:
                for name, limit in (("scorecardBinding", MAX_BINDING_BYTES), ("scorecard", MAX_SCORECARD_BYTES)):
                    selected = data["selection"][name]
                    if selected is not None:
                        limits[selected["path"]] = limit
            for path, limit in limits.items():
                if records[path].size > limit:
                    raise ArchiveIntegrityError("Archive control exceeds its retained byte limit.")
            controls = {}
            for member in members[1:]:
                member_bytes = _read_member(archive, member, records[member.filename], retain=member.filename in limits)
                if member.filename in limits:
                    controls[member.filename] = member_bytes
        result = ExportArchiveIntegrity(archive_ref, inventory_ref, len(records), data["totalFileBytes"])
        return _VerifiedArchive(result, data, records, controls)
    except Exception:
        raise ArchiveIntegrityError("Export archive integrity could not be verified for this attempt.") from None


def verify_export_archive(output_root: Path, archive_ref: StoredArtifact,
                          inventory_ref: StoredArtifact, *, run_id: str,
                          attempt_id: str) -> ExportArchiveIntegrity:
    """Verify the actual ZIP without reading original evidence or writing files.

    Both references must come from a trusted caller. Only archive_ref is read
    from output_root/run_id; inventory_ref anchors the exact inventory.json
    bytes and need not exist locally. The complete ZIP is held in memory using
    the storage API, while payload members are read in bounded chunks.

    This checks the ZIP and inventory contract, not the bundled record graph,
    media decoding, private evidence, ledger, human or provider authenticity,
    or release outcomes. A consistent replacement of both trusted references
    is outside the integrity guarantee. No extraction, clock, or network is used.
    """
    return _read_verified_archive(output_root, archive_ref, inventory_ref,
                                  run_id=run_id, attempt_id=attempt_id).integrity
