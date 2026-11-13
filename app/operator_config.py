"""Read the operator's fixed approval inputs from private local configuration."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat

from app.preparation_state import _canonical_directory, reference
from core.json_data import unique_fields
from core.storage import StoredArtifact, _parent_directory
from features.inference.pricing import MAX_PRICE_RECORD_BYTES


MAX_CONFIG_BYTES = 16 * 1024
CONFIG_NAME = "operator-approval.json"


class OperatorConfigError(ValueError):
    """The private approval configuration could not be verified."""


@dataclass(frozen=True)
class OperatorApprovalConfig:
    output_root: Path
    approved_by: str
    price_quote: StoredArtifact


def _reject_constant(value):
    raise ValueError("Nonstandard JSON number.")


def read_operator_config(state_dir: Path) -> OperatorApprovalConfig:
    """Read one bounded regular file; never create configuration or directories."""
    try:
        state_dir = Path(os.path.abspath(state_dir))
        with _parent_directory(state_dir, [CONFIG_NAME], create=False) as directory:
            info = os.fstat(directory)
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise OperatorConfigError()
            descriptor = os.open(CONFIG_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
            with os.fdopen(descriptor, "rb") as file:
                info = os.fstat(file.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_mode & 0o077 or not 0 < info.st_size <= MAX_CONFIG_BYTES):
                    raise OperatorConfigError()
                body = file.read(MAX_CONFIG_BYTES + 1)
        if len(body) != info.st_size:
            raise OperatorConfigError()
        value = json.loads(body.decode("utf-8"), object_pairs_hook=unique_fields,
                           parse_constant=_reject_constant)
        if (type(value) is not dict or set(value) != {
                "schemaVersion", "manifestType", "outputRoot", "approvedBy", "priceQuote"}
                or value["schemaVersion"] != "1.4"
                or value["manifestType"] != "local_operator_approval_config"
                or type(value["outputRoot"]) is not str
                or not 0 < len(value["outputRoot"]) <= 4096
                or type(value["approvedBy"]) is not str
                or not value["approvedBy"].strip() or len(value["approvedBy"]) > 200):
            raise OperatorConfigError()
        output = _canonical_directory(value["outputRoot"])
        if str(output) != value["outputRoot"]:
            raise OperatorConfigError()
        quote = reference(value["priceQuote"])
        if (quote.size > MAX_PRICE_RECORD_BYTES
                or quote.path != f"provider_pricing/{quote.sha256}.json"):
            raise OperatorConfigError()
        return OperatorApprovalConfig(output, value["approvedBy"], quote)
    except Exception:
        raise OperatorConfigError("Operator approval configuration is invalid.") from None
