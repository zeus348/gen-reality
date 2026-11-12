"""Save the experiment clock before a new run processes any media."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import time

from core.budget import BudgetLedger
from core.json_data import unique_fields
from core.storage import ArtifactExistsError, StoredArtifact, _parent_directory, publish_bytes


RUN_START_PATH = "run-start.json"
MAX_RUN_START_BYTES = 4096
MAX_ELAPSED_SECONDS = 180 * 60
MAX_UNIX_SECONDS = 253_402_300_799


class RunStartError(ValueError):
    """The original experiment start cannot be saved or verified safely."""


@dataclass(frozen=True)
class RunStart:
    run_id: str
    started_unix: float
    deadline_unix: float
    record: StoredArtifact


def _valid_time(value) -> bool:
    return (type(value) in (int, float) and 0 <= value <= MAX_UNIX_SECONDS
            and math.isfinite(value))


def _invalid_constant(value):
    raise ValueError("Run-start JSON must contain finite standard numbers.")


def _read_start(root: Path, run_id: str, *, sync_directory: bool = False) -> RunStart:
    with _parent_directory(root, [RUN_START_PATH], create=False) as directory:
        descriptor = os.open(RUN_START_PATH, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
        with os.fdopen(descriptor, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_RUN_START_BYTES:
                raise RunStartError("Run start must be a bounded regular file.")
            contents = file.read(info.st_size + 1)
        if sync_directory:
            # A concurrent publisher may have linked the file but not yet
            # synced this directory. Make the start durable before media work.
            os.fsync(directory)
    if len(contents) != info.st_size:
        raise RunStartError("Run-start file size changed during reading.")
    data = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields,
                      parse_constant=_invalid_constant)
    fields = {"schemaVersion", "manifestType", "runId", "startedAtUnix", "deadlineAtUnix", "maxElapsedSeconds"}
    if (type(data) is not dict or set(data) != fields
            or data["schemaVersion"] != "1.4" or data["manifestType"] != "experiment_start"
            or data["runId"] != run_id or type(data["maxElapsedSeconds"]) is not int
            or data["maxElapsedSeconds"] != MAX_ELAPSED_SECONDS):
        raise RunStartError("Run-start fields do not match this experiment.")
    started, deadline = data["startedAtUnix"], data["deadlineAtUnix"]
    if (not _valid_time(started) or not _valid_time(deadline)
            or deadline != started + MAX_ELAPSED_SECONDS):
        raise RunStartError("Run-start timestamps do not match its fixed duration.")
    record = StoredArtifact(RUN_START_PATH, sha256(contents).hexdigest(), len(contents))
    return RunStart(run_id, started, deadline, record)


def read_run_start(output_root: Path, *, run_id: str) -> RunStart:
    """Read the fixed record from the trusted private run root without a clock."""
    try:
        BudgetLedger._run_id(run_id)
        return _read_start(Path(output_root) / run_id, run_id)
    except Exception:
        raise RunStartError("Run-start record could not be verified.") from None


def start_run(output_root: Path, *, run_id: str, clock=time.time) -> RunStart:
    """Create the start once; never give an existing experiment a new deadline.

    A run with other files but no start is legacy or incomplete work. Do not
    adopt it. Concurrent writers may retry after the winning record is durable.
    A stranded pending file blocks this run until trusted repair or a new run ID.
    The caller must keep the output root private and call this before media work.
    """
    try:
        BudgetLedger._run_id(run_id)
        root = Path(output_root) / run_id
        try:
            return _read_start(root, run_id, sync_directory=True)
        except FileNotFoundError:
            pass
        with _parent_directory(root, [RUN_START_PATH], create=True) as directory:
            with os.scandir(directory) as entries:
                occupied = next(entries, None) is not None
        if occupied:
            # Another starter may have finished since the first read.
            return _read_start(root, run_id, sync_directory=True)
        started = clock()
        if not _valid_time(started) or started > MAX_UNIX_SECONDS - MAX_ELAPSED_SECONDS:
            raise RunStartError("Run start needs a valid UTC clock.")
        deadline = started + MAX_ELAPSED_SECONDS
        data = {"schemaVersion": "1.4", "manifestType": "experiment_start", "runId": run_id,
                "startedAtUnix": started, "deadlineAtUnix": deadline,
                "maxElapsedSeconds": MAX_ELAPSED_SECONDS}
        contents = (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        try:
            record = publish_bytes(root, RUN_START_PATH, contents)
        except ArtifactExistsError:
            return _read_start(root, run_id, sync_directory=True)
        return RunStart(run_id, started, deadline, record)
    except Exception:
        raise RunStartError("Run start could not be saved or verified; existing files were kept.") from None
