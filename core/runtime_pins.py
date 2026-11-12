"""Keep operator-owned receipt pins outside the run and export directories."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import NamedTuple

from core.budget import BudgetLedger
from core.storage import StoredArtifact


FORMAT_VERSION = "runtime-pins-1"
MAX_DATABASE_BYTES = 16 * 1024 * 1024
MAX_PINS = 10_000
MAX_CELL_BYTES = 16 * 1024
MAX_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_QUERY_STEPS = 1_000_000
QUERY_STEP_INTERVAL = 1000
LOCK_TIMEOUT_SECONDS = 0.25
PAGE_SIZE_BYTES = 4096
_SQLITE_SUFFIXES = ("", "-journal", "-wal", "-shm")
_KINDS = {
    "runtime_create_context": r"create-context\.json",
    "runtime_create_result": r"create-result\.json",
    "runtime_task_get": r"task-gets/[0-9a-f]{32}\.json",
    "runtime_download_intent": r"downloads/[0-9a-f]{32}/intent\.json",
    "runtime_download_result": r"downloads/[0-9a-f]{32}/result\.json",
}
_TABLES = {
    "runtime_pin_owner": """CREATE TABLE runtime_pin_owner (
        owner_id INTEGER PRIMARY KEY CHECK (owner_id = 1),
        format_version TEXT NOT NULL,
        run_id TEXT NOT NULL,
        output_root TEXT NOT NULL,
        ledger_path TEXT NOT NULL
    )""",
    "runtime_pins": """CREATE TABLE runtime_pins (
        pin_id INTEGER PRIMARY KEY CHECK (pin_id > 0),
        attempt_id TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        provider_task_id TEXT,
        receipt_kind TEXT NOT NULL,
        receipt_path TEXT NOT NULL UNIQUE,
        receipt_sha256 TEXT NOT NULL,
        receipt_size INTEGER NOT NULL
    )""",
}
_TRIGGERS = {}
for _table in _TABLES:
    for _operation in ("UPDATE", "DELETE"):
        _name = f"{_table}_no_{_operation.lower()}"
        _TRIGGERS[_name] = (f"CREATE TRIGGER {_name} BEFORE {_operation} ON {_table} "
                           "BEGIN SELECT RAISE(ABORT, 'Runtime pins are immutable.'); END")
_TRIGGERS["runtime_pin_owner_no_replace"] = """CREATE TRIGGER runtime_pin_owner_no_replace
    BEFORE INSERT ON runtime_pin_owner WHEN EXISTS (SELECT 1 FROM runtime_pin_owner)
    BEGIN SELECT RAISE(ABORT, 'Runtime pin owner already exists.'); END"""
_TRIGGERS["runtime_pins_no_replace"] = """CREATE TRIGGER runtime_pins_no_replace
    BEFORE INSERT ON runtime_pins
    WHEN EXISTS (SELECT 1 FROM runtime_pins WHERE pin_id = NEW.pin_id OR receipt_path = NEW.receipt_path)
    BEGIN SELECT RAISE(ABORT, 'Runtime pin already exists.'); END"""


class RuntimePinError(ValueError):
    """The private receipt index cannot support this operation."""


class _Owner(NamedTuple):
    format_version: str
    run_id: str
    output_root: str
    ledger_path: str


@dataclass(frozen=True)
class RuntimePin:
    attempt_id: str
    payload_sha256: str
    provider_task_id: str | None
    receipt_kind: str
    receipt: StoredArtifact


@dataclass(frozen=True)
class RuntimePinSnapshot:
    run_id: str
    pins: tuple[RuntimePin, ...]
    sha256: str


def _owner(output_root, ledger_path, run_id):
    BudgetLedger._run_id(run_id)
    root, ledger = Path(output_root).resolve(), Path(ledger_path).resolve()
    if any(len(str(path).encode("utf-8")) > 4096 for path in (root, ledger)):
        raise RuntimePinError("Runtime pin owner path exceeds its limit.")
    return _Owner(FORMAT_VERSION, run_id, str(root), str(ledger))


def _require_separate_sqlite_files(index: Path, ledger: Path):
    """Protect main files and current or future SQLite sidecars in both directions."""
    index_files = {Path(str(index) + suffix).resolve() for suffix in _SQLITE_SUFFIXES}
    ledger_files = {Path(str(ledger) + suffix).resolve() for suffix in _SQLITE_SUFFIXES}
    if index_files & ledger_files:
        raise RuntimePinError("Runtime pin files cannot overlap ledger files.")

    def existing_identity(path):
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        return info.st_dev, info.st_ino

    ledger_identities = {existing_identity(path) for path in ledger_files} - {None}
    if any(existing_identity(path) in ledger_identities for path in index_files):
        raise RuntimePinError("Runtime pin files cannot alias ledger files.")


def _check_paths(path, owner, *, allow_missing=False):
    """The operator owns the private parent and its ancestors."""
    root, ledger = Path(owner.output_root), Path(owner.ledger_path)
    parent = path.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_mode & 0o077
            or parent.st_uid != os.getuid()):
        raise RuntimePinError("Runtime pins need an existing private parent directory.")
    if path.resolve().is_relative_to(root):
        raise RuntimePinError("Runtime pins must be separate from run files and the ledger.")
    _require_separate_sqlite_files(path, ledger)
    ledger_info = ledger.lstat()
    if not stat.S_ISREG(ledger_info.st_mode):
        raise RuntimePinError("Runtime pins need an existing regular ledger.")
    for suffix in _SQLITE_SUFFIXES:
        candidate = Path(str(path) + suffix)
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            if suffix or allow_missing:
                continue
            raise RuntimePinError("Runtime pin index does not exist.") from None
        if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077
                or info.st_uid != os.getuid() or info.st_size > MAX_DATABASE_BYTES):
            raise RuntimePinError("Runtime pin index is not a private bounded regular file.")
        if suffix in ("-wal", "-shm"):
            raise RuntimePinError("Runtime pin index cannot use WAL sidecars.")


@contextmanager
def _connection(path, *, writable=False):
    mode = "rw" if writable else "ro"
    connection = sqlite3.connect(path.as_uri() + "?mode=" + mode, uri=True,
                                 timeout=LOCK_TIMEOUT_SECONDS, isolation_level=None)
    try:
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_CELL_BYTES)
        connection.execute("PRAGMA trusted_schema=OFF")
        if not writable:
            connection.execute("PRAGMA query_only=ON")
        steps = 0

        def bounded_query():
            nonlocal steps
            steps += QUERY_STEP_INTERVAL
            return steps > MAX_QUERY_STEPS

        connection.set_progress_handler(bounded_query, QUERY_STEP_INTERVAL)
        yield connection
    finally:
        connection.close()


def _inspect(connection, owner):
    expected = {name: ("table", sql) for name, sql in _TABLES.items()}
    expected.update({name: ("trigger", sql) for name, sql in _TRIGGERS.items()})
    expected["sqlite_autoindex_runtime_pins_1"] = ("index", None)
    rows = connection.execute("SELECT name, type, sql FROM sqlite_schema ORDER BY name LIMIT ?",
                              (len(expected) + 1,)).fetchall()
    if {name: (kind, sql) for name, kind, sql in rows} != expected:
        raise RuntimePinError("Runtime pin index has a foreign schema.")
    found = connection.execute(
        "SELECT format_version, run_id, output_root, ledger_path FROM runtime_pin_owner LIMIT 2").fetchall()
    if found != [owner]:
        raise RuntimePinError("Runtime pin index belongs to another owner.")
    if connection.execute("PRAGMA page_size").fetchone() != (PAGE_SIZE_BYTES,):
        raise RuntimePinError("Runtime pin index uses an unsupported page size.")
    if connection.execute("PRAGMA journal_mode").fetchone() != ("delete",):
        raise RuntimePinError("Runtime pin index uses an unsupported journal mode.")


def _validate_pin(pin):
    if type(pin) is not RuntimePin or type(pin.receipt) is not StoredArtifact:
        raise RuntimePinError("Runtime pin requires the fixed receipt identity.")
    BudgetLedger._attempt_id(pin.attempt_id)
    for value in (pin.payload_sha256, pin.receipt.sha256):
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimePinError("Runtime pin hash is invalid.")
    if pin.provider_task_id is not None and (type(pin.provider_task_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", pin.provider_task_id) is None):
        raise RuntimePinError("Runtime pin task ID is invalid.")
    pattern = _KINDS.get(pin.receipt_kind)
    prefix = f"attempts/{pin.attempt_id}/runtime/"
    if (pattern is None or type(pin.receipt.path) is not str
            or re.fullmatch(re.escape(prefix) + pattern, pin.receipt.path) is None):
        raise RuntimePinError("Runtime pin path does not match its attempt and kind.")
    if type(pin.receipt.size) is not int or not 0 < pin.receipt.size <= 16 * 1024:
        raise RuntimePinError("Runtime pin receipt size exceeds its limit.")
    if pin.receipt_kind == "runtime_create_context":
        if pin.provider_task_id is not None:
            raise RuntimePinError("Creation context cannot claim a provider task.")
    elif pin.receipt_kind != "runtime_create_result" and pin.provider_task_id is None:
        raise RuntimePinError("Task observations need a provider task ID.")


def _values(pin):
    return (pin.attempt_id, pin.payload_sha256, pin.provider_task_id, pin.receipt_kind,
            pin.receipt.path, pin.receipt.sha256, pin.receipt.size)


_COLUMNS = "attempt_id, payload_sha256, provider_task_id, receipt_kind, receipt_path, receipt_sha256, receipt_size"


def _from_row(row):
    pin = RuntimePin(*row[:4], StoredArtifact(*row[4:]))
    _validate_pin(pin)
    return pin


def read_runtime_pins(path: Path, *, output_root: Path, ledger_path: Path,
                      run_id: str) -> RuntimePinSnapshot:
    """Read the explicitly selected external index without schema or row writes."""
    try:
        path = Path(path).absolute()
        owner = _owner(output_root, ledger_path, run_id)
        _check_paths(path, owner)
        with _connection(path) as connection:
            connection.execute("BEGIN")
            _inspect(connection, owner)
            rows = connection.execute(f"SELECT {_COLUMNS} FROM runtime_pins ORDER BY receipt_path LIMIT ?",
                                      (MAX_PINS + 1,))
            pins, size = [], 0
            for row in rows:
                pin = _from_row(row)
                size += len(json.dumps(asdict(pin), sort_keys=True).encode("utf-8"))
                if len(pins) >= MAX_PINS or size > MAX_CAPTURE_BYTES:
                    raise RuntimePinError("Runtime pin capture exceeds its limits.")
                pins.append(pin)
        body = json.dumps({"owner": owner, "pins": [asdict(pin) for pin in pins]},
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
        return RuntimePinSnapshot(run_id, tuple(pins), sha256(body).hexdigest())
    except Exception:
        raise RuntimePinError("Runtime pin index could not be read for this owner.") from None


class RuntimePinStore:
    """Capture fresh receipts only; this local index is not provider proof."""

    def __init__(self, path: Path, *, output_root: Path, ledger_path: Path, run_id: str):
        try:
            self.path = Path(path).absolute()
            self._owner = _owner(output_root, ledger_path, run_id)
            _check_paths(self.path, self._owner, allow_missing=True)
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                read_runtime_pins(self.path, output_root=output_root, ledger_path=ledger_path, run_id=run_id)
            else:
                os.close(descriptor)
                with _connection(self.path, writable=True) as connection:
                    connection.execute("PRAGMA synchronous=FULL")
                    connection.execute(f"PRAGMA page_size={PAGE_SIZE_BYTES}")
                    connection.execute(f"PRAGMA max_page_count={MAX_DATABASE_BYTES // PAGE_SIZE_BYTES}")
                    connection.execute("BEGIN IMMEDIATE")
                    for sql in (*_TABLES.values(), *_TRIGGERS.values()):
                        connection.execute(sql)
                    connection.execute("INSERT INTO runtime_pin_owner VALUES (1, ?, ?, ?, ?)", self._owner)
                    connection.execute("COMMIT")
            # A prior creation may have committed before its directory sync failed.
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception:
            raise RuntimePinError("Runtime pin index could not be opened for this owner.") from None

    def require_owner(self, output_root: Path, ledger_path: Path, run_id: str):
        if _owner(output_root, ledger_path, run_id) != self._owner:
            raise RuntimePinError("Runtime pin index belongs to another owner.")

    def matches(self, pin: RuntimePin) -> bool:
        """Missing pins stay missing. An existing different pin is an error."""
        try:
            _validate_pin(pin)
            _check_paths(self.path, self._owner)
            with _connection(self.path) as connection:
                connection.execute("BEGIN")
                _inspect(connection, self._owner)
                row = connection.execute(f"SELECT {_COLUMNS} FROM runtime_pins WHERE receipt_path=? LIMIT 1",
                                         (pin.receipt.path,)).fetchone()
                if row is None:
                    return False
                if _from_row(row) != pin:
                    raise RuntimePinError("Runtime receipt differs from its retained pin.")
            return True
        except Exception:
            raise RuntimePinError("Runtime receipt does not match its retained pin.") from None

    def pin(self, pin: RuntimePin):
        """Commit one fresh receipt identity; never replace an earlier pin."""
        try:
            _validate_pin(pin)
            _check_paths(self.path, self._owner)
            # Inspect foreign files before opening a writable SQLite connection.
            with _connection(self.path) as connection:
                _inspect(connection, self._owner)
            with _connection(self.path, writable=True) as connection:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute(f"PRAGMA max_page_count={MAX_DATABASE_BYTES // PAGE_SIZE_BYTES}")
                connection.execute("BEGIN IMMEDIATE")
                _inspect(connection, self._owner)
                row = connection.execute(f"SELECT {_COLUMNS} FROM runtime_pins WHERE receipt_path=? LIMIT 1",
                                         (pin.receipt.path,)).fetchone()
                if row is not None:
                    if _from_row(row) != pin:
                        raise RuntimePinError("Runtime receipt differs from its retained pin.")
                else:
                    count = connection.execute("SELECT count(*) FROM runtime_pins").fetchone()[0]
                    if count >= MAX_PINS:
                        raise RuntimePinError("Runtime pin count exceeds its limit.")
                    connection.execute(f"INSERT INTO runtime_pins ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                       _values(pin))
                connection.execute("COMMIT")
        except Exception:
            raise RuntimePinError("Runtime receipt pin could not be saved.") from None
