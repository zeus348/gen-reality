"""Store the local editor's fixed selections outside its immutable media files."""

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sqlite3
import stat

from core.json_data import unique_fields
from core.runtime_pins import _require_separate_sqlite_files
from core.storage import StoredArtifact, _parent_directory


MAX_DATABASE_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 65_536
MAX_RUNS = 64
MAX_INPUTS_PER_RUN = 256
MAX_ATTEMPTS_PER_RUN = 256
MAX_REVISION = 2_147_483_647
PAGE_SIZE = 4096
LOCK_TIMEOUT = 0.25
FORMAT = "local-preparation-1"


class ServiceError(ValueError):
    """Expose a fixed code, never SQLite details or private filesystem paths."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def encode(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(body.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ServiceError("CAPACITY_EXCEEDED")
    return body


def _invalid_number(value):
    raise ValueError("Nonstandard JSON number.")


def decode(body):
    if type(body) is not str or len(body.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ServiceError("STATE_UNAVAILABLE")
    return json.loads(body, object_pairs_hook=unique_fields, parse_constant=_invalid_number)


def reference(value):
    if type(value) is not dict or set(value) != {"path", "sha256", "size"}:
        raise ServiceError("STATE_UNAVAILABLE")
    record = StoredArtifact(**value)
    if (type(record.path) is not str or len(record.path) > 1024
            or type(record.sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", record.sha256) is None
            or type(record.size) is not int or not 0 < record.size <= 512 * 1024 * 1024):
        raise ServiceError("STATE_UNAVAILABLE")
    return record


def check_run_id(run_id):
    if type(run_id) is not str or re.fullmatch(r"run-[0-9a-f]{32}", run_id) is None:
        raise ServiceError("INVALID_INPUT")


def check_revision(revision):
    if type(revision) is not int or not 0 <= revision <= MAX_REVISION:
        raise ServiceError("INVALID_INPUT")


_TABLES = {
    "preparation_owner": """CREATE TABLE preparation_owner (
        owner_id INTEGER PRIMARY KEY CHECK (owner_id=1),
        format_version TEXT NOT NULL, output_root TEXT NOT NULL, ledger_path TEXT NOT NULL
    )""",
    "preparation_runs": """CREATE TABLE preparation_runs (
        run_id TEXT PRIMARY KEY, start_record TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision BETWEEN 0 AND 2147483647),
        source_id TEXT, asset_id TEXT, placement TEXT, prepared TEXT
    )""",
    "preparation_inputs": """CREATE TABLE preparation_inputs (
        input_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES preparation_runs(run_id),
        kind TEXT NOT NULL CHECK (kind IN ('source','asset')), record TEXT NOT NULL
    )""",
    "preparation_attempts": """CREATE TABLE preparation_attempts (
        attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES preparation_runs(run_id),
        revision INTEGER NOT NULL, source_id TEXT NOT NULL REFERENCES preparation_inputs(input_id),
        asset_id TEXT NOT NULL REFERENCES preparation_inputs(input_id), placement TEXT NOT NULL,
        prompt TEXT NOT NULL, status TEXT NOT NULL, manifest TEXT, error_code TEXT,
        CHECK ((status='BUILDING' AND manifest IS NULL AND error_code IS NULL)
            OR (status='PREPARED' AND manifest IS NOT NULL AND error_code IS NULL)
            OR (status='FAILED' AND manifest IS NULL AND error_code='WORK_FAILED')
            OR (status='INTERRUPTED' AND manifest IS NULL AND error_code='WORK_INTERRUPTED'))
    )""",
}
_TRIGGERS = {}
for _table, _key in (("preparation_owner", "owner_id"), ("preparation_runs", "run_id"),
                     ("preparation_inputs", "input_id"), ("preparation_attempts", "attempt_id")):
    _TRIGGERS[_table + "_no_delete"] = (
        f"CREATE TRIGGER {_table}_no_delete BEFORE DELETE ON {_table} "
        "BEGIN SELECT RAISE(ABORT, 'Preparation history cannot be deleted.'); END")
    _TRIGGERS[_table + "_no_replace"] = (
        f"CREATE TRIGGER {_table}_no_replace BEFORE INSERT ON {_table} "
        f"WHEN EXISTS (SELECT 1 FROM {_table} WHERE {_key}=NEW.{_key}) "
        "BEGIN SELECT RAISE(ABORT, 'Preparation history cannot be replaced.'); END")
for _table in ("preparation_owner", "preparation_inputs"):
    _TRIGGERS[_table + "_no_update"] = (
        f"CREATE TRIGGER {_table}_no_update BEFORE UPDATE ON {_table} "
        "BEGIN SELECT RAISE(ABORT, 'Preparation identity cannot change.'); END")
_TRIGGERS["preparation_runs_revision"] = """CREATE TRIGGER preparation_runs_revision
    BEFORE UPDATE ON preparation_runs
    WHEN NEW.run_id!=OLD.run_id OR NEW.start_record!=OLD.start_record OR NEW.revision!=OLD.revision+1
    BEGIN SELECT RAISE(ABORT, 'Preparation edits require the next revision.'); END"""
_TRIGGERS["preparation_attempts_transition"] = """CREATE TRIGGER preparation_attempts_transition
    BEFORE UPDATE ON preparation_attempts
    WHEN OLD.status!='BUILDING' OR NEW.status NOT IN ('PREPARED','FAILED','INTERRUPTED')
      OR NEW.attempt_id!=OLD.attempt_id OR NEW.run_id!=OLD.run_id OR NEW.revision!=OLD.revision
      OR NEW.source_id!=OLD.source_id OR NEW.asset_id!=OLD.asset_id
      OR NEW.placement!=OLD.placement OR NEW.prompt!=OLD.prompt
    BEGIN SELECT RAISE(ABORT, 'Preparation attempts have fixed inputs.'); END"""


def _canonical_directory(path):
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if not stat.S_ISDIR(info.st_mode):
            raise ServiceError("STATE_UNAVAILABLE")
    return path


def _check_state_files(state_dir, output_root, ledger_path, path, *, allow_missing=False):
    with _parent_directory(state_dir, [path.name], create=False) as directory:
        info = os.fstat(directory)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ServiceError("STATE_UNAVAILABLE")
    if state_dir.is_relative_to(output_root):
        raise ServiceError("STATE_UNAVAILABLE")
    if not stat.S_ISREG(ledger_path.lstat().st_mode):
        raise ServiceError("STATE_UNAVAILABLE")
    _require_separate_sqlite_files(path, ledger_path)
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(path) + suffix)
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            if suffix or allow_missing:
                continue
            raise
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_size > MAX_DATABASE_BYTES):
            raise ServiceError("STATE_UNAVAILABLE")


@contextmanager
def _state_connection(path, *, write=False):
    connection = sqlite3.connect(path.as_uri() + ("?mode=rw" if write else "?mode=ro"),
                                 uri=True, timeout=LOCK_TIMEOUT, isolation_level=None)
    try:
        connection.row_factory = sqlite3.Row
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_RECORD_BYTES)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA foreign_keys=ON")
        steps = 0

        def bounded_query():
            nonlocal steps
            steps += 1000
            return steps > 2_000_000

        connection.set_progress_handler(bounded_query, 1000)
        if write:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(f"PRAGMA max_page_count={MAX_DATABASE_BYTES // PAGE_SIZE}")
            connection.execute("PRAGMA wal_autocheckpoint=64")
            connection.execute(f"PRAGMA journal_size_limit={MAX_DATABASE_BYTES}")
        else:
            connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def _inspect_state(connection, owner):
    expected = {name: ("table", sql) for name, sql in _TABLES.items()}
    expected.update({name: ("trigger", sql) for name, sql in _TRIGGERS.items()})
    for table in ("preparation_runs", "preparation_inputs", "preparation_attempts"):
        expected[f"sqlite_autoindex_{table}_1"] = ("index", None)
    rows = connection.execute("SELECT name,type,sql FROM sqlite_schema LIMIT ?", (len(expected) + 1,))
    if {row["name"]: (row["type"], row["sql"]) for row in rows} != expected:
        raise ServiceError("STATE_UNAVAILABLE")
    rows = connection.execute("SELECT format_version,output_root,ledger_path "
                              "FROM preparation_owner LIMIT 2").fetchall()
    if [tuple(row) for row in rows] != [owner]:
        raise ServiceError("STATE_UNAVAILABLE")
    if (connection.execute("PRAGMA page_size").fetchone()[0] != PAGE_SIZE
            or connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal"):
        raise ServiceError("STATE_UNAVAILABLE")


def read_registered_run_start(state_dir, output_root, ledger_path, *, run_id) -> StoredArtifact:
    """Read one registered start without startup recovery or a writer connection."""
    try:
        check_run_id(run_id)
        state_dir = _canonical_directory(state_dir)
        output_root = _canonical_directory(output_root)
        ledger_path = Path(os.path.abspath(ledger_path))
        path = state_dir / "preparation.sqlite"
        owner = (FORMAT, str(output_root), str(ledger_path))
        _check_state_files(state_dir, output_root, ledger_path, path)
        with _state_connection(path) as connection:
            connection.execute("BEGIN")
            _inspect_state(connection, owner)
            row = connection.execute(
                "SELECT start_record FROM preparation_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise ServiceError("RUN_NOT_FOUND")
            record = reference(decode(row["start_record"]))
        _check_state_files(state_dir, output_root, ledger_path, path)
        return record
    except ServiceError:
        raise
    except Exception:
        raise ServiceError("STATE_UNAVAILABLE") from None


class State:
    """Four fixed tables. Never hold a transaction during media work."""

    def __init__(self, state_dir, output_root, ledger_path):
        self.state_dir = _canonical_directory(state_dir)
        self.output_root = _canonical_directory(output_root)
        self.ledger_path = Path(os.path.abspath(ledger_path))
        self.path = self.state_dir / "preparation.sqlite"
        self.owner = (FORMAT, str(self.output_root), str(self.ledger_path))
        try:
            self._check_files(allow_missing=True)
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                with self.connection() as connection:
                    self._inspect(connection)
            else:
                os.close(descriptor)
                with self.connection(write=True, initialize=True) as connection:
                    connection.execute(f"PRAGMA page_size={PAGE_SIZE}")
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("BEGIN IMMEDIATE")
                    for sql in (*_TABLES.values(), *_TRIGGERS.values()):
                        connection.execute(sql)
                    connection.execute("INSERT INTO preparation_owner VALUES (1,?,?,?)", self.owner)
                    connection.execute("COMMIT")
            with _parent_directory(self.state_dir, [self.path.name], create=False) as directory:
                os.fsync(directory)
            with self.transaction() as connection:
                connection.execute("UPDATE preparation_attempts SET status='INTERRUPTED', "
                                   "error_code='WORK_INTERRUPTED' WHERE status='BUILDING'")
        except Exception:
            raise ServiceError("STATE_UNAVAILABLE") from None

    def _check_files(self, *, allow_missing=False):
        _check_state_files(self.state_dir, self.output_root, self.ledger_path,
                           self.path, allow_missing=allow_missing)

    @contextmanager
    def connection(self, *, write=False, initialize=False):
        self._check_files()
        if write and not initialize:
            with self.connection() as check:
                self._inspect(check)
        with _state_connection(self.path, write=write) as connection:
            yield connection

    def _inspect(self, connection):
        _inspect_state(connection, self.owner)

    @contextmanager
    def transaction(self):
        with self.connection(write=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._inspect(connection)
                yield connection
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _run(connection, run_id):
        row = connection.execute("SELECT * FROM preparation_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ServiceError("RUN_NOT_FOUND")
        return dict(row)

    @staticmethod
    def _require_revision(row, expected):
        if row["revision"] != expected:
            raise ServiceError("REVISION_CONFLICT")
        if expected >= MAX_REVISION:
            raise ServiceError("CAPACITY_EXCEEDED")

    def create(self, run_id, start):
        with self.transaction() as connection:
            if connection.execute("SELECT count(*) FROM preparation_runs").fetchone()[0] >= MAX_RUNS:
                raise ServiceError("CAPACITY_EXCEEDED")
            connection.execute("INSERT INTO preparation_runs VALUES (?,?,0,NULL,NULL,NULL,NULL)",
                               (run_id, encode(asdict(start))))

    def require_capacity(self, run_id=None, *, attempts=False):
        with self.connection() as connection:
            self._inspect(connection)
            if run_id is None:
                count = connection.execute("SELECT count(*) FROM preparation_runs").fetchone()[0]
                maximum = MAX_RUNS
            else:
                table = "preparation_attempts" if attempts else "preparation_inputs"
                count = connection.execute(f"SELECT count(*) FROM {table} WHERE run_id=?", (run_id,)).fetchone()[0]
                maximum = MAX_ATTEMPTS_PER_RUN if attempts else MAX_INPUTS_PER_RUN
            if count >= maximum:
                raise ServiceError("CAPACITY_EXCEEDED")

    def get(self, run_id):
        with self.connection() as connection:
            connection.execute("BEGIN")
            self._inspect(connection)
            result = self._run(connection, run_id)
            result["inputs"] = {}
            for column, kind in (("source_id", "source"), ("asset_id", "asset")):
                if result[column] is not None:
                    row = connection.execute("SELECT * FROM preparation_inputs WHERE input_id=? AND run_id=? AND kind=?",
                                             (result[column], run_id, kind)).fetchone()
                    if row is None:
                        raise ServiceError("STATE_UNAVAILABLE")
                    result["inputs"][kind] = dict(row)
            result["attempts"] = [dict(row) for row in connection.execute(
                "SELECT * FROM preparation_attempts WHERE run_id=? ORDER BY rowid LIMIT ?",
                (run_id, MAX_ATTEMPTS_PER_RUN + 1))]
            if len(result["attempts"]) > MAX_ATTEMPTS_PER_RUN:
                raise ServiceError("STATE_UNAVAILABLE")
            return result

    def set_placement(self, run_id, revision, placement):
        with self.transaction() as connection:
            row = self._run(connection, run_id)
            self._require_revision(row, revision)
            connection.execute("UPDATE preparation_runs SET placement=?,prepared=NULL,revision=revision+1 WHERE run_id=?",
                               (encode(placement), run_id))

    def add_input(self, run_id, revision, input_id, kind, record):
        with self.transaction() as connection:
            count = connection.execute("SELECT count(*) FROM preparation_inputs WHERE run_id=?", (run_id,)).fetchone()[0]
            if count >= MAX_INPUTS_PER_RUN:
                raise ServiceError("CAPACITY_EXCEEDED")
            row = self._run(connection, run_id)
            connection.execute("INSERT INTO preparation_inputs VALUES (?,?,?,?)",
                               (input_id, run_id, kind, encode(asdict(record))))
            selected = row["revision"] == revision
            if selected:
                self._require_revision(row, revision)
                column = "source_id" if kind == "source" else "asset_id"
                connection.execute(f"UPDATE preparation_runs SET {column}=?,prepared=NULL,revision=revision+1 WHERE run_id=?",
                                   (input_id, run_id))
        if not selected:
            raise ServiceError("REVISION_CONFLICT")

    def begin_attempt(self, snapshot, attempt_id, prompt):
        with self.transaction() as connection:
            run_id, revision = snapshot["run_id"], snapshot["revision"]
            self._require_revision(self._run(connection, run_id), revision)
            count = connection.execute("SELECT count(*) FROM preparation_attempts WHERE run_id=?", (run_id,)).fetchone()[0]
            if count >= MAX_ATTEMPTS_PER_RUN:
                raise ServiceError("CAPACITY_EXCEEDED")
            connection.execute("INSERT INTO preparation_attempts VALUES (?,?,?,?,?,?,?,'BUILDING',NULL,NULL)",
                (attempt_id, run_id, revision, snapshot["source_id"], snapshot["asset_id"], snapshot["placement"], prompt))

    def finish_attempt(self, attempt_id, manifest):
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM preparation_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None or row["status"] != "BUILDING":
                raise ServiceError("STATE_UNAVAILABLE")
            body = encode(asdict(manifest))
            connection.execute("UPDATE preparation_attempts SET status='PREPARED',manifest=? WHERE attempt_id=?",
                               (body, attempt_id))
            run = self._run(connection, row["run_id"])
            if run["revision"] == row["revision"]:
                self._require_revision(run, row["revision"])
                connection.execute("UPDATE preparation_runs SET prepared=?,revision=revision+1 WHERE run_id=?",
                                   (body, row["run_id"]))

    def fail_attempt(self, attempt_id):
        with self.transaction() as connection:
            connection.execute("UPDATE preparation_attempts SET status='FAILED',error_code='WORK_FAILED' "
                               "WHERE attempt_id=? AND status='BUILDING'", (attempt_id,))
