"""Bind one registered preparation to one provider operation without spending."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3

from app import preparation_state
from core.budget import BudgetLedger
from core.run_start import read_run_start
from core.storage import StoredArtifact
from features.inference.preflight import read_checked_request, _read_record, _match_fields
from features.inference.prepared import read_prepared_attempt


MAX_INTENTS_PER_RUN = preparation_state.MAX_ATTEMPTS_PER_RUN
MAX_INTENTS = preparation_state.MAX_RUNS * MAX_INTENTS_PER_RUN
MAX_RECORD_BYTES = 65_536


class ProviderIntentError(ValueError):
    """Expose fixed codes without private database details or client tokens."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ProviderIntent:
    run_id: str
    key_sha256: str
    attempt_id: str
    mode: str
    preparation: StoredArtifact
    input_sha256: str
    checked_request: StoredArtifact | None

    @property
    def reservation_key(self):
        return f"provider-intent-v1:{self.run_id}:{self.key_sha256}"


_TABLE = """CREATE TABLE provider_intents (
    run_id TEXT NOT NULL REFERENCES approved_runs(run_id)
        CHECK (length(run_id)=36 AND substr(run_id,1,4)='run-'
               AND substr(run_id,5) NOT GLOB '*[^0-9a-f]*'),
    key_sha256 TEXT NOT NULL CHECK (length(key_sha256)=64 AND key_sha256 NOT GLOB '*[^0-9a-f]*'),
    attempt_id TEXT NOT NULL UNIQUE
        CHECK (length(attempt_id)=36 AND substr(attempt_id,1,4)='att-'
               AND substr(attempt_id,5) NOT GLOB '*[^0-9a-f]*'),
    mode TEXT NOT NULL CHECK (mode IN ('A','B')),
    preparation_path TEXT NOT NULL CHECK (preparation_path='attempts/'||attempt_id||'/manifest.json'),
    preparation_sha256 TEXT NOT NULL
        CHECK (length(preparation_sha256)=64 AND preparation_sha256 NOT GLOB '*[^0-9a-f]*'),
    preparation_size INTEGER NOT NULL
        CHECK (typeof(preparation_size)='integer' AND preparation_size BETWEEN 1 AND 65536),
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256)=64 AND input_sha256 NOT GLOB '*[^0-9a-f]*'),
    checked_path TEXT, checked_sha256 TEXT, checked_size INTEGER,
    PRIMARY KEY (run_id,key_sha256),
    CHECK ((checked_path IS NULL AND checked_sha256 IS NULL AND checked_size IS NULL)
        OR (checked_path IS NOT NULL AND checked_sha256 IS NOT NULL AND checked_size IS NOT NULL
            AND checked_path='attempts/'||attempt_id||'/provider_checks/'||checked_sha256||'.json'
            AND length(checked_sha256)=64 AND checked_sha256 NOT GLOB '*[^0-9a-f]*'
            AND typeof(checked_size)='integer' AND checked_size BETWEEN 1 AND 65536))
)"""
_TRIGGERS = {
    "provider_intents_no_delete": """CREATE TRIGGER provider_intents_no_delete
        BEFORE DELETE ON provider_intents
        BEGIN SELECT RAISE(ABORT, 'Provider intents cannot be deleted.'); END""",
    "provider_intents_no_replace": """CREATE TRIGGER provider_intents_no_replace
        BEFORE INSERT ON provider_intents
        WHEN EXISTS (SELECT 1 FROM provider_intents
            WHERE (run_id=NEW.run_id AND key_sha256=NEW.key_sha256) OR attempt_id=NEW.attempt_id)
        BEGIN SELECT RAISE(ABORT, 'Provider intents cannot be replaced.'); END""",
    "provider_intents_fixed_identity": """CREATE TRIGGER provider_intents_fixed_identity
        BEFORE UPDATE ON provider_intents
        WHEN NEW.run_id IS NOT OLD.run_id OR NEW.key_sha256 IS NOT OLD.key_sha256
          OR NEW.attempt_id IS NOT OLD.attempt_id OR NEW.mode IS NOT OLD.mode
          OR NEW.preparation_path IS NOT OLD.preparation_path
          OR NEW.preparation_sha256 IS NOT OLD.preparation_sha256
          OR NEW.preparation_size IS NOT OLD.preparation_size
          OR NEW.input_sha256 IS NOT OLD.input_sha256
          OR OLD.checked_path IS NOT NULL OR NEW.checked_path IS NULL
        BEGIN SELECT RAISE(ABORT, 'Provider intents have one fixed request selection.'); END""",
}


def _schema(connection):
    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_RECORD_BYTES)
    connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16_384)
    expected = {"provider_intents": ("table", _TABLE),
                "sqlite_autoindex_provider_intents_1": ("index", None),
                "sqlite_autoindex_provider_intents_2": ("index", None)}
    expected.update({name: ("trigger", sql) for name, sql in _TRIGGERS.items()})
    rows = connection.execute("SELECT name,type,sql FROM sqlite_schema WHERE tbl_name='provider_intents' LIMIT ?",
                              (len(expected) + 1,)).fetchall()
    actual = {row["name"]: (row["type"], row["sql"]) for row in rows}
    return actual, expected


def _inspect(connection):
    actual, expected = _schema(connection)
    if actual != expected:
        raise ProviderIntentError("STATE_UNAVAILABLE")


def _ids(run_id, attempt_id):
    try:
        preparation_state.check_run_id(run_id)
        BudgetLedger._attempt_id(attempt_id)
    except Exception:
        raise ProviderIntentError("INVALID_INPUT") from None


def _key_hash(key):
    if (type(key) is not str or not 1 <= len(key) <= 128
            or any(not 0x21 <= ord(character) <= 0x7e for character in key)):
        raise ProviderIntentError("INVALID_INPUT")
    return sha256(key.encode("ascii")).hexdigest()


def _reference(record, attempt_id, *, checked=False):
    if type(record) is not StoredArtifact:
        raise ProviderIntentError("INVALID_INPUT")
    try:
        preparation_state.reference(asdict(record))
    except Exception:
        raise ProviderIntentError("INVALID_INPUT") from None
    expected = (f"attempts/{attempt_id}/provider_checks/{record.sha256}.json" if checked
                else f"attempts/{attempt_id}/manifest.json")
    if record.path != expected or record.size > MAX_RECORD_BYTES:
        raise ProviderIntentError("INVALID_INPUT")
    return record


def _input_hash(run_id, attempt_id, mode, manifest):
    data = {"formatVersion": "provider-intent-1", "model": "aleph2", "runId": run_id,
            "attemptId": attempt_id, "mode": mode, "preparation": asdict(manifest)}
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _intent(row):
    try:
        _ids(row["run_id"], row["attempt_id"])
        if row["mode"] not in ("A", "B"):
            raise ValueError("Invalid mode.")
        for name in ("key_sha256", "input_sha256"):
            if type(row[name]) is not str or re.fullmatch(r"[0-9a-f]{64}", row[name]) is None:
                raise ValueError("Invalid hash.")
        manifest = _reference(StoredArtifact(row["preparation_path"], row["preparation_sha256"],
                                              row["preparation_size"]), row["attempt_id"])
        checked = None
        fields = (row["checked_path"], row["checked_sha256"], row["checked_size"])
        if fields != (None, None, None):
            checked = _reference(StoredArtifact(*fields), row["attempt_id"], checked=True)
        if row["input_sha256"] != _input_hash(row["run_id"], row["attempt_id"], row["mode"], manifest):
            raise ValueError("Input identity changed.")
        return ProviderIntent(row["run_id"], row["key_sha256"], row["attempt_id"], row["mode"],
                              manifest, row["input_sha256"], checked)
    except Exception:
        raise ProviderIntentError("STATE_UNAVAILABLE") from None


def _find(connection, run_id, key_hash, attempt_id):
    rows = connection.execute("SELECT * FROM provider_intents WHERE "
                              "(run_id=? AND key_sha256=?) OR attempt_id=? LIMIT 2",
                              (run_id, key_hash, attempt_id)).fetchall()
    return tuple(_intent(row) for row in rows)


def _matching(rows, run_id, key_hash, attempt_id, mode):
    if not rows:
        return None
    if len(rows) != 1 or (rows[0].run_id, rows[0].key_sha256, rows[0].attempt_id, rows[0].mode) != (
            run_id, key_hash, attempt_id, mode):
        raise ProviderIntentError("INTENT_CONFLICT")
    return rows[0]


class ProviderIntents:
    """Keep fixed intent rows on the already-owned ledger. Make no provider calls."""

    def __init__(self, coordinator, preparation):
        try:
            self.coordinator = coordinator
            self.preparation = preparation
            self._ledger = coordinator.ledger
            self._ledger_path = Path(os.path.abspath(self._ledger.path))
            self._state = preparation._state
            self._output_root = preparation.output_root
            self._state_dir = preparation.state_dir
            with coordinator._mutex:
                self._require_owner()
                with self._ledger._transaction() as connection:
                    actual, expected = _schema(connection)
                    if not actual:
                        connection.execute(_TABLE)
                        for sql in _TRIGGERS.values():
                            connection.execute(sql)
                    elif actual != expected:
                        raise ProviderIntentError("STATE_UNAVAILABLE")
                    _inspect(connection)
        except ProviderIntentError:
            raise
        except Exception:
            raise ProviderIntentError("STATE_UNAVAILABLE") from None

    def _require_owner(self):
        try:
            self.coordinator._require_open()
            self.preparation._require_open()
        except Exception:
            raise ProviderIntentError("SERVICE_CLOSED") from None
        if (self.preparation.coordinator is not self.coordinator
                or getattr(self.coordinator, "_preparation_service", None) is not self.preparation
                or self.coordinator.ledger is not self._ledger
                or self.preparation._state is not self._state
                or Path(os.path.abspath(self._ledger.path)) != self._ledger_path
                or self._state.ledger_path != self._ledger_path
                or self.preparation.output_root != self._output_root
                or self._state.output_root != self._output_root
                or self.preparation.state_dir != self._state_dir
                or self._state.state_dir != self._state_dir):
            raise ProviderIntentError("OWNER_MISMATCH")

    def _prepared(self, run_id, attempt_id):
        self._require_owner()
        # Read only immutable history, without joining the newer draft's inputs.
        with self._state.connection() as connection:
            connection.execute("BEGIN")
            self._state._inspect(connection)
            run = self._state._run(connection, run_id)
            row = connection.execute("SELECT status,manifest FROM preparation_attempts "
                                     "WHERE run_id=? AND attempt_id=? LIMIT 1", (run_id, attempt_id)).fetchone()
        if row is None or row["status"] != "PREPARED" or row["manifest"] is None:
            raise ProviderIntentError("INPUT_REQUIRED")
        start = preparation_state.reference(preparation_state.decode(run["start_record"]))
        if read_run_start(self._output_root, run_id=run_id).record != start:
            raise ProviderIntentError("STATE_UNAVAILABLE")
        manifest = _reference(preparation_state.reference(preparation_state.decode(row["manifest"])), attempt_id)
        return read_prepared_attempt(self._output_root / run_id, manifest, run_id=run_id, attempt_id=attempt_id)

    def admit(self, run_id, *, prepared_attempt_id, mode, idempotency_key) -> ProviderIntent:
        _ids(run_id, prepared_attempt_id)
        if type(mode) is not str or mode not in ("A", "B"):
            raise ProviderIntentError("INVALID_INPUT")
        key_hash = _key_hash(idempotency_key)
        try:
            with self.coordinator._mutex:
                self._require_owner()
                with self._ledger._connection() as connection:
                    _inspect(connection)
                    existing = _matching(_find(connection, run_id, key_hash, prepared_attempt_id),
                                         run_id, key_hash, prepared_attempt_id, mode)
                    if existing is not None:
                        return existing
            prepared = self._prepared(run_id, prepared_attempt_id)
            prepared.media_for_mode(mode)
            digest = _input_hash(run_id, prepared_attempt_id, mode, prepared.manifest)
            with self.coordinator._mutex:
                self._require_owner()
                with self._ledger._transaction() as connection:
                    _inspect(connection)
                    existing = _matching(_find(connection, run_id, key_hash, prepared_attempt_id),
                                         run_id, key_hash, prepared_attempt_id, mode)
                    if existing is not None:
                        return existing
                    if connection.execute("SELECT 1 FROM approved_runs WHERE run_id=?", (run_id,)).fetchone() is None:
                        raise ProviderIntentError("APPROVAL_REQUIRED")
                    for where, parameters, limit in (("WHERE run_id=?", (run_id,), MAX_INTENTS_PER_RUN),
                                                     ("", (), MAX_INTENTS)):
                        count = connection.execute(
                            "SELECT count(*) FROM (SELECT 1 FROM provider_intents " + where + " LIMIT ?)",
                            (*parameters, limit + 1)).fetchone()[0]
                        if count >= limit:
                            raise ProviderIntentError("CAPACITY_EXCEEDED")
                    connection.execute("INSERT INTO provider_intents VALUES (?,?,?,?,?,?,?,?,NULL,NULL,NULL)",
                        (run_id, key_hash, prepared_attempt_id, mode, prepared.manifest.path,
                         prepared.manifest.sha256, prepared.manifest.size, digest))
                    return _matching(_find(connection, run_id, key_hash, prepared_attempt_id),
                                     run_id, key_hash, prepared_attempt_id, mode)
        except ProviderIntentError:
            raise
        except Exception:
            raise ProviderIntentError("STATE_UNAVAILABLE") from None

    def get(self, run_id, *, prepared_attempt_id) -> ProviderIntent:
        _ids(run_id, prepared_attempt_id)
        try:
            with self.coordinator._mutex:
                self._require_owner()
                with self._ledger._connection() as connection:
                    _inspect(connection)
                    rows = connection.execute("SELECT * FROM provider_intents WHERE run_id=? AND attempt_id=? LIMIT 2",
                                              (run_id, prepared_attempt_id)).fetchall()
                    if not rows:
                        raise ProviderIntentError("INTENT_NOT_FOUND")
                    if len(rows) != 1:
                        raise ProviderIntentError("STATE_UNAVAILABLE")
                    return _intent(rows[0])
        except ProviderIntentError:
            raise
        except Exception:
            raise ProviderIntentError("STATE_UNAVAILABLE") from None

    def select_checked(self, run_id, *, prepared_attempt_id, checked_request) -> ProviderIntent:
        _ids(run_id, prepared_attempt_id)
        _reference(checked_request, prepared_attempt_id, checked=True)
        try:
            intent = self.get(run_id, prepared_attempt_id=prepared_attempt_id)
            if intent.checked_request is not None and intent.checked_request != checked_request:
                raise ProviderIntentError("INTENT_CONFLICT")
            prepared = self._prepared(run_id, prepared_attempt_id)
            if prepared.manifest != intent.preparation:
                raise ProviderIntentError("INTENT_CONFLICT")
            root = self._output_root / run_id
            checked = read_checked_request(root, checked_request, run_id=run_id, attempt_id=prepared_attempt_id)
            if (checked.frozen.mode != intent.mode or checked.frozen.preparation_sha256 != intent.preparation.sha256
                    or checked.frozen.prompt != prepared.prompt or checked.source_kind != prepared.source_kind):
                raise ProviderIntentError("INTENT_CONFLICT")
            _match_fields(_read_record(root, checked_request), {"preparation": asdict(intent.preparation)})
            with self.coordinator._mutex:
                self._require_owner()
                with self._ledger._transaction() as connection:
                    _inspect(connection)
                    current = _matching(_find(connection, run_id, intent.key_sha256, prepared_attempt_id),
                                        run_id, intent.key_sha256, prepared_attempt_id, intent.mode)
                    if current is None or current.preparation != intent.preparation:
                        raise ProviderIntentError("INTENT_CONFLICT")
                    if current.checked_request is not None:
                        if current.checked_request != checked_request:
                            raise ProviderIntentError("INTENT_CONFLICT")
                        return current
                    connection.execute("UPDATE provider_intents SET checked_path=?,checked_sha256=?,checked_size=? "
                                       "WHERE run_id=? AND key_sha256=? AND checked_path IS NULL",
                                       (checked_request.path, checked_request.sha256, checked_request.size,
                                        run_id, intent.key_sha256))
                    return _matching(_find(connection, run_id, intent.key_sha256, prepared_attempt_id),
                                     run_id, intent.key_sha256, prepared_attempt_id, intent.mode)
        except ProviderIntentError:
            raise
        except Exception:
            raise ProviderIntentError("STATE_UNAVAILABLE") from None
