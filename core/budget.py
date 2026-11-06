"""Reserve integer USD cents from an operator-approved SQLite budget."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4


class BudgetError(ValueError):
    """A budget operation is invalid or has no server approval."""


class BudgetExceeded(BudgetError):
    """The run has reached its money, attempt, or time limit."""


class IdempotencyConflict(BudgetError):
    """An existing key belongs to a different frozen payload."""


@dataclass(frozen=True)
class Reservation:
    run_id: str
    attempt_id: str
    payload_sha256: str
    cost_cents: int
    state: str


class BudgetLedger:
    """Local single-host ledger. Parent directory must already be private and durable.

    Only trusted operator code may call approve_run. Do not expose approval or
    quoted costs as client-controlled fields. Reservations do not permit a paid
    POST yet; submission state and charge reconciliation are separate work.
    """

    def __init__(self, path: Path, *, clock=time.time):
        self.path = Path(path).absolute()
        self.clock = clock
        if (not self.path.parent.is_dir() or self.path.parent.is_symlink()
                or self.path.is_symlink() or self.path.parent.stat().st_mode & 0o077):
            raise BudgetError("Budget database needs a private existing parent directory.")
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS approved_runs (
                    run_id TEXT PRIMARY KEY,
                    approved_by TEXT NOT NULL,
                    limit_cents INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    cost_cents INTEGER NOT NULL,
                    started_unix REAL NOT NULL,
                    deadline_unix REAL NOT NULL,
                    last_checked_unix REAL NOT NULL,
                    time_stopped INTEGER NOT NULL DEFAULT 0 CHECK (time_stopped IN (0, 1))
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    run_id TEXT NOT NULL REFERENCES approved_runs(run_id),
                    key_sha256 TEXT NOT NULL,
                    attempt_id TEXT NOT NULL UNIQUE,
                    payload_sha256 TEXT NOT NULL,
                    cost_cents INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK (state = 'RESERVED'),
                    PRIMARY KEY (run_id, key_sha256)
                );
            """)
        # SQLite syncs database/WAL contents; persist newly created directory entries.
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def _now(self):
        now = self.clock()
        # Bound before isfinite so an enormous integer cannot overflow conversion.
        if (type(now) not in (int, float) or not 0 <= now <= 253_402_300_799
                or not math.isfinite(now)):
            raise BudgetError("Budget clock must provide a valid UTC timestamp.")
        return now

    @staticmethod
    def _run_id(run_id):
        if type(run_id) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id) is None:
            raise BudgetError("Run ID must contain 1 to 64 letters, digits, underscores, or hyphens.")

    def approve_run(self, run_id: str, *, approved_by: str, limit_cents: int,
                    max_attempts: int, cost_cents: int, max_elapsed_minutes: int = 180):
        """Record one operator approval and trusted quote; never increase it in place."""
        self._run_id(run_id)
        if type(approved_by) is not str or not approved_by.strip() or len(approved_by) > 200:
            raise BudgetError("Budget approval needs an operator identity.")
        for value, maximum in [(limit_cents, 1000), (max_attempts, 6),
                               (max_elapsed_minutes, 180), (cost_cents, limit_cents)]:
            if type(value) is not int or not 1 <= value <= maximum:
                raise BudgetError("Budget values must be positive integers within the P0 limits.")
        now = self._now()
        try:
            with self._transaction() as connection:
                connection.execute("INSERT INTO approved_runs (run_id, approved_by, limit_cents, max_attempts, "
                                   "cost_cents, started_unix, deadline_unix, last_checked_unix) "
                                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                   (run_id, approved_by, limit_cents, max_attempts, cost_cents,
                                    now, now + max_elapsed_minutes * 60, now))
        except sqlite3.IntegrityError:
            raise BudgetError("This run already has a server-side approval.") from None

    def reserve(self, run_id: str, key: str, payload_sha256: str) -> Reservation:
        """Atomically reserve one quoted charge; repeated keys return the same attempt.

        A repeated key with a different payload conflicts. Lookups still work
        after the deadline, but cannot create another reservation or paid POST.
        Raw idempotency keys are never stored.
        """
        self._run_id(run_id)
        if type(key) is not str or not key.strip() or len(key) > 512:
            raise BudgetError("Idempotency key must contain from 1 to 512 UTF-8 bytes.")
        try:
            key_bytes = key.encode("utf-8")
        except UnicodeEncodeError:
            raise BudgetError("Idempotency key must be valid UTF-8.") from None
        if len(key_bytes) > 512:
            raise BudgetError("Idempotency key must contain from 1 to 512 UTF-8 bytes.")
        if type(payload_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", payload_sha256) is None:
            raise BudgetError("Payload needs a lowercase SHA-256 hash.")
        key_sha256 = sha256(key_bytes).hexdigest()
        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM reservations WHERE run_id=? AND key_sha256=?",
                                          (run_id, key_sha256)).fetchone()
            if existing is not None:
                if existing["payload_sha256"] != payload_sha256:
                    raise IdempotencyConflict("Idempotency key belongs to a different payload.")
                return self._reservation(existing)
            run = connection.execute("SELECT * FROM approved_runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise BudgetError("Run has no server-side budget approval.")
            if run["time_stopped"]:
                raise BudgetExceeded("Run elapsed-time limit has been reached.")
            now = self._now()
            if now < run["last_checked_unix"]:
                raise BudgetError("Budget clock moved backwards; new reservations are blocked.")
            if now >= run["deadline_unix"]:
                connection.execute("UPDATE approved_runs SET time_stopped=1 WHERE run_id=?", (run_id,))
                reservation = None
            else:
                usage = connection.execute("SELECT count(*) AS attempts, coalesce(sum(cost_cents), 0) AS cents "
                                           "FROM reservations WHERE run_id=?", (run_id,)).fetchone()
                if usage["attempts"] >= run["max_attempts"] or usage["cents"] + run["cost_cents"] > run["limit_cents"]:
                    raise BudgetExceeded("Run money or attempt limit has been reached.")
                attempt_id = "att-" + uuid4().hex
                connection.execute("INSERT INTO reservations VALUES (?, ?, ?, ?, ?, 'RESERVED')",
                                   (run_id, key_sha256, attempt_id, payload_sha256, run["cost_cents"]))
                connection.execute("UPDATE approved_runs SET last_checked_unix=? WHERE run_id=?", (now, run_id))
                reservation = Reservation(run_id, attempt_id, payload_sha256, run["cost_cents"], "RESERVED")
        # Raise only after committing the stop, so an error cannot roll it back.
        if reservation is None:
            raise BudgetExceeded("Run elapsed-time limit has been reached.")
        return reservation

    @staticmethod
    def _reservation(row) -> Reservation:
        return Reservation(row["run_id"], row["attempt_id"], row["payload_sha256"], row["cost_cents"], row["state"])
