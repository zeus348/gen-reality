"""Commit before a paid create call, and never repeat an uncertain submission."""

from dataclasses import dataclass
import fcntl
import os
import re
import stat
from threading import RLock
from typing import Callable

from core.budget import BudgetError, BudgetExceeded, BudgetLedger


class SubmissionError(ValueError):
    """A submission cannot change state safely."""


@dataclass(frozen=True)
class Submission:
    attempt_id: str
    state: str
    provider_task_id: str | None = None
    evidence_sha256: str | None = None


class SubmissionCoordinator:
    """One active coordinator per private Unix ledger, with concurrent callers.

    Hold the process lock until all create calls finish. A new owner marks any
    abandoned SUBMITTING rows as UNKNOWN. This prevents local replay; it cannot
    prove whether a provider accepted a request before the connection failed.
    """

    def __init__(self, ledger: BudgetLedger):
        self.ledger = ledger
        self._mutex = RLock()
        self._lock = None
        descriptor = os.open(str(ledger.path) + ".submission.lock",
                             os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SubmissionError("Submission lock must be a regular file.")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SubmissionError("A submission coordinator already owns this ledger.") from None
            with ledger._connection() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS submission_jobs (
                        attempt_id TEXT PRIMARY KEY REFERENCES reservations(attempt_id),
                        state TEXT NOT NULL,
                        provider_task_id TEXT UNIQUE,
                        evidence_sha256 TEXT
                    )
                """)
            with ledger._transaction() as connection:
                connection.execute("UPDATE submission_jobs SET state='SUBMISSION_UNKNOWN' WHERE state='SUBMITTING'")
            self._lock = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def close(self):
        """Wait for the current create call before releasing ownership."""
        with self._mutex:
            if self._lock is not None:
                os.close(self._lock)
                self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def _require_open(self):
        if self._lock is None:
            raise SubmissionError("Submission coordinator is closed.")

    def get(self, attempt_id: str) -> Submission:
        self._require_open()
        with self.ledger._connection() as connection:
            row = connection.execute("SELECT * FROM submission_jobs WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is not None:
                return Submission(**dict(row))
            if connection.execute("SELECT 1 FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone() is None:
                raise SubmissionError("Submission has no matching reservation.")
        return Submission(attempt_id, "RESERVED")

    def _claim(self, attempt_id: str) -> bool:
        """Recheck authority and persist the one-way create boundary."""
        with self.ledger._transaction() as connection:
            if connection.execute("SELECT 1 FROM submission_jobs WHERE attempt_id=?", (attempt_id,)).fetchone():
                return False
            reservation = connection.execute("SELECT * FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone()
            if reservation is None:
                raise SubmissionError("Submission has no matching reservation.")
            run_id = reservation["run_id"]
            run = connection.execute("SELECT * FROM approved_runs WHERE run_id=?", (run_id,)).fetchone()
            now = self.ledger._now()
            if now < run["last_checked_unix"]:
                raise BudgetError("Budget clock moved backwards; submission is blocked.")
            if run["time_stopped"] or now >= run["deadline_unix"]:
                connection.execute("UPDATE approved_runs SET time_stopped=1 WHERE run_id=?", (run_id,))
                claimed = False
            else:
                attempts, cents, quote_exceeded = self.ledger._usage(connection, run_id)
                if quote_exceeded or cents > run["limit_cents"] or attempts > run["max_attempts"]:
                    raise BudgetExceeded("Run spending limits block submission.")
                if connection.execute("SELECT 1 FROM actual_charges WHERE attempt_id=?", (attempt_id,)).fetchone():
                    raise SubmissionError("A settled reservation cannot start a new submission.")
                connection.execute("INSERT INTO submission_jobs VALUES (?, 'SUBMITTING', NULL, NULL)", (attempt_id,))
                connection.execute("UPDATE approved_runs SET last_checked_unix=? WHERE run_id=?", (now, run_id))
                claimed = True
        if not claimed:
            raise BudgetExceeded("Run elapsed-time limit has been reached.")
        return True

    @staticmethod
    def _task_id(task_id):
        if type(task_id) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id) is None:
            raise SubmissionError("Provider task ID is invalid.")

    def submit_once(self, attempt_id: str, create_task: Callable[[], str]) -> Submission:
        """Call a trusted adapter once. The adapter must disable all create retries.

        Any exception or malformed response is uncertain and keeps the cost held.
        Process exits leave SUBMITTING for the next owner to recover as UNKNOWN.
        """
        with self._mutex:
            self._require_open()
            if not self._claim(attempt_id):
                return self.get(attempt_id)
            try:
                task_id = create_task()
                self._task_id(task_id)
                with self.ledger._transaction() as connection:
                    connection.execute("UPDATE submission_jobs SET state='QUEUED', provider_task_id=? WHERE attempt_id=?",
                                       (task_id, attempt_id))
            except Exception:
                # Even a local persistence failure after acceptance cannot permit replay.
                with self.ledger._transaction() as connection:
                    connection.execute("UPDATE submission_jobs SET state='SUBMISSION_UNKNOWN' WHERE attempt_id=?",
                                       (attempt_id,))
            return self.get(attempt_id)

    def record_verified_task(self, attempt_id: str, task_id: str, evidence_sha256: str) -> Submission:
        """A trusted operator may reconcile UNKNOWN using verified task evidence."""
        self._task_id(task_id)
        if type(evidence_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None:
            raise SubmissionError("Reconciliation needs a lowercase evidence SHA-256 hash.")
        with self._mutex:
            self._require_open()
            with self.ledger._transaction() as connection:
                row = connection.execute("SELECT * FROM submission_jobs WHERE attempt_id=?", (attempt_id,)).fetchone()
                if row is None or row["state"] != "SUBMISSION_UNKNOWN":
                    raise SubmissionError("Only an unknown submission can be reconciled.")
                connection.execute("UPDATE submission_jobs SET state='QUEUED', provider_task_id=?, evidence_sha256=? "
                                   "WHERE attempt_id=?", (task_id, evidence_sha256, attempt_id))
            return self.get(attempt_id)
