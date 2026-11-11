"""Record task observations without repeating a paid create call."""

from dataclasses import dataclass
import math
import re
from typing import Callable

from features.inference.submission import SubmissionCoordinator, SubmissionError


PROVIDER_STATUSES = ("PENDING", "THROTTLED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED")


@dataclass(frozen=True)
class TaskObservation:
    """A trusted GET result and the hash of its saved provider response."""

    task_id: str
    status: str
    evidence_sha256: str


@dataclass(frozen=True)
class PollResult:
    attempt_id: str
    state: str
    provider_status: str | None
    poll_status: str
    evidence_sha256: str | None


class TaskPoller:
    """Use the coordinator's ownership lock and append each polling outcome.

    The retrieval callback must use GET only and save its response evidence.
    A local timeout leaves the task state and held cost unchanged. Provider
    success is only an observation until raw video validation completes.
    """

    def __init__(self, coordinator: SubmissionCoordinator):
        self.coordinator = coordinator
        with coordinator._mutex:
            coordinator._require_open()
            with coordinator.ledger._connection() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS poll_events (
                        event_id INTEGER PRIMARY KEY,
                        attempt_id TEXT NOT NULL REFERENCES submission_jobs(attempt_id),
                        observed_unix REAL NOT NULL,
                        provider_status TEXT,
                        poll_status TEXT NOT NULL,
                        evidence_sha256 TEXT
                    )
                """)

    def get(self, attempt_id: str) -> PollResult:
        """Return local state and the last known provider observation."""
        with self.coordinator._mutex:
            job = self.coordinator.get(attempt_id)
            with self.coordinator.ledger._connection() as connection:
                row = connection.execute("SELECT * FROM poll_events WHERE attempt_id=? "
                                         "ORDER BY event_id DESC LIMIT 1", (attempt_id,)).fetchone()
            if row is None:
                return PollResult(attempt_id, job.state, None, "NOT_POLLED", None)
            return PollResult(attempt_id, job.state, row["provider_status"],
                              row["poll_status"], row["evidence_sha256"])

    def _record(self, previous: PollResult, state: str, provider_status: str | None,
                poll_status: str, evidence_sha256: str | None) -> PollResult:
        ledger = self.coordinator.ledger
        now = ledger._now()
        with ledger._transaction() as connection:
            connection.execute("UPDATE submission_jobs SET state=? WHERE attempt_id=?",
                               (state, previous.attempt_id))
            connection.execute("INSERT INTO poll_events "
                               "(attempt_id, observed_unix, provider_status, poll_status, evidence_sha256) "
                               "VALUES (?, ?, ?, ?, ?)",
                               (previous.attempt_id, now, provider_status, poll_status, evidence_sha256))
        return PollResult(previous.attempt_id, state, provider_status, poll_status, evidence_sha256)

    def poll_once(self, attempt_id: str, retrieve_task: Callable[[str], TaskObservation],
                  *, deadline_unix: float) -> PollResult:
        """Fetch the known task once, or record that the local deadline was reached.

        Call again with a new deadline to resume. Failed or canceled tasks remain
        terminal. Polling never records a final charge or releases a reservation.
        """
        if (type(deadline_unix) not in (int, float) or not 0 <= deadline_unix <= 253_402_300_799
                or not math.isfinite(deadline_unix)):
            raise ValueError("Polling deadline must be a valid UTC timestamp.")
        with self.coordinator._mutex:
            previous = self.get(attempt_id)
            if previous.state in ("SUCCEEDED", "FAILED", "CANCELED"):
                return previous
            job = self.coordinator.get(attempt_id)
            if previous.state not in ("QUEUED", "RUNNING") or job.provider_task_id is None:
                raise SubmissionError("Polling requires a known queued or running task.")
            if self.coordinator.ledger._now() >= deadline_unix:
                return self._record(previous, previous.state, previous.provider_status,
                                    "POLL_DEADLINE_REACHED", previous.evidence_sha256)
            try:
                observation = retrieve_task(job.provider_task_id)
                if (not isinstance(observation, TaskObservation) or observation.task_id != job.provider_task_id
                        or observation.status not in PROVIDER_STATUSES
                        or type(observation.evidence_sha256) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", observation.evidence_sha256) is None):
                    raise ValueError("Provider task observation is invalid.")
                if (previous.provider_status in ("SUCCEEDED", "FAILED", "CANCELLED")
                        and observation.status != previous.provider_status):
                    raise ValueError("Provider observation conflicts with its terminal status.")
            except TimeoutError:
                return self._record(previous, previous.state, previous.provider_status,
                                    "POLL_DEADLINE_REACHED", previous.evidence_sha256)
            except Exception:
                return self._record(previous, previous.state, previous.provider_status,
                                    "POLL_ERROR", previous.evidence_sha256)
            state = previous.state
            if observation.status == "RUNNING":
                state = "RUNNING"
            elif observation.status == "FAILED":
                state = "FAILED"
            elif observation.status == "CANCELLED":
                state = "CANCELED"
            return self._record(previous, state, observation.status, "OBSERVED", observation.evidence_sha256)
