"""Commit local success only after raw output and metadata are durable."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

from core.storage import ArtifactExistsError, StoredArtifact, publish_bytes, read_verified
from features.inference.polling import TaskPoller
from features.inference.submission import SubmissionCoordinator
from features.output.validate import validate_raw_video


class OutputError(ValueError):
    """Output cannot be recorded as a completed local task."""


@dataclass(frozen=True)
class CompletedOutput:
    run_id: str
    attempt_id: str
    provider_task_id: str
    provider_evidence_sha256: str
    raw: StoredArtifact
    metadata: StoredArtifact
    completed_unix: float


def _publish_or_verify(root: Path, path: str, contents: bytes) -> StoredArtifact:
    """Resume an interrupted publication only when the existing bytes match."""
    expected = StoredArtifact(path, sha256(contents).hexdigest(), len(contents))
    try:
        return publish_bytes(root, path, contents)
    except ArtifactExistsError:
        read_verified(root, expected)
        return expected


class OutputCompleter:
    """One trusted coordinator owns the ledger and application output root.

    The download callback receives the known task ID and saved success-evidence
    hash. It must download that task's output using GET only. This class checks
    content; it does not establish live-provider provenance or visual quality.
    """

    def __init__(self, coordinator: SubmissionCoordinator, output_root: Path):
        self.coordinator = coordinator
        self.output_root = Path(output_root)
        self.poller = TaskPoller(coordinator)
        with coordinator._mutex:
            coordinator._require_open()
            with coordinator.ledger._connection() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS completed_outputs (
                        attempt_id TEXT PRIMARY KEY REFERENCES submission_jobs(attempt_id),
                        provider_task_id TEXT NOT NULL,
                        provider_evidence_sha256 TEXT NOT NULL,
                        raw_path TEXT NOT NULL, raw_sha256 TEXT NOT NULL, raw_size INTEGER NOT NULL,
                        metadata_path TEXT NOT NULL, metadata_sha256 TEXT NOT NULL, metadata_size INTEGER NOT NULL,
                        completed_unix REAL NOT NULL
                    )
                """)

    def get(self, attempt_id: str) -> CompletedOutput | None:
        with self.coordinator._mutex:
            self.coordinator._require_open()
            with self.coordinator.ledger._connection() as connection:
                row = connection.execute("SELECT o.*, r.run_id FROM completed_outputs o "
                                         "JOIN reservations r USING (attempt_id) WHERE attempt_id=?",
                                         (attempt_id,)).fetchone()
            if row is None:
                return None
            job = self.coordinator.get(attempt_id)
            if job.state != "SUCCEEDED" or job.provider_task_id != row["provider_task_id"]:
                raise OutputError("Output record conflicts with the current task.")
            raw = StoredArtifact(row["raw_path"], row["raw_sha256"], row["raw_size"])
            metadata = StoredArtifact(row["metadata_path"], row["metadata_sha256"], row["metadata_size"])
            root = self.output_root / row["run_id"]
            read_verified(root, raw)
            read_verified(root, metadata)
            return CompletedOutput(row["run_id"], attempt_id, job.provider_task_id,
                                   row["provider_evidence_sha256"], raw, metadata, row["completed_unix"])

    def complete(self, attempt_id: str, download: Callable[[str, str], bytes]) -> CompletedOutput:
        """Validate, publish, then commit success and its artifact records together."""
        with self.coordinator._mutex:
            existing = self.get(attempt_id)
            if existing is not None:
                return existing
            job = self.coordinator.get(attempt_id)
            observed = self.poller.get(attempt_id)
            if (job.state not in ("QUEUED", "RUNNING") or observed.provider_status != "SUCCEEDED"
                    or job.provider_task_id is None or observed.evidence_sha256 is None):
                raise OutputError("Output completion requires a known task with provider success.")
            with self.coordinator.ledger._connection() as connection:
                run_id = connection.execute("SELECT run_id FROM reservations WHERE attempt_id=?",
                                            (attempt_id,)).fetchone()["run_id"]
            try:
                contents = download(job.provider_task_id, observed.evidence_sha256)
            except Exception:
                raise OutputError("Output download failed; resume the same task.") from None
            self.coordinator._require_open()
            video = validate_raw_video(contents)
            root = self.output_root / run_id
            prefix = f"attempts/{attempt_id}/output/"
            raw = _publish_or_verify(root, prefix + "raw.mp4", video.video_bytes)
            details = {
                "schemaVersion": "1.4", "manifestType": "validated_raw_output",
                "runId": run_id, "attemptId": attempt_id, "providerTaskId": job.provider_task_id,
                "raw": asdict(raw), "codec": video.codec, "width": video.width, "height": video.height,
                "durationSeconds": video.duration_seconds, "frameRate": video.frame_rate,
                "frameSeconds": video.frame_seconds, "audioPresent": video.audio_present,
                "rawContentValidation": "PASS", "visualEffectDemonstrated": "NOT_RUN",
                "automatedWorkflowComplete": "NOT_RUN",
            }
            metadata_bytes = (json.dumps(details, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            metadata = _publish_or_verify(root, prefix + "metadata.json", metadata_bytes)
            now = self.coordinator.ledger._now()
            with self.coordinator.ledger._transaction() as connection:
                current = connection.execute("SELECT * FROM submission_jobs WHERE attempt_id=?", (attempt_id,)).fetchone()
                latest = connection.execute("SELECT * FROM poll_events WHERE attempt_id=? ORDER BY event_id DESC LIMIT 1",
                                            (attempt_id,)).fetchone()
                if (current["state"] not in ("QUEUED", "RUNNING") or current["provider_task_id"] != job.provider_task_id
                        or latest["provider_status"] != "SUCCEEDED" or latest["evidence_sha256"] != observed.evidence_sha256):
                    raise OutputError("Provider task or success evidence changed during completion.")
                connection.execute("INSERT INTO completed_outputs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                   (attempt_id, job.provider_task_id, observed.evidence_sha256,
                                    raw.path, raw.sha256, raw.size, metadata.path, metadata.sha256, metadata.size, now))
                connection.execute("UPDATE submission_jobs SET state='SUCCEEDED' WHERE attempt_id=?", (attempt_id,))
            return CompletedOutput(run_id, attempt_id, job.provider_task_id, observed.evidence_sha256, raw, metadata, now)
