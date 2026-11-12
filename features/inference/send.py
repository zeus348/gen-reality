"""Recheck the saved request and authority at the final SDK send boundary."""

from dataclasses import asdict
from pathlib import Path

import httpx

from features.inference.frozen import submit_frozen
from features.inference.preflight import CheckedRequest, PreflightError, _match_fields, _read_record
from features.inference.runtime import RuntimeEvidence
from features.inference.submission import Submission, SubmissionCoordinator, SubmissionError


def submit_checked(coordinator: SubmissionCoordinator, checked: CheckedRequest,
                    output_root: Path, api_key: str, *, transport: httpx.BaseTransport | None = None,
                    runtime: RuntimeEvidence | None = None) -> Submission:
    """Send once through the coordinator using a verified, current check record.

    The trusted application supplies the check and the same provider account key
    used for uploads and account checks. Known or uncertain submissions return
    their existing state without another create or a freshness requirement.
    A failure after the durable claim remains UNKNOWN and keeps its reservation.
    """
    with coordinator._mutex:
        coordinator._require_open()
        coordinator.ledger._run_id(checked.run_id)
        coordinator.ledger._attempt_id(checked.attempt_id)
        if runtime is not None:
            runtime.require_identity(output_root, checked.run_id, checked.attempt_id)
        with coordinator.ledger._connection() as connection:
            row = connection.execute("SELECT run_id FROM reservations WHERE attempt_id=?", (checked.attempt_id,)).fetchone()
        if row is None or row["run_id"] != checked.run_id:
            raise SubmissionError("Checked request has no matching run reservation.")
        if coordinator.get(checked.attempt_id).state != "RESERVED":
            if runtime is not None:
                runtime.bind_existing(coordinator)
            return submit_frozen(coordinator, checked.attempt_id, checked.frozen, output_root, api_key,
                                 transport=transport, runtime=runtime)
        root = Path(output_root) / checked.run_id

        def verify_check():
            if not checked.evidence.path.startswith(f"attempts/{checked.attempt_id}/provider_checks/"):
                raise PreflightError("Request check belongs to another attempt.")
            _match_fields(_read_record(root, checked.evidence), {
                "schemaVersion": "1.4", "manifestType": "checked_provider_request", "runId": checked.run_id,
                "attemptId": checked.attempt_id, "sourceKind": checked.source_kind,
                "frozen": asdict(checked.frozen), "payloadSha256": checked.frozen.payload_sha256,
                "checkedAtUnix": checked.checked_unix, "validUntilUnix": checked.valid_until_unix,
                "spendingAuthorized": False})

        verify_check()
        if runtime is not None:
            runtime.bind_checked(coordinator, checked)
        checked.require_current(coordinator.ledger._now())

        def before_send():
            verify_check()
            coordinator.require_send_allowed(checked.attempt_id, check_current=checked.require_current)

        return submit_frozen(coordinator, checked.attempt_id, checked.frozen, output_root, api_key,
                             transport=transport, before_send=before_send, runtime=runtime)
