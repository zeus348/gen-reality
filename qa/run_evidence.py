"""Join captured ledger, archive, and experiment-clock facts without I/O."""

from collections import defaultdict
from dataclasses import dataclass
import math
import re

from core import run_audit, run_snapshot
from core.run_audit import AuditAssessment, AuditIssue, RunAudit
from core.run_start import MAX_ELAPSED_SECONDS, MAX_RUN_START_BYTES, RunStart
from core.storage import MAX_ARTIFACT_BYTES, StoredArtifact, _path_parts
from features.review.attempt_records import ExportAttemptFacts
from features.review.integrity import ExportArchiveIntegrity
from features.review.preparation import ExportPreparationIntegrity
from features.review.review_records import ExportReviewIntegrity
from features.review.scorecard_records import ExportScorecardIntegrity


MAX_ARCHIVES = run_snapshot.MAX_TABLE_ROWS
_JOB_STATES = {"SUBMITTING", "SUBMISSION_UNKNOWN", "QUEUED", "RUNNING", "FAILED", "CANCELED", "SUCCEEDED"}


class RunEvidenceError(ValueError):
    """The audit needs bounded immutable values from the existing readers."""


@dataclass(frozen=True)
class AttemptEvidence:
    reservation_index: int
    attempt_id: str | None
    reservation_state: str | None
    submission_state: str | None
    provider_task_id: str | None
    archive: StoredArtifact | None
    inventory: StoredArtifact | None
    links: AuditAssessment


@dataclass(frozen=True)
class RunEvidenceAudit:
    run_id: str
    snapshot_sha256: str | None
    run_start: StoredArtifact | None
    ledger: RunAudit | None
    archive_links: AuditAssessment
    archive_coverage: AuditAssessment
    experiment_window: AuditAssessment
    attempts: tuple[AttemptEvidence, ...]
    issues: tuple[AuditIssue, ...]


def _matches(value, pattern):
    return type(value) is str and re.fullmatch(pattern, value) is not None


def _attempt(value):
    return _matches(value, r"att-[0-9a-f]{32}")


def _task(value):
    return _matches(value, r"[A-Za-z0-9_-]{1,128}")


def _time(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 253_402_300_799


def _reference(value, *, limit=MAX_ARTIFACT_BYTES):
    if (type(value) is not StoredArtifact or type(value.size) is not int
            or not 0 < value.size <= limit or not _matches(value.sha256, r"[0-9a-f]{64}")):
        raise RunEvidenceError("Artifact references must have bounded paths, sizes, and hashes.")
    _path_parts(value.path)


def _archive_references(facts):
    """Check the immutable input shape, without repeating archive verification."""
    if (type(facts) is not ExportAttemptFacts or not _matches(facts.run_id, r"[A-Za-z0-9_-]{1,64}")
            or not _attempt(facts.attempt_id) or facts.mode not in ("A", "B")
            or facts.source_kind_declared not in ("clean_camera", "synthetic_fixture")
            or type(facts.sdk_version) is not str or len(facts.sdk_version) > 32
            or not _matches(facts.sdk_version, r"[0-9]+\.[0-9]+\.[0-9]+")
            or not _task(facts.provider_task_id) or not _matches(facts.payload_sha256, r"[0-9a-f]{64}")
            or not _matches(facts.request_body_sha256, r"[0-9a-f]{64}")
            or type(facts.reserved_cents) is not int or not 1 <= facts.reserved_cents <= 1000
            or not _time(facts.completed_unix) or type(facts.graph) is not ExportScorecardIntegrity
            or type(facts.graph.review) is not ExportReviewIntegrity
            or type(facts.graph.review.preparation) is not ExportPreparationIntegrity
            or type(facts.graph.review.preparation.integrity) is not ExportArchiveIntegrity):
        raise RunEvidenceError("Archive facts must use the existing bounded immutable contract.")
    integrity = facts.graph.review.preparation.integrity
    metadata = facts.graph.review.raw_metadata
    for record in (integrity.archive, integrity.inventory, metadata, facts.raw_output,
                   facts.request_check, facts.request, facts.frozen_request, facts.provider_success):
        _reference(record)
    return integrity.archive, integrity.inventory, metadata


def _note(issues, category, code, attempt=None, *, missing=False, sequence=None, event=None):
    issues.append(AuditIssue(category, code, missing, attempt if _attempt(attempt) else None, sequence, event))


def _assessment(issues, *, default="PASS", shared=None):
    missing = any(issue.missing for issue in issues) or shared is not None and shared.coverage == "INCOMPLETE"
    failed = any(not issue.missing for issue in issues) or shared is not None and shared.verdict == "FAIL"
    verdict = "FAIL" if failed else "BLOCKED" if missing else default
    return AuditAssessment(verdict, "INCOMPLETE" if missing else "COMPLETE")


def _groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.attempt_id].append(row)
    return grouped


def _one_row(groups, attempt, table, missing_tables, issues):
    rows = groups.get(attempt, ())
    if len(rows) != 1:
        _note(issues, "archive_links", table.upper() + ("_MISSING" if not rows else "_DUPLICATE"), attempt,
              missing=not rows and table in missing_tables)
        return None
    return rows[0]


def _check_archive(facts, metadata, snapshot, reservations, jobs, completed, issues):
    attempt = facts.attempt_id
    reservation = _one_row(reservations, attempt, "reservations", snapshot.missing_tables, issues)
    job = _one_row(jobs, attempt, "submission_jobs", snapshot.missing_tables, issues)
    output = _one_row(completed, attempt, "completed_outputs", snapshot.missing_tables, issues)
    if reservation is not None:
        for code, actual, expected in (
            ("ARCHIVE_RESERVATION_RUN_MISMATCH", facts.run_id, reservation.run_id),
            ("ARCHIVE_PAYLOAD_MISMATCH", facts.payload_sha256, reservation.payload_sha256),
            ("ARCHIVE_QUOTE_MISMATCH", facts.reserved_cents, reservation.cost_cents),
        ):
            if actual != expected:
                _note(issues, "archive_links", code, attempt)
    if job is not None:
        if job.state != "SUCCEEDED":
            _note(issues, "archive_links", "ARCHIVE_JOB_STATE_MISMATCH", attempt)
        if facts.provider_task_id != job.provider_task_id:
            _note(issues, "archive_links", "ARCHIVE_JOB_TASK_MISMATCH", attempt)
    if output is not None:
        for code, actual, expected in (
            ("ARCHIVE_OUTPUT_TASK_MISMATCH", facts.provider_task_id, output.provider_task_id),
            ("ARCHIVE_SUCCESS_MISMATCH", facts.provider_success.sha256, output.provider_evidence_sha256),
            ("ARCHIVE_RAW_MISMATCH", facts.raw_output,
             StoredArtifact(output.raw_path, output.raw_sha256, output.raw_size)),
            ("ARCHIVE_METADATA_MISMATCH", metadata,
             StoredArtifact(output.metadata_path, output.metadata_sha256, output.metadata_size)),
            ("ARCHIVE_COMPLETION_TIME_MISMATCH", facts.completed_unix, output.completed_unix),
        ):
            if actual != expected:
                _note(issues, "archive_links", code, attempt)


def _check_experiment(run_id, snapshot, started, ledger, issues):
    category = "experiment_window"
    if ledger is None:
        _note(issues, category, "LEDGER_MISSING", missing=True)
    else:
        if ledger.elapsed.verdict == "FAIL":
            _note(issues, category, "LEDGER_AUTHORITY_INVALID")
        if ledger.elapsed.coverage == "INCOMPLETE":
            _note(issues, category, "LEDGER_AUTHORITY_INCOMPLETE", missing=True)
    if started is None:
        _note(issues, category, "EXPERIMENT_START_MISSING", missing=True)
        return
    if (started.run_id != run_id or started.record.path != "run-start.json"
            or started.deadline_unix != started.started_unix + MAX_ELAPSED_SECONDS):
        _note(issues, category, "EXPERIMENT_START_INVALID")
        return
    if snapshot is None:
        return
    if snapshot.run_id != run_id:
        _note(issues, category, "LEDGER_RUN_MISMATCH")
    for approval in snapshot.approved_runs:
        if not (started.started_unix <= approval.started_unix < approval.deadline_unix <= started.deadline_unix):
            _note(issues, category, "APPROVAL_OUTSIDE_EXPERIMENT")

    def check_time(value, attempt, *, sequence=None, event=None):
        if value is None:
            _note(issues, category, "AUTHORITY_TIME_MISSING", attempt, missing=True, sequence=sequence, event=event)
        elif not _time(value) or not started.started_unix <= value < started.deadline_unix:
            _note(issues, category, "AUTHORITY_OUTSIDE_EXPERIMENT", attempt, sequence=sequence, event=event)

    for row in snapshot.run_events:
        if row.event_kind in ("APPROVAL_RECORDED", "RESERVATION_RECORDED"):
            check_time(row.recorded_unix, row.attempt_id, sequence=row.sequence)
    for row in snapshot.submission_events:
        if row.event_kind in ("CLAIMED", "SEND_GATE_PASSED"):
            check_time(row.recorded_unix, row.attempt_id, event=row.event_id)


def audit_run_evidence(*, run_id: str, snapshot: run_snapshot.RunSnapshot | None,
                       run_start: RunStart | None, archives: tuple[ExportAttemptFacts, ...] = ()) -> RunEvidenceAudit:
    """Compare reader-produced facts without claiming their external origin.

    The caller selects the private ledger/start and retains archive references.
    This function performs no reads, writes, provider calls, or clock queries.
    Runtime receipts, recorded tests, human approval, and outcomes remain separate.
    Missing inputs remain explicit; known ledger failures are never replaced by
    missing archive or clock coverage. Malformed API values raise a safe error.
    """
    try:
        if (not _matches(run_id, r"[A-Za-z0-9_-]{1,64}") or type(archives) is not tuple
                or len(archives) > MAX_ARCHIVES or snapshot is not None and type(snapshot) is not run_snapshot.RunSnapshot):
            raise RunEvidenceError("Audit inputs must use the fixed immutable contract.")
        references = tuple(_archive_references(facts) for facts in archives)
        if run_start is not None:
            if (type(run_start) is not RunStart or not _matches(run_start.run_id, r"[A-Za-z0-9_-]{1,64}")
                    or not _time(run_start.started_unix) or not _time(run_start.deadline_unix)):
                raise RunEvidenceError("Run start must contain bounded immutable clock facts.")
            _reference(run_start.record, limit=MAX_RUN_START_BYTES)
        ledger = run_audit.audit_run(snapshot) if snapshot is not None else None
        issues = []
        selected = _groups(archives)
        for attempt, choices in selected.items():
            if len(choices) != 1:
                _note(issues, "archive_links", "ARCHIVE_SELECTION_DUPLICATE", attempt)
        for facts in archives:
            if facts.run_id != run_id:
                _note(issues, "archive_links", "ARCHIVE_RUN_MISMATCH", facts.attempt_id)
        attempts = []
        if snapshot is None:
            _note(issues, "archive_coverage", "LEDGER_MISSING", missing=True)
            if archives:
                _note(issues, "archive_links", "LEDGER_MISSING", missing=True)
        else:
            if snapshot.run_id != run_id:
                _note(issues, "archive_links", "LEDGER_RUN_MISMATCH")
            reservations = _groups(snapshot.reservations)
            jobs, completed = _groups(snapshot.submission_jobs), _groups(snapshot.completed_outputs)
            for table in ("reservations", "submission_jobs", "completed_outputs"):
                if table in snapshot.missing_tables:
                    _note(issues, "archive_coverage", "TABLE_MISSING:" + table, missing=True)
            for facts, (_, _, metadata) in zip(archives, references):
                _check_archive(facts, metadata, snapshot, reservations, jobs, completed, issues)
            for row in snapshot.completed_outputs:
                if row.attempt_id not in selected:
                    _note(issues, "archive_coverage", "COMPLETED_ARCHIVE_MISSING", row.attempt_id, missing=True)

            # Keep every captured reservation, including duplicate and invalid rows.
            by_attempt, shared = defaultdict(list), []
            for issue in issues:
                if issue.attempt_id is None:
                    shared.append(issue)
                else:
                    by_attempt[issue.attempt_id].append(issue)
            selected_refs = {facts.attempt_id: refs for facts, refs in zip(archives, references)
                             if len(selected[facts.attempt_id]) == 1}
            shared_assessment = _assessment(shared)
            for index, row in enumerate(snapshot.reservations):
                attempt = row.attempt_id if _attempt(row.attempt_id) else None
                job_rows = jobs.get(row.attempt_id, ())
                job = job_rows[0] if len(job_rows) == 1 else None
                refs = selected_refs.get(row.attempt_id)
                local = by_attempt.get(attempt, [])
                attempts.append(AttemptEvidence(
                    index, attempt, row.state if row.state == "RESERVED" else None,
                    job.state if job is not None and job.state in _JOB_STATES else None,
                    job.provider_task_id if job is not None and _task(job.provider_task_id) else None,
                    refs[0] if refs else None, refs[1] if refs else None,
                    _assessment(local, default="PASS" if refs else "NOT_RUN", shared=shared_assessment),
                ))
        _check_experiment(run_id, snapshot, run_start, ledger, issues)
        return RunEvidenceAudit(
            run_id, ledger.snapshot_sha256 if ledger else None, run_start.record if run_start else None, ledger,
            _assessment([issue for issue in issues if issue.category == "archive_links"],
                        default="PASS" if archives else "NOT_RUN"),
            _assessment([issue for issue in issues if issue.category == "archive_coverage"]),
            _assessment([issue for issue in issues if issue.category == "experiment_window"]),
            tuple(attempts), tuple(issues),
        )
    except Exception:
        raise RunEvidenceError("Run evidence could not be assessed from the supplied immutable values.") from None
