"""Join typed attempt facts to complete ledger rows without reading new evidence."""

from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
import socket
import sqlite3
import time
import unittest
from unittest.mock import patch

from core import run_audit as ledger_audit
from core import run_snapshot as captured
from core.budget import BudgetLedger
from core.run_start import RunStart
from core.storage import StoredArtifact
from features.review.attempt_records import ExportAttemptFacts
from features.review.integrity import ExportArchiveIntegrity
from features.review.preparation import ExportPreparationIntegrity
from features.review.review_records import ExportReviewIntegrity
from features.review.scorecard_records import ExportScorecardIntegrity
from qa import run_evidence as evidence
import test_run_audit as fixtures


RUN, ATTEMPT_A, ATTEMPT_B = fixtures.RUN, fixtures.ATTEMPT_A, fixtures.ATTEMPT_B
_AUDIT_LEDGER = ledger_audit.audit_run
PASS = ledger_audit.AuditAssessment("PASS", "COMPLETE")
NOT_RUN = ledger_audit.AuditAssessment("NOT_RUN", "COMPLETE")
BLOCKED = ledger_audit.AuditAssessment("BLOCKED", "INCOMPLETE")


def two_completed_attempts():
    first = fixtures.complete_snapshot()
    approved = first.approved_runs[0]._replace(last_checked_unix=1009.0)
    reserved = first.reservations[0]._replace(attempt_id=ATTEMPT_B, key_sha256="6" * 64, payload_sha256="7" * 64)
    submission = tuple(row._replace(event_id=row.event_id + 3, attempt_id=ATTEMPT_B,
        recorded_unix=row.recorded_unix + 6, provider_task_id="task-b" if row.provider_task_id else None)
        for row in first.submission_events)
    journal = (captured.RunEventRow(6, RUN, ATTEMPT_B, "RESERVATION_RECORDED", 1007.0, None,
                                   fixtures.record_json(fixtures.reservation_record(reserved))),) + tuple(
        captured.RunEventRow(index + 7, RUN, ATTEMPT_B, "SUBMISSION_EVENT", row.recorded_unix,
                             row.event_id, None) for index, row in enumerate(submission))
    completed = first.completed_outputs[0]._replace(attempt_id=ATTEMPT_B, provider_task_id="task-b",
        provider_evidence_sha256="f" * 64, raw_path=f"attempts/{ATTEMPT_B}/output/raw.mp4", raw_sha256="8" * 64,
        metadata_path=f"attempts/{ATTEMPT_B}/output/metadata.json", metadata_sha256="9" * 64, completed_unix=1012.0)
    return fixtures.resign(first, approved_runs=(approved,), reservations=first.reservations + (reserved,),
        submission_jobs=first.submission_jobs + (captured.SubmissionJobRow(ATTEMPT_B, "SUCCEEDED", "task-b", None),),
        submission_events=first.submission_events + submission, run_events=first.run_events + journal,
        poll_events=first.poll_events + (captured.PollEventRow(2, ATTEMPT_B, 1011.0, "SUCCEEDED", "OBSERVED", "f" * 64),),
        completed_outputs=first.completed_outputs + (completed,))


def archive_facts(value, index, mode):
    reservation, output = value.reservations[index], value.completed_outputs[index]
    prefix = f"attempts/{reservation.attempt_id}/"
    archive_hash, inventory_hash = ("a", "b") if mode == "A" else ("c", "d")
    archive = StoredArtifact(prefix + f"export/archives/{archive_hash * 64}.zip", archive_hash * 64, 1200)
    inventory = StoredArtifact(prefix + f"export/inventories/{inventory_hash * 64}.json", inventory_hash * 64, 300)
    archived = ExportArchiveIntegrity(archive, inventory, 20, 900)
    preparation = ExportPreparationIntegrity(archived, StoredArtifact(prefix + "manifest.json", "1" * 64, 100),
                                            "synthetic_fixture")
    reviewed = ExportReviewIntegrity(preparation, StoredArtifact(prefix + "review/manifest.json", "2" * 64, 200),
        StoredArtifact(output.metadata_path, output.metadata_sha256, output.metadata_size))
    graph = ExportScorecardIntegrity(reviewed, "NOT_SELECTED", None, None, None)
    return ExportAttemptFacts(value.run_id, reservation.attempt_id, mode, "synthetic_fixture", "3.3.0",
        output.provider_task_id, reservation.payload_sha256, "3" * 64, reservation.cost_cents, output.completed_unix,
        StoredArtifact(output.raw_path, output.raw_sha256, output.raw_size),
        StoredArtifact(prefix + f"provider_checks/{'4' * 64}.json", "4" * 64, 250),
        StoredArtifact(prefix + "provider/request.json", "3" * 64, 300),
        StoredArtifact(prefix + "provider/frozen_request.json", reservation.payload_sha256, 400),
        StoredArtifact(f"provider_responses/{output.provider_evidence_sha256}.json", output.provider_evidence_sha256, 500), graph)


class RunEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.value = two_completed_attempts()
        self.start = RunStart(RUN, 1000.0, 11800.0, StoredArtifact("run-start.json", "0" * 64, 200))
        self.archives = (archive_facts(self.value, 0, "A"), archive_facts(self.value, 1, "B"))

    def assess(self, **changes):
        arguments = {"run_id": RUN, "snapshot": self.value, "run_start": self.start, "archives": self.archives} | changes
        return evidence.audit_run_evidence(**arguments)

    @staticmethod
    def change_approval(value, **changes):
        approved = value.approved_runs[0]._replace(**changes)
        events = tuple(row._replace(record_json=fixtures.record_json(fixtures.approval_record(approved)))
                       if row.event_kind == "APPROVAL_RECORDED" else row for row in value.run_events)
        return fixtures.resign(value, approved_runs=(approved,), run_events=events)

    @staticmethod
    def change_authority_time(value, kind, timestamp):
        changed_ids = {row.event_id for row in value.submission_events if row.event_kind == kind}
        submission = tuple(row._replace(recorded_unix=timestamp) if row.event_id in changed_ids else row
                           for row in value.submission_events)
        events = tuple(row._replace(recorded_unix=timestamp)
                       if row.event_kind == kind or row.submission_event_id in changed_ids else row
                       for row in value.run_events)
        return fixtures.resign(value, submission_events=submission, run_events=events)

    def test_complete_two_mode_join_preserves_exact_frozen_ledger_without_io(self):
        expected_ledger = _AUDIT_LEDGER(self.value)
        self.assertEqual(expected_ledger.record_consistency, PASS)
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No database")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("No file reads")), \
                patch.object(Path, "write_bytes", side_effect=AssertionError("No writes")), \
                patch.object(socket, "socket", side_effect=AssertionError("No network")), \
                patch.object(time, "time", side_effect=AssertionError("No clock")), \
                patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No ledger writer")):
            result = self.assess()
        self.assertIsInstance(result, evidence.RunEvidenceAudit)
        self.assertEqual((result.run_id, result.snapshot_sha256, result.run_start),
                         (RUN, self.value.sha256, self.start.record))
        self.assertEqual(result.ledger, expected_ledger)
        self.assertEqual((result.archive_links, result.archive_coverage, result.experiment_window), (PASS, PASS, PASS))
        self.assertEqual(result.issues, ())
        expected = tuple(evidence.AttemptEvidence(index, fact.attempt_id, "RESERVED", "SUCCEEDED",
            fact.provider_task_id, fact.graph.review.preparation.integrity.archive,
            fact.graph.review.preparation.integrity.inventory, PASS) for index, fact in enumerate(self.archives))
        self.assertEqual(result.attempts, expected)
        self.assertIs(type(result.attempts), tuple)
        with self.assertRaises(FrozenInstanceError):
            result.run_id = "other"
        with self.assertRaises(FrozenInstanceError):
            result.attempts[0].links.verdict = "FAIL"
        with self.assertRaises(FrozenInstanceError):
            result.attempts[0].archive.sha256 = "f" * 64
        with self.assertRaises(FrozenInstanceError):
            result.ledger.current_limits.verdict = "FAIL"
        for field in ("visualEffectDemonstrated", "automatedWorkflowComplete", "releaseClassification",
                      "providerProvenance", "humanProvenance"):
            self.assertNotIn(field, asdict(result))

    def test_missing_snapshot_start_and_history_remain_incomplete(self):
        absent = self.assess(snapshot=None, run_start=None, archives=())
        self.assertIsNone(absent.ledger)
        self.assertIsNone(absent.snapshot_sha256)
        self.assertIsNone(absent.run_start)
        self.assertEqual(absent.attempts, ())
        self.assertEqual(absent.archive_coverage, BLOCKED)
        self.assertEqual(absent.experiment_window, BLOCKED)
        missing_start = self.assess(run_start=None)
        self.assertEqual(missing_start.experiment_window, BLOCKED)
        self.assertEqual(missing_start.ledger, _AUDIT_LEDGER(self.value))
        self.assertEqual(missing_start.archive_links, PASS)
        late_gate = self.change_authority_time(self.value, "SEND_GATE_PASSED", 11800.0)
        invalid_ledger = _AUDIT_LEDGER(late_gate)
        self.assertEqual(invalid_ledger.elapsed.verdict, "FAIL")
        result = self.assess(snapshot=late_gate, run_start=None)
        self.assertEqual(result.ledger, invalid_ledger)
        self.assertEqual(result.experiment_window, ledger_audit.AuditAssessment("FAIL", "INCOMPLETE"))
        for table in ("run_events", "submission_events", "approved_runs"):
            value = fixtures.resign(self.value, **{table: ()}, missing_tables=(table,))
            with self.subTest(table=table):
                result = self.assess(snapshot=value)
                self.assertEqual(result.experiment_window.coverage, "INCOMPLETE")
                self.assertEqual(result.ledger, _AUDIT_LEDGER(value))

    def test_all_noncompleted_reservations_remain_visible_without_archive_gaps(self):
        states = (None, "SUBMITTING", "SUBMISSION_UNKNOWN", "QUEUED", "RUNNING", "FAILED")
        value = fixtures.reservations_snapshot(len(states))
        jobs = tuple(captured.SubmissionJobRow(row.attempt_id, state,
            None if state in ("SUBMITTING", "SUBMISSION_UNKNOWN") else f"task-{index}", None)
            for index, (row, state) in enumerate(zip(value.reservations, states)) if state is not None)
        value = fixtures.resign(value, submission_jobs=jobs)
        result = self.assess(snapshot=value, archives=())
        self.assertEqual(result.archive_links, NOT_RUN)
        self.assertEqual(result.archive_coverage, PASS)
        self.assertEqual(len(result.attempts), len(states))
        for index, (row, state) in enumerate(zip(result.attempts, states)):
            self.assertEqual((row.reservation_index, row.attempt_id, row.reservation_state, row.submission_state),
                             (index, value.reservations[index].attempt_id, "RESERVED", state))
            self.assertEqual(row.links, NOT_RUN)
            self.assertIsNone(row.archive)
            self.assertIsNone(row.inventory)
        unknown = value.reservations[2].attempt_id
        self.assertIn(unknown, result.ledger.unknown_attempt_ids)
        self.assertEqual(result.ledger.held_reserve_cents, 840)
        canceled = fixtures.resign(value, submission_jobs=jobs[:-1] + (jobs[-1]._replace(state="CANCELED"),))
        self.assertEqual(self.assess(snapshot=canceled, archives=()).attempts[-1].links, NOT_RUN)

    def test_missing_completed_archives_block_coverage_and_the_affected_row(self):
        result = self.assess(archives=self.archives[:1])
        self.assertEqual(result.archive_links, PASS)
        self.assertEqual(result.archive_coverage, BLOCKED)
        self.assertEqual(result.attempts[0].links, PASS)
        self.assertEqual(result.attempts[1].links, BLOCKED)
        self.assertIsNone(result.attempts[1].archive)
        empty = self.assess(archives=())
        self.assertEqual(empty.archive_links, NOT_RUN)
        self.assertEqual(empty.archive_coverage, BLOCKED)
        self.assertTrue(all(row.links == BLOCKED for row in empty.attempts))

    def test_duplicate_rows_archives_and_unreserved_archives_are_not_collapsed(self):
        for table in ("reservations", "submission_jobs", "completed_outputs"):
            rows = getattr(self.value, table)
            value = fixtures.resign(self.value, **{table: rows + rows[:1]})
            with self.subTest(table=table):
                result = self.assess(snapshot=value)
                self.assertEqual(result.archive_links.verdict, "FAIL")
                self.assertEqual(result.ledger, _AUDIT_LEDGER(value))
                self.assertEqual(len(result.attempts), len(value.reservations))
                self.assertEqual([row.reservation_index for row in result.attempts], list(range(len(value.reservations))))
                self.assertTrue(all(row.links.verdict == "FAIL" for row in result.attempts if row.attempt_id == ATTEMPT_A))
        duplicated = self.assess(archives=self.archives + self.archives[:1])
        self.assertEqual(duplicated.archive_links.verdict, "FAIL")
        self.assertEqual(len(duplicated.attempts), 2)
        single = fixtures.complete_snapshot()
        unreserved = self.assess(snapshot=single)
        self.assertEqual(unreserved.archive_links.verdict, "FAIL")
        self.assertEqual(len(unreserved.attempts), 1)
        self.assertEqual(unreserved.attempts[0].attempt_id, ATTEMPT_A)

    def test_every_archive_join_field_matches_the_selected_ledger_rows(self):
        first = self.archives[0]
        changed_success = replace(first.provider_success, sha256="f" * 64,
                                  path=f"provider_responses/{'f' * 64}.json")
        changes = [replace(first, run_id="other-run"), replace(first, provider_task_id="other-task"),
            replace(first, payload_sha256="f" * 64, frozen_request=replace(first.frozen_request, sha256="f" * 64)),
            replace(first, reserved_cents=141), replace(first, completed_unix=first.completed_unix + 1),
            replace(first, provider_success=changed_success)]
        for field, value in (("path", f"attempts/{ATTEMPT_A}/output/other.mp4"), ("sha256", "f" * 64), ("size", 101)):
            changes.append(replace(first, raw_output=replace(first.raw_output, **{field: value})))
        for field, value in (("path", f"attempts/{ATTEMPT_A}/output/other.json"), ("sha256", "f" * 64), ("size", 201)):
            metadata = replace(first.graph.review.raw_metadata, **{field: value})
            graph = replace(first.graph, review=replace(first.graph.review, raw_metadata=metadata))
            changes.append(replace(first, graph=graph))
        for index, changed in enumerate(changes):
            with self.subTest(field=index):
                result = self.assess(archives=(changed, self.archives[1]))
                self.assertEqual(result.archive_links.verdict, "FAIL")
                self.assertEqual(result.attempts[0].links.verdict, "FAIL")
                self.assertEqual(result.ledger, _AUDIT_LEDGER(self.value))
        other_job = self.value.submission_jobs[0]._replace(provider_task_id="other-task")
        value = fixtures.resign(self.value, submission_jobs=(other_job, self.value.submission_jobs[1]))
        self.assertEqual(self.assess(snapshot=value).archive_links.verdict, "FAIL")
        for table in ("submission_jobs", "completed_outputs"):
            value = fixtures.resign(self.value, **{table: ()}, missing_tables=(table,))
            with self.subTest(missing_table=table):
                self.assertEqual(self.assess(snapshot=value).archive_links, BLOCKED)

    def test_run_start_identity_duration_and_approval_containment_are_checked(self):
        changes = [replace(self.start, run_id="other-run"), replace(self.start, deadline_unix=11799.0),
                   replace(self.start, deadline_unix=11801.0),
                   replace(self.start, started_unix=1001.0, deadline_unix=11801.0),
                   replace(self.start, started_unix=999.0, deadline_unix=11799.0)]
        for changed in changes:
            with self.subTest(start=changed):
                self.assertEqual(self.assess(run_start=changed).experiment_window.verdict, "FAIL")
        value = self.change_approval(self.value, deadline_unix=11799.0)
        start = replace(self.start, started_unix=999.0, deadline_unix=11799.0)
        self.assertEqual(self.assess(snapshot=value, run_start=start).experiment_window, PASS)

    def test_authority_times_include_start_and_exclude_the_exact_deadline(self):
        value = fixtures.complete_snapshot()
        for timestamp in (1000.0, 11799.999999):
            changed = value
            for kind in ("RESERVATION_RECORDED", "CLAIMED", "SEND_GATE_PASSED"):
                changed = self.change_authority_time(changed, kind, timestamp)
            changed = self.change_approval(changed, last_checked_unix=max(1003.0, timestamp))
            with self.subTest(valid=timestamp):
                self.assertEqual(self.assess(snapshot=changed, archives=self.archives[:1]).experiment_window, PASS)
        for kind in ("APPROVAL_RECORDED", "RESERVATION_RECORDED", "CLAIMED", "SEND_GATE_PASSED"):
            for timestamp in (999.999999, 11800.0):
                changed = self.change_authority_time(value, kind, timestamp)
                with self.subTest(kind=kind, invalid=timestamp):
                    self.assertEqual(self.assess(snapshot=changed, archives=self.archives[:1]).experiment_window.verdict, "FAIL")
                missing_approval = fixtures.resign(changed, approved_runs=(), missing_tables=("approved_runs",))
                with self.subTest(kind=kind, invalid=timestamp, approval_missing=True):
                    baseline = _AUDIT_LEDGER(missing_approval)
                    self.assertEqual(baseline.elapsed, BLOCKED)
                    result = self.assess(snapshot=missing_approval, archives=self.archives[:1])
                    self.assertEqual(result.ledger, baseline)
                    self.assertEqual(result.experiment_window, ledger_audit.AuditAssessment("FAIL", "INCOMPLETE"))
                    self.assertTrue(any(issue.category == "experiment_window"
                                        and issue.code == "AUTHORITY_OUTSIDE_EXPERIMENT" and not issue.missing
                                        for issue in result.issues))

    def test_missing_authority_times_are_incomplete_without_erasing_core_findings(self):
        for kind in ("APPROVAL_RECORDED", "RESERVATION_RECORDED", "CLAIMED", "SEND_GATE_PASSED"):
            changed = self.change_authority_time(self.value, kind, None)
            with self.subTest(kind=kind):
                result = self.assess(snapshot=changed)
                self.assertEqual(result.experiment_window, BLOCKED)
                self.assertEqual(result.ledger, _AUDIT_LEDGER(changed))

    def test_late_observations_and_null_charge_time_do_not_extend_spending_checks(self):
        outputs = tuple(row._replace(completed_unix=20000.0 + index) for index, row in enumerate(self.value.completed_outputs))
        polls = tuple(row._replace(observed_unix=19000.0 + index) for index, row in enumerate(self.value.poll_events))
        value = fixtures.resign(self.value, completed_outputs=outputs, poll_events=polls)
        value = self.change_authority_time(value, "TASK_RECORDED", 18000.0)
        charge = captured.ActualChargeRow(ATTEMPT_A, 140, "5" * 64)
        event = captured.RunEventRow(10, RUN, ATTEMPT_A, "CHARGE_RECORDED", None, None,
                                    fixtures.record_json({"actual_cents": 140, "evidence_sha256": "5" * 64}))
        value = fixtures.resign(value, actual_charges=(charge,), run_events=value.run_events + (event,))
        archives = tuple(archive_facts(value, index, mode) for index, mode in enumerate(("A", "B")))
        first = archives[0]
        # The ledger has no request-body hash or private response byte size to compare.
        first = replace(first, request_body_sha256="f" * 64, request=replace(first.request, sha256="f" * 64),
                        provider_success=replace(first.provider_success, size=501))
        result = self.assess(snapshot=value, archives=(first, archives[1]))
        self.assertEqual((result.experiment_window, result.archive_links, result.archive_coverage), (PASS, PASS, PASS))
        self.assertEqual(result.ledger, _AUDIT_LEDGER(value))
        self.assertEqual(result.ledger.actual_or_reserved_cents, 280)

    def test_known_failures_and_invalid_identifiers_survive_missing_evidence_safely(self):
        for value in (fixtures.reservations_snapshot(7), fixtures.reservations_snapshot(2, quote=501)):
            with self.subTest(reservations=len(value.reservations)):
                result = self.assess(snapshot=value, run_start=None, archives=())
                self.assertEqual(result.ledger, _AUDIT_LEDGER(value))
                self.assertEqual(result.ledger.current_limits.verdict, "FAIL")
                self.assertEqual(result.experiment_window, BLOCKED)
                self.assertEqual(len(result.attempts), len(value.reservations))
        mismatched = self.assess(run_id="other-run")
        self.assertEqual(mismatched.ledger, _AUDIT_LEDGER(self.value))
        self.assertEqual(mismatched.archive_links.verdict, "FAIL")
        private = "private request token"
        reservation = self.value.reservations[0]._replace(attempt_id=private, state=private)
        job = captured.SubmissionJobRow(private, private, private, None)
        value = fixtures.resign(self.value, reservations=(reservation,), submission_jobs=(job,),
            completed_outputs=(), submission_events=(), poll_events=(), run_events=())
        result = self.assess(snapshot=value, run_start=None, archives=())
        self.assertNotIn(private, repr(result))
        self.assertEqual((result.attempts[0].attempt_id, result.attempts[0].reservation_state,
                          result.attempts[0].submission_state, result.attempts[0].provider_task_id), (None, None, None, None))
        self.assertEqual(result.ledger.record_consistency.verdict, "FAIL")

    def test_malformed_mutable_and_oversized_inputs_raise_safe_errors(self):
        first = self.archives[0]
        cases = [{"run_id": "../private-token"}, {"run_id": True}, {"snapshot": {}},
            {"snapshot": replace(self.value, reservations=list(self.value.reservations))},
            {"run_start": {}}, {"run_start": replace(self.start, started_unix=True)},
            {"run_start": replace(self.start, deadline_unix=float("nan"))},
            {"run_start": replace(self.start, record=replace(self.start.record, size=True))},
            {"run_start": replace(self.start, record=replace(self.start.record, sha256="private-token"))},
            {"archives": list(self.archives)}, {"archives": ({},)},
            {"archives": (replace(first, graph=None),)},
            {"archives": (replace(first, completed_unix=float("inf")),)},
            {"archives": (replace(first, reserved_cents=True),)},
            {"archives": (replace(first, payload_sha256="private-token"),)},
            {"archives": (replace(first, sdk_version="1.2.private-token"),)},
            {"archives": (replace(first, sdk_version="1" * 29 + ".1.1"),)},
            {"archives": (replace(first, raw_output=replace(first.raw_output, path="../private-token")),)}]
        for index, changes in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(evidence.RunEvidenceError) as caught:
                self.assess(**changes)
            self.assertNotIn("private-token", str(caught.exception))
        longest_sdk = replace(first, sdk_version="1" * 28 + ".1.1")
        self.assertEqual(self.assess(archives=(longest_sdk, self.archives[1])).archive_links, PASS)
        with patch.object(evidence, "MAX_ARCHIVES", 2):
            self.assertEqual(self.assess().archive_links, PASS)
        with patch.object(evidence, "MAX_ARCHIVES", 1), self.assertRaises(evidence.RunEvidenceError):
            self.assess()


if __name__ == "__main__":
    unittest.main()
