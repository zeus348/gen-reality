"""Audit frozen local records and sequence-ordered spending evidence without IO."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import socket
import sqlite3
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from core import run_audit as audit
from core import run_snapshot as snapshot
from core.budget import BudgetLedger
from features.inference.submission import SubmissionCoordinator
from features.inference.polling import TaskPoller
from features.output.complete import OutputCompleter


RUN = "audit-run"
ATTEMPT_A, ATTEMPT_B = "att-" + "a" * 32, "att-" + "b" * 32
TABLES = tuple(name for name, _ in snapshot._TABLES)
HASH_SNAPSHOT = snapshot._logical_hash


def record_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def resign(value, **changes):
    """Keep semantic fault fixtures anchored to their actual captured facts."""
    changed = replace(value, **changes)
    captured = tuple(getattr(changed, name) for name in TABLES)
    return replace(changed, sha256=HASH_SNAPSHOT(changed.run_id, captured,
        changed.missing_tables, changed.relationship_issues))


def approval_record(approval):
    return {field: getattr(approval, field) for field in
            ("approved_by", "limit_cents", "max_attempts", "cost_cents", "started_unix", "deadline_unix")}


def reservation_record(reservation):
    return {field: getattr(reservation, field) for field in ("key_sha256", "payload_sha256", "cost_cents")}


def complete_snapshot():
    approved = snapshot.ApprovedRunRow(RUN, "fixture operator", 1000, 6, 140, 1000.0, 11800.0, 1003.0, 0)
    reserved = snapshot.ReservationRow(RUN, "a" * 64, ATTEMPT_A, "b" * 64, 140, "RESERVED")
    submission = (
        snapshot.SubmissionEventRow(1, ATTEMPT_A, "CLAIMED", 1002.0, None, None),
        snapshot.SubmissionEventRow(2, ATTEMPT_A, "SEND_GATE_PASSED", 1003.0, None, None),
        snapshot.SubmissionEventRow(3, ATTEMPT_A, "TASK_RECORDED", 1004.0, "task-a", None),
    )
    journal = (
        snapshot.RunEventRow(1, RUN, None, "APPROVAL_RECORDED", 1000.0, None, record_json(approval_record(approved))),
        snapshot.RunEventRow(2, RUN, ATTEMPT_A, "RESERVATION_RECORDED", 1001.0, None, record_json(reservation_record(reserved))),
        *[snapshot.RunEventRow(index + 3, RUN, row.attempt_id, "SUBMISSION_EVENT", row.recorded_unix,
                              row.event_id, None) for index, row in enumerate(submission)],
    )
    value = snapshot.RunSnapshot(RUN, (approved,), (reserved,), (),
        (snapshot.SubmissionJobRow(ATTEMPT_A, "SUCCEEDED", "task-a", None),), submission, journal,
        (snapshot.PollEventRow(1, ATTEMPT_A, 1005.0, "SUCCEEDED", "OBSERVED", "c" * 64),),
        (snapshot.CompletedOutputRow(ATTEMPT_A, "task-a", "c" * 64,
            f"attempts/{ATTEMPT_A}/output/raw.mp4", "d" * 64, 100,
            f"attempts/{ATTEMPT_A}/output/metadata.json", "e" * 64, 200, 1006.0),), (), (), "0" * 64)
    return resign(value)


def refund_snapshot(*, charge_first=True, charge_time=None, actual=0, quote=700, max_attempts=6):
    approved = snapshot.ApprovedRunRow(RUN, "fixture operator", 1000, max_attempts, quote, 1000.0, 11800.0, 1002.0, 0)
    first = snapshot.ReservationRow(RUN, "a" * 64, ATTEMPT_A, "c" * 64, quote, "RESERVED")
    second = snapshot.ReservationRow(RUN, "b" * 64, ATTEMPT_B, "d" * 64, quote, "RESERVED")
    charge = snapshot.ActualChargeRow(ATTEMPT_A, actual, "e" * 64)
    reserve_b = snapshot.RunEventRow(4 if charge_first else 3, RUN, ATTEMPT_B, "RESERVATION_RECORDED", 1002.0, None,
                                     record_json(reservation_record(second)))
    charged = snapshot.RunEventRow(3 if charge_first else 4, RUN, ATTEMPT_A, "CHARGE_RECORDED", charge_time, None,
        record_json({"actual_cents": actual, "evidence_sha256": charge.evidence_sha256}))
    journal = (snapshot.RunEventRow(1, RUN, None, "APPROVAL_RECORDED", 1000.0, None, record_json(approval_record(approved))),
        snapshot.RunEventRow(2, RUN, ATTEMPT_A, "RESERVATION_RECORDED", 1001.0, None, record_json(reservation_record(first))),
        *sorted((reserve_b, charged)))
    value = snapshot.RunSnapshot(RUN, (approved,), (first, second), (charge,), (), (), journal, (), (), (), (), "0" * 64)
    return resign(value)


def reservations_snapshot(count, *, quote=140, limit=1000, max_attempts=6):
    approved = snapshot.ApprovedRunRow(RUN, "fixture operator", limit, max_attempts, quote,
                                      1000.0, 11800.0, 1010.0, 0)
    reserved = tuple(snapshot.ReservationRow(RUN, f"{index + 1:064x}", f"att-{index + 1:032x}",
        f"{index + 10:064x}", quote, "RESERVED") for index in range(count))
    journal = (snapshot.RunEventRow(1, RUN, None, "APPROVAL_RECORDED", 1000.0, None,
                                   record_json(approval_record(approved))),) + tuple(
        snapshot.RunEventRow(index + 2, RUN, row.attempt_id, "RESERVATION_RECORDED", 1001.0 + index,
                             None, record_json(reservation_record(row))) for index, row in enumerate(reserved))
    return resign(snapshot.RunSnapshot(RUN, (approved,), reserved, (), (), (), journal, (), (), (), (), "0" * 64))


class RunAuditTests(unittest.TestCase):
    def setUp(self):
        self.value = complete_snapshot()

    def assess(self, value=None):
        return audit.audit_run(self.value if value is None else value)

    def assert_pass(self, result):
        for category in ("record_consistency", "current_limits", "spending_replay", "elapsed"):
            self.assertEqual(getattr(result, category), audit.AuditAssessment("PASS", "COMPLETE"), category)
        self.assertEqual(result.coverage, "COMPLETE")
        self.assertEqual(result.issues, ())

    def change_event(self, value, index, **changes):
        rows = list(value.run_events)
        rows[index] = rows[index]._replace(**changes)
        return resign(value, run_events=tuple(rows))

    def change_event_record(self, value, index, change):
        data = json.loads(value.run_events[index].record_json)
        change(data)
        return self.change_event(value, index, record_json=record_json(data))

    def change_approval(self, value, **changes):
        approved = value.approved_runs[0]._replace(**changes)
        value = resign(value, approved_runs=(approved,))
        return self.change_event(value, 0, record_json=record_json(approval_record(approved)))

    def assert_issue(self, result, category, code, *, missing=False):
        self.assertTrue(any(issue.category == category and issue.code == code and issue.missing is missing
                            for issue in result.issues), (category, code, result.issues))

    def submission_history(self, events, *, state="SUCCEEDED", task="task-a", evidence=None, keep_output=True):
        rows = tuple(snapshot.SubmissionEventRow(index + 1, ATTEMPT_A, *fields) for index, fields in enumerate(events))
        links = tuple(snapshot.RunEventRow(index + 3, RUN, ATTEMPT_A, "SUBMISSION_EVENT", row.recorded_unix, row.event_id, None)
                      for index, row in enumerate(rows))
        return resign(self.value, submission_events=rows, run_events=self.value.run_events[:2] + links,
            submission_jobs=(snapshot.SubmissionJobRow(ATTEMPT_A, state, task, evidence),),
            poll_events=self.value.poll_events if keep_output else (),
            completed_outputs=self.value.completed_outputs if keep_output else ())

    def test_complete_consistent_history_returns_exact_frozen_summary_without_io_or_clock(self):
        with patch.object(sqlite3, "connect", side_effect=AssertionError("No database IO")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("No files")), \
                patch.object(Path, "write_bytes", side_effect=AssertionError("No writes")), \
                patch.object(socket, "socket", side_effect=AssertionError("No network")), \
                patch.object(time, "time", side_effect=AssertionError("No clock")), \
                patch.object(BudgetLedger, "__init__", side_effect=AssertionError("No writer")):
            result = self.assess()
        self.assert_pass(result)
        self.assertEqual((result.run_id, result.snapshot_sha256), (RUN, self.value.sha256))
        self.assertEqual((result.reservation_count, result.recorded_actual_cents, result.held_reserve_cents,
                          result.actual_or_reserved_cents), (1, 0, 140, 140))
        self.assertEqual((result.unknown_attempt_ids, result.in_flight_attempt_ids, result.unresolved_billing_attempt_ids),
                         ((), (), (ATTEMPT_A,)))
        self.assertEqual((result.limit_cents, result.max_attempts, result.approval_started_unix, result.deadline_unix,
                          result.time_basis), (1000, 6, 1000.0, 11800.0, "APPROVAL_START"))
        self.assertEqual(result.boundaries, (
            audit.SpendingBoundary(2, "RESERVATION_RECORDED", ATTEMPT_A, 1, 140, "PASS"),
            audit.SpendingBoundary(3, "CLAIMED", ATTEMPT_A, 1, 140, "PASS"),
            audit.SpendingBoundary(4, "SEND_GATE_PASSED", ATTEMPT_A, 1, 140, "PASS")))
        with self.assertRaises(FrozenInstanceError):
            result.coverage = "INCOMPLETE"
        with self.assertRaises(FrozenInstanceError):
            result.current_limits.verdict = "FAIL"
        with self.assertRaises(FrozenInstanceError):
            result.boundaries[0].verdict = "FAIL"
        self.assertFalse(hasattr(result, "visualEffectDemonstrated"))
        self.assertFalse(hasattr(result, "automatedWorkflowComplete"))

    def test_malformed_snapshot_structure_raises_and_stale_hash_is_a_record_failure(self):
        cases = [None, {}, replace(self.value, sha256="not-a-hash"), replace(self.value, run_id="../invalid"),
            replace(self.value, reservations=list(self.value.reservations)),
            resign(self.value, reservations=(tuple(self.value.reservations[0]),)),
            replace(self.value, missing_tables=("invented_table",)),
            replace(self.value, relationship_issues=["not-a-tuple"]),
            replace(self.value, missing_tables=TABLES + ("extra",)),
            replace(self.value, reservations=(self.value.reservations[0]._replace(cost_cents=True),)),
            replace(self.value, reservations=(self.value.reservations[0]._replace(cost_cents=2 ** 63),)),
            replace(self.value, approved_runs=(self.value.approved_runs[0]._replace(started_unix=float("nan")),)),
            replace(self.value, approved_runs=(self.value.approved_runs[0]._replace(approved_by="\ud800"),)),
            replace(self.value, approved_runs=(self.value.approved_runs[0]._replace(approved_by="x" * 1025),)),
            replace(self.value, approved_runs=(self.value.approved_runs[0]._replace(approved_by="😀" * 257),)),
            replace(self.value, run_events=(self.value.run_events[0]._replace(record_json="x" * 4097),)),
            replace(self.value, reservations=self.value.reservations * (snapshot.MAX_TABLE_ROWS + 1))]
        for index, value in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(audit.RunAuditError):
                audit.audit_run(value)
        stale = self.assess(replace(self.value, sha256="f" * 64))
        self.assertEqual(stale.record_consistency, audit.AuditAssessment("FAIL", "COMPLETE"))
        self.assertEqual(stale.snapshot_sha256, self.value.sha256)
        self.assert_issue(stale, "records", "SNAPSHOT_HASH_MISMATCH")

    def test_sequence_replay_detects_transient_overspend_hidden_by_a_later_refund(self):
        allowed = self.assess(refund_snapshot(charge_first=True))
        self.assert_pass(allowed)
        self.assertEqual((allowed.recorded_actual_cents, allowed.held_reserve_cents, allowed.actual_or_reserved_cents), (0, 700, 700))
        self.assertEqual(allowed.boundaries[-1], audit.SpendingBoundary(4, "RESERVATION_RECORDED", ATTEMPT_B, 2, 700, "PASS"))
        rejected = self.assess(refund_snapshot(charge_first=False))
        self.assertEqual(rejected.current_limits.verdict, "PASS")
        self.assertEqual(rejected.spending_replay.verdict, "FAIL")
        self.assertEqual(rejected.boundaries[-1], audit.SpendingBoundary(3, "RESERVATION_RECORDED", ATTEMPT_B, 2, 1400, "FAIL"))

    def test_refunds_never_restore_the_attempt_count(self):
        result = self.assess(refund_snapshot(max_attempts=1))
        self.assertEqual((result.reservation_count, result.actual_or_reserved_cents), (2, 700))
        self.assertEqual(result.current_limits.verdict, "FAIL")
        self.assertEqual(result.spending_replay.verdict, "FAIL")
        self.assertEqual(result.boundaries[-1].attempt_count, 2)

    def test_null_or_backward_charge_observation_time_does_not_reorder_spending(self):
        for recorded_at in (None, 0.0, 20000.0):
            with self.subTest(recorded_at=recorded_at):
                result = self.assess(refund_snapshot(charge_time=recorded_at))
                self.assert_pass(result)
                self.assertEqual(result.boundaries[-1].accounted_cents, 700)

    def test_literal_p0_and_reduced_caps_apply_at_each_spending_boundary(self):
        for count, quote, limit, max_attempts in ((6, 140, 1000, 6), (2, 500, 1000, 6), (2, 200, 400, 2)):
            with self.subTest(valid=(count, quote, limit, max_attempts)):
                result = self.assess(reservations_snapshot(count, quote=quote, limit=limit, max_attempts=max_attempts))
                self.assert_pass(result)
                self.assertEqual((result.boundaries[-1].attempt_count, result.boundaries[-1].accounted_cents),
                                 (count, count * quote))
        for count, quote, limit, maximum, issue in ((7, 140, 1000, 6, "ATTEMPT_CAP_EXCEEDED"),
                (2, 501, 1000, 6, "MONEY_CAP_EXCEEDED"), (3, 140, 1000, 2, "ATTEMPT_CAP_EXCEEDED"),
                (2, 201, 400, 6, "MONEY_CAP_EXCEEDED")):
            with self.subTest(invalid=(count, quote, limit, maximum)):
                result = self.assess(reservations_snapshot(count, quote=quote, limit=limit, max_attempts=maximum))
                self.assert_issue(result, "limits", issue)
                self.assert_issue(result, "replay", "BOUNDARY_CAP_EXCEEDED")
                self.assertEqual(result.boundaries[-1].verdict, "FAIL")

    def test_actual_charges_replace_holds_and_exact_integer_sums_do_not_overflow(self):
        value = refund_snapshot(actual=200, quote=300)
        result = self.assess(value)
        self.assert_pass(result)
        self.assertEqual((result.recorded_actual_cents, result.held_reserve_cents, result.actual_or_reserved_cents),
                         (200, 300, 500))
        self.assertEqual(result.unresolved_billing_attempt_ids, (ATTEMPT_B,))
        maximum = 2 ** 63 - 1
        charges = tuple(snapshot.ActualChargeRow(row.attempt_id, maximum, "f" * 64) for row in value.reservations)
        rows = value.run_events[:2] + (value.run_events[-1]._replace(sequence=3),) + tuple(
            snapshot.RunEventRow(index + 4, RUN, row.attempt_id, "CHARGE_RECORDED", None, None,
                record_json({"actual_cents": row.actual_cents, "evidence_sha256": row.evidence_sha256}))
            for index, row in enumerate(charges))
        result = self.assess(resign(value, actual_charges=charges, run_events=rows))
        self.assertEqual((result.recorded_actual_cents, result.held_reserve_cents, result.actual_or_reserved_cents),
                         (2 * maximum, 0, 2 * maximum))
        self.assertEqual(result.current_limits.verdict, "FAIL")
        self.assertEqual(result.spending_replay.verdict, "PASS")
        charge = snapshot.ActualChargeRow(ATTEMPT_A, 141, "f" * 64)
        late = snapshot.RunEventRow(6, RUN, ATTEMPT_A, "CHARGE_RECORDED", 20000.0, None,
                                   record_json({"actual_cents":141, "evidence_sha256":"f" * 64}))
        result = self.assess(resign(self.value, actual_charges=(charge,), run_events=self.value.run_events + (late,)))
        self.assertEqual(result.overquote_attempt_ids, (ATTEMPT_A,))
        self.assertEqual(result.current_limits, audit.AuditAssessment("PASS", "COMPLETE"))
        self.assertEqual(result.spending_replay, audit.AuditAssessment("PASS", "COMPLETE"))
        self.assertTrue(all(row.verdict == "PASS" and row.accounted_cents == 140 for row in result.boundaries))

    def test_known_invalid_money_still_fails_when_approval_or_tables_are_missing(self):
        value = refund_snapshot(actual=-1)
        value = resign(value, approved_runs=(), missing_tables=("approved_runs",))
        result = self.assess(value)
        self.assertEqual(result.current_limits, audit.AuditAssessment("FAIL", "INCOMPLETE"))
        self.assert_issue(result, "limits", "MONEY_RECORDS_INVALID")
        self.assert_issue(result, "limits", "APPROVAL_MISSING", missing=True)
        self.assertEqual((result.recorded_actual_cents, result.held_reserve_cents, result.actual_or_reserved_cents),
                         (None, None, None))
        self.assertEqual(result.spending_replay.verdict, "FAIL")
        orphan = resign(refund_snapshot(), actual_charges=(snapshot.ActualChargeRow("att-" + "f" * 32, 0, "f" * 64),))
        self.assert_issue(self.assess(orphan), "records", "CHARGE_ORPHAN")
        sentinel = "private-request-body-sentinel"
        value = resign(self.value, reservations=(self.value.reservations[0]._replace(attempt_id=sentinel),),
            submission_jobs=(snapshot.SubmissionJobRow(sentinel, "SUBMISSION_UNKNOWN", None, None),),
            run_events=(self.value.run_events[0], self.value.run_events[1]._replace(attempt_id=sentinel)),
            submission_events=(), poll_events=(), completed_outputs=())
        result = self.assess(value)
        self.assert_issue(result, "records", "RESERVATION_INVALID")
        self.assertNotIn(sentinel, repr(result))

    def test_absent_legacy_evidence_blocks_coverage_without_fabricating_approval(self):
        empty = resign(self.value, **{name: () for name in TABLES}, missing_tables=TABLES)
        result = self.assess(empty)
        for field in ("record_consistency", "current_limits", "spending_replay", "elapsed"):
            self.assertEqual(getattr(result, field), audit.AuditAssessment("BLOCKED", "INCOMPLETE"))
        self.assertIsNone(result.limit_cents)
        self.assertIsNone(result.approval_started_unix)
        self.assertEqual(result.boundaries, ())
        value = resign(self.value, run_events=(), missing_tables=("run_events",))
        result = self.assess(value)
        self.assertEqual(result.current_limits, audit.AuditAssessment("PASS", "COMPLETE"))
        self.assertEqual(result.spending_replay, audit.AuditAssessment("BLOCKED", "INCOMPLETE"))
        self.assert_issue(result, "replay", "FINANCIAL_HISTORY_MISSING", missing=True)
        self.assert_issue(result, "elapsed", "AUTHORITY_HISTORY_MISSING", missing=True)

    def test_financial_record_json_is_exact_and_wrong_records_are_not_missing_evidence(self):
        original = self.value.run_events[1].record_json
        payload = json.loads(original)
        variants = ["{", json.dumps(payload, indent=2), original[:-1] + ',"cost_cents":140}',
                    record_json({**payload, "cost_cents":141}), record_json({**payload, "extra":0})]
        for raw in variants:
            with self.subTest(raw=raw):
                result = self.assess(self.change_event(self.value, 1, record_json=raw))
                self.assertEqual(result.record_consistency.verdict, "FAIL")
                self.assertEqual(result.spending_replay, audit.AuditAssessment("FAIL", "COMPLETE"))
                self.assert_issue(result, "replay", "RUN_RECORD_MISMATCH")
        missing = self.assess(resign(self.value, run_events=self.value.run_events[:1] + self.value.run_events[2:]))
        self.assert_issue(missing, "replay", "FINANCIAL_HISTORY_MISSING", missing=True)

    def test_duplicate_financial_and_journal_records_fail_with_localized_immutable_issues(self):
        cases = [(resign(self.value, reservations=self.value.reservations * 2), "records", "RESERVATION_DUPLICATE"),
            (resign(self.value, approved_runs=self.value.approved_runs * 2), "limits", "APPROVAL_DUPLICATE"),
            (resign(self.value, run_events=self.value.run_events + (self.value.run_events[1],)), "replay", "RUN_SEQUENCE_DUPLICATE"),
            (resign(self.value, run_events=self.value.run_events + (self.value.run_events[1]._replace(sequence=9),)),
             "replay", "FINANCIAL_HISTORY_DUPLICATE")]
        for value, category, code in cases:
            with self.subTest(code=code):
                self.assert_issue(self.assess(value), category, code)
        result = self.assess(self.change_event(self.value, 1, record_json="{}"))
        issue = next(issue for issue in result.issues if issue.code == "RUN_RECORD_MISMATCH")
        self.assertEqual((issue.attempt_id, issue.sequence, issue.event_id, issue.missing), (ATTEMPT_A, 2, None, False))
        with self.assertRaises(FrozenInstanceError):
            issue.code = "changed"
        self.assertEqual(result, self.assess(self.change_event(self.value, 1, record_json="{}")))

    def test_submission_links_require_exact_identity_time_order_and_unique_coverage(self):
        for change in ({"run_id":"another-run"}, {"sequence":0}):
            with self.subTest(invalid_journal_identity=change):
                self.assert_issue(self.assess(self.change_event(self.value, 2, **change)), "replay", "RUN_EVENT_INVALID")
        for change in ({"submission_event_id":99}, {"attempt_id":ATTEMPT_B}, {"recorded_unix":1002.5}, {"record_json":"{}"}):
            with self.subTest(change=change):
                self.assert_issue(self.assess(self.change_event(self.value, 2, **change)), "replay", "SUBMISSION_LINK_MISMATCH")
        duplicate = resign(self.value, run_events=self.value.run_events + (self.value.run_events[2]._replace(sequence=9),))
        self.assert_issue(self.assess(duplicate), "replay", "SUBMISSION_LINK_DUPLICATE")
        reversed_links = self.change_event(self.value, 2, sequence=4)
        reversed_links = self.change_event(reversed_links, 3, sequence=3)
        self.assert_issue(self.assess(reversed_links), "replay", "SUBMISSION_ORDER_MISMATCH")

    def test_authority_deadline_is_exclusive_and_fractional_short_approvals_are_valid(self):
        before = 1000.0 + 10800.0 - 0.000001
        for gate, valid in ((before, True), (11800.0, False)):
            value = self.submission_history((("CLAIMED",1002.0,None,None), ("SEND_GATE_PASSED",gate,None,None),
                                             ("TASK_RECORDED",1004.0,"task-a",None)))
            value = self.change_approval(value, last_checked_unix=before)
            result = self.assess(value)
            if valid:
                self.assert_pass(result)
            else:
                self.assert_issue(result, "elapsed", "AUTHORITY_TIME_INVALID")
        short = reservations_snapshot(1)
        short = self.change_approval(short, deadline_unix=1000.5, last_checked_unix=1000.25)
        short = self.change_event(short, 1, recorded_unix=1000.25)
        self.assert_pass(self.assess(short))
        for timestamp in (999.0, -1.0, 253_402_300_800.0):
            with self.subTest(timestamp=timestamp):
                self.assert_issue(self.assess(self.change_event(self.value, 1, recorded_unix=timestamp)),
                                  "elapsed", "AUTHORITY_TIME_INVALID")
        missing = self.assess(self.change_event(self.value, 1, recorded_unix=None))
        self.assert_issue(missing, "elapsed", "AUTHORITY_TIME_MISSING", missing=True)

    def test_authority_times_are_monotone_but_highwater_need_not_equal_last_event(self):
        maximum = 253_402_300_799.0
        upper = self.change_approval(reservations_snapshot(1), started_unix=maximum,
                                     deadline_unix=maximum + 60, last_checked_unix=maximum)
        upper = resign(upper, run_events=tuple(row._replace(recorded_unix=maximum) for row in upper.run_events))
        self.assert_pass(self.assess(upper))
        self.assert_issue(self.assess(self.change_approval(upper, last_checked_unix=maximum + 1)),
                          "elapsed", "APPROVAL_TIME_INVALID")
        self.assert_pass(self.assess(self.change_approval(self.value, last_checked_unix=1100.0, time_stopped=1)))
        self.assert_issue(self.assess(self.change_approval(self.value, last_checked_unix=1002.5)),
                          "elapsed", "AUTHORITY_TIME_INVALID")
        value = self.submission_history((("CLAIMED",1002.0,None,None), ("SEND_GATE_PASSED",1001.5,None,None),
                                         ("TASK_RECORDED",1004.0,"task-a",None)))
        self.assert_issue(self.assess(value), "elapsed", "AUTHORITY_TIME_INVALID")
        self.assert_issue(self.assess(self.change_event(self.value, 0, recorded_unix=1000.5)),
                          "elapsed", "AUTHORITY_TIME_INVALID")

    def test_multiple_gates_and_global_sequence_gaps_are_valid_without_recounting_attempts(self):
        value = self.submission_history((("CLAIMED",1002.0,None,None), ("SEND_GATE_PASSED",1003.0,None,None),
            ("SEND_GATE_PASSED",1003.0,None,None), ("TASK_RECORDED",None,"task-a",None)))
        submissions = tuple(row._replace(event_id=row.event_id * 3) for row in value.submission_events)
        journal = tuple(row._replace(sequence=row.sequence * 5,
            submission_event_id=row.submission_event_id * 3 if row.submission_event_id is not None else None)
            for row in value.run_events)
        result = self.assess(resign(value, submission_events=submissions, run_events=journal))
        self.assert_pass(result)
        self.assertEqual([row.event_kind for row in result.boundaries],
                         ["RESERVATION_RECORDED", "CLAIMED", "SEND_GATE_PASSED", "SEND_GATE_PASSED"])
        self.assertTrue(all(row.attempt_count == 1 and row.accounted_cents == 140 for row in result.boundaries))

    def test_missing_send_gate_is_incomplete_without_inventing_a_record_failure(self):
        value = self.submission_history((("CLAIMED",1002.0,None,None), ("TASK_RECORDED",1004.0,"task-a",None)))
        result = self.assess(value)
        self.assertEqual(result.record_consistency, audit.AuditAssessment("PASS", "COMPLETE"))
        self.assertEqual(result.spending_replay, audit.AuditAssessment("BLOCKED", "INCOMPLETE"))
        self.assertEqual(result.elapsed, audit.AuditAssessment("BLOCKED", "INCOMPLETE"))
        self.assert_issue(result, "replay", "SEND_GATE_MISSING", missing=True)

    def test_unknown_recovery_reconciliation_and_non_authority_times_preserve_reserves(self):
        prefix = (("CLAIMED",1002.0,None,None), ("SEND_GATE_PASSED",1003.0,None,None))
        for observed in (None, 0.0, 20000.0):
            value = self.submission_history(prefix + (("OWNER_RECOVERED_UNKNOWN",observed,None,None),),
                                            state="SUBMISSION_UNKNOWN", task=None, keep_output=False)
            result = self.assess(value)
            self.assert_pass(result)
            self.assertEqual((result.unknown_attempt_ids, result.in_flight_attempt_ids, result.held_reserve_cents),
                             ((ATTEMPT_A,), (), 140))
            value = self.submission_history(prefix + (("SUBMISSION_UNKNOWN",observed,None,None),
                ("TASK_RECONCILED",observed,"task-a","f" * 64)), state="QUEUED", evidence="f" * 64, keep_output=False)
            result = self.assess(value)
            self.assert_pass(result)
            self.assertEqual(result.in_flight_attempt_ids, (ATTEMPT_A,))
        invalid = self.submission_history(prefix + (("TASK_RECONCILED",1004.0,"task-a","f" * 64),),
                                         state="QUEUED", evidence="f" * 64, keep_output=False)
        self.assert_issue(self.assess(invalid), "records", "SUBMISSION_TRANSITION_INVALID")

    def test_poll_success_retention_and_provider_cancellation_follow_writer_states(self):
        for prior_running, local in ((False, "QUEUED"), (True, "RUNNING")):
            polls = ((snapshot.PollEventRow(1, ATTEMPT_A, 1005.0, "RUNNING", "OBSERVED", "a" * 64),)
                     if prior_running else ()) + (self.value.poll_events[0]._replace(event_id=2),)
            value = resign(self.value, completed_outputs=(), poll_events=polls,
                submission_jobs=(self.value.submission_jobs[0]._replace(state=local),))
            self.assert_pass(self.assess(value))
        for status in ("POLL_ERROR", "POLL_DEADLINE_REACHED"):
            later = self.value.poll_events[0]._replace(event_id=2, observed_unix=20000.0, poll_status=status)
            value = resign(self.value, poll_events=self.value.poll_events + (later,),
                completed_outputs=(self.value.completed_outputs[0]._replace(completed_unix=20001.0),))
            self.assert_pass(self.assess(value))
            self.assert_issue(self.assess(resign(value, poll_events=value.poll_events[:1] +
                (later._replace(evidence_sha256="f" * 64),))), "records", "POLL_RETAINED_EVIDENCE_MISMATCH")
        canceled = resign(self.value, completed_outputs=(),
            poll_events=(self.value.poll_events[0]._replace(provider_status="CANCELLED"),),
            submission_jobs=(self.value.submission_jobs[0]._replace(state="CANCELED"),))
        self.assert_pass(self.assess(canceled))

    def test_bad_poll_and_output_records_do_not_block_independent_financial_replay(self):
        cases = [resign(self.value, poll_events=(self.value.poll_events[0]._replace(poll_status="invalid"),)),
            resign(self.value, completed_outputs=(self.value.completed_outputs[0]._replace(raw_path="private/raw.mp4"),)),
            resign(self.value, completed_outputs=(self.value.completed_outputs[0]._replace(provider_evidence_sha256="f" * 64),)),
            resign(self.value, poll_events=self.value.poll_events +
                (self.value.poll_events[0]._replace(event_id=2, provider_status="RUNNING"),))]
        for index, value in enumerate(cases):
            with self.subTest(case=index):
                result = self.assess(value)
                self.assertEqual(result.record_consistency.verdict, "FAIL")
                self.assertEqual(result.current_limits.verdict, "PASS")
                self.assertEqual(result.spending_replay, audit.AuditAssessment("PASS", "COMPLETE"))
                self.assertTrue(all(boundary.verdict == "PASS" for boundary in result.boundaries))

    def test_replay_rejects_spending_before_authority_and_after_settlement(self):
        swapped = self.change_event(self.value, 0, sequence=2)
        swapped = self.change_event(swapped, 1, sequence=1)
        self.assert_issue(self.assess(swapped), "replay", "BOUNDARY_BEFORE_APPROVAL")
        swapped = self.change_event(self.value, 1, sequence=3)
        swapped = self.change_event(swapped, 2, sequence=2)
        self.assert_issue(self.assess(swapped), "replay", "BOUNDARY_BEFORE_RESERVATION")
        value = refund_snapshot(actual=701)
        self.assert_issue(self.assess(value), "replay", "PRIOR_QUOTE_EXCEEDED")
        charge = snapshot.ActualChargeRow(ATTEMPT_A, 0, "f" * 64)
        row = snapshot.RunEventRow(3, RUN, ATTEMPT_A, "CHARGE_RECORDED", None, None,
                                   record_json({"actual_cents":0, "evidence_sha256":"f" * 64}))
        journal = self.value.run_events[:2] + (row,) + tuple(item._replace(sequence=item.sequence + 1)
                                                                           for item in self.value.run_events[2:])
        result = self.assess(resign(self.value, actual_charges=(charge,), run_events=journal))
        self.assert_issue(result, "replay", "SETTLED_SPENDING_BOUNDARY")

    def test_current_application_histories_with_zero_or_repeated_send_checks_are_supported(self):
        for gate_count in (0, 2):
            with self.subTest(gate_count=gate_count), TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "ledger.sqlite"
                ledger = BudgetLedger(path, clock=lambda:1001.0)
                ledger.approve_run(RUN, approved_by="fixture operator", limit_cents=1000, max_attempts=6, cost_cents=140)
                ledger.reserve(RUN, "fixture-key", "c" * 64, attempt_id=ATTEMPT_A)
                with SubmissionCoordinator(ledger) as owner:
                    TaskPoller(owner)
                    OutputCompleter(owner, root / "output")

                    def fake_create():
                        for _ in range(gate_count):
                            owner.require_send_allowed(ATTEMPT_A)
                        return "fixture-task"

                    owner.submit_once(ATTEMPT_A, fake_create)
                    captured = snapshot.read_run_snapshot(path, run_id=RUN)
                result = self.assess(captured)
                if gate_count:
                    self.assert_pass(result)
                else:
                    self.assertEqual(result.record_consistency.verdict, "PASS")
                    self.assertEqual(result.spending_replay, audit.AuditAssessment("BLOCKED", "INCOMPLETE"))
                self.assertEqual(result.actual_or_reserved_cents, 140)
                self.assertEqual(result.in_flight_attempt_ids, (ATTEMPT_A,))


if __name__ == "__main__":
    unittest.main()
