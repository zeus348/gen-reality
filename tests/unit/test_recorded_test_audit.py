"""Audit literal retained test reports without invoking pytest or a shell."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ElementTree

from core.run_audit import AuditAssessment
from core.storage import StoredArtifact
from qa import recorded_test_audit as audit
from qa import p0_cases, test_registry, test_results


RUN = "recorded-coverage-run"
EXECUTION = "a" * 32
SOURCE_SHA256 = sha256(b"independently approved source ZIP identity").hexdigest()
SOURCE_SIZE = 47
REGISTRY_BYTES = test_registry.registry_bytes()
REGISTRY_SHA256 = sha256(REGISTRY_BYTES).hexdigest()
PYTEST_OPTIONS = tuple(test_registry.PYTEST_OPTIONS)
SUITES = dict(test_registry.SUITES)
PARSE_REPORTS = test_results.parse_reports
POLICY_BYTES = p0_cases.policy_bytes()
POLICY_VERSION = p0_cases.POLICY_VERSION
EXPECTED_NODES = (
    'tests/ui/test_review_page_browser.py::ReviewPageBrowserTests::test_native_views_samples_and_playback_stay_within_one_local_attempt',
    'tests/unit/test_account.py::AccountTests::test_balance_and_model_limits_block_the_account_check',
    'tests/unit/test_alignment.py::AlignmentTests::test_24_fps_source_matches_30_fps_raw_without_changing_timestamps',
    'tests/unit/test_archive_attempt_facts.py::ArchiveAttemptFactsTests::test_invalid_references_missing_and_corrupt_payloads_are_rejected_safely',
    'tests/unit/test_archive_attempt_facts.py::ArchiveAttemptFactsTests::test_selected_unsigned_cards_and_invalid_bindings_are_rejected',
    'tests/unit/test_archive_attempt_facts.py::ArchiveAttemptFactsTests::test_task_raw_and_private_references_must_match_the_inventory',
    'tests/unit/test_asset.py::AssetTests::test_binary_alpha_is_valid',
    'tests/unit/test_asset.py::AssetTests::test_missing_and_corrupt_files_have_safe_errors',
    'tests/unit/test_asset.py::AssetTests::test_opaque_empty_and_nonzero_minimum_alpha_are_rejected',
    'tests/unit/test_asset.py::AssetTests::test_original_hash_and_partial_alpha_are_preserved',
    'tests/unit/test_asset.py::AssetTests::test_oversized_dimensions_are_rejected_before_decode',
    'tests/unit/test_asset.py::AssetTests::test_oversized_file_is_rejected_before_decode',
    'tests/unit/test_budget.py::BudgetTests::test_money_and_attempt_caps_are_independent',
    'tests/unit/test_budget.py::BudgetTests::test_reservation_uses_approved_quote_and_persists_across_restart',
    'tests/unit/test_budget.py::BudgetTests::test_same_key_with_changed_payload_conflicts',
    'tests/unit/test_budget.py::BudgetTests::test_twenty_concurrent_requests_reserve_only_one_available_charge',
    'tests/unit/test_charges.py::ChargeTests::test_concurrent_reconciliation_cannot_double_spend_released_money',
    'tests/unit/test_charges.py::ChargeTests::test_final_charge_replaces_reservation_instead_of_adding_to_it',
    'tests/unit/test_charges.py::ChargeTests::test_over_budget_billing_is_retained_and_stops_new_reservations',
    'tests/unit/test_checked_send.py::CheckedSendTests::test_charge_over_quote_before_send_blocks_new_spending',
    'tests/unit/test_export_archive.py::ExportArchiveTests::test_real_modes_and_selections_preserve_exact_members_crc_and_fixed_metadata',
    'tests/unit/test_export_inventory.py::ExportInventoryTests::test_no_directory_scan_or_private_url_secret_html_or_other_attempt_is_selected',
    'tests/unit/test_frozen_request.py::FrozenRequestTests::test_each_frozen_input_changes_the_reservation_fingerprint',
    'tests/unit/test_frozen_request.py::FrozenRequestTests::test_mode_b_uses_its_frozen_guide_and_omits_keyframes',
    'tests/unit/test_guidance.py::GuidanceTests::test_real_compilation_preserves_inputs_and_renders_every_frame',
    'tests/unit/test_guide_quality.py::GuideQualityTests::test_rgb_psnr_and_ssim_on_all_frames_outside_edge_band',
    'tests/unit/test_keyframe.py::KeyframeTests::test_exact_alpha_blend_matches_independent_reference',
    'tests/unit/test_keyframe.py::KeyframeTests::test_repeated_render_has_identical_png_hash',
    'tests/unit/test_media.py::MediaTests::test_bad_paths_and_corrupt_media_have_safe_errors',
    'tests/unit/test_media.py::MediaTests::test_only_the_first_five_seconds_are_used',
    'tests/unit/test_media.py::MediaTests::test_original_bytes_and_hashes_are_preserved',
    'tests/unit/test_media.py::MediaTests::test_portrait_is_pillarboxed_without_crop',
    'tests/unit/test_media.py::MediaTests::test_real_output_has_exact_media_format_and_timebase',
    'tests/unit/test_output_completion.py::OutputCompletionTests::test_artifacts_precede_success_and_fixture_content_never_claims_workflow_pass',
    'tests/unit/test_output_completion.py::OutputCompletionTests::test_corrupt_download_cannot_create_artifacts_or_local_success',
    'tests/unit/test_placement.py::PlacementTests::test_center_and_aspect_ratio',
    'tests/unit/test_placement.py::PlacementTests::test_invalid_fractions_are_rejected',
    'tests/unit/test_placement.py::PlacementTests::test_zero_height_and_rotation_are_rejected',
    'tests/unit/test_prepare.py::PrepareTests::test_real_cli_exports_verified_snapshot_without_secrets_or_paid_claims',
    'tests/unit/test_prepare.py::PrepareTests::test_second_placement_keeps_first_attempt_unchanged',
    'tests/unit/test_raw_origin.py::RawOriginTests::test_alignment_uses_elapsed_nearest_ties_span_and_source_epsilon',
    'tests/unit/test_raw_video.py::RawVideoTests::test_corrupt_late_audio_is_rejected_even_when_video_and_metadata_pass',
    'tests/unit/test_retrieve.py::RetrievalTests::test_sdk_get_timeout_resumes_after_restart_without_another_create',
    'tests/unit/test_review_evidence.py::ReviewEvidenceTests::test_mode_a_reuses_actual_static_keyframe_and_preserves_original_source_reference',
    'tests/unit/test_review_evidence.py::ReviewEvidenceTests::test_mode_b_samples_real_24_fps_guide_and_30_fps_raw_frames',
    'tests/unit/test_review_evidence.py::ReviewEvidenceTests::test_repeat_and_other_attempt_keep_existing_review_bytes_unchanged',
    'tests/unit/test_review_page.py::ReviewPageTests::test_page_is_immutable_repeatable_and_independent_of_another_attempt',
    'tests/unit/test_run_audit.py::RunAuditTests::test_duplicate_financial_and_journal_records_fail_with_localized_immutable_issues',
    'tests/unit/test_run_audit.py::RunAuditTests::test_known_invalid_money_still_fails_when_approval_or_tables_are_missing',
    'tests/unit/test_run_audit.py::RunAuditTests::test_literal_p0_and_reduced_caps_apply_at_each_spending_boundary',
    'tests/unit/test_run_audit.py::RunAuditTests::test_refunds_never_restore_the_attempt_count',
    'tests/unit/test_run_start.py::RunDeadlineTests::test_exact_experiment_deadline_blocks_final_send_and_stays_stopped',
    'tests/unit/test_runtime_chain.py::RuntimeChainTests::test_injected_receipts_keep_valid_bindings_but_fail_default_network',
    'tests/unit/test_runtime_chain.py::RuntimeChainTests::test_relabelled_mock_receipts_cannot_replace_retained_pins',
    'tests/unit/test_runtime_evidence.py::RuntimeEvidenceTests::test_injected_constructed_sdk_client_cannot_claim_default_transport',
    'tests/unit/test_runway.py::RunwayTests::test_connection_failures_and_timeouts_make_one_post_each',
    'tests/unit/test_runway.py::RunwayTests::test_modes_send_exact_payloads_through_the_pinned_sdk',
    'tests/unit/test_runway.py::RunwayTests::test_retryable_http_errors_make_one_post_each',
    'tests/unit/test_scorecard.py::ScorecardTests::test_absent_or_unsigned_valid_draft_is_not_run',
    'tests/unit/test_scorecard.py::ScorecardTests::test_complete_declared_review_can_pass_with_two_or_more_plays',
    'tests/unit/test_scorecard.py::ScorecardTests::test_each_dimension_must_pass_without_averaging',
    'tests/unit/test_scorecard.py::ScorecardTests::test_nonblank_review_time_must_be_an_actual_utc_timestamp',
    'tests/unit/test_scorecard.py::ScorecardTests::test_scores_plays_inspected_times_and_required_observation_are_strict',
    'tests/unit/test_submission.py::SubmissionTests::test_process_exit_recovers_unknown_without_replay',
    'tests/unit/test_submission.py::SubmissionTests::test_submitting_is_committed_before_the_create_call',
    'tests/unit/test_submission.py::SubmissionTests::test_timeout_keeps_money_and_prevents_replay',
    'tests/unit/test_submit_command.py::SubmitCommandTests::test_restores_exact_request_and_repeated_command_needs_no_key',
)
UNIT_NODES = (
    "tests/unit/test_media.py::MediaTests::test_real_output_has_exact_media_format_and_timebase",
    "tests/unit/test_media.py::MediaTests::test_original_bytes_and_hashes_are_preserved",
    "tests/unit/test_media.py::MediaTests::test_only_the_first_five_seconds_are_used",
    "tests/unit/test_media.py::MediaTests::test_portrait_is_pillarboxed_without_crop",
)
METRIC_NODE = "tests/unit/test_guide_quality.py::GuideQualityTests::test_rgb_psnr_and_ssim_on_all_frames_outside_edge_band"
VIEWER_NODE = ("tests/ui/test_review_page_browser.py::ReviewPageBrowserTests::"
               "test_native_views_samples_and_playback_stay_within_one_local_attempt")
INTEGRATION_NODE = "tests/integration/test_probe.py::test_probe"
PASS = AuditAssessment("PASS", "COMPLETE")
BLOCKED = AuditAssessment("BLOCKED", "INCOMPLETE")


def json_bytes(data):
    return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def evidence_record(path, contents):
    digest = sha256(contents).hexdigest()
    return audit.TestRecord(StoredArtifact(path(digest) if callable(path) else path, digest, len(contents)), contents)


def codec_report():
    return {"frames": 120, "color": "8-bit RGB", "edgeExclusionPixels": 2, "pixelsPerFrame": 919552,
            "regions": [{"crop": crop, "minimumPsnrDb": 30.0, "minimumSsim": 0.95} for crop in
                        ([574, 720, 0, 0], [574, 720, 706, 0], [132, 294, 574, 0],
                         [132, 294, 574, 426], [124, 124, 578, 298])]}


def passed_events(nodes):
    events = [{"$report_type": "SessionStart", "pytest_version": "9.2.0"}]
    events.extend({"$report_type": "TestReport", "nodeid": node, "when": phase,
                   "outcome": "passed", "sections": []}
                  for node in nodes for phase in ("setup", "call", "teardown"))
    events.append({"$report_type": "SessionFinish", "exitstatus": 0})
    return events


def normalized_facts(events, exit_code):
    facts = []
    for row in events:
        kind = row.get("$report_type")
        if kind not in ("CollectReport", "TestReport", "SubTestReport"):
            continue
        facts.append({"reportType": kind, "nodeId": row["nodeid"],
            "phase": "collect" if kind == "CollectReport" else row["when"], "outcome": row["outcome"],
            "hasXfail": "wasxfail" in row, "wasXfail": row.get("wasxfail"),
            "subtest": row.get("_subtest.context") if kind == "SubTestReport" else None})
    return {"reports": facts, "nodeIds": sorted({row["nodeid"] for row in events if row.get("$report_type") == "TestReport"}),
            "subtestReports": sum(row["reportType"] == "SubTestReport" for row in facts), "sessionExitCode": exit_code}


def junit_report(events):
    root = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(root, "testsuite", name="pytest", tests="0", failures="0", errors="0", skipped="0")
    nodes = sorted({row["nodeid"] for row in events if row.get("$report_type") == "TestReport"})
    counts = {"failure": 0, "error": 0, "skipped": 0}
    for node in nodes:
        path, *parts = node.split("::")
        names = [path.removesuffix(".py").replace("/", "."), *parts]
        case = ElementTree.SubElement(suite, "testcase", classname=".".join(names[:-1]), name=names[-1])
        reports = [row for row in events if row.get("nodeid") == node and row.get("$report_type") in ("TestReport", "SubTestReport")]
        failed = next((row for row in reports if row["outcome"] == "failed"), None)
        if failed:
            kind = "failure" if failed["when"] == "call" else "error"
            ElementTree.SubElement(case, kind, message="fixture failure")
            counts[kind] += 1
        elif any(row["outcome"] == "skipped" for row in reports):
            ElementTree.SubElement(case, "skipped", message="fixture skip")
            counts["skipped"] += 1
    suite.set("tests", str(len(nodes)))
    for key, attribute in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
        suite.set(attribute, str(counts[key]))
    return root


def report_issues(events, exit_code):
    issues = []
    for row in events:
        if row.get("$report_type") not in ("CollectReport", "TestReport", "SubTestReport"):
            continue
        if row["outcome"] == "failed":
            issues.append("TEST_FAILURE")
        if "wasxfail" in row:
            issues.append("TEST_XPASS" if row["outcome"] == "passed" else "TEST_XFAIL")
        elif row["outcome"] == "skipped":
            issues.append("TEST_SKIPPED")
    if not any(row.get("$report_type") == "TestReport" for row in events):
        issues.append("NO_TESTS")
    if exit_code:
        issues.append("NO_TESTS" if exit_code == 5 else "TEST_FAILURE")
    return list(dict.fromkeys(issues))


def make_evidence(*, specifications=None, missing_suites=(), receipt_changes=None, run_change=None,
                  registry_contents=REGISTRY_BYTES, execution_id=EXECUTION):
    """Refresh each containing reference without borrowing auditor behavior."""
    specifications = specifications or {}
    source = StoredArtifact(f"qa/test-sources/{SOURCE_SHA256}.zip", SOURCE_SHA256, SOURCE_SIZE)
    registry = evidence_record(lambda digest: f"qa/test-registries/{digest}.json", registry_contents)
    suites, summaries, receipts, saved_events = [], [], {}, {}
    defaults = {"unit": UNIT_NODES, "integration_offline": (INTEGRATION_NODE,), "ui": (VIEWER_NODE,)}
    for name, default_nodes in defaults.items():
        specification = specifications.get(name, {})
        events = passed_events(specification.get("nodes", default_nodes))
        if specification.get("events"):
            specification["events"](events)
        tests = [row for row in events if row.get("$report_type") == "TestReport"]
        failed = any(row.get("outcome") == "failed" for row in events)
        exit_code = specification.get("exit_code", 1 if failed else 0 if tests else 5)
        events[-1]["exitstatus"] = exit_code
        cases = normalized_facts(events, exit_code)
        if specification.get("cases"):
            specification["cases"](cases)
        junit = junit_report(events)
        if specification.get("junit"):
            specification["junit"](junit)
        payloads = {"stdout": b"", "stderr": b"", "junit": ElementTree.tostring(junit),
                    "events": b"".join(json.dumps(row, allow_nan=False).encode() + b"\n" for row in events),
                    "cases": json_bytes(cases)} | specification.get("payloads", {})
        suffixes = {"stdout": "stdout.txt", "stderr": "stderr.txt", "junit": "junit.xml",
                    "events": "events.jsonl", "cases": "cases.json"}
        artifacts = {role: evidence_record(lambda digest, role=role: f"qa/test-files/{digest}.{suffixes[role]}", body)
                     for role, body in payloads.items()}
        packages = {package: None for package in ("iniconfig", "pluggy", "packaging", "pygments", "Pillow", "jsonschema", "httpx", "runwayml", "playwright")}
        packages.update(pytest="9.2.0", **{"pytest-reportlog": "1.0.0"})
        path, marker = SUITES[name]
        receipt = {"schemaVersion": "1.4", "manifestType": "recorded_test_execution", "runId": RUN, "suite": name,
            "source": asdict(source), "registry": asdict(registry.record), "executed": True,
            "argv": ["/historical/operator/venv/python", *PYTEST_OPTIONS,
                     "-o", "markers=live_provider: explicitly authorized provider work",
                     "--junitxml=../reports/junit.xml", "--report-log=../reports/events.jsonl", "-m", marker, path],
            "runtime": {"python": "3.11.9", "packages": packages, "tools": {"ffmpeg": None, "ffprobe": None}},
            "startedAtUnix": 1000.0, "finishedAtUnix": 1001.0, "elapsedSeconds": 1.0, "exitCode": exit_code,
            "issues": report_issues(events, exit_code), "sourceUnchanged": True, "reportsComplete": True,
            "artifacts": {role: asdict(record.record) for role, record in artifacts.items()}}
        if name == "ui":
            receipt["runtime"]["tools"]["browser"] = None
        if name in missing_suites:
            receipt.update(executed=False, argv=None, startedAtUnix=None, finishedAtUnix=None, elapsedSeconds=None,
                           exitCode=None, issues=["SUITE_MISSING"], sourceUnchanged=None, reportsComplete=False,
                           artifacts={role: None for role in artifacts})
            artifacts = {role: None for role in artifacts}
        if name in (receipt_changes or {}):
            receipt_changes[name](receipt)
        record = evidence_record(f"qa/test-runs/{execution_id}/{name}.json", json_bytes(receipt))
        suites.append(audit.SuiteRecords(name, record, **artifacts))
        summaries.append({"name": name, "record": asdict(record.record), "executed": receipt["executed"],
                          "exit_code": receipt["exitCode"], "issues": receipt["issues"]})
        receipts[name], saved_events[name] = receipt, events
    run_data = {"schemaVersion": "1.4", "manifestType": "recorded_test_run", "runId": RUN,
                "source": asdict(source), "registry": asdict(registry.record), "suites": summaries}
    if run_change:
        run_change(run_data)
    run = evidence_record(f"qa/test-runs/{execution_id}/run.json", json_bytes(run_data))
    context = audit.ApprovedTestContext(RUN, SOURCE_SHA256, SOURCE_SIZE, REGISTRY_SHA256, run.record)
    return SimpleNamespace(context=context, evidence=audit.RecordedTestEvidence(run, registry, tuple(suites)),
                           run_data=run_data, receipts=receipts, events=saved_events, source=source)


def fail_phase(events, node=UNIT_NODES[0], phase="call"):
    next(row for row in events if row.get("nodeid") == node and row.get("when") == phase)["outcome"] = "failed"
    if phase == "setup":
        events[:] = [row for row in events if not (row.get("nodeid") == node and row.get("when") == "call")]


def add_subtest(events, *, outcome="passed", xfail=False):
    index = next(index for index, row in enumerate(events) if row.get("nodeid") == UNIT_NODES[0] and row.get("when") == "call")
    row = {"$report_type": "SubTestReport", "nodeid": UNIT_NODES[0], "when": "call", "outcome": outcome,
           "_subtest.context": {"msg": None, "kwargs": {"case": "private-subtest-token"}}}
    if xfail:
        row["wasxfail"] = "fixture expectation"
    events.insert(index, row)


def add_metrics(events, report=None, *, node=METRIC_NODE, phase="call", copies=1):
    selected = next(row for row in events if row.get("nodeid") == node and row.get("when") == phase)
    line = "T006 codec fixture: " + json.dumps(codec_report() if report is None else report)
    selected["sections"] = [["Captured stdout call", (line + "\n") * copies]]


class RecordedTestAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_evidence()

    def assess(self, fixture=None, *, context=None, evidence=None):
        fixture = self.fixture if fixture is None else fixture
        return audit.audit_recorded_tests(fixture.context if context is None else context,
                                          fixture.evidence if evidence is None else evidence)

    @staticmethod
    def coverage(result, test_id):
        return next(row for row in result.coverage if row.test_id == test_id)

    @staticmethod
    def suite(result, name):
        return next(row for row in result.suites if row.name == name)

    def test_anchored_execution_has_frozen_exact_component_coverage_without_io(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No reads")), \
                patch.object(Path, "write_bytes", side_effect=AssertionError("No writes")), \
                patch.object(sqlite3, "connect", side_effect=AssertionError("No database")), \
                patch.object(socket, "socket", side_effect=AssertionError("No network")), \
                patch.object(subprocess, "Popen", side_effect=AssertionError("No process")), \
                patch.object(time, "time", side_effect=AssertionError("No clock")):
            result = self.assess()
        self.assertIsInstance(result, audit.RecordedTestAudit)
        self.assertEqual((result.run_id, result.source, result.registry_sha256, result.execution),
                         (RUN, self.fixture.source, REGISTRY_SHA256, self.fixture.context.execution))
        self.assertEqual((result.policy_version, result.policy_sha256), (POLICY_VERSION, sha256(POLICY_BYTES).hexdigest()))
        self.assertEqual((result.integrity, result.execution_result), (PASS, PASS))
        self.assertEqual([row.test_id for row in result.coverage], [f"T{index:03d}" for index in range(1, 24)])
        component = self.coverage(result, "T001")
        self.assertEqual(component.assessment, PASS)
        self.assertEqual((component.clauses[0].required_count, component.clauses[0].executed_count,
                          component.clauses[0].passed_count), (4, 4, 4))
        unit = self.suite(result, "unit")
        self.assertEqual((unit.executed, unit.exit_code, unit.node_count, unit.subtest_count), (True, 0, 4, 0))
        self.assertEqual((unit.integrity, unit.execution), (PASS, PASS))
        self.assertIs(type(result.suites), tuple)
        self.assertIs(type(result.coverage), tuple)
        self.assertIsNone(result.codec_metrics)
        with self.assertRaises(FrozenInstanceError):
            result.run_id = "other"
        with self.assertRaises(FrozenInstanceError):
            component.clauses[0].passed_count = 99
        with self.assertRaises(FrozenInstanceError):
            unit.execution.verdict = "FAIL"

    def test_no_selected_execution_or_evidence_stays_not_run(self):
        for context, evidence in ((self.fixture.context, None),
                (replace(self.fixture.context, execution=None), self.fixture.evidence),
                (replace(self.fixture.context, execution=None), None)):
            with self.subTest(selected=context.execution is not None, supplied=evidence is not None):
                result = audit.audit_recorded_tests(context, evidence)
                self.assertEqual(result.execution_result, AuditAssessment("NOT_RUN", "INCOMPLETE"))
                self.assertNotEqual(self.coverage(result, "T001").assessment.verdict, "PASS")
        partial = replace(self.fixture.evidence, run_record=None)
        self.assertEqual(self.assess(evidence=partial).integrity.coverage, "INCOMPLETE")

    def test_independent_source_registry_run_and_execution_anchors_are_enforced(self):
        context = self.fixture.context
        changes = [replace(context, run_id="other-run"), replace(context, source_sha256="f" * 64),
                   replace(context, source_size=SOURCE_SIZE + 1), replace(context, registry_sha256="f" * 64),
                   replace(context, execution=replace(context.execution, sha256="f" * 64))]
        for changed in changes:
            with self.subTest(context=changed):
                self.assertEqual(self.assess(context=changed).integrity.verdict, "FAIL")
        wrong_registry = make_evidence(registry_contents=REGISTRY_BYTES.replace(b"offline-tests-1", b"offline-tests-2"))
        self.assertEqual(self.assess(wrong_registry).integrity.verdict, "FAIL")
        asserted = replace(wrong_registry.context, registry_sha256=wrong_registry.evidence.registry.record.sha256)
        self.assertEqual(self.assess(wrong_registry, context=asserted).integrity.verdict, "FAIL")

    def test_hash_and_size_checks_cover_every_retained_record(self):
        evidence = self.fixture.evidence
        targets = [("run_record", None), ("registry", None)] + [(field, index)
            for index in range(3) for field in ("receipt", "stdout", "stderr", "junit", "events", "cases")]
        for field, index in targets:
            original = getattr(evidence, field) if index is None else getattr(evidence.suites[index], field)
            changed_records = [replace(original, record=replace(original.record, sha256="f" * 64)),
                               replace(original, record=replace(original.record, size=original.record.size + 1))]
            if original.contents:
                equivalent = (original.contents[:-1] + b" " if original.contents.endswith(b"\n")
                              else original.contents.replace(b" ", b"\t", 1))
                self.assertNotEqual(equivalent, original.contents)
                self.assertEqual(len(equivalent), len(original.contents))
                changed_records.append(replace(original, contents=equivalent))
            for changed in changed_records:
                if index is None:
                    selected = replace(evidence, **{field: changed})
                else:
                    suites = list(evidence.suites)
                    suites[index] = replace(suites[index], **{field: changed})
                    selected = replace(evidence, suites=tuple(suites))
                with self.subTest(role=field, suite=index, reference=changed.record):
                    self.assertEqual(self.assess(evidence=selected).integrity.verdict, "FAIL")
        fixture = make_evidence(specifications={"unit": {"payloads": {"stdout": b"alpha"}}})
        unit = fixture.evidence.suites[0]
        changed = replace(unit, stdout=replace(unit.stdout, contents=b"omega"))
        selected = replace(fixture.evidence, suites=(changed, *fixture.evidence.suites[1:]))
        self.assertEqual(self.assess(fixture, evidence=selected).integrity.verdict, "FAIL")

    def test_cross_execution_duplicate_and_unlisted_suites_cannot_be_combined(self):
        other = make_evidence(execution_id="b" * 32)
        mixed = replace(self.fixture.evidence, suites=(other.evidence.suites[0], *self.fixture.evidence.suites[1:]))
        self.assertEqual(self.assess(evidence=mixed).integrity.verdict, "FAIL")
        duplicate = replace(self.fixture.evidence, suites=(self.fixture.evidence.suites[0],
                            self.fixture.evidence.suites[0], self.fixture.evidence.suites[2]))
        self.assertEqual(self.assess(evidence=duplicate).integrity.verdict, "FAIL")
        unlisted = make_evidence(run_change=lambda data: data["suites"].pop())
        self.assertEqual(self.assess(unlisted).integrity.verdict, "FAIL")
        duplicate_summary = make_evidence(run_change=lambda data: data["suites"].__setitem__(1, data["suites"][0]))
        self.assertEqual(self.assess(duplicate_summary).integrity.verdict, "FAIL")

    def test_suite_summary_identity_and_report_links_are_exact(self):
        changes = [lambda data: data["suites"][0].update(executed=False),
                   lambda data: data["suites"][0].update(exit_code=1),
                   lambda data: data["suites"][0].update(issues=["TEST_FAILURE"])]
        for change in changes:
            self.assertEqual(self.assess(make_evidence(run_change=change)).integrity.verdict, "FAIL")
        receipt_changes = [lambda data: data.update(runId="other-run"), lambda data: data.update(suite="ui"),
            lambda data: data["source"].update(sha256="f" * 64),
            lambda data: data["registry"].update(sha256="f" * 64),
            lambda data: data["artifacts"]["events"].update(sha256="f" * 64)]
        for index, change in enumerate(receipt_changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(make_evidence(receipt_changes={"unit": change})).integrity.verdict, "FAIL")

    def test_only_fixed_commands_and_pinned_test_versions_are_accepted(self):
        changes = [lambda data: data["argv"].append("--collect-only"),
            lambda data: data["argv"].__setitem__(data["argv"].index("pytest_reportlog.plugin"), "foreign_plugin"),
            lambda data: data["argv"].__setitem__(-2, "live_provider"),
            lambda data: data["argv"].__setitem__(-1, "tests/ui"),
            lambda data: data["runtime"]["packages"].update(pytest="0.0.1"),
            lambda data: data["runtime"]["packages"].update({"pytest-reportlog": "0.0.1"})]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.assertEqual(self.assess(make_evidence(receipt_changes={"unit": change})).integrity.verdict, "FAIL")
        historical = make_evidence(receipt_changes={"unit": lambda data: data["argv"].__setitem__(0, "/another/trusted/python")})
        self.assertEqual(self.assess(historical).execution_result, PASS)

    def test_source_process_and_report_flags_cannot_override_missing_execution(self):
        changes = [lambda data: data.update(sourceUnchanged=False, issues=["SOURCE_CHANGED"]),
            lambda data: data.update(reportsComplete=False),
            lambda data: data.update(issues=["PROCESS_TIMEOUT"]),
            lambda data: data.update(issues=["OUTPUT_LIMIT"]),
            lambda data: data.update(issues=["TEST_SKIPPED"])]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                result = self.assess(make_evidence(receipt_changes={"unit": change}))
                self.assertNotEqual(self.suite(result, "unit").execution.verdict, "PASS")
                self.assertNotEqual(self.coverage(result, "T001").assessment.verdict, "PASS")
        changed = self.assess(make_evidence(receipt_changes={"unit": changes[0]}))
        self.assertEqual(changed.execution_result.verdict, "FAIL")

    def test_normalized_cases_are_recomputed_from_raw_reports(self):
        changes = [lambda data: data.update(nodeIds=[]), lambda data: data.update(subtestReports=99),
                   lambda data: data["reports"][0].update(outcome="failed"),
                   lambda data: data.update(sessionExitCode=1)]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                fixture = make_evidence(specifications={"unit": {"cases": change}})
                self.assertEqual(self.assess(fixture).integrity.verdict, "FAIL")
        fixture = make_evidence(specifications={"unit": {"nodes": UNIT_NODES[:-1],
            "cases": lambda data: data["nodeIds"].append(UNIT_NODES[-1])}})
        self.assertEqual(self.assess(fixture).integrity.verdict, "FAIL")

    def test_failed_setup_call_teardown_and_subtests_remain_failures(self):
        mutations = [lambda events, phase=phase: fail_phase(events, phase=phase) for phase in ("setup", "call", "teardown")]
        mutations.append(lambda events: add_subtest(events, outcome="failed"))
        for index, mutation in enumerate(mutations):
            fixture = make_evidence(specifications={"unit": {"events": mutation}}, missing_suites=("integration_offline",))
            with self.subTest(case=index):
                result = self.assess(fixture)
                self.assertEqual(result.execution_result.verdict, "FAIL")
                self.assertEqual(self.coverage(result, "T001").assessment.verdict, "FAIL")
                self.assertEqual(self.suite(result, "integration_offline").execution.coverage, "INCOMPLETE")
        fixture = make_evidence(specifications={"unit": {"events": lambda events: add_subtest(events)}})
        self.assertEqual(self.suite(self.assess(fixture), "unit").subtest_count, 1)
        self.assertEqual(self.coverage(self.assess(fixture), "T001").assessment, PASS)

    def test_skips_xfail_xpass_and_no_tests_never_count_as_passed_coverage(self):
        def change_call(events, outcome, xfail):
            row = next(row for row in events if row.get("nodeid") == UNIT_NODES[0] and row.get("when") == "call")
            row["outcome"] = outcome
            if xfail:
                row["wasxfail"] = "fixture expectation"
        for outcome, xfail in (("skipped", False), ("skipped", True), ("passed", True)):
            fixture = make_evidence(specifications={"unit": {"events": lambda events: change_call(events, outcome, xfail)}})
            with self.subTest(outcome=outcome, xfail=xfail):
                result = self.assess(fixture)
                self.assertNotEqual(result.execution_result.verdict, "PASS")
                self.assertNotEqual(self.coverage(result, "T001").assessment.verdict, "PASS")
                self.assertLess(self.coverage(result, "T001").clauses[0].passed_count, 4)
        empty = self.assess(make_evidence(specifications={"unit": {"nodes": ()}}))
        self.assertNotEqual(empty.execution_result.verdict, "PASS")
        self.assertEqual(self.suite(empty, "unit").node_count, 0)

    def test_exact_required_nodes_and_complete_sessions_cannot_be_replaced_by_counts(self):
        partial = make_evidence(specifications={"unit": {"nodes": (*UNIT_NODES[:-1], UNIT_NODES[-1] + "_extra")}})
        result = self.assess(partial)
        self.assertEqual(self.suite(result, "unit").node_count, 4)
        self.assertNotEqual(self.coverage(result, "T001").assessment.verdict, "PASS")
        self.assertEqual(self.coverage(result, "T001").clauses[0].passed_count, 3)
        events = self.fixture.evidence.suites[0].events.contents
        malformed = [b"\n".join(events.splitlines()[:-1]) + b"\n",
                     events.replace(b'"exitstatus": 0', b'"exitstatus": 1')]
        for contents in malformed:
            fixture = make_evidence(specifications={"unit": {"payloads": {"events": contents}}})
            self.assertEqual(self.assess(fixture).integrity.verdict, "FAIL")
        repeated = make_evidence(specifications={"unit": {"events": lambda events: events.insert(2, deepcopy(events[1]))}})
        self.assertEqual(self.assess(repeated).integrity.verdict, "FAIL")

    def test_junit_failure_error_skip_and_aggregate_counters_cannot_hide_behind_passed_events(self):
        def add_status(root, tag):
            ElementTree.SubElement(next(root.iter("testcase")), tag, message="private-junit-token")
            field = {"failure": "failures", "error": "errors", "skipped": "skipped"}[tag]
            next(root.iter("testsuite")).set(field, "1")
        changes = [lambda root, tag=tag: add_status(root, tag) for tag in ("failure", "error", "skipped")]
        changes.extend(lambda root, field=field: next(root.iter("testsuite")).set(field, "1")
                       for field in ("failures", "errors", "skipped"))
        for index, change in enumerate(changes):
            fixture = make_evidence(specifications={"unit": {"junit": change}})
            records = fixture.evidence.suites[0]
            # The unchanged historical parser checks identities, so the new audit owns this guard.
            self.assertEqual(PARSE_REPORTS(records.events.contents, records.junit.contents, 0)[1], ())
            with self.subTest(case=index):
                result = self.assess(fixture)
                self.assertEqual(result.integrity.verdict, "FAIL")
                self.assertNotEqual(self.coverage(result, "T001").assessment.verdict, "PASS")
                self.assertNotIn("private-junit-token", repr(result))
        def move_failure(root):
            cases = list(root.iter("testcase"))
            original = next(case for case in cases if case.find("failure") is not None)
            failure = original.find("failure")
            original.remove(failure)
            next(case for case in cases if case is not original).append(failure)
        fixture = make_evidence(specifications={"unit": {"events": fail_phase, "junit": move_failure}})
        records = fixture.evidence.suites[0]
        self.assertEqual(PARSE_REPORTS(records.events.contents, records.junit.contents, 1)[1], ("TEST_FAILURE",))
        result = self.assess(fixture)
        self.assertEqual(result.integrity.verdict, "FAIL")
        self.assertIn("JUNIT_STATUS_MISMATCH", self.suite(result, "unit").issues)

    def test_t006_metrics_require_the_exact_call_stdout_and_single_record(self):
        nodes = (*UNIT_NODES, METRIC_NODE)
        fixture = make_evidence(specifications={"unit": {"nodes": nodes, "events": add_metrics}})
        result = self.assess(fixture)
        self.assertEqual(self.coverage(result, "T006").assessment, PASS)
        self.assertEqual((result.codec_metrics.frames, result.codec_metrics.color,
                          result.codec_metrics.edge_exclusion_pixels, result.codec_metrics.pixels_per_frame),
                         (120, "8-bit RGB", 2, 919552))
        self.assertEqual(tuple(region.crop for region in result.codec_metrics.regions),
                         tuple(tuple(region["crop"]) for region in codec_report()["regions"]))
        for region in result.codec_metrics.regions:
            self.assertEqual((region.minimum_psnr_db, region.lossless, region.minimum_ssim), (30.0, False, 0.95))
        for change in (None, lambda events: add_metrics(events, node=UNIT_NODES[0]),
                       lambda events: add_metrics(events, phase="setup")):
            candidate = make_evidence(specifications={"unit": {"nodes": nodes, "events": change}})
            result = self.assess(candidate)
            self.assertIsNone(result.codec_metrics)
            self.assertEqual(self.coverage(result, "T006").assessment.coverage, "INCOMPLETE")
        duplicate = make_evidence(specifications={"unit": {"nodes": nodes, "events": lambda events: add_metrics(events, copies=2)}})
        self.assertEqual(self.coverage(self.assess(duplicate), "T006").assessment.verdict, "FAIL")

    def test_t006_metrics_enforce_shape_thresholds_and_lossless_infinity(self):
        changes = [lambda data: data.update(frames=119), lambda data: data.update(frames=True),
            lambda data: data.update(color="YUV"), lambda data: data.update(edgeExclusionPixels=1),
            lambda data: data.update(pixelsPerFrame=1), lambda data: data["regions"].pop(),
            lambda data: data["regions"][0].update(crop=[574, 720, 1, 0]),
            lambda data: data["regions"][0].update(minimumPsnrDb=29.999),
            lambda data: data["regions"][0].update(minimumSsim=0.94999),
            lambda data: data["regions"][0].update(minimumPsnrDb=float("nan")),
            lambda data: data["regions"][0].update(minimumPsnrDb=float("-inf")),
            lambda data: data["regions"][0].update(minimumSsim=float("inf"))]
        for index, change in enumerate(changes):
            report = codec_report()
            change(report)
            fixture = make_evidence(specifications={"unit": {"nodes": (METRIC_NODE,),
                "events": lambda events: add_metrics(events, report)}})
            with self.subTest(case=index):
                result = self.assess(fixture)
                self.assertIsNone(result.codec_metrics)
                self.assertEqual(self.coverage(result, "T006").assessment.verdict, "FAIL")
        report = codec_report()
        report["regions"][0]["minimumPsnrDb"] = float("inf")
        fixture = make_evidence(specifications={"unit": {"nodes": (METRIC_NODE,), "events": lambda events: add_metrics(events, report)}})
        result = self.assess(fixture)
        self.assertEqual(self.coverage(result, "T006").assessment, PASS)
        self.assertEqual((result.codec_metrics.regions[0].minimum_psnr_db, result.codec_metrics.regions[0].lossless), (None, True))

    def test_missing_suites_keep_other_passes_and_fixed_p0_gaps_explicit(self):
        result = self.assess(make_evidence(missing_suites=("integration_offline",)))
        self.assertEqual(self.coverage(result, "T001").assessment, PASS)
        self.assertEqual(self.suite(result, "integration_offline").execution.coverage, "INCOMPLETE")
        self.assertNotEqual(result.execution_result.verdict, "PASS")
        nodes = EXPECTED_NODES
        unit = tuple(node for node in nodes if node.startswith("tests/unit/"))
        ui = tuple(node for node in nodes if node.startswith("tests/ui/"))
        all_available = make_evidence(specifications={"unit": {"nodes": unit, "events": add_metrics}, "ui": {"nodes": ui}})
        result = self.assess(all_available)
        self.assertEqual(result.execution_result, PASS)
        for test_id in ("T010", "T014", "T015", "T017", "T019", "T020", "T021", "T022", "T023"):
            with self.subTest(test_id=test_id):
                coverage = self.coverage(result, test_id)
                self.assertNotEqual(coverage.assessment.verdict, "PASS")
                missing = [clause for clause in coverage.clauses if clause.required_count == 0]
                self.assertTrue(missing)
                self.assertTrue(all(clause.assessment.coverage == "INCOMPLETE" and clause.reason_codes for clause in missing))
        self.assertNotEqual(self.coverage(self.assess(), "T023").assessment.verdict, "PASS")
        for field in ("liveVerified", "visualEffectDemonstrated", "automatedWorkflowComplete", "releaseClassification"):
            self.assertNotIn(field, asdict(result))

    def test_fixed_policy_preserves_all_reviewed_nodes_and_external_gaps(self):
        self.assertEqual(tuple(test_id for test_id, _ in audit.P0_CASES), tuple(f"T{index:03d}" for index in range(1, 24)))
        actual_nodes = {node for _, clauses in audit.P0_CASES for clause in clauses for node in clause.required_nodes}
        self.assertEqual(len(actual_nodes), 67)
        self.assertEqual(actual_nodes, set(EXPECTED_NODES))
        clauses = {clause.clause_id: clause for _, selected in audit.P0_CASES for clause in selected}
        self.assertEqual(clauses["T001.1"].required_nodes, UNIT_NODES)
        self.assertEqual(clauses["T006.1"].required_nodes, (METRIC_NODE,))
        for clause_id, kind, code in (("T017.1", "qa", "IMPLEMENTATION_MISSING"),
                ("T022.2", "qa", "IMPLEMENTATION_MISSING"), ("T023.2", "workflow", "IMPLEMENTATION_MISSING"),
                ("T018.2", "human_review", "EXTERNAL_EVIDENCE_REQUIRED")):
            with self.subTest(clause=clause_id):
                clause = clauses[clause_id]
                self.assertEqual((clause.kind, clause.required_nodes, clause.missing_code), (kind, (), code))
        self.assertEqual(clauses["T023.1"].kind, "viewer")
        self.assertEqual(clauses["T023.1"].required_nodes, (VIEWER_NODE,))

    def test_untrusted_json_scalar_shapes_and_secrets_never_reach_public_results(self):
        changes = [lambda data: data.update(executed=1), lambda data: data.update(exitCode=True),
            lambda data: data.update(startedAtUnix="private-token"), lambda data: data.update(issues=["private-token"]),
            lambda data: data.update(unexpected="private-token"), lambda data: data["runtime"].update(python=True)]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                result = self.assess(make_evidence(receipt_changes={"unit": change}))
                self.assertEqual(result.integrity.verdict, "FAIL")
                self.assertNotIn("private-token", repr(result))
        for contents in (b"not JSON", b'{"private-token":NaN}', b'{"nodeIds":[],"nodeIds":[]}'):
            fixture = make_evidence(specifications={"unit": {"payloads": {"cases": contents}}})
            result = self.assess(fixture)
            self.assertEqual(result.integrity.verdict, "FAIL")
            self.assertNotIn("private-token", repr(result))
        def private_failure(events):
            fail_phase(events)
            for row in events:
                if row.get("outcome") == "failed":
                    row["longrepr"] = "private-token failure body"
            add_subtest(events, outcome="failed")
        fixture = make_evidence(specifications={"unit": {"nodes": (*UNIT_NODES, "tests/unit/private-token.py::Hidden::test_private-token"),
            "events": private_failure,
            "payloads": {"stdout": b"private-token", "stderr": b"private-token"}}})
        result = self.assess(fixture)
        self.assertEqual(result.execution_result.verdict, "FAIL")
        self.assertNotIn("private-token", repr(result))
        self.assertNotIn("private-subtest-token", repr(result))

    def test_mutable_malformed_and_oversized_api_values_raise_safe_errors(self):
        context, evidence = self.fixture.context, self.fixture.evidence
        cases = [(None, evidence), (replace(context, run_id="../private-token"), evidence),
            (replace(context, source_size=True), evidence), (replace(context, source_size=32 * 1024 * 1024 + 1), evidence),
            (replace(context, source_sha256="private-token"), evidence), (context, {}),
            (context, replace(evidence, suites=list(evidence.suites))),
            (context, replace(evidence, registry=replace(evidence.registry, contents=bytearray(REGISTRY_BYTES)))),
            (context, replace(evidence, registry=replace(evidence.registry, record=replace(evidence.registry.record, size=True))))]
        for index, (selected_context, selected_evidence) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(audit.RecordedTestAuditError) as caught:
                    audit.audit_recorded_tests(selected_context, selected_evidence)
                self.assertNotIn("private-token", str(caught.exception))
        oversized = evidence_record(evidence.run_record.record.path, b"x" * (64 * 1024 + 1))
        with self.assertRaises(audit.RecordedTestAuditError):
            self.assess(evidence=replace(evidence, run_record=oversized))
        oversized = evidence_record(evidence.registry.record.path, b"x" * (16 * 1024 + 1))
        with self.assertRaises(audit.RecordedTestAuditError):
            self.assess(evidence=replace(evidence, registry=oversized))


if __name__ == "__main__":
    unittest.main()
