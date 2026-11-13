"""Check one externally selected offline test execution without running code."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import xml.etree.ElementTree as ElementTree

from core.json_data import unique_fields
from core.run_audit import AuditAssessment, AuditIssue
from core.storage import StoredArtifact, _path_parts
from qa.p0_cases import METRIC_CROPS, P0_CASES, POLICY_VERSION, T006_NODE, policy_bytes
from qa.record_tests import MAX_SOURCE_BYTES
from qa.run_evidence import _assessment, _matches, _time
from qa.test_registry import DEFAULT_SUITES, PYTEST_OPTIONS, PYTEST_VERSION, REPORTLOG_VERSION, SUITES, registry_bytes
from qa.test_results import MAX_REPORT_BYTES, MAX_STREAM_BYTES, bin_xml_escape, mangle_test_address, parse_reports


MAX_RECEIPT_BYTES = 64 * 1024
MAX_REGISTRY_BYTES = 16 * 1024
MAX_CAPTURE_BYTES = 320 * 1024 * 1024
_REPORTS = ("stdout", "stderr", "junit", "events", "cases")
_SUFFIXES = dict(zip(_REPORTS, ("stdout.txt", "stderr.txt", "junit.xml", "events.jsonl", "cases.json")))
_PACKAGES = ("pytest", "pytest-reportlog", "iniconfig", "pluggy", "packaging", "pygments",
             "Pillow", "jsonschema", "httpx", "runwayml", "playwright")
_PROCESS_ISSUES = {"PROCESS_TIMEOUT", "OUTPUT_LIMIT", "PROCESS_START_FAILED", "PROCESS_IO_FAILED"}
_REPORT_ISSUES = {"TEST_FAILURE", "TEST_XPASS", "TEST_XFAIL", "TEST_SKIPPED", "NO_TESTS"}
_ISSUES = _PROCESS_ISSUES | _REPORT_ISSUES | {"SUITE_MISSING", "REPORT_INVALID", "SOURCE_CHANGED", "CLOCK_MOVED_BACKWARDS"}
_MISSING_ISSUES = {"TEST_XFAIL", "TEST_SKIPPED", "NO_TESTS", "SUITE_MISSING"}
_RECEIPT_FIELDS = {"schemaVersion", "manifestType", "runId", "suite", "source", "registry", "runtime",
                   "executed", "argv", "startedAtUnix", "finishedAtUnix", "elapsedSeconds", "exitCode",
                   "issues", "sourceUnchanged", "reportsComplete", "artifacts"}


class RecordedTestAuditError(ValueError):
    """The audit needs fixed immutable values within the documented limits."""


@dataclass(frozen=True)
class TestRecord:
    record: StoredArtifact
    contents: bytes


@dataclass(frozen=True)
class ApprovedTestContext:
    run_id: str
    source_sha256: str
    source_size: int
    registry_sha256: str
    execution: StoredArtifact | None


@dataclass(frozen=True)
class SuiteRecords:
    name: str
    receipt: TestRecord | None = None
    stdout: TestRecord | None = None
    stderr: TestRecord | None = None
    junit: TestRecord | None = None
    events: TestRecord | None = None
    cases: TestRecord | None = None


@dataclass(frozen=True)
class RecordedTestEvidence:
    run_record: TestRecord | None = None
    registry: TestRecord | None = None
    suites: tuple[SuiteRecords, ...] = ()


@dataclass(frozen=True)
class CodecRegion:
    crop: tuple[int, ...]
    minimum_psnr_db: float | None
    lossless: bool
    minimum_ssim: float


@dataclass(frozen=True)
class CodecMetrics:
    frames: int
    color: str
    edge_exclusion_pixels: int
    pixels_per_frame: int
    regions: tuple[CodecRegion, ...]


@dataclass(frozen=True)
class SuiteAudit:
    name: str
    record: StoredArtifact | None
    executed: bool | None
    exit_code: int | None
    node_count: int
    subtest_count: int
    integrity: AuditAssessment
    execution: AuditAssessment
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ClauseCoverage:
    clause_id: str
    kind: str
    required_count: int
    executed_count: int
    passed_count: int
    assessment: AuditAssessment
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class TestCoverage:
    test_id: str
    clauses: tuple[ClauseCoverage, ...]
    assessment: AuditAssessment


@dataclass(frozen=True)
class RecordedTestAudit:
    run_id: str
    source: StoredArtifact
    registry_sha256: str
    execution: StoredArtifact | None
    policy_version: str
    policy_sha256: str
    integrity: AuditAssessment
    execution_result: AuditAssessment
    suites: tuple[SuiteAudit, ...]
    coverage: tuple[TestCoverage, ...]
    codec_metrics: CodecMetrics | None
    issues: tuple[AuditIssue, ...]


def _note(issues, category, code, *, missing=False):
    issue = AuditIssue(category, code, missing)
    if issue not in issues:
        issues.append(issue)


def _reference(value, limit, *, empty=False):
    if (type(value) is not StoredArtifact or type(value.size) is not int
            or not (0 if empty else 1) <= value.size <= limit or not _matches(value.sha256, r"[0-9a-f]{64}")):
        raise RecordedTestAuditError("Test references need bounded sizes and hashes.")
    _path_parts(value.path)


def _record_shape(value, limit, *, empty=False):
    if value is None:
        return 0
    if type(value) is not TestRecord or type(value.contents) is not bytes or len(value.contents) > limit:
        raise RecordedTestAuditError("Test records need bounded immutable bytes.")
    _reference(value.record, limit, empty=empty)
    return len(value.contents)


def _check_inputs(context, evidence):
    if (type(context) is not ApprovedTestContext or not _matches(context.run_id, r"[A-Za-z0-9_-]{1,64}")
            or not _matches(context.source_sha256, r"[0-9a-f]{64}")
            or type(context.source_size) is not int or not 0 < context.source_size <= MAX_SOURCE_BYTES
            or not _matches(context.registry_sha256, r"[0-9a-f]{64}")):
        raise RecordedTestAuditError("Test context must use its fixed bounded identity.")
    if context.execution is not None:
        _reference(context.execution, MAX_RECEIPT_BYTES)
    if evidence is None:
        return
    if (type(evidence) is not RecordedTestEvidence or type(evidence.suites) is not tuple
            or len(evidence.suites) > len(DEFAULT_SUITES)):
        raise RecordedTestAuditError("Test evidence must contain at most the three fixed suites.")
    size = _record_shape(evidence.run_record, MAX_RECEIPT_BYTES)
    size += _record_shape(evidence.registry, MAX_REGISTRY_BYTES)
    for suite in evidence.suites:
        if type(suite) is not SuiteRecords or type(suite.name) is not str or len(suite.name) > 64:
            raise RecordedTestAuditError("Suite evidence must use its fixed fields.")
        size += _record_shape(suite.receipt, MAX_RECEIPT_BYTES)
        for name in _REPORTS:
            stream = name in ("stdout", "stderr")
            size += _record_shape(getattr(suite, name), MAX_STREAM_BYTES if stream else MAX_REPORT_BYTES, empty=stream)
    if size > MAX_CAPTURE_BYTES:
        raise RecordedTestAuditError("Test evidence exceeds the total byte limit.")


def _verified(item, path, issues, category):
    if item is None:
        _note(issues, category, "RECORD_MISSING", missing=True)
        return None
    if (item.record.path != path or len(item.contents) != item.record.size
            or sha256(item.contents).hexdigest() != item.record.sha256):
        _note(issues, category, "RECORD_MISMATCH")
        return None
    return item.contents


def _reject_constant(value):
    raise ValueError("Receipt numbers must be finite.")


def _json(contents):
    body = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_fields, parse_constant=_reject_constant)
    if type(body) is not dict:
        raise ValueError("A receipt must be an object.")
    json.dumps(body, allow_nan=False)
    return body


def _fields(body, expected):
    if type(body) is not dict or set(body) != set(expected):
        raise ValueError("Receipt fields differ from the fixed format.")


def _identity(body, context, source, registry):
    for field, limit in (("source", MAX_SOURCE_BYTES), ("registry", MAX_REGISTRY_BYTES)):
        _fields(body[field], ("path", "sha256", "size"))
        _reference(StoredArtifact(**body[field]), limit)
    if (body["schemaVersion"] != "1.4" or body["runId"] != context.run_id
            or body["source"] != asdict(source) or body["registry"] != asdict(registry)):
        raise ValueError("Receipt identity differs from the approved execution.")


def _issue_list(value):
    if (type(value) is not list or len(value) > len(_ISSUES) or any(type(item) is not str or item not in _ISSUES for item in value)
            or len(value) != len(set(value))):
        raise ValueError("Receipt issues do not use the fixed vocabulary.")


def _top_records(context, evidence, source, registry, issues):
    execution = None
    if context.execution is None:
        _note(issues, "integrity", "EXECUTION_ANCHOR_MISSING", missing=True)
        # Use an untrusted path only to inspect its graph. The missing external
        # anchor still blocks every suite; the path never grants authority.
        if evidence and evidence.run_record and _matches(evidence.run_record.record.path, r"qa/test-runs/[0-9a-f]{32}/run\.json"):
            execution = evidence.run_record.record.path.rsplit("/", 1)[0]
    else:
        path = context.execution.path
        if not _matches(path, r"qa/test-runs/[0-9a-f]{32}/run\.json"):
            _note(issues, "integrity", "EXECUTION_PATH_INVALID")
        else:
            execution = path.rsplit("/", 1)[0]
    if context.registry_sha256 != registry.sha256:
        _note(issues, "integrity", "APPROVED_REGISTRY_MISMATCH")
    if evidence is None:
        _note(issues, "integrity", "TEST_EVIDENCE_MISSING", missing=True)
        return execution, {}
    registry_body = _verified(evidence.registry, registry.path, issues, "integrity")
    if registry_body is not None and registry_body != registry_bytes():
        _note(issues, "integrity", "REGISTRY_BYTES_MISMATCH")
    selected = evidence.run_record
    if selected is not None and context.execution is not None and selected.record != context.execution:
        _note(issues, "integrity", "EXECUTION_ANCHOR_MISMATCH")
    contents = _verified(selected, execution + "/run.json" if execution else "", issues, "integrity")
    if contents is None:
        return execution, {}
    try:
        body = _json(contents)
        _fields(body, ("schemaVersion", "manifestType", "runId", "source", "registry", "suites"))
        _identity(body, context, source, registry)
        if body["manifestType"] != "recorded_test_run" or type(body["suites"]) is not list or not 1 <= len(body["suites"]) <= 3:
            raise ValueError("Run suite list is invalid.")
        summaries = {}
        for summary in body["suites"]:
            _fields(summary, ("name", "record", "executed", "exit_code", "issues"))
            name = summary["name"]
            if type(name) is not str or name not in SUITES or name in summaries or type(summary["executed"]) is not bool:
                raise ValueError("Run suite identity is invalid.")
            _issue_list(summary["issues"])
            if summary["exit_code"] is not None and (type(summary["exit_code"]) is not int or not -128 <= summary["exit_code"] <= 255):
                raise ValueError("Run suite exit code is invalid.")
            _fields(summary["record"], ("path", "sha256", "size"))
            reference = StoredArtifact(**summary["record"])
            _reference(reference, MAX_RECEIPT_BYTES)
            if reference.path != execution + "/" + name + ".json":
                raise ValueError("Suite execution differs from the selected run.")
            summaries[name] = summary
        return execution, summaries
    except Exception:
        _note(issues, "integrity", "RUN_RECEIPT_INVALID")
        return execution, {}


def _runtime_shape(runtime):
    _fields(runtime, ("python", "packages", "tools"))
    if type(runtime["python"]) is not str or not _matches(runtime["python"], r"[0-9]+\.[0-9]+\.[0-9]+") or len(runtime["python"]) > 32:
        raise ValueError("Recorded Python version is invalid.")
    _fields(runtime["packages"], _PACKAGES)
    if any(value is not None and (type(value) is not str or not 1 <= len(value) <= 128)
           for value in runtime["packages"].values()):
        raise ValueError("Recorded package version is invalid.")
    if (runtime["packages"]["pytest"] != PYTEST_VERSION
            or runtime["packages"]["pytest-reportlog"] != REPORTLOG_VERSION):
        raise ValueError("Recorded pytest tools differ from the fixed pins.")
    tools = runtime["tools"]
    if (type(tools) is not dict or set(tools) not in ({"ffmpeg", "ffprobe"}, {"ffmpeg", "ffprobe", "browser"})
            or any(value is not None and (type(value) is not str or not 1 <= len(value) <= 1024) for value in tools.values())):
        raise ValueError("Recorded media tools are invalid.")


def _receipt_shape(body, name, context, source, registry):
    _fields(body, _RECEIPT_FIELDS)
    _identity(body, context, source, registry)
    if body["manifestType"] != "recorded_test_execution" or body["suite"] != name:
        raise ValueError("Suite receipt identity differs from the selection.")
    _runtime_shape(body["runtime"])
    _issue_list(body["issues"])
    _fields(body["artifacts"], _REPORTS)
    if (type(body["executed"]) is not bool or type(body["reportsComplete"]) is not bool
            or body["sourceUnchanged"] is not None and type(body["sourceUnchanged"]) is not bool):
        raise ValueError("Suite booleans are invalid.")
    if "SUITE_MISSING" in body["issues"]:
        empty_fields = ("argv", "startedAtUnix", "finishedAtUnix", "elapsedSeconds", "exitCode", "sourceUnchanged")
        if (body["issues"] != ["SUITE_MISSING"] or body["executed"] or body["reportsComplete"]
                or any(body[key] is not None for key in empty_fields)
                or any(value is not None for value in body["artifacts"].values())):
            raise ValueError("Missing suite receipt contains execution claims.")
        return
    path, marker = SUITES[name]
    expected = [*PYTEST_OPTIONS, "-o", "markers=live_provider: explicitly authorized provider work",
                "--junitxml=../reports/junit.xml", "--report-log=../reports/events.jsonl", "-m", marker, path]
    argv = body["argv"]
    if (type(argv) is not list or len(argv) != len(expected) + 1 or type(argv[0]) is not str
            or not 1 <= len(argv[0]) <= 4096 or any(ord(char) < 32 for char in argv[0]) or argv[1:] != expected):
        raise ValueError("Test command differs from the fixed suite.")
    if (not _time(body["startedAtUnix"]) or not _time(body["finishedAtUnix"])
            or type(body["elapsedSeconds"]) not in (int, float) or not math.isfinite(body["elapsedSeconds"])
            or body["elapsedSeconds"] < 0 or body["sourceUnchanged"] is None
            or body["exitCode"] is not None and (type(body["exitCode"]) is not int or not -128 <= body["exitCode"] <= 255)):
        raise ValueError("Test process facts are invalid.")
    if body["executed"] != (body["exitCode"] is not None):
        raise ValueError("Executed state differs from the process exit.")


def _reports(records, receipt, issues):
    payloads = {}
    for name in _REPORTS:
        item = getattr(records, name)
        declared = receipt["artifacts"][name]
        if declared is not None:
            try:
                _fields(declared, ("path", "sha256", "size"))
                stream = name in ("stdout", "stderr")
                _reference(StoredArtifact(**declared), MAX_STREAM_BYTES if stream else MAX_REPORT_BYTES, empty=stream)
            except Exception:
                _note(issues, "integrity", name.upper() + "_REFERENCE_INVALID")
        if item is None and declared is None:
            if receipt["executed"]:
                _note(issues, "integrity", name.upper() + "_MISSING", missing=True)
            continue
        if item is None:
            _note(issues, "integrity", name.upper() + "_MISSING", missing=True)
            continue
        if declared != asdict(item.record):
            _note(issues, "integrity", name.upper() + "_REFERENCE_MISMATCH")
        path = f"qa/test-files/{item.record.sha256}.{_SUFFIXES[name]}"
        payloads[name] = _verified(item, path, issues, "integrity")
    return payloads


def _canonical_matches(value, contents):
    if contents is None:
        return False
    offset = 0
    encoder = json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False)
    for text in encoder.iterencode(value):
        chunk = text.encode("utf-8")
        end = offset + len(chunk)
        if end > MAX_REPORT_BYTES or contents[offset:end] != chunk:
            return False
        offset = end
    return offset + 1 == len(contents) and contents[offset:] == b"\n"


def _junit_status(junit, reports):
    """Check the pinned writer's status evidence in addition to node identity."""
    xml = ElementTree.fromstring(junit)
    statuses = {name: sum(1 for _ in xml.iter(name)) for name in ("failure", "error", "skipped")}
    for suite in xml.iter("testsuite"):
        for field, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            count = suite.get(field)
            if count is None or not _matches(count, r"[0-9]{1,6}"):
                raise ValueError("JUnit counters are missing or invalid.")
            if int(count) != sum(1 for _ in suite.iter(tag)):
                raise ValueError("JUnit counters contradict its status elements.")
    if xml.tag == "testsuites":
        for field, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            count = xml.get(field)
            if count is not None and (not _matches(count, r"[0-9]{1,6}") or int(count) != statuses[tag]):
                raise ValueError("JUnit aggregate counters contradict its status elements.")
    failed = any(row["outcome"] == "failed" for row in reports)
    skipped = any(row["outcome"] == "skipped" for row in reports)
    if bool(statuses["failure"] + statuses["error"]) != failed or bool(statuses["skipped"]) != skipped:
        raise ValueError("JUnit status contradicts the recorded report outcomes.")
    expected, actual = {}, {}
    for row in reports:
        if row["reportType"] == "CollectReport" and row["outcome"] == "passed":
            continue
        names = mangle_test_address(row["nodeId"])
        identity = (".".join(names[:-1]), bin_xml_escape(names[-1]))
        status = expected.setdefault(identity, set())
        if row["outcome"] == "failed":
            status.add("failure" if row["phase"] == "call" else "error")
        elif row["outcome"] == "skipped":
            status.add("skipped")
    for case in xml.iter("testcase"):
        identity = (case.get("classname"), case.get("name"))
        actual.setdefault(identity, set()).update(child.tag for child in case if child.tag in statuses)
    if actual != expected:
        raise ValueError("JUnit node statuses contradict their recorded phases.")


def _parse_suite(receipt, payloads, issues):
    facts = None
    try:
        facts, report_issues = parse_reports(payloads.get("events"), payloads.get("junit"), receipt["exitCode"])
        for code in report_issues:
            _note(issues, "execution", code, missing=code in _MISSING_ISSUES)
        if payloads.get("cases") is not None and not _canonical_matches(facts, payloads["cases"]):
            _note(issues, "integrity", "CASES_MISMATCH")
        try:
            _junit_status(payloads["junit"], facts["reports"])
        except Exception:
            _note(issues, "integrity", "JUNIT_STATUS_MISMATCH")
        if set(receipt["issues"]) & _REPORT_ISSUES != set(report_issues):
            _note(issues, "integrity", "REPORT_ISSUES_MISMATCH")
        if "REPORT_INVALID" in receipt["issues"]:
            _note(issues, "integrity", "REPORT_INVALID_CLAIM_MISMATCH")
    except Exception:
        missing = payloads.get("events") is None or payloads.get("junit") is None
        _note(issues, "integrity", "REPORT_MISSING" if missing else "REPORT_INVALID", missing=missing)
    return facts


def _suite_audit(name, records, summary, directory, context, source, registry, shared):
    issues, receipt, facts, payloads = [], None, None, {}
    if summary is None:
        _note(issues, "integrity", "SUITE_SELECTION_MISSING", missing=True)
    if records is None or records.receipt is None:
        _note(issues, "integrity", "SUITE_EVIDENCE_MISSING", missing=True)
    else:
        contents = _verified(records.receipt, (directory or "") + "/" + name + ".json", issues, "integrity")
        if summary is not None and summary["record"] != asdict(records.receipt.record):
            _note(issues, "integrity", "SUITE_REFERENCE_MISMATCH")
        if contents is not None:
            try:
                receipt = _json(contents)
                _receipt_shape(receipt, name, context, source, registry)
                if summary is not None and (summary["executed"] != receipt["executed"]
                        or summary["exit_code"] != receipt["exitCode"] or summary["issues"] != receipt["issues"]):
                    _note(issues, "integrity", "SUITE_SUMMARY_MISMATCH")
            except Exception:
                receipt = None
                _note(issues, "integrity", "SUITE_RECEIPT_INVALID")
    if receipt is not None:
        for code in receipt["issues"]:
            _note(issues, "execution", code, missing=code in _MISSING_ISSUES)
        payloads = _reports(records, receipt, issues)
        if "SUITE_MISSING" not in receipt["issues"]:
            facts = _parse_suite(receipt, payloads, issues)
            if not receipt["sourceUnchanged"]:
                _note(issues, "execution", "SOURCE_CHANGED")
            if (receipt["sourceUnchanged"] is False) != ("SOURCE_CHANGED" in receipt["issues"]):
                _note(issues, "integrity", "SOURCE_STATUS_MISMATCH")
            if (receipt["finishedAtUnix"] < receipt["startedAtUnix"]) != ("CLOCK_MOVED_BACKWARDS" in receipt["issues"]):
                _note(issues, "integrity", "CLOCK_STATUS_MISMATCH")
            complete = receipt["executed"] and facts is not None and not set(receipt["issues"]) & _PROCESS_ISSUES
            if receipt["reportsComplete"] != complete:
                _note(issues, "integrity", "REPORT_COMPLETENESS_MISMATCH")
            if not complete:
                _note(issues, "execution", "EXECUTION_INCOMPLETE", missing=True)
    integrity = _assessment([issue for issue in issues if issue.category == "integrity"], shared=shared)
    execution = _assessment([issue for issue in issues if issue.category == "execution"], shared=integrity)
    audit = SuiteAudit(name, records.receipt.record if records and records.receipt else None,
                       receipt["executed"] if receipt else None, receipt["exitCode"] if receipt else None,
                       len(facts["nodeIds"]) if facts else 0, facts["subtestReports"] if facts else 0,
                       integrity, execution, tuple(dict.fromkeys(issue.code for issue in issues)))
    return audit, facts, payloads, issues


def _codec_value(body):
    _fields(body, ("frames", "color", "edgeExclusionPixels", "pixelsPerFrame", "regions"))
    if (type(body["frames"]) is not int or body["frames"] != 120 or body["color"] != "8-bit RGB"
            or type(body["edgeExclusionPixels"]) is not int or body["edgeExclusionPixels"] != 2
            or type(body["pixelsPerFrame"]) is not int or body["pixelsPerFrame"] != 919552
            or type(body["regions"]) is not list or len(body["regions"]) != len(METRIC_CROPS)):
        raise ValueError("Codec fixture population differs from the fixed policy.")
    regions = []
    for row, crop in zip(body["regions"], METRIC_CROPS):
        _fields(row, ("crop", "minimumPsnrDb", "minimumSsim"))
        psnr, ssim = row["minimumPsnrDb"], row["minimumSsim"]
        if (type(row["crop"]) is not list or len(row["crop"]) != 4
                or any(type(value) is not int for value in row["crop"]) or tuple(row["crop"]) != crop
                or type(psnr) not in (int, float) or math.isnan(psnr) or psnr < 30
                or type(ssim) not in (int, float) or not math.isfinite(ssim) or not 0.95 <= ssim <= 1):
            raise ValueError("Codec fixture region or threshold is invalid.")
        lossless = math.isinf(psnr)
        regions.append(CodecRegion(crop, None if lossless else float(psnr), lossless, float(ssim)))
    return CodecMetrics(120, "8-bit RGB", 2, 919552, tuple(regions))


def _codec_metrics(events, issues):
    if events is None:
        _note(issues, "codec", "CODEC_METRICS_MISSING", missing=True)
        return None
    found = []
    try:
        for line in events.splitlines():
            row = json.loads(line.decode("utf-8"), object_pairs_hook=unique_fields, parse_constant=_reject_constant)
            if (row.get("$report_type") != "TestReport" or row.get("nodeid") != T006_NODE or row.get("when") != "call"):
                continue
            sections = row.get("sections", [])
            if type(sections) is not list:
                raise ValueError("Captured codec output sections are invalid.")
            for section in sections:
                if type(section) is not list or len(section) != 2 or any(type(value) is not str for value in section):
                    raise ValueError("Captured codec output section is invalid.")
                if section[0] == "Captured stdout call":
                    found.extend(text.removeprefix("T006 codec fixture: ") for text in section[1].splitlines()
                                 if text.startswith("T006 codec fixture: "))
        if not found:
            _note(issues, "codec", "CODEC_METRICS_MISSING", missing=True)
            return None
        if len(found) != 1:
            raise ValueError("Codec metrics must occur exactly once in the selected call.")
        body = json.loads(found[0], object_pairs_hook=unique_fields)
        return _codec_value(body)
    except Exception:
        _note(issues, "codec", "CODEC_METRICS_INVALID")
        return None


def _combine(assessments, *, empty="NOT_RUN"):
    values = tuple(assessments)
    if not values:
        return AuditAssessment(empty, "INCOMPLETE")
    incomplete = any(value.coverage == "INCOMPLETE" for value in values)
    failed = any(value.verdict == "FAIL" for value in values)
    return AuditAssessment("FAIL" if failed else "BLOCKED" if incomplete else "PASS",
                           "INCOMPLETE" if incomplete else "COMPLETE")


def _node_result(node, suites, facts):
    suite_name = "ui" if node.startswith("tests/ui/") else "integration_offline" if node.startswith("tests/integration/") else "unit"
    suite = suites[suite_name]
    rows = [row for row in facts.get(suite_name, {}).get("reports", ()) if row["nodeId"] == node]
    parents = [row for row in rows if row["reportType"] == "TestReport"]
    issues = []
    if not parents:
        _note(issues, "coverage", "REQUIRED_NODE_MISSING", missing=True)
    elif [row["phase"] for row in parents] != ["setup", "call", "teardown"]:
        _note(issues, "coverage", "REQUIRED_PHASE_MISSING", missing=True)
    for row in rows:
        if row["outcome"] == "failed":
            _note(issues, "coverage", "REQUIRED_TEST_FAILED")
        if row["hasXfail"]:
            _note(issues, "coverage", "REQUIRED_TEST_XPASS" if row["outcome"] == "passed" else "REQUIRED_TEST_XFAIL",
                  missing=row["outcome"] != "passed")
        elif row["outcome"] == "skipped":
            _note(issues, "coverage", "REQUIRED_TEST_SKIPPED", missing=True)
    # A failure in a different test does not erase this node's own facts. The
    # selected suite's process, source and report integrity must still hold.
    for code in suite.issues:
        if code in _PROCESS_ISSUES | {"SOURCE_CHANGED", "CLOCK_MOVED_BACKWARDS"}:
            _note(issues, "coverage", code)
        elif code == "EXECUTION_INCOMPLETE":
            _note(issues, "coverage", code, missing=True)
    if suite.executed is not True:
        _note(issues, "coverage", "REQUIRED_SUITE_NOT_EXECUTED", missing=True)
    assessment = _assessment(issues, shared=suite.integrity)
    return bool(parents), assessment, tuple(issue.code for issue in issues)


def _coverage(suites, facts, codec_issues):
    results = []
    for test_id, specs in P0_CASES:
        clauses = []
        for spec in specs:
            if not spec.required_nodes:
                clauses.append(ClauseCoverage(spec.clause_id, spec.kind, 0, 0, 0,
                                               AuditAssessment("NOT_RUN", "INCOMPLETE"), (spec.missing_code,)))
                continue
            nodes = [_node_result(node, suites, facts) for node in spec.required_nodes]
            assessments = [row[1] for row in nodes]
            reasons = [code for row in nodes for code in row[2]]
            if T006_NODE in spec.required_nodes:
                assessments.append(_assessment(codec_issues))
                reasons.extend(issue.code for issue in codec_issues)
            clauses.append(ClauseCoverage(spec.clause_id, spec.kind, len(nodes), sum(row[0] for row in nodes),
                                           sum(row[1].verdict == "PASS" for row in nodes), _combine(assessments),
                                           tuple(dict.fromkeys(reasons))))
        results.append(TestCoverage(test_id, tuple(clauses), _combine(clause.assessment for clause in clauses)))
    return tuple(results)


def audit_recorded_tests(context: ApprovedTestContext,
                         evidence: RecordedTestEvidence | None) -> RecordedTestAudit:
    """Verify one selected execution and fixed offline coverage, without I/O.

    The trusted caller separately approves the source identity and retains the
    runner's execution reference. Imported evidence cannot supply those anchors.
    This pure audit does not read the retained source ZIP or authenticate an
    operator. A later loader must verify that ZIP against the approved identity.
    External live and human clauses remain explicitly unresolved.
    """
    try:
        _check_inputs(context, evidence)
        source = StoredArtifact(f"qa/test-sources/{context.source_sha256}.zip", context.source_sha256, context.source_size)
        registry_body = registry_bytes()
        digest = sha256(registry_body).hexdigest()
        registry = StoredArtifact(f"qa/test-registries/{digest}.json", digest, len(registry_body))
        issues = []
        directory, summaries = _top_records(context, evidence, source, registry, issues)
        selected = {}
        for records in evidence.suites if evidence else ():
            if records.name not in SUITES:
                _note(issues, "integrity", "SUITE_UNKNOWN")
            elif records.name in selected:
                _note(issues, "integrity", "SUITE_DUPLICATE")
            else:
                selected[records.name] = records
                if summaries and records.name not in summaries:
                    _note(issues, "integrity", "SUITE_NOT_SELECTED")
        shared = _assessment(issues)
        suites, facts, codec_issues, codec = {}, {}, [], None
        for name in DEFAULT_SUITES:
            suite, parsed, payloads, findings = _suite_audit(
                name, selected.get(name), summaries.get(name), directory, context, source, registry, shared)
            suites[name] = suite
            if parsed is not None:
                facts[name] = parsed
            if name == "unit":
                codec = _codec_metrics(payloads.get("events") if parsed is not None else None, codec_issues)
            for issue in findings:
                _note(issues, name + "." + issue.category, issue.code, missing=issue.missing)
        integrity = _combine(suite.integrity for suite in suites.values())
        execution = _combine(suite.execution for suite in suites.values())
        if evidence is None or context.execution is None:
            if integrity.verdict != "FAIL":
                integrity = AuditAssessment("NOT_RUN", "INCOMPLETE")
            if execution.verdict != "FAIL":
                execution = AuditAssessment("NOT_RUN", "INCOMPLETE")
        coverage = _coverage(suites, facts, codec_issues)
        issues.extend(codec_issues)
        return RecordedTestAudit(context.run_id, source, context.registry_sha256, context.execution,
                                 POLICY_VERSION, sha256(policy_bytes()).hexdigest(), integrity, execution,
                                 tuple(suites.values()), coverage, codec, tuple(issues))
    except Exception:
        raise RecordedTestAuditError("Recorded tests could not be assessed from the supplied bounded values.") from None
