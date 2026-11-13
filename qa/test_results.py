"""Run fixed pytest commands and retain bounded process and report facts."""

from dataclasses import dataclass
from collections import Counter
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import time
import xml.etree.ElementTree as ElementTree

from _pytest.junitxml import bin_xml_escape, mangle_test_address

from core.json_data import unique_fields
from qa.test_registry import PYTEST_VERSION


MAX_SUITE_SECONDS = 1200
MAX_STREAM_BYTES = 4 * 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_REPORT_EVENTS = 100_000
MAX_REPORT_LINE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ProcessCapture:
    executed: bool
    exit_code: int | None
    started: float
    finished: float
    elapsed: float
    stdout: bytes
    stderr: bytes
    issues: tuple[str, ...]


def child_environment() -> dict[str, str]:
    """Keep runtime paths and proxy trust, but remove test-command overrides."""
    environment = dict(os.environ)
    for name in (
        "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST",
        "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT",
        "PYTHONWARNINGS", "PYTHONPYCACHEPREFIX", "RUN_LIVE_PROVIDER_TESTS",
        "RUNWAY_API_KEY", "RUNWAYML_API_SECRET",
    ):
        environment.pop(name, None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _stop(process: subprocess.Popen) -> None:
    """Kill the child session, including media tools that still hold its pipes."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _reports_within_limit(paths: tuple[Path, ...]) -> bool:
    for path in paths:
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_REPORT_BYTES:
            return False
    return True


def run_process(argv: list[str], cwd: Path, environment: dict[str, str], *,
                report_paths: tuple[Path, ...] = (), timeout: float | None = None) -> ProcessCapture:
    """Drain both pipes with byte limits and stop the process on a hard bound."""
    started, start_tick = time.time(), time.monotonic()
    timeout = MAX_SUITE_SECONDS if timeout is None else timeout
    output = {"stdout": bytearray(), "stderr": bytearray()}
    issues = []
    process = None
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        with selectors.DefaultSelector() as selector:
            for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map() or process.poll() is None:
                if time.monotonic() - start_tick >= timeout:
                    issues.append("PROCESS_TIMEOUT")
                    _stop(process)
                    break
                if not _reports_within_limit(report_paths):
                    issues.append("OUTPUT_LIMIT")
                    _stop(process)
                    break
                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fileobj.fileno(), 65_536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    retained = output[key.data]
                    remaining = MAX_STREAM_BYTES - len(retained)
                    retained.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        issues.append("OUTPUT_LIMIT")
                        _stop(process)
                        break
                if issues:
                    break
            process.wait(timeout=5)
            # A child can finish its last report between the final poll and exit.
            if not _reports_within_limit(report_paths):
                issues.append("OUTPUT_LIMIT")
    except OSError:
        issues.append("PROCESS_START_FAILED" if process is None else "PROCESS_IO_FAILED")
        if process is not None:
            _stop(process)
    finally:
        if process is not None:
            # A completed leader can leave descendants holding its pipe ends.
            try:
                _stop(process)
            finally:
                process.stdout.close()
                process.stderr.close()
    finished, elapsed = time.time(), time.monotonic() - start_tick
    if (not all(type(value) in (int, float) and math.isfinite(value)
                for value in (started, finished, elapsed))
            or not 0 <= started <= 253_402_300_799
            or not 0 <= finished <= 253_402_300_799 or elapsed < 0):
        raise ValueError("Test process clock is invalid.")
    if finished < started:
        issues.append("CLOCK_MOVED_BACKWARDS")
    return ProcessCapture(process is not None, process.returncode if process else None,
                          started, finished, elapsed, bytes(output["stdout"]),
                          bytes(output["stderr"]), tuple(dict.fromkeys(issues)))


def read_report(path: Path) -> bytes | None:
    """Read a bounded regular report without following links or blocking on a FIFO."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_REPORT_BYTES:
                return None
            contents = stream.read(info.st_size + 1)
        return contents if len(contents) == info.st_size else None
    except OSError:
        return None


def _reject_constant(value):
    raise ValueError("Test report numbers must be finite.")


def _case_fact(event: dict) -> dict:
    node = event.get("nodeid")
    outcome = event.get("outcome")
    kind = event["$report_type"]
    phase = "collect" if kind == "CollectReport" else event.get("when")
    if (type(node) is not str or len(node) > 4096
            or outcome not in ("passed", "failed", "skipped")
            or phase not in ("collect", "setup", "call", "teardown")):
        raise ValueError("Test report identity or outcome is invalid.")
    subtest = event.get("_subtest.context") if kind == "SubTestReport" else None
    if kind == "SubTestReport":
        if (phase != "call" or type(subtest) is not dict
                or set(subtest) != {"msg", "kwargs"}
                or subtest["msg"] is not None and type(subtest["msg"]) is not str
                or type(subtest["kwargs"]) is not dict
                or any(type(key) is not str or type(value) is not str
                       for key, value in subtest["kwargs"].items())):
            raise ValueError("Subtest context is invalid.")
    has_xfail = "wasxfail" in event
    if has_xfail and type(event["wasxfail"]) is not str:
        raise ValueError("Expected-failure context is invalid.")
    return {"reportType": kind, "nodeId": node, "phase": phase, "outcome": outcome,
            "hasXfail": has_xfail, "wasXfail": event.get("wasxfail"), "subtest": subtest}


def _check_junit_nodes(xml: ElementTree.Element, phases: dict, facts: list[dict]) -> None:
    """Check parent identities with the pinned pytest writer's naming rules."""
    expected = Counter()
    failed_calls = {fact["nodeId"] for fact in facts
                    if fact["phase"] == "call" and fact["outcome"] == "failed"}
    # Pytest writes a second case when both call and teardown fail.
    case_counts = {node: 2 if node in failed_calls and ("teardown", "failed") in observed else 1
                   for node, observed in phases.items()}
    for fact in facts:
        if fact["reportType"] == "CollectReport" and fact["outcome"] != "passed":
            if fact["nodeId"] in case_counts:
                raise ValueError("JUnit collection identities are repeated.")
            case_counts[fact["nodeId"]] = 1
    for node, count in case_counts.items():
        names = mangle_test_address(node)
        identity = (".".join(names[:-1]), bin_xml_escape(names[-1]))
        if identity in expected:
            raise ValueError("JUnit test identities are ambiguous.")
        expected[identity] = count

    actual = Counter()
    for case in xml.iter("testcase"):
        identity = (case.get("classname"), case.get("name"))
        if None in identity:
            raise ValueError("JUnit test identity is missing.")
        actual[identity] += 1
    if actual != expected:
        raise ValueError("JUnit tests do not match the recorded pytest events.")


def parse_reports(events: bytes | None, junit: bytes | None, exit_code: int | None) -> tuple[dict, tuple[str, ...]]:
    """Retain every parent phase and subtest; never infer success from its parent."""
    if events is None or junit is None:
        raise ValueError("Test reports are missing.")
    if (len(events) > MAX_REPORT_BYTES or len(junit) > MAX_REPORT_BYTES
            or b"<!DOCTYPE" in junit or b"<!ENTITY" in junit):
        raise ValueError("Test reports exceed their fixed format.")
    xml = ElementTree.fromstring(junit)
    if xml.tag not in ("testsuite", "testsuites"):
        raise ValueError("JUnit root is invalid.")
    lines = events.splitlines()
    if not 2 <= len(lines) <= MAX_REPORT_EVENTS:
        raise ValueError("Test report count is invalid.")
    facts, phases, issues = [], {}, []
    for index, line in enumerate(lines):
        if not 0 < len(line) <= MAX_REPORT_LINE_BYTES:
            raise ValueError("Test report line exceeds its limit.")
        event = json.loads(line.decode("utf-8"), object_pairs_hook=unique_fields,
                           parse_constant=_reject_constant)
        if type(event) is not dict:
            raise ValueError("Test report must be an object.")
        kind = event.get("$report_type")
        if index == 0:
            if kind != "SessionStart" or event.get("pytest_version") != PYTEST_VERSION:
                raise ValueError("Test session does not use the pinned pytest version.")
            continue
        if index == len(lines) - 1:
            if (kind != "SessionFinish" or type(event.get("exitstatus")) is not int
                    or event["exitstatus"] != exit_code or not 0 <= exit_code <= 5):
                raise ValueError("Test session exit does not match its process.")
            continue
        if kind in ("TestReport", "SubTestReport", "CollectReport"):
            fact = _case_fact(event)
            facts.append(fact)
            if kind == "TestReport":
                phases.setdefault(fact["nodeId"], []).append((fact["phase"], fact["outcome"]))
            if fact["outcome"] == "failed":
                issues.append("TEST_FAILURE")
            if fact["hasXfail"]:
                issues.append("TEST_XPASS" if fact["outcome"] == "passed" else "TEST_XFAIL")
            elif fact["outcome"] == "skipped":
                issues.append("TEST_SKIPPED")
        elif kind != "WarningMessage":
            raise ValueError("Test report event kind is invalid.")
    for node, observed in phases.items():
        expected = ["setup", "call", "teardown"] if observed[0] == ("setup", "passed") else ["setup", "teardown"]
        if not node or [phase for phase, _ in observed] != expected:
            raise ValueError("Test phases are incomplete or repeated.")
    if any(fact["reportType"] == "SubTestReport" and fact["nodeId"] not in phases for fact in facts):
        raise ValueError("Subtest has no parent test.")
    _check_junit_nodes(xml, phases, facts)
    if not phases:
        issues.append("NO_TESTS")
    if exit_code != 0:
        issues.append("NO_TESTS" if exit_code == 5 else "TEST_FAILURE")
    return {"reports": facts, "nodeIds": sorted(phases),
            "subtestReports": sum(fact["reportType"] == "SubTestReport" for fact in facts),
            "sessionExitCode": exit_code}, tuple(dict.fromkeys(issues))
