"""Record real pytest runs from small source ZIPs without running this suite again."""

from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
import warnings
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from core.storage import StoredArtifact, publish_bytes, read_verified
from qa import record_tests as runner


RUN = "recorded-tests-run"
UNIT_FILE = "tests/unit/test_probe.py"
UI_FILE = "tests/ui/test_view.py"
MARKER = b"approved source bytes"
PASS_TEST = '''from pathlib import Path

def test_from_archive():
    root = Path(__file__).resolve().parents[2]
    assert (root / "marker.txt").read_bytes() == b"approved source bytes"
    print("T006 codec fixture: {\\"probe\\": true}")
'''.encode()


class RecordedTestsTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.sources = self.root / "sources"
        self.evidence = self.root / "evidence"
        self.run_root = self.evidence / RUN

    @staticmethod
    def zip_bytes(entries):
        """Build candidate bytes without using the production extraction code."""
        output = BytesIO()
        with warnings.catch_warnings(), ZipFile(output, "w", compression=ZIP_STORED) as archive:
            warnings.simplefilter("ignore", UserWarning)
            for name, contents, mode in entries:
                info = ZipInfo(name)
                info.create_system = 3
                info.external_attr = mode << 16
                archive.writestr(info, contents)
        return output.getvalue()

    def archive(self, files=None, *, extra=()):
        files = {UNIT_FILE: PASS_TEST, "marker.txt": MARKER} if files is None else files
        entries = [(name, contents, stat.S_IFREG | 0o600) for name, contents in files.items()]
        contents = self.zip_bytes([*entries, *extra])
        digest = sha256(contents).hexdigest()
        path = f"archives/{digest}.zip"
        existing = self.sources / path
        if existing.exists():
            self.assertEqual(existing.read_bytes(), contents)
            return StoredArtifact(path, digest, len(contents))
        return publish_bytes(self.sources, path, contents)

    def record(self, source=None, *, suites=("unit",), **changes):
        source = self.archive() if source is None else source
        arguments = {"run_id": RUN, "suites": suites} | changes
        return runner.record_offline_tests(self.sources, source, self.evidence, **arguments)

    def data(self, record):
        return json.loads(read_verified(self.run_root, record))

    def suite_data(self, result, name="unit"):
        selected = next(item for item in result.suites if item.name == name)
        return selected, self.data(selected.record)

    def artifact(self, receipt, name):
        value = receipt["artifacts"][name]
        return None if value is None else read_verified(self.run_root, StoredArtifact(**value))

    def events(self, receipt):
        return [json.loads(line) for line in self.artifact(receipt, "events").splitlines()]

    def all_saved_bytes(self):
        return {path.relative_to(self.run_root): path.read_bytes()
                for path in self.run_root.rglob("*") if path.is_file()}

    def cli(self, source, *, suite="unit", **changes):
        values = {"source-root": self.sources, "source-path": source.path,
                  "source-sha256": source.sha256, "source-size": source.size,
                  "evidence-root": self.evidence, "run-id": RUN, "suite": suite} | changes
        arguments = [item for name, value in values.items() for item in ("--" + name, str(value))]
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run([sys.executable, "-B", "-m", "qa.record_tests", *arguments],
            cwd=Path(__file__).resolve().parents[2], env=environment,
            capture_output=True, text=True, timeout=20)

    def test_runs_exact_zip_and_records_frozen_hashed_evidence(self):
        source = self.archive()
        original = read_verified(self.sources, source)
        result = self.record(source)
        self.assertIsInstance(result, runner.RecordedTestRun)
        self.assertEqual(result.run_id, RUN)
        self.assertEqual(read_verified(self.run_root, result.source), original)
        self.assertEqual(result.source.sha256, source.sha256)
        registry = read_verified(self.run_root, result.registry)
        self.assertEqual(sha256(registry).hexdigest(), result.registry.sha256)
        self.assertTrue(json.loads(registry))
        self.assertIs(type(result.suites), tuple)
        self.assertEqual(len(result.suites), 1)
        suite, receipt = self.suite_data(result)
        self.assertIsInstance(suite, runner.RecordedSuite)
        self.assertTrue(suite.executed)
        self.assertEqual((suite.exit_code, suite.issues), (0, ()))
        self.assertTrue(receipt["executed"])
        self.assertEqual(receipt["exitCode"], 0)
        self.assertTrue(receipt["sourceUnchanged"])
        self.assertTrue(receipt["reportsComplete"])
        self.assertEqual(receipt["source"], asdict(result.source))
        self.assertEqual(receipt["registry"], asdict(result.registry))
        self.assertLessEqual(receipt["startedAtUnix"], receipt["finishedAtUnix"])
        self.assertGreaterEqual(receipt["elapsedSeconds"], 0)
        self.assertIn(b"T006 codec fixture", self.artifact(receipt, "junit"))
        self.assertTrue(json.loads(self.artifact(receipt, "cases")))
        for value in receipt["artifacts"].values():
            if value is not None:
                read_verified(self.run_root, StoredArtifact(**value))
        self.assertEqual(read_verified(self.sources, source), original)
        with self.assertRaises(FrozenInstanceError):
            result.run_id = "other"
        with self.assertRaises(FrozenInstanceError):
            suite.executed = False
        for data in (asdict(result), receipt, self.data(result.record)):
            for field in ("visualEffectDemonstrated", "automatedWorkflowComplete", "releaseClassification"):
                self.assertNotIn(field, data)

    def test_preserves_real_phases_subtests_skips_failures_and_passing_output(self):
        body = '''import unittest
import pytest

class Probe(unittest.TestCase):
    def test_pass(self):
        print('T006 codec fixture: {"probe": true}')
    def test_subtests(self):
        for value in (0, 1):
            with self.subTest(value=value):
                self.assertEqual(value, 0)
    def test_skip(self):
        self.skipTest("fixture skip")
    @unittest.expectedFailure
    def test_expected_failure(self):
        self.fail("fixture expected failure")
    def test_error(self):
        raise ValueError("fixture call error")

class SetupError(unittest.TestCase):
    def setUp(self):
        raise RuntimeError("fixture setup error")
    def test_body(self):
        pass

class TeardownError(unittest.TestCase):
    def tearDown(self):
        raise RuntimeError("fixture teardown error")
    def test_body(self):
        pass

@pytest.mark.xfail(reason="fixture expectation")
def test_unexpected_success():
    pass

@pytest.fixture
def broken_setup():
    raise RuntimeError("pytest setup error")

def test_setup_phase(broken_setup):
    pass

@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("pytest teardown error")

def test_teardown_phase(broken_teardown):
    pass

def test_call_and_teardown_fail(broken_teardown):
    raise RuntimeError("pytest call and teardown error")

@pytest.mark.parametrize("value", [1, 2], ids=["punctuation::[x]", "second"])
def test_parameterized(value):
    assert value > 0
'''.encode()
        suite, receipt = self.suite_data(self.record(self.archive({UNIT_FILE: body})))
        self.assertNotEqual(suite.exit_code, 0)
        self.assertTrue(receipt["reportsComplete"])
        for issue in ("TEST_FAILURE", "TEST_SKIPPED", "TEST_XFAIL", "TEST_XPASS"):
            self.assertIn(issue, suite.issues)
        events = self.events(receipt)
        subtests = [row for row in events if row.get("$report_type") == "SubTestReport"]
        self.assertEqual([row["outcome"] for row in subtests], ["passed", "failed"])
        self.assertEqual([row["_subtest.context"]["kwargs"]["value"] for row in subtests], ["0", "1"])
        reports = [row for row in events if row.get("$report_type") == "TestReport"]
        self.assertEqual({row["when"] for row in reports}, {"setup", "call", "teardown"})
        self.assertTrue(any(row["outcome"] == "skipped" for row in reports))
        self.assertTrue(any(row.get("wasxfail") for row in reports))
        self.assertTrue(any(row["nodeid"].endswith("Probe::test_subtests")
                            and row["when"] == "call" and row["outcome"] == "passed" for row in reports))
        self.assertTrue(any(row["$report_type"] == "SessionFinish" and row["exitstatus"] != 0 for row in events))
        self.assertIn(b"T006 codec fixture", self.artifact(receipt, "events"))
        self.assertIn(b"T006 codec fixture", self.artifact(receipt, "junit"))
        self.assertIn(b"fixture setup error", self.artifact(receipt, "events"))
        self.assertIn(b"fixture teardown error", self.artifact(receipt, "events"))
        cases = json.loads(self.artifact(receipt, "cases"))
        facts = cases["reports"]
        self.assertEqual(cases["sessionExitCode"], suite.exit_code)
        self.assertEqual(cases["subtestReports"], 2)
        self.assertEqual(cases["nodeIds"], sorted({row["nodeid"] for row in reports}))
        case_events = [row for row in events
                       if row.get("$report_type") in ("CollectReport", "TestReport", "SubTestReport")]
        self.assertEqual([(fact["reportType"], fact["nodeId"], fact["phase"], fact["outcome"])
                          for fact in facts],
                         [(row["$report_type"], row["nodeid"],
                           "collect" if row["$report_type"] == "CollectReport" else row["when"], row["outcome"])
                          for row in case_events])
        self.assertEqual([fact["subtest"] for fact in facts if fact["reportType"] == "SubTestReport"],
                         [row["_subtest.context"] for row in subtests])
        self.assertTrue(any(fact["phase"] == "setup" and fact["outcome"] == "failed"
                            and fact["nodeId"].endswith("::test_setup_phase") for fact in facts))
        self.assertTrue(any(fact["phase"] == "teardown" and fact["outcome"] == "failed"
                            and fact["nodeId"].endswith("::test_teardown_phase") for fact in facts))
        self.assertTrue(any(fact["hasXfail"] and fact["outcome"] == "passed" for fact in facts))
        self.assertTrue(any(fact["hasXfail"] and fact["outcome"] == "skipped" for fact in facts))

    def test_missing_and_empty_suites_do_not_block_other_suites(self):
        source = self.archive({UNIT_FILE: PASS_TEST, UI_FILE: PASS_TEST, "marker.txt": MARKER})
        result = self.record(source, suites=("unit", "integration_offline", "ui"))
        self.assertEqual([suite.name for suite in result.suites], ["unit", "integration_offline", "ui"])
        unit, _ = self.suite_data(result, "unit")
        missing, receipt = self.suite_data(result, "integration_offline")
        ui, _ = self.suite_data(result, "ui")
        self.assertEqual((unit.exit_code, ui.exit_code), (0, 0))
        self.assertFalse(missing.executed)
        self.assertIsNone(missing.exit_code)
        self.assertIn("SUITE_MISSING", missing.issues)
        self.assertIsNone(receipt["argv"])
        self.assertFalse(receipt["reportsComplete"])
        empty = self.archive({UNIT_FILE: PASS_TEST, UI_FILE: PASS_TEST, "marker.txt": MARKER,
                              "tests/integration/README.txt": b"No integration tests yet."})
        result = self.record(empty, suites=("integration_offline", "unit"))
        suite, _ = self.suite_data(result, "integration_offline")
        self.assertTrue(suite.executed)
        self.assertIn("NO_TESTS", suite.issues)
        self.assertEqual(self.suite_data(result, "unit")[0].exit_code, 0)
        collect_failure = self.archive({UNIT_FILE: b"raise RuntimeError('collection failed')\n",
                                        UI_FILE: PASS_TEST, "marker.txt": MARKER})
        result = self.record(collect_failure, suites=("unit", "ui"))
        failed, receipt = self.suite_data(result, "unit")
        self.assertIn("TEST_FAILURE", failed.issues)
        self.assertNotIn("REPORT_INVALID", failed.issues)
        self.assertTrue(receipt["reportsComplete"])
        self.assertEqual(self.suite_data(result, "ui")[0].exit_code, 0)

    def test_source_reference_failures_precede_any_process(self):
        source = self.archive()
        with patch.object(runner.test_results.subprocess, "Popen",
                          side_effect=RuntimeError("Test process must not start")) as process:
            for record in (replace(source, sha256="f" * 64), replace(source, size=True),
                           replace(source, path="../private-source.zip")):
                with self.subTest(record=record), self.assertRaises(runner.RunnerError):
                    self.record(record)
            process.assert_not_called()
        self.assertFalse(self.run_root.exists())

    def test_rejects_unsafe_zip_members_before_any_process(self):
        cases = [
            [("../escaped.py", b"not executed", stat.S_IFREG | 0o600)],
            [("/absolute.py", b"not executed", stat.S_IFREG | 0o600)],
            [(UNIT_FILE, PASS_TEST, stat.S_IFREG | 0o600)],
            [("linked.txt", b"../outside", stat.S_IFLNK | 0o777)],
        ]
        with patch.object(runner.test_results.subprocess, "Popen",
                          side_effect=RuntimeError("Test process must not start")) as process:
            for entries in cases:
                source = self.archive(extra=entries)
                with self.subTest(entries=entries), self.assertRaises(runner.RunnerError):
                    self.record(source)
            with self.assertRaises(runner.RunnerError):
                self.record(self.archive({}))
            process.assert_not_called()
        self.assertFalse((self.root / "escaped.py").exists())

    def test_archive_byte_member_count_and_expansion_limits_are_checked(self):
        source = self.archive()
        sizes = {"MAX_SOURCE_BYTES": source.size - 1, "MAX_SOURCE_FILES": 1,
                 "MAX_SOURCE_FILE_BYTES": len(PASS_TEST) - 1,
                 "MAX_SOURCE_TOTAL_BYTES": len(PASS_TEST) + len(MARKER) - 1}
        with patch.object(runner.test_results.subprocess, "Popen",
                          side_effect=RuntimeError("Test process must not start")) as process:
            for name, maximum in sizes.items():
                with self.subTest(limit=name), patch.object(runner, name, maximum), self.assertRaises(runner.RunnerError):
                    self.record(source)
            process.assert_not_called()
        with patch.object(runner, "MAX_SOURCE_BYTES", source.size), \
                patch.object(runner, "MAX_SOURCE_FILES", 2), \
                patch.object(runner, "MAX_SOURCE_FILE_BYTES", len(PASS_TEST)), \
                patch.object(runner, "MAX_SOURCE_TOTAL_BYTES", len(PASS_TEST) + len(MARKER)):
            self.assertEqual(self.record(source).suites[0].exit_code, 0)

    def test_fixed_command_and_environment_ignore_external_pytest_options(self):
        body = '''import os
from pathlib import Path

def test_clean_environment():
    assert os.environ.get("PYTEST_ADDOPTS") is None
    assert os.environ.get("PYTEST_PLUGINS") is None
    assert os.environ.get("RUN_LIVE_PROVIDER_TESTS") is None
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    root = Path(__file__).resolve().parents[2]
    assert not (root / ".pytest_cache").exists()
'''.encode()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "pytest.py").write_text('raise RuntimeError("external pytest replacement")\n')
        (outside / "outside_plugin.py").write_text('raise RuntimeError("external pytest plugin")\n')
        hostile = {"PYTEST_ADDOPTS": "--collect-only", "PYTEST_PLUGINS": "outside_plugin",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0", "RUN_LIVE_PROVIDER_TESTS": "1",
                   "PYTHONPATH": str(outside), "PYTHONDONTWRITEBYTECODE": "0"}
        with patch.dict(os.environ, hostile):
            result = self.record(self.archive({UNIT_FILE: body}))
            self.assertEqual(os.environ["PYTEST_ADDOPTS"], "--collect-only")
        suite, receipt = self.suite_data(result)
        self.assertEqual((suite.exit_code, suite.issues), (0, ()))
        self.assertIn("no:cacheprovider", receipt["argv"])
        self.assertIn("pytest_reportlog.plugin", receipt["argv"])
        self.assertNotIn("--collect-only", receipt["argv"])
        self.assertTrue(receipt["sourceUnchanged"])
        self.assertTrue(any(row.get("when") == "call" for row in self.events(receipt)))

    def test_timeout_records_incomplete_execution(self):
        source = self.archive({UNIT_FILE: b"import time\ndef test_wait():\n    time.sleep(1.2)\n"})
        started = time.monotonic()
        with patch.object(runner.test_results, "MAX_SUITE_SECONDS", 0.15):
            suite, receipt = self.suite_data(self.record(source))
        self.assertIn("PROCESS_TIMEOUT", suite.issues)
        self.assertTrue(suite.executed)
        self.assertFalse(receipt["reportsComplete"])
        self.assertLess(time.monotonic() - started, 8)

    def test_keyboard_interrupt_kills_descendant_after_leader_exit(self):
        pid_file = self.root / "descendant.pid"
        child_code = '''from pathlib import Path
import subprocess
import sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
Path("descendant.pid").write_text(str(child.pid))
'''
        start_process = runner.test_results.subprocess.Popen
        leaders = []
        descendant_pid = None

        def track_process(*arguments, **options):
            process = start_process(*arguments, **options)
            leaders.append(process)
            return process

        def interrupt_after_leader_exit(*arguments, **options):
            nonlocal descendant_pid
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if pid_file.exists() and leaders[0].poll() is not None:
                    descendant_pid = int(pid_file.read_text())
                    self.assertGreater(descendant_pid, 1)
                    self.assertNotEqual(descendant_pid, os.getpid())
                    self.assertEqual(os.getpgid(descendant_pid), leaders[0].pid)
                    raise KeyboardInterrupt
                time.sleep(0.01)
            self.fail("The child did not publish its descendant PID and exit within three seconds.")

        def descendant_is_running():
            try:
                process_info = Path(f"/proc/{descendant_pid}/stat").read_text()
            except FileNotFoundError:
                return False
            return process_info.rsplit(")", 1)[1].split()[0] != "Z"

        try:
            with patch.object(runner.test_results.subprocess, "Popen", side_effect=track_process), \
                    patch.object(runner.test_results.selectors.DefaultSelector, "select",
                                 side_effect=interrupt_after_leader_exit):
                with self.assertRaises(KeyboardInterrupt):
                    runner.test_results.run_process(
                        [sys.executable, "-B", "-c", child_code], self.root,
                        runner.test_results.child_environment(), timeout=5,
                    )
            self.assertIsNotNone(descendant_pid)
            self.assertEqual(leaders[0].returncode, 0)
            deadline = time.monotonic() + 3
            while descendant_is_running() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(descendant_is_running(), "The descendant survived runner cancellation.")
        finally:
            for process in leaders:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    def test_stream_and_report_limits_keep_retained_bytes_bounded(self):
        stream = b'''import sys
def test_output(capfd):
    with capfd.disabled():
        sys.stdout.write("x" * 262144)
        sys.stdout.flush()
'''
        report = b'def test_report():\n    print("x" * 131072)\n'
        cases = [("MAX_STREAM_BYTES", stream, ("stdout", "stderr")),
                 ("MAX_REPORT_BYTES", report, ("junit", "events"))]
        for constant, body, names in cases:
            with self.subTest(limit=constant), patch.object(runner.test_results, constant, 1024):
                suite, receipt = self.suite_data(self.record(self.archive({UNIT_FILE: body})))
                self.assertIn("OUTPUT_LIMIT", suite.issues)
                self.assertFalse(receipt["reportsComplete"])
                for name in names:
                    value = receipt["artifacts"][name]
                    if value is not None:
                        self.assertLessEqual(value["size"], 1024)
                        self.assertLessEqual(len(self.artifact(receipt, name)), 1024)

        # Model the report changing after the last periodic observation. The
        # real child has exited with both pipes at EOF before draining starts.
        report_path = self.root / "late-report.xml"
        start_process = runner.test_results.subprocess.Popen
        check_reports = runner.test_results._reports_within_limit

        def finished_process(*args, **kwargs):
            process = start_process(*args, **kwargs)
            process.wait(timeout=5)
            return process

        observations = 0

        def observe_report(paths):
            nonlocal observations
            observations += 1
            return True if observations == 1 else check_reports(paths)

        command = [sys.executable, "-B", "-c",
                   "from pathlib import Path; Path('late-report.xml').write_bytes(b'x' * 2048)"]
        with patch.object(runner.test_results, "MAX_REPORT_BYTES", 1024), \
                patch.object(runner.test_results.subprocess, "Popen", side_effect=finished_process), \
                patch.object(runner.test_results, "_reports_within_limit", side_effect=observe_report):
            capture = runner.test_results.run_process(
                command, self.root, runner.test_results.child_environment(), report_paths=(report_path,))
        self.assertEqual(capture.exit_code, 0)
        self.assertGreaterEqual(observations, 2)
        self.assertIn("OUTPUT_LIMIT", capture.issues)

    def test_malformed_or_missing_reports_are_not_complete(self):
        for statement in [
                'Path(session.config.option.report_log).write_text("not JSON\\n")',
                'path = Path(session.config.option.report_log); path.write_text("\\n".join(path.read_text().splitlines()[:-1]) + "\\n")',
                'Path(session.config.option.xmlpath).write_text("not XML")',
                'path = Path(session.config.option.report_log); rows = [json.loads(line) for line in path.read_text().splitlines()]; '
                'rows = [row for row in rows if not (row.get("$report_type") == "TestReport" and row.get("nodeid", "").endswith("::test_dropped"))]; '
                'path.write_text("".join(json.dumps(row) + "\\n" for row in rows))']:
            hook = ('from pathlib import Path\nimport pytest\nimport json\n'
                    '@pytest.hookimpl(hookwrapper=True)\ndef pytest_sessionfinish(session, exitstatus):\n'
                    '    yield\n    ' + statement + '\n').encode()
            with self.subTest(statement=statement):
                source = self.archive({UNIT_FILE: PASS_TEST + b"\ndef test_dropped():\n    pass\n",
                                       "marker.txt": MARKER, "conftest.py": hook})
                suite, receipt = self.suite_data(self.record(source))
                self.assertEqual(suite.exit_code, 0)
                self.assertIn("REPORT_INVALID", suite.issues)
                self.assertFalse(receipt["reportsComplete"])

    def test_source_changes_are_reported_and_next_suite_gets_fresh_bytes(self):
        body = b'''from pathlib import Path
def test_change_source():
    root = Path(__file__).resolve().parents[2]
    (root / "marker.txt").write_bytes(b"changed")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "added.txt").write_text("unexpected source file")
'''
        source = self.archive({UNIT_FILE: body, UI_FILE: PASS_TEST, "marker.txt": MARKER})
        original = read_verified(self.sources, source)
        result = self.record(source, suites=("unit", "ui"))
        changed, changed_receipt = self.suite_data(result, "unit")
        later, later_receipt = self.suite_data(result, "ui")
        self.assertEqual((changed.exit_code, later.exit_code), (0, 0))
        self.assertIn("SOURCE_CHANGED", changed.issues)
        self.assertFalse(changed_receipt["sourceUnchanged"])
        self.assertEqual(later.issues, ())
        self.assertTrue(later_receipt["sourceUnchanged"])
        self.assertEqual(read_verified(self.sources, source), original)
        linked_root = b'''from pathlib import Path
import shutil
def test_replace_source_root():
    root = Path(__file__).resolve().parents[2]
    replacement = root.with_name("copied-source")
    shutil.copytree(root, replacement)
    root.rename(root.with_name("original-source"))
    root.symlink_to(replacement, target_is_directory=True)
'''
        suite, receipt = self.suite_data(self.record(self.archive({UNIT_FILE: linked_root})))
        self.assertEqual(suite.exit_code, 0)
        self.assertIn("SOURCE_CHANGED", suite.issues)
        self.assertFalse(receipt["sourceUnchanged"])

    def test_repeated_runs_and_publication_failure_preserve_prior_evidence(self):
        source = self.archive()
        first = self.record(source)
        before = self.all_saved_bytes()
        second = self.record(source)
        self.assertNotEqual(second.record.path, first.record.path)
        for path, contents in before.items():
            self.assertEqual((self.run_root / path).read_bytes(), contents)
        publish = runner.publish_bytes

        def fail_final(root, path, contents):
            try:
                data = json.loads(contents)
            except (ValueError, UnicodeError):
                data = None
            if type(data) is dict and data.get("manifestType") == "recorded_test_run":
                raise OSError("private publication token")
            return publish(root, path, contents)

        before = self.all_saved_bytes()
        with patch.object(runner, "publish_bytes", side_effect=fail_final), self.assertRaises(runner.RunnerError) as caught:
            self.record(source)
        self.assertNotIn("private publication token", str(caught.exception))
        for path, contents in before.items():
            self.assertEqual((self.run_root / path).read_bytes(), contents)
        self.assertEqual(read_verified(self.run_root, first.record), before[Path(first.record.path)])

    def test_invalid_ids_and_suite_selections_fail_before_any_process(self):
        source = self.archive()
        cases = [{"run_id": "../outside"}, {"run_id": True}, {"suites": ()},
                 {"suites": ("unit", "unit")}, {"suites": ("shell-command",)}, {"suites": "unit"}]
        with patch.object(runner.test_results.subprocess, "Popen",
                          side_effect=RuntimeError("Test process must not start")) as process:
            for changes in cases:
                with self.subTest(changes=changes), self.assertRaises(runner.RunnerError):
                    self.record(source, **changes)
            process.assert_not_called()

    def test_cli_runs_fixed_suite_and_returns_verified_record(self):
        result = self.cli(self.archive())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        value = json.loads(result.stdout)
        saved = StoredArtifact(**value["record"])
        data = self.data(saved)
        self.assertEqual(data["runId"], RUN)
        self.assertEqual(data["manifestType"], "recorded_test_run")
        self.assertNotIn("releaseClassification", data)

    def test_cli_failures_are_safe_and_missing_suites_are_nonzero(self):
        source = self.archive()
        bad = self.cli(source, **{"source-sha256": "private-not-a-sha"})
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual(bad.stdout, "")
        self.assertNotIn("private-not-a-sha", bad.stderr)
        self.assertNotIn(str(self.root), bad.stderr)
        missing = self.cli(source, suite="integration_offline")
        self.assertNotEqual(missing.returncode, 0)
        self.assertEqual(missing.stderr, "")
        data = json.loads(missing.stdout)
        self.assertTrue(self.data(StoredArtifact(**data["record"])))


if __name__ == "__main__":
    unittest.main()
