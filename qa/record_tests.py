"""Record fixed offline test suites from an operator-approved source ZIP."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import stat
from struct import Struct
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from core.budget import BudgetLedger
from core.storage import ArtifactExistsError, StoredArtifact, publish_bytes, read_verified
from qa import test_results
from qa.test_registry import DEFAULT_SUITES, PYTEST_OPTIONS, PYTEST_VERSION, REPORTLOG_VERSION, SUITES, registry_bytes


MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_FILES = 1024
MAX_SOURCE_FILE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 64 * 1024 * 1024
_END = Struct("<4s4H2IH")


class RunnerError(ValueError):
    """The selected source or recorded test execution cannot be handled safely."""


@dataclass(frozen=True)
class RecordedSuite:
    name: str
    record: StoredArtifact
    executed: bool
    exit_code: int | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RecordedTestRun:
    run_id: str
    source: StoredArtifact
    registry: StoredArtifact
    suites: tuple[RecordedSuite, ...]
    record: StoredArtifact


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _source_name(name: str) -> None:
    # Source trees may contain .gitignore. Storage record paths have a different,
    # deliberately narrower alphabet; do not use that alphabet for source files.
    parts = name.split("/")
    if (not 1 <= len(name) <= 1024 or len(parts) > 16
            or any(part in (".", "..") or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", part) is None
                   for part in parts)):
        raise RunnerError("Source ZIP member name is invalid.")


def _source_files(contents: bytes) -> tuple[tuple[str, bytes], ...]:
    """Bound ZIP directory allocation, then let zipfile check each member's CRC."""
    if len(contents) < _END.size:
        raise RunnerError("Source ZIP end record is missing.")
    signature, disk, directory_disk, disk_count, count, size, offset, comment = _END.unpack_from(
        contents, len(contents) - _END.size)
    if (signature != b"PK\x05\x06" or disk != 0 or directory_disk != 0
            or not 1 <= count <= MAX_SOURCE_FILES or disk_count != count or comment != 0
            or not count * 46 <= size <= count * (46 + 1024)
            or offset + size + _END.size != len(contents)
            or contents[-(_END.size + 20):-(_END.size + 16)] == b"PK\x06\x07"):
        raise RunnerError("Source ZIP directory exceeds its fixed bounds.")
    files, names, total = [], set(), 0
    with ZipFile(BytesIO(contents), allowZip64=False) as archive:
        members = archive.infolist()
        if len(members) != count or archive.comment:
            raise RunnerError("Source ZIP count or comment is invalid.")
        for member in members:
            name = member.filename
            _source_name(name)
            total += member.file_size
            if (name in names or member.orig_filename != name or member.is_dir()
                    or member.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
                    or member.flag_bits & 1 or member.extra or member.comment
                    or stat.S_IFMT(member.external_attr >> 16) not in (0, stat.S_IFREG)
                    or not 0 <= member.file_size <= MAX_SOURCE_FILE_BYTES
                    or total > MAX_SOURCE_TOTAL_BYTES):
                raise RunnerError("Source ZIP members exceed their fixed contract.")
            with archive.open(member) as stream:
                body = stream.read(MAX_SOURCE_FILE_BYTES + 1)
            if len(body) != member.file_size:
                raise RunnerError("Source ZIP member size is invalid.")
            names.add(name)
            files.append((name, body))
    # A file cannot also serve as an ancestor directory of another member.
    if any("/".join(name.split("/")[:index]) in names
           for name in names for index in range(1, len(name.split("/")))):
        raise RunnerError("Source ZIP file paths conflict.")
    return tuple(files)


def _extract_source(files: tuple[tuple[str, bytes], ...], root: Path) -> None:
    root.mkdir(mode=0o700)
    for name, contents in files:
        path = root / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(contents)


def _source_unchanged(root: Path, files: tuple[tuple[str, bytes], ...]) -> bool:
    expected = {name: (len(contents), sha256(contents).digest()) for name, contents in files}
    found, directory_count = set(), 0
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            return False
        for folder, directories, filenames in os.walk(root, followlinks=False):
            directory_count += 1
            if directory_count > MAX_SOURCE_FILES * 16:
                return False
            if any((Path(folder) / name).is_symlink() for name in directories):
                return False
            for name in filenames:
                path = Path(folder) / name
                relative = str(path.relative_to(root))
                if relative not in expected:
                    return False
                size, digest = expected[relative]
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                        return False
                    contents = stream.read(size + 1)
                if len(contents) != size or sha256(contents).digest() != digest:
                    return False
                found.add(relative)
        return found == set(expected)
    except OSError:
        return False


def _publish(root: Path, path: str, contents: bytes) -> StoredArtifact:
    expected = StoredArtifact(path, sha256(contents).hexdigest(), len(contents))
    try:
        return publish_bytes(root, path, contents)
    except ArtifactExistsError:
        read_verified(root, expected)
        return expected


def _content_file(root: Path, group: str, suffix: str, contents: bytes) -> StoredArtifact:
    return _publish(root, f"qa/{group}/{sha256(contents).hexdigest()}.{suffix}", contents)


def _runtime_info(suites: tuple[str, ...]) -> dict:
    packages = {}
    for name in ("pytest", "pytest-reportlog", "iniconfig", "pluggy", "packaging",
                 "pygments", "Pillow", "jsonschema", "httpx", "runwayml", "playwright"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    if packages["pytest"] != PYTEST_VERSION or packages["pytest-reportlog"] != REPORTLOG_VERSION:
        raise RunnerError("Recorded tests require the pinned pytest and reportlog versions.")
    commands = {"ffmpeg": ["ffmpeg", "-version"], "ffprobe": ["ffprobe", "-version"]}
    if "ui" in suites:
        commands["browser"] = [os.environ.get("GEN_REALITY_BROWSER_BIN", "/usr/bin/chromium"), "--version"]
    tools = {}
    with TemporaryDirectory(prefix="gen-reality-test-versions-") as folder:
        for name, argv in commands.items():
            captured = test_results.run_process(argv, Path(folder), test_results.child_environment(), timeout=10)
            first_line = captured.stdout.splitlines()[:1]
            tools[name] = (first_line[0][:1024].decode("utf-8", errors="replace")
                           if captured.exit_code == 0 and not captured.issues and first_line else None)
    return {"python": platform.python_version(), "packages": packages, "tools": tools}


def _base_receipt(run_id: str, suite: str, source: StoredArtifact,
                  registry: StoredArtifact, runtime: dict) -> dict:
    return {
        "schemaVersion": "1.4", "manifestType": "recorded_test_execution",
        "runId": run_id, "suite": suite, "source": asdict(source), "registry": asdict(registry),
        "runtime": runtime, "executed": False, "argv": None,
        "startedAtUnix": None, "finishedAtUnix": None, "elapsedSeconds": None,
        "exitCode": None, "issues": [], "sourceUnchanged": None, "reportsComplete": False,
        "artifacts": {name: None for name in ("stdout", "stderr", "junit", "events", "cases")},
    }


def _record_suite(root: Path, files: tuple[tuple[str, bytes], ...], *, run_id: str,
                  suite: str, source: StoredArtifact, registry: StoredArtifact,
                  runtime: dict, execution_id: str) -> RecordedSuite:
    data = _base_receipt(run_id, suite, source, registry, runtime)
    with TemporaryDirectory(prefix="gen-reality-recorded-tests-") as folder:
        work = Path(folder)
        source_dir, reports = work / "source", work / "reports"
        _extract_source(files, source_dir)
        reports.mkdir(mode=0o700)
        selected_path, marker = SUITES[suite]
        if not (source_dir / selected_path).is_dir():
            data["issues"] = ["SUITE_MISSING"]
        else:
            argv = [sys.executable, *PYTEST_OPTIONS,
                    "-o", "markers=live_provider: explicitly authorized provider work",
                    "--junitxml=../reports/junit.xml", "--report-log=../reports/events.jsonl",
                    "-m", marker, selected_path]
            capture = test_results.run_process(
                argv, source_dir, test_results.child_environment(),
                report_paths=(reports / "junit.xml", reports / "events.jsonl"),
            )
            data.update(executed=capture.executed, argv=argv, startedAtUnix=capture.started,
                        finishedAtUnix=capture.finished, elapsedSeconds=capture.elapsed,
                        exitCode=capture.exit_code, issues=list(capture.issues))
            payloads = {"stdout": capture.stdout, "stderr": capture.stderr,
                        "junit": test_results.read_report(reports / "junit.xml"),
                        "events": test_results.read_report(reports / "events.jsonl")}
            try:
                cases, issues = test_results.parse_reports(payloads["events"], payloads["junit"], capture.exit_code)
                payloads["cases"] = _json_bytes(cases)
                if len(payloads["cases"]) > test_results.MAX_REPORT_BYTES:
                    raise ValueError("Normalized test facts exceed their limit.")
                data["issues"].extend(issues)
                data["reportsComplete"] = capture.executed and not any(
                    issue.startswith("PROCESS_") or issue == "OUTPUT_LIMIT" for issue in capture.issues)
            except Exception:
                payloads["cases"] = None
                data["issues"].append("REPORT_INVALID")
            data["sourceUnchanged"] = _source_unchanged(source_dir, files)
            if not data["sourceUnchanged"]:
                data["issues"].append("SOURCE_CHANGED")
            suffixes = {"stdout": "stdout.txt", "stderr": "stderr.txt", "junit": "junit.xml",
                        "events": "events.jsonl", "cases": "cases.json"}
            for name, contents in payloads.items():
                if contents is not None:
                    data["artifacts"][name] = asdict(_content_file(root, "test-files", suffixes[name], contents))
    data["issues"] = list(dict.fromkeys(data["issues"]))
    record = _publish(root, f"qa/test-runs/{execution_id}/{suite}.json", _json_bytes(data))
    return RecordedSuite(suite, record, data["executed"], data["exitCode"], tuple(data["issues"]))


def record_offline_tests(source_root: Path, source_ref: StoredArtifact, evidence_root: Path, *,
                         run_id: str, suites: tuple[str, ...] = DEFAULT_SUITES) -> RecordedTestRun:
    """Execute approved source bytes, with a fresh extraction for each fixed suite.

    The trusted operator supplies the source hash. This executes source code;
    it is not a sandbox for uploaded or untrusted programs. Missing suites and
    failed test runs receive records without preventing the other requested
    suites. Raw pytest facts do not establish acceptance coverage or release.
    """
    try:
        BudgetLedger._run_id(run_id)
        if (type(suites) is not tuple or not suites or len(suites) > len(SUITES)
                or any(type(name) is not str or name not in SUITES for name in suites)
                or len(set(suites)) != len(suites)):
            raise RunnerError("Select each fixed offline suite at most once.")
        if (not isinstance(source_ref, StoredArtifact) or type(source_ref.size) is not int
                or not 0 < source_ref.size <= MAX_SOURCE_BYTES):
            raise RunnerError("Source ZIP size exceeds its limit.")
        contents = read_verified(Path(source_root), source_ref)
        files = _source_files(contents)
        runtime = _runtime_info(suites)
        root = Path(evidence_root) / run_id
        source = _content_file(root, "test-sources", "zip", contents)
        registry = _content_file(root, "test-registries", "json", registry_bytes())
        execution_id = uuid4().hex
        results = tuple(_record_suite(root, files, run_id=run_id, suite=suite, source=source,
                                      registry=registry, runtime=runtime, execution_id=execution_id)
                        for suite in suites)
        data = {"schemaVersion": "1.4", "manifestType": "recorded_test_run", "runId": run_id,
                "source": asdict(source), "registry": asdict(registry),
                "suites": [asdict(result) for result in results]}
        record = _publish(root, f"qa/test-runs/{execution_id}/run.json", _json_bytes(data))
        return RecordedTestRun(run_id, source, registry, results, record)
    except Exception:
        raise RunnerError("Offline test evidence could not be recorded.") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-size", type=int, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", action="append", choices=tuple(SUITES))
    arguments = parser.parse_args(argv)
    try:
        source = StoredArtifact(arguments.source_path, arguments.source_sha256, arguments.source_size)
        result = record_offline_tests(
            arguments.source_root, source, arguments.evidence_root, run_id=arguments.run_id,
            suites=tuple(arguments.suite) if arguments.suite else DEFAULT_SUITES,
        )
    except Exception:
        print("Offline tests could not be recorded. Check the approved source, tools, and private storage.",
              file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if all(item.executed and item.exit_code == 0 and not item.issues for item in result.suites) else 1


if __name__ == "__main__":
    raise SystemExit(main())
