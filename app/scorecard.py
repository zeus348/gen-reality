"""Bind an entered scorecard to saved review evidence without provider calls."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import sys
import time

from core.budget import BudgetError, BudgetLedger
from core.storage import StoredArtifact, _parent_directory
from features.inference.submission import SubmissionCoordinator
from features.review.bind import bind_scorecard
from features.review.score import MAX_SCORECARD_BYTES, ScorecardError, assess_scorecard


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("Scorecard command arguments are invalid.")


def _read_scorecard(path: Path) -> bytes:
    """Read a bounded regular file without following any symlink components."""
    path = Path(path).absolute()
    with _parent_directory(path.parent, [path.name], create=False) as directory:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_SCORECARD_BYTES:
                raise ScorecardError("Scorecard must be a regular file containing 1 byte to 64 KiB.")
            contents = file.read(MAX_SCORECARD_BYTES + 1)
    if len(contents) != info.st_size:
        raise ScorecardError("Scorecard size changed while it was read.")
    return contents


def bind_scorecard_file(output_root: Path, ledger_path: Path, review_manifest: StoredArtifact,
                        scorecard_path: Path, *, run_id: str, attempt_id: str,
                        clock=time.time) -> StoredArtifact:
    """Validate entered bytes before opening the existing trusted ledger.

    No reviewer fields are supplied or changed here. The coordinator can recover
    abandoned SUBMITTING jobs as SUBMISSION_UNKNOWN, with their reserves held.
    This operation creates no approval, reservation, or provider request.
    """
    contents = _read_scorecard(scorecard_path)
    assessment = assess_scorecard(contents, run_id=run_id, attempt_id=attempt_id)
    if assessment.rubric_verdict == "NOT_RUN":
        raise ScorecardError("A complete reviewer name, UTC time, and signature are required.")
    if not stat.S_ISREG(Path(ledger_path).lstat().st_mode):
        raise BudgetError("Scorecard binding requires an existing trusted budget database.")
    ledger = BudgetLedger(ledger_path, clock=clock)
    with SubmissionCoordinator(ledger) as coordinator:
        return bind_scorecard(coordinator, output_root, review_manifest, contents,
                              run_id=run_id, attempt_id=attempt_id)


def main(argv=None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--review-path", required=True, help="Path relative to the run directory.")
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--review-size", type=int, required=True)
    parser.add_argument("--scorecard-file", type=Path, required=True)
    try:
        arguments = parser.parse_args(argv)
        manifest = StoredArtifact(arguments.review_path, arguments.review_sha256, arguments.review_size)
        binding = bind_scorecard_file(arguments.output_root, arguments.ledger, manifest, arguments.scorecard_file,
                                      run_id=arguments.run_id, attempt_id=arguments.attempt_id)
        result = {"runId": arguments.run_id, "attemptId": arguments.attempt_id, "binding": asdict(binding)}
    except Exception:
        print("Scorecard could not be verified or bound to this attempt.", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
