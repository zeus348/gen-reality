"""Submit one saved request using an existing operator-approved budget."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
import time

import httpx

from core import run_start
from core.budget import BudgetError, BudgetLedger, IdempotencyConflict
from core.storage import StoredArtifact
from features.inference.preflight import read_checked_request
from features.inference.send import submit_checked
from features.inference.submission import Submission, SubmissionCoordinator


def submit_saved_request(output_root: Path, ledger_path: Path, evidence: StoredArtifact, *,
                         run_id: str, attempt_id: str, api_key_env: str = "RUNWAY_API_KEY",
                         clock=time.time, transport: httpx.BaseTransport | None = None) -> Submission:
    """Restore, reserve if needed, and send through the checked submission path.

    This operator command does not create approval, refresh evidence, or upload
    media. Existing reservations retain their original identity. A repeat of
    known or uncertain work needs no current key, fresh evidence, or run-start
    file. New work requires saved start evidence and a capped approval.
    """
    BudgetLedger._run_id(run_id)
    BudgetLedger._attempt_id(attempt_id)
    if type(api_key_env) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", api_key_env) is None:
        raise ValueError("Supply an environment variable name for the API key.")
    checked = read_checked_request(Path(output_root) / run_id, evidence,
                                   run_id=run_id, attempt_id=attempt_id)
    if not Path(ledger_path).is_file():
        raise BudgetError("Submission requires an existing approved budget database.")
    ledger = BudgetLedger(ledger_path, clock=clock)
    with SubmissionCoordinator(ledger) as coordinator:
        with ledger._connection() as connection:
            approval = connection.execute(
                "SELECT cost_cents, started_unix, deadline_unix FROM approved_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            reservation = connection.execute("SELECT * FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone()
        if approval is None or approval["cost_cents"] != checked.frozen.estimated_cents:
            raise BudgetError("Saved request needs an existing approval with the same quote.")
        if reservation is not None:
            if (reservation["run_id"] != run_id
                    or reservation["payload_sha256"] != checked.frozen.payload_sha256
                    or reservation["cost_cents"] != checked.frozen.estimated_cents):
                raise IdempotencyConflict("Saved request does not match the existing reservation.")
            if coordinator.get(attempt_id).state != "RESERVED":
                return submit_checked(coordinator, checked, output_root, "", transport=transport)
        started = run_start.read_run_start(output_root, run_id=run_id)
        if not (started.started_unix <= approval["started_unix"]
                < approval["deadline_unix"] <= started.deadline_unix):
            raise BudgetError("New submission needs approval within the saved experiment deadline.")
        checked.require_current(ledger._now())
        api_key = os.environ.get(api_key_env, "")
        if not api_key.strip():
            raise ValueError("The named environment variable must contain a Runway API key.")
        if reservation is None:
            ledger.reserve(run_id, f"saved-request:{run_id}:{attempt_id}",
                           checked.frozen.payload_sha256, attempt_id=attempt_id)
        return submit_checked(coordinator, checked, output_root, api_key, transport=transport)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--check-path", required=True, help="Path relative to the run directory.")
    parser.add_argument("--check-sha256", required=True)
    parser.add_argument("--check-size", type=int, required=True)
    parser.add_argument("--api-key-env", default="RUNWAY_API_KEY", help="Environment variable name; never the key value.")
    arguments = parser.parse_args(argv)
    try:
        evidence = StoredArtifact(arguments.check_path, arguments.check_sha256, arguments.check_size)
        result = submit_saved_request(arguments.output_root, arguments.ledger, evidence,
                                      run_id=arguments.run_id, attempt_id=arguments.attempt_id,
                                      api_key_env=arguments.api_key_env)
    except Exception:
        # SDK, storage, and database errors can include sensitive input values.
        print("Submission could not proceed. Check saved evidence, budget, and the API key environment variable.",
              file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), sort_keys=True))
    return 2 if result.state == "SUBMISSION_UNKNOWN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
